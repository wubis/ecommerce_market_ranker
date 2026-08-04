"""Protocol-safe ranking evaluation and validation-only champion promotion."""

from __future__ import annotations

import json
import math
import resource
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from market_rank.artifacts import (
    ArtifactDependency,
    ArtifactExistsError,
    ArtifactStore,
    LoadedArtifact,
)
from market_rank.config import RankingEvaluationConfig, ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.evaluation.metrics import (
    CLOSED_POOL_PROTOCOL,
    END_TO_END_PROTOCOL,
    EsciLabel,
    Judgment,
    evaluate_ranked_products,
)
from market_rank.evaluation.retrieval import (
    CANDIDATE_DIRECTORY,
    RETRIEVAL_EVALUATION_FILENAME,
    RetrievalEvaluationManifest,
    load_retrieval_evaluation_manifest,
)
from market_rank.evaluation.retrieval import (
    QUERY_METRIC_DIRECTORY as RETRIEVAL_QUERY_METRIC_DIRECTORY,
)
from market_rank.features.artifact import (
    CANDIDATE_MATRIX_DIRECTORY,
    CLOSED_MATRIX_DIRECTORY,
    FEATURE_ARTIFACT_FILENAME,
    PARSED_QUERIES_FILENAME,
    RankingFeatureManifest,
    load_ranking_feature_manifest,
)
from market_rank.features.registry import FEATURE_NAMES
from market_rank.ranking.population import stable_rank_predictions
from market_rank.ranking.training import (
    RANKING_MODELS_FILENAME,
    LoadedRankers,
    RankingModelsManifest,
    load_rankers,
    load_ranking_models_manifest,
    ranking_models_artifact_id,
)

RANKING_EVALUATION_FILENAME = "ranking-evaluation.json"
ACTIVE_RELEVANCE_FILENAME = "active-relevance.json"
RUN_FILENAME = "run.json"
PREDICTIONS_FILENAME = "predictions.parquet"
QUERY_METRICS_FILENAME = "query-metrics.parquet"
METRICS_FILENAME = "metrics.parquet"
COMPARISONS_FILENAME = "comparisons.parquet"
FAILURE_ANALYSIS_FILENAME = "failure-analysis.parquet"

Profile = Literal["development", "portfolio"]
ProjectEvaluationSplit = Literal["validation", "test"]
EvaluationStage = Literal["rrf", "pointwise", "lambdamart"]
Protocol = Literal["closed_pool_task1_v1", "end_to_end_diagnostic_v1"]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]

_STAGES: tuple[EvaluationStage, ...] = ("rrf", "pointwise", "lambdamart")
_LABELS: dict[int, EsciLabel] = {0: "I", 1: "C", 2: "S", 3: "E"}
_GAINS: dict[EsciLabel, float] = {"I": 0.0, "C": 0.01, "S": 0.1, "E": 1.0}
_THRESHOLDS: tuple[tuple[str, frozenset[EsciLabel]], ...] = (
    ("exact", frozenset({"E"})),
    ("exact_substitute", frozenset({"E", "S"})),
)
_HIGHER_IS_BETTER = frozenset(
    {
        "ndcg_official_gain",
        "precision",
        "map",
        "mrr",
        "exact_hit",
        "judged_recall",
        "judged_mrr",
        "known_judgment_coverage",
    }
)


class RankingEvaluationError(RuntimeError):
    """Base error for ranking evaluation, experiment tracking, and promotion."""


class RankingEvaluationBuildError(RankingEvaluationError):
    """Raised when compatible parents cannot produce a valid ranking evaluation."""


class RankingEvaluationValidationError(RankingEvaluationError):
    """Raised when a persisted evaluation or active contract is incompatible."""


class RankingEvaluationResourceError(RankingEvaluationBuildError):
    """Raised when evaluation exceeds the configured RSS limit."""

    def __init__(self, measurement: RankingEvaluationResourceMeasurement) -> None:
        super().__init__(
            f"ranking evaluation peak RSS {measurement.peak_rss_bytes} exceeds "
            f"the {measurement.rss_limit_bytes}-byte limit"
        )
        self.measurement = measurement


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RankingEvaluationResourceMeasurement(_StrictModel):
    load_peak_rss_bytes: int = Field(strict=True, ge=0)
    evaluation_peak_rss_bytes: int = Field(strict=True, ge=0)
    promotion_peak_rss_bytes: int = Field(strict=True, ge=0)
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    artifact_payload_bytes: int = Field(strict=True, ge=0)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        observed = max(
            self.load_peak_rss_bytes,
            self.evaluation_peak_rss_bytes,
            self.promotion_peak_rss_bytes,
        )
        if self.peak_rss_bytes != observed:
            raise ValueError("ranking-evaluation peak RSS differs from phase observations")
        if self.passed != (self.peak_rss_bytes <= self.rss_limit_bytes):
            raise ValueError("ranking-evaluation resource status differs from the RSS gate")
        return self


class ChampionCandidate(_StrictModel):
    stage: EvaluationStage
    eligible: bool = Field(strict=True)
    selection_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ndcg_by_cutoff: tuple[tuple[int, float], ...] = Field(min_length=1)
    delta_from_rrf: tuple[tuple[int, float], ...] = Field(min_length=1)
    decision_reason: str = Field(strict=True, min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        cutoffs = tuple(cutoff for cutoff, _ in self.ndcg_by_cutoff)
        delta_cutoffs = tuple(cutoff for cutoff, _ in self.delta_from_rrf)
        if cutoffs != tuple(sorted(set(cutoffs))) or delta_cutoffs != cutoffs:
            raise ValueError("champion evidence cutoffs must be aligned, unique, and sorted")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for _, value in self.ndcg_by_cutoff
        ):
            raise ValueError("champion NDCG values must be finite in [0,1]")
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0 for _, value in self.delta_from_rrf
        ):
            raise ValueError("champion NDCG deltas must be finite in [-1,1]")
        if not math.isclose(
            self.selection_score,
            sum(value for _, value in self.ndcg_by_cutoff) / len(self.ndcg_by_cutoff),
            abs_tol=1e-12,
        ):
            raise ValueError("champion selection score must be the mean configured NDCG")
        return self


class ActiveRelevanceContract(_StrictModel):
    """Exactly one relevance stage selected without consulting project test."""

    schema_version: Literal[1] = 1
    contract_version: Literal["active-relevance-v1"] = "active-relevance-v1"
    selected_stage: EvaluationStage
    selected_model_id: Literal["pointwise", "lambdamart"] | None = None
    selection_protocol: Literal["closed_pool_task1_v1"] = CLOSED_POOL_PROTOCOL
    selection_split: Literal["validation"] = "validation"
    test_evaluated: Literal[False] = False
    selection_metric: Literal["mean_ndcg_official_gain_across_configured_cutoffs"] = (
        "mean_ndcg_official_gain_across_configured_cutoffs"
    )
    selection_cutoffs: tuple[int, ...] = Field(min_length=1)
    selection_query_count: int = Field(strict=True, ge=1)
    active_score_field: Literal["hybrid_rrf_score", "pointwise_score", "lambdamart_score"]
    active_score_comparable: Literal[True] = True
    fallback_contract: Literal["rrf-on-model-failure-v1"] = "rrf-on-model-failure-v1"
    fallback_stage: Literal["rrf"] = "rrf"
    ranking_models_artifact_id: str = Field(strict=True, min_length=1)
    ranking_models_manifest_sha256: Sha256Digest
    feature_artifact_id: str = Field(strict=True, min_length=1)
    feature_registry_sha256: Sha256Digest
    candidates: tuple[ChampionCandidate, ChampionCandidate, ChampionCandidate]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if tuple(candidate.stage for candidate in self.candidates) != _STAGES:
            raise ValueError("champion candidates must be ordered RRF, pointwise, LambdaMART")
        if any(
            tuple(cutoff for cutoff, _ in candidate.ndcg_by_cutoff) != self.selection_cutoffs
            for candidate in self.candidates
        ):
            raise ValueError("champion candidate cutoffs differ from the selection contract")
        if any(abs(delta) > 1e-12 for _, delta in self.candidates[0].delta_from_rrf):
            raise ValueError("RRF baseline deltas must be zero")
        expected_model = None if self.selected_stage == "rrf" else self.selected_stage
        if self.selected_model_id != expected_model:
            raise ValueError("active model identity differs from the selected stage")
        expected_score = {
            "rrf": "hybrid_rrf_score",
            "pointwise": "pointwise_score",
            "lambdamart": "lambdamart_score",
        }[self.selected_stage]
        if self.active_score_field != expected_score:
            raise ValueError("active score field differs from the selected stage")
        if not next(
            candidate for candidate in self.candidates if candidate.stage == self.selected_stage
        ).eligible:
            raise ValueError("selected active relevance candidate is ineligible")
        return self


class AblationRecord(_StrictModel):
    ablation_id: Literal["ABL-01", "ABL-02", "ABL-03", "ABL-04", "ABL-05"]
    status: Literal["inherited", "evaluated"]
    protocol: str = Field(strict=True, min_length=1)
    evidence_artifact_id: str = Field(strict=True, min_length=1)
    query_count: int = Field(strict=True, ge=1)


class RankingExperimentRun(_StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(strict=True, min_length=1)
    status: Literal["completed"] = "completed"
    config_sha256: Sha256Digest
    profile: Profile
    selection_split: Literal["validation"] = "validation"
    test_evaluated: Literal[False] = False
    protocols: tuple[Literal["closed_pool_task1_v1"], Literal["end_to_end_diagnostic_v1"]] = (
        CLOSED_POOL_PROTOCOL,
        END_TO_END_PROTOCOL,
    )
    ranking_models_artifact_id: str = Field(strict=True, min_length=1)
    ranking_models_manifest_sha256: Sha256Digest
    ranking_features_artifact_id: str = Field(strict=True, min_length=1)
    ranking_features_manifest_sha256: Sha256Digest
    retrieval_evaluation_artifact_id: str = Field(strict=True, min_length=1)
    retrieval_evaluation_manifest_sha256: Sha256Digest
    active_relevance: ActiveRelevanceContract
    ablations: tuple[AblationRecord, ...]
    metric_rows: int = Field(strict=True, ge=1)
    comparison_rows: int = Field(strict=True, ge=1)
    failure_analysis_rows: int = Field(strict=True, ge=1)
    resource: RankingEvaluationResourceMeasurement

    @model_validator(mode="after")
    def validate_ablations(self) -> Self:
        if tuple(item.ablation_id for item in self.ablations) != (
            "ABL-01",
            "ABL-02",
            "ABL-03",
            "ABL-04",
            "ABL-05",
        ):
            raise ValueError("experiment must account for required ABL-01 through ABL-05")
        return self


class RankingEvaluationCheck(_StrictModel):
    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class RankingEvaluationManifest(_StrictModel):
    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    profile: Profile
    component_version: Literal["ranking-eval-v1"] = "ranking-eval-v1"
    selection_split: Literal["validation"] = "validation"
    test_evaluated: Literal[False] = False
    protocols: tuple[Literal["closed_pool_task1_v1"], Literal["end_to_end_diagnostic_v1"]] = (
        CLOSED_POOL_PROTOCOL,
        END_TO_END_PROTOCOL,
    )
    closed_population_id: Literal["esci_task1_us_judged_pool_validation_v1"] = (
        "esci_task1_us_judged_pool_validation_v1"
    )
    diagnostic_population_id: Literal["end_to_end_hybrid_union_validation_v1"] = (
        "end_to_end_hybrid_union_validation_v1"
    )
    ranking_models_artifact_id: str = Field(strict=True, min_length=1)
    ranking_models_manifest_sha256: Sha256Digest
    ranking_features_artifact_id: str = Field(strict=True, min_length=1)
    ranking_features_manifest_sha256: Sha256Digest
    retrieval_evaluation_artifact_id: str = Field(strict=True, min_length=1)
    retrieval_evaluation_manifest_sha256: Sha256Digest
    closed_cutoffs: tuple[int, ...] = Field(min_length=1)
    diagnostic_cutoffs: tuple[int, ...] = Field(min_length=1)
    bootstrap_method: Literal["normalized-query-group-v1"] = "normalized-query-group-v1"
    bootstrap_replicates: int = Field(strict=True, ge=100)
    bootstrap_seed: int = Field(strict=True, ge=0)
    validation_queries: int = Field(strict=True, ge=1)
    normalized_query_groups: int = Field(strict=True, ge=1)
    closed_rows: int = Field(strict=True, ge=1)
    candidate_rows: int = Field(strict=True, ge=0)
    prediction_rows: int = Field(strict=True, ge=1)
    query_metric_rows: int = Field(strict=True, ge=1)
    aggregate_metric_rows: int = Field(strict=True, ge=1)
    comparison_rows: int = Field(strict=True, ge=1)
    failure_analysis_rows: int = Field(strict=True, ge=1)
    active_relevance: ActiveRelevanceContract
    resource: RankingEvaluationResourceMeasurement
    checks: tuple[RankingEvaluationCheck, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        for cutoffs in (self.closed_cutoffs, self.diagnostic_cutoffs):
            if cutoffs != tuple(sorted(set(cutoffs))):
                raise ValueError("ranking-evaluation cutoffs must be unique and sorted")
        check_ids = tuple(check.check_id for check in self.checks)
        if check_ids != tuple(sorted(set(check_ids))):
            raise ValueError("ranking-evaluation checks must be unique and sorted")
        if (
            self.active_relevance.ranking_models_artifact_id != self.ranking_models_artifact_id
            or self.active_relevance.ranking_models_manifest_sha256
            != self.ranking_models_manifest_sha256
            or self.active_relevance.feature_artifact_id != self.ranking_features_artifact_id
        ):
            raise ValueError("active-relevance lineage differs from evaluation lineage")
        return self


@dataclass(frozen=True, slots=True)
class RankingEvaluationBuildResult:
    artifact: LoadedArtifact
    manifest: RankingEvaluationManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class FrozenRankingEvaluation:
    """One fixed-split evaluation that performs no champion selection."""

    split: ProjectEvaluationSplit
    predictions: pl.DataFrame
    query_metrics: pl.DataFrame
    aggregate_metrics: pl.DataFrame
    comparisons: pl.DataFrame
    failure_analysis: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _Dependencies:
    models: LoadedArtifact
    models_manifest: RankingModelsManifest
    features: LoadedArtifact
    features_manifest: RankingFeatureManifest
    retrieval: LoadedArtifact
    retrieval_manifest: RetrievalEvaluationManifest


@dataclass(frozen=True, slots=True)
class _EvaluationInputs:
    closed: pl.DataFrame
    candidates: pl.DataFrame
    contexts: pl.DataFrame


_PREDICTION_SCHEMA = pl.Schema(
    {
        "profile": pl.String,
        "protocol": pl.String,
        "population_id": pl.String,
        "query_id": pl.Int64,
        "normalized_query_sha256": pl.String,
        "project_split": pl.String,
        "stage": pl.String,
        "product_id": pl.String,
        "score": pl.Float64,
        "rank": pl.UInt32,
        "judged": pl.Boolean,
        "label_id": pl.UInt8,
        "gain": pl.Float32,
        "active_relevance": pl.Boolean,
    }
)

_QUERY_METRIC_SCHEMA = pl.Schema(
    {
        "profile": pl.String,
        "protocol": pl.String,
        "population_id": pl.String,
        "query_id": pl.Int64,
        "normalized_query_sha256": pl.String,
        "project_split": pl.String,
        "source": pl.String,
        "query_length_bucket": pl.String,
        "lexical_specificity_bucket": pl.String,
        "brand_presence": pl.String,
        "color_presence": pl.String,
        "model_presence": pl.String,
        "compatibility_presence": pl.String,
        "judgment_composition": pl.String,
        "stage": pl.String,
        "threshold_id": pl.String,
        "metric": pl.String,
        "cutoff": pl.UInt32,
        "value": pl.Float64,
        "returned_count": pl.UInt32,
        "judged_count": pl.UInt32,
        "unjudged_count": pl.UInt32,
        "relevant_judgment_count": pl.UInt32,
        "empty_result": pl.Boolean,
    }
)

_AGGREGATE_SCHEMA = pl.Schema(
    {
        "profile": pl.String,
        "protocol": pl.String,
        "population_id": pl.String,
        "stage": pl.String,
        "threshold_id": pl.String,
        "metric": pl.String,
        "cutoff": pl.UInt32,
        "slice_dimension": pl.String,
        "slice_value": pl.String,
        "mean": pl.Float64,
        "median": pl.Float64,
        "ci95_lower": pl.Float64,
        "ci95_upper": pl.Float64,
        "query_count": pl.UInt32,
        "normalized_query_groups": pl.UInt32,
        "empty_query_count": pl.UInt32,
        "returned_count": pl.UInt64,
        "judged_count": pl.UInt64,
        "unjudged_count": pl.UInt64,
        "relevant_judgment_count": pl.UInt64,
        "bootstrap_replicates": pl.UInt32,
        "bootstrap_method": pl.String,
    }
)

_COMPARISON_SCHEMA = pl.Schema(
    {
        "profile": pl.String,
        "protocol": pl.String,
        "ablation_id": pl.String,
        "treatment_stage": pl.String,
        "baseline_stage": pl.String,
        "metric_direction": pl.String,
        "threshold_id": pl.String,
        "metric": pl.String,
        "cutoff": pl.UInt32,
        "mean_improvement": pl.Float64,
        "median_improvement": pl.Float64,
        "ci95_lower": pl.Float64,
        "ci95_upper": pl.Float64,
        "query_count": pl.UInt32,
        "normalized_query_groups": pl.UInt32,
        "win_count": pl.UInt32,
        "tie_count": pl.UInt32,
        "loss_count": pl.UInt32,
        "bootstrap_replicates": pl.UInt32,
        "bootstrap_method": pl.String,
    }
)

_FAILURE_SCHEMA = pl.Schema(
    {
        "profile": pl.String,
        "protocol": pl.String,
        "ablation_id": pl.String,
        "query_id": pl.Int64,
        "normalized_query_sha256": pl.String,
        "source": pl.String,
        "query_length_bucket": pl.String,
        "lexical_specificity_bucket": pl.String,
        "brand_presence": pl.String,
        "color_presence": pl.String,
        "model_presence": pl.String,
        "compatibility_presence": pl.String,
        "judgment_composition": pl.String,
        "metric": pl.String,
        "cutoff": pl.UInt32,
        "baseline_stage": pl.String,
        "treatment_stage": pl.String,
        "baseline_value": pl.Float64,
        "treatment_value": pl.Float64,
        "delta": pl.Float64,
        "outcome": pl.String,
        "selection_method": pl.String,
    }
)


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def ranking_evaluation_artifact_id(
    release: ResolvedReleaseManifest, config_sha256: str, profile: Profile
) -> str:
    return "/".join(
        (
            "ranking-evaluation",
            release.manifest.dataset_version,
            profile,
            "ranking-eval-v1",
            config_sha256,
        )
    )


def load_ranking_evaluation_manifest(path: Path) -> RankingEvaluationManifest:
    try:
        return RankingEvaluationManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise RankingEvaluationValidationError(
            f"cannot load ranking-evaluation manifest {path}: {exc}"
        ) from exc


def load_active_relevance_contract(path: Path) -> ActiveRelevanceContract:
    try:
        return ActiveRelevanceContract.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise RankingEvaluationValidationError(
            f"cannot load active-relevance contract {path}: {exc}"
        ) from exc


def _load_dependencies(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    store: ArtifactStore,
) -> _Dependencies:
    try:
        models = store.load(ranking_models_artifact_id(release, config.sha256, profile))
        models_manifest = load_ranking_models_manifest(models.path / RANKING_MODELS_FILENAME)
        features = store.load(models_manifest.feature_artifact_id)
        features_manifest = load_ranking_feature_manifest(features.path / FEATURE_ARTIFACT_FILENAME)
        retrieval = store.load(features_manifest.retrieval_evaluation_artifact_id)
        retrieval_manifest = load_retrieval_evaluation_manifest(
            retrieval.path / RETRIEVAL_EVALUATION_FILENAME
        )
    except (OSError, RuntimeError) as exc:
        raise RankingEvaluationBuildError(
            "compatible Goldfish 009-011 artifacts are required before ranking evaluation"
        ) from exc
    if (
        models_manifest.artifact_id != models.manifest.artifact_id
        or models_manifest.config_sha256 != config.sha256
        or models_manifest.profile != profile
        or models_manifest.feature_artifact_id != features.manifest.artifact_id
        or models_manifest.feature_manifest_sha256 != features.manifest_sha256
        or features_manifest.artifact_id != features.manifest.artifact_id
        or features_manifest.config_sha256 != config.sha256
        or features_manifest.profile != profile
        or features_manifest.retrieval_evaluation_artifact_id != retrieval.manifest.artifact_id
        or features_manifest.retrieval_evaluation_manifest_sha256 != retrieval.manifest_sha256
        or retrieval_manifest.artifact_id != retrieval.manifest.artifact_id
        or retrieval_manifest.profile != profile
        or models_manifest.feature_names != FEATURE_NAMES
    ):
        raise RankingEvaluationBuildError("ranking-evaluation parent lineage is incompatible")
    return _Dependencies(
        models,
        models_manifest,
        features,
        features_manifest,
        retrieval,
        retrieval_manifest,
    )


def _artifact_dependency(dependencies: _Dependencies) -> tuple[ArtifactDependency, ...]:
    return (
        ArtifactDependency(
            artifact_id=dependencies.models.manifest.artifact_id,
            manifest_sha256=dependencies.models.manifest_sha256,
        ),
    )


def _context_rows(
    closed: pl.DataFrame,
    retrieval_context: pl.DataFrame,
    parsed: pl.DataFrame,
) -> pl.DataFrame:
    median_specificity = float(
        closed.group_by("query_id")
        .agg(pl.col("query_lexical_specificity").first())
        .select(pl.col("query_lexical_specificity").median())
        .item()
    )
    label_counts = closed.group_by("query_id").agg(
        *(pl.col("label_id").eq(label).sum().alias(f"label_{label}") for label in range(4))
    )
    base = (
        closed.group_by("query_id")
        .agg(
            pl.col("normalized_query_sha256").first(),
            pl.col("project_split").first(),
            pl.col("query_token_count").first(),
            pl.col("query_lexical_specificity").first(),
            pl.col("query_brand_detected").first(),
            pl.col("query_color_detected").first(),
            pl.col("query_model_token_count").first(),
        )
        .join(label_counts, on="query_id", how="inner", validate="1:1")
        .join(retrieval_context, on="query_id", how="left", validate="1:1")
        .join(parsed, on="query_id", how="left", validate="1:1")
    )
    if base["source"].null_count() or base["compatibility_present"].null_count():
        raise RankingEvaluationBuildError("validation query context is incomplete")
    return (
        base.with_columns(
            pl.when(pl.col("query_token_count") <= 2)
            .then(pl.lit("1-2"))
            .when(pl.col("query_token_count") <= 5)
            .then(pl.lit("3-5"))
            .otherwise(pl.lit("6+"))
            .alias("query_length_bucket"),
            pl.when(pl.col("query_lexical_specificity") <= median_specificity)
            .then(pl.lit("low"))
            .otherwise(pl.lit("high"))
            .alias("lexical_specificity_bucket"),
            pl.when(pl.col("query_brand_detected") == 1)
            .then(pl.lit("present"))
            .otherwise(pl.lit("absent"))
            .alias("brand_presence"),
            pl.when(pl.col("query_color_detected") == 1)
            .then(pl.lit("present"))
            .otherwise(pl.lit("absent"))
            .alias("color_presence"),
            pl.when(pl.col("query_model_token_count") > 0)
            .then(pl.lit("present"))
            .otherwise(pl.lit("absent"))
            .alias("model_presence"),
            pl.when(pl.col("compatibility_present"))
            .then(pl.lit("present"))
            .otherwise(pl.lit("absent"))
            .alias("compatibility_presence"),
            pl.when(pl.col("label_3") > 0)
            .then(pl.lit("exact_present"))
            .when(pl.col("label_2") > 0)
            .then(pl.lit("substitute_without_exact"))
            .when(pl.col("label_1") > pl.col("label_0"))
            .then(pl.lit("complement_heavy"))
            .otherwise(pl.lit("irrelevant_heavy"))
            .alias("judgment_composition"),
        )
        .select(
            "query_id",
            "normalized_query_sha256",
            "project_split",
            "source",
            "query_length_bucket",
            "lexical_specificity_bucket",
            "brand_presence",
            "color_presence",
            "model_presence",
            "compatibility_presence",
            "judgment_composition",
        )
        .sort("query_id")
    )


def _read_evaluation_inputs(
    dependencies: _Dependencies,
    config: ResolvedConfig,
    *,
    split: ProjectEvaluationSplit = "validation",
) -> _EvaluationInputs:
    closed_path = str(dependencies.features.path / CLOSED_MATRIX_DIRECTORY / "*.parquet")
    candidate_path = str(dependencies.features.path / CANDIDATE_MATRIX_DIRECTORY / "*.parquet")
    hybrid_path = str(dependencies.retrieval.path / CANDIDATE_DIRECTORY / "*.parquet")
    retrieval_metric_path = str(
        dependencies.retrieval.path / RETRIEVAL_QUERY_METRIC_DIRECTORY / "*.parquet"
    )
    try:
        closed = (
            pl.scan_parquet(closed_path)
            .filter(pl.col("project_split") == split)
            .select(
                "profile",
                "query_id",
                "normalized_query_sha256",
                "project_split",
                "product_id",
                "label_id",
                "gain",
                *FEATURE_NAMES,
            )
            .collect()
            .sort("query_id", "product_id")
        )
        candidates = (
            pl.scan_parquet(candidate_path)
            .filter(pl.col("project_split") == split)
            .select(
                "profile",
                "query_id",
                "normalized_query_sha256",
                "project_split",
                "product_id",
                *FEATURE_NAMES,
            )
            .collect()
            .sort("query_id", "product_id")
        )
        hybrid = (
            pl.scan_parquet(hybrid_path)
            .filter((pl.col("project_split") == split) & (pl.col("stage") == "hybrid"))
            .select("query_id", "product_id", "rank", "rrf_score")
            .collect()
        )
        retrieval_context = (
            pl.scan_parquet(retrieval_metric_path)
            .filter(pl.col("project_split") == split)
            .select("query_id", "source")
            .unique()
            .collect()
        )
        parsed = (
            pl.scan_parquet(dependencies.features.path / PARSED_QUERIES_FILENAME)
            .select(
                "query_id",
                (
                    (pl.col("compatibility_tokens").list.len() > 0)
                    | (pl.col("compatibility_phrases").list.len() > 0)
                ).alias("compatibility_present"),
            )
            .collect()
        )
    except pl.exceptions.PolarsError as exc:
        raise RankingEvaluationBuildError(f"cannot read ranking-evaluation inputs: {exc}") from exc
    if closed.is_empty():
        raise RankingEvaluationBuildError(f"project {split} has no closed judged rows")
    limits = config.config.ranking_evaluation
    if closed.height > limits.max_closed_rows or candidates.height > limits.max_candidate_rows:
        raise RankingEvaluationBuildError(f"exact {split} evaluation population exceeds row limits")
    for name, frame in (("closed", closed), ("candidate", candidates)):
        if frame.select(pl.struct("query_id", "product_id").n_unique()).item() != frame.height:
            raise RankingEvaluationBuildError(f"{name} evaluation keys are not unique")
    candidate_keys = candidates.select("query_id", "product_id")
    hybrid_keys = hybrid.select("query_id", "product_id")
    if (
        candidate_keys.join(hybrid_keys, on=("query_id", "product_id"), how="anti").height
        or hybrid_keys.join(candidate_keys, on=("query_id", "product_id"), how="anti").height
    ):
        raise RankingEvaluationBuildError(
            "retrieved-union features and original hybrid candidates are not identical"
        )
    candidates = candidates.join(
        hybrid, on=("query_id", "product_id"), how="left", validate="1:1"
    ).sort("query_id", "product_id")
    for group in candidates.partition_by("query_id", maintain_order=True):
        if sorted(cast(list[int], group["rank"].to_list())) != list(range(1, group.height + 1)):
            raise RankingEvaluationBuildError("hybrid candidate ranks are not contiguous")
    contexts = _context_rows(closed, retrieval_context, parsed)
    return _EvaluationInputs(closed, candidates, contexts)


def _score_columns(frame: pl.DataFrame, rankers: LoadedRankers) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(
            pl.Series("pointwise_score", [], dtype=pl.Float64),
            pl.Series("lambdamart_score", [], dtype=pl.Float64),
        )
    matrix = np.ascontiguousarray(frame.select(FEATURE_NAMES).to_numpy(), dtype=np.float32)
    return frame.with_columns(
        pl.Series("pointwise_score", rankers.predict("pointwise", matrix), dtype=pl.Float64),
        pl.Series("lambdamart_score", rankers.predict("lambdamart", matrix), dtype=pl.Float64),
    )


def _prediction_rows(
    frame: pl.DataFrame,
    *,
    profile: Profile,
    protocol: Protocol,
    population_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in frame.partition_by("query_id", maintain_order=True):
        product_ids = tuple(cast(list[str], group["product_id"].to_list()))
        score_columns = {
            "rrf": "closed_rrf_score" if protocol == CLOSED_POOL_PROTOCOL else "rrf_score",
            "pointwise": "pointwise_score",
            "lambdamart": "lambdamart_score",
        }
        for stage in _STAGES:
            scores = np.asarray(group[score_columns[stage]].to_numpy(), dtype=np.float64)
            if protocol == END_TO_END_PROTOCOL and stage == "rrf":
                ranks = tuple(cast(list[int], group["rank"].to_list()))
            else:
                ranks = stable_rank_predictions(product_ids, scores)
            for offset, product_id in enumerate(product_ids):
                judged = protocol == CLOSED_POOL_PROTOCOL
                rows.append(
                    {
                        "profile": profile,
                        "protocol": protocol,
                        "population_id": population_id,
                        "query_id": int(group["query_id"][offset]),
                        "normalized_query_sha256": str(group["normalized_query_sha256"][offset]),
                        "project_split": str(group["project_split"][offset]),
                        "stage": stage,
                        "product_id": product_id,
                        "score": float(scores[offset]),
                        "rank": ranks[offset],
                        "judged": judged,
                        "label_id": int(group["label_id"][offset]) if judged else None,
                        "gain": float(group["gain"][offset]) if judged else None,
                        "active_relevance": False,
                    }
                )
    return rows


def _judgments_by_query(closed: pl.DataFrame) -> dict[int, tuple[Judgment, ...]]:
    result: dict[int, tuple[Judgment, ...]] = {}
    for group in closed.partition_by("query_id", maintain_order=True):
        query_id = int(group["query_id"][0])
        result[query_id] = tuple(
            Judgment(
                product_id=str(product_id),
                label=_LABELS[int(label_id)],
                gain=_GAINS[_LABELS[int(label_id)]],
            )
            for product_id, label_id in group.select("product_id", "label_id").iter_rows()
        )
    return result


def _context_mapping(contexts: pl.DataFrame) -> dict[int, dict[str, Any]]:
    return {int(row["query_id"]): row for row in contexts.iter_rows(named=True)}


def _append_metric(
    rows: list[dict[str, object]],
    *,
    profile: Profile,
    population_id: str,
    context: dict[str, Any],
    stage: EvaluationStage,
    threshold_id: str,
    record: Any,
    empty: bool,
    relevant_judgment_count: int | None = None,
) -> None:
    rows.append(
        {
            "profile": profile,
            "protocol": record.protocol,
            "population_id": population_id,
            "query_id": context["query_id"],
            "normalized_query_sha256": context["normalized_query_sha256"],
            "project_split": context["project_split"],
            "source": context["source"],
            "query_length_bucket": context["query_length_bucket"],
            "lexical_specificity_bucket": context["lexical_specificity_bucket"],
            "brand_presence": context["brand_presence"],
            "color_presence": context["color_presence"],
            "model_presence": context["model_presence"],
            "compatibility_presence": context["compatibility_presence"],
            "judgment_composition": context["judgment_composition"],
            "stage": stage,
            "threshold_id": threshold_id,
            "metric": record.metric,
            "cutoff": record.cutoff,
            "value": record.value,
            "returned_count": record.returned_count,
            "judged_count": record.judged_count,
            "unjudged_count": record.unjudged_count,
            "relevant_judgment_count": (
                record.relevant_judgment_count
                if relevant_judgment_count is None
                else relevant_judgment_count
            ),
            "empty_result": empty,
        }
    )


def _query_metric_rows(
    predictions: pl.DataFrame,
    inputs: _EvaluationInputs,
    config: ResolvedConfig,
    profile: Profile,
    *,
    split: ProjectEvaluationSplit = "validation",
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    judgments = _judgments_by_query(inputs.closed)
    contexts = _context_mapping(inputs.contexts)
    ranking = config.config.ranking_evaluation
    population_ids = {
        CLOSED_POOL_PROTOCOL: f"esci_task1_us_judged_pool_{split}_v1",
        END_TO_END_PROTOCOL: f"end_to_end_hybrid_union_{split}_v1",
    }
    for protocol, cutoffs in (
        (CLOSED_POOL_PROTOCOL, ranking.closed_cutoffs),
        (END_TO_END_PROTOCOL, ranking.diagnostic_cutoffs),
    ):
        protocol_predictions = predictions.filter(pl.col("protocol") == protocol)
        for query_id in sorted(contexts):
            context = contexts[query_id]
            for stage in _STAGES:
                ranked_ids = tuple(
                    cast(
                        list[str],
                        protocol_predictions.filter(
                            (pl.col("query_id") == query_id) & (pl.col("stage") == stage)
                        )
                        .sort("rank")["product_id"]
                        .to_list(),
                    )
                )
                for cutoff in cutoffs:
                    for threshold_id, labels in _THRESHOLDS:
                        records = evaluate_ranked_products(
                            protocol,
                            ranked_ids,
                            judgments[query_id],
                            k=cutoff,
                            relevant_labels=labels,
                        )
                        for record in records:
                            if protocol == CLOSED_POOL_PROTOCOL:
                                if record.metric == "ndcg_official_gain":
                                    if threshold_id != "exact":
                                        continue
                                    metric_threshold = "official_gain"
                                elif record.metric == "exact_hit":
                                    if threshold_id != "exact":
                                        continue
                                    metric_threshold = "exact"
                                else:
                                    metric_threshold = threshold_id
                            else:
                                metric_threshold = threshold_id
                            _append_metric(
                                rows,
                                profile=profile,
                                population_id=population_ids[protocol],
                                context=context,
                                stage=stage,
                                threshold_id=metric_threshold,
                                record=record,
                                empty=not ranked_ids,
                                relevant_judgment_count=(
                                    sum(item.gain > 0.0 for item in judgments[query_id])
                                    if metric_threshold == "official_gain"
                                    else None
                                ),
                            )
    return pl.DataFrame(rows, schema=_QUERY_METRIC_SCHEMA).sort(
        "protocol", "query_id", "stage", "threshold_id", "metric", "cutoff"
    )


def _stable_seed(seed: int, key: str) -> int:
    digest = sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _summary_values(
    frame: pl.DataFrame,
    *,
    config: ResolvedConfig,
    key: str,
) -> tuple[float, float, float, float, int]:
    grouped = (
        frame.group_by("normalized_query_sha256")
        .agg(pl.col("value").sum().alias("value_sum"), pl.len().alias("query_count"))
        .sort("normalized_query_sha256")
    )
    sums = grouped["value_sum"].to_numpy()
    counts = grouped["query_count"].to_numpy()
    group_count = grouped.height
    evaluation = config.config.evaluation
    rng = np.random.default_rng(_stable_seed(config.config.runtime.seed, key))
    samples = np.empty(evaluation.bootstrap_replicates, dtype=np.float64)
    completed = 0
    while completed < evaluation.bootstrap_replicates:
        size = min(
            evaluation.bootstrap_batch_replicates, evaluation.bootstrap_replicates - completed
        )
        indices = rng.integers(0, group_count, size=(size, group_count))
        samples[completed : completed + size] = sums[indices].sum(axis=1) / counts[indices].sum(
            axis=1
        )
        completed += size
    values = frame["value"].to_numpy()
    return (
        float(values.mean()),
        float(np.median(values)),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
        group_count,
    )


def build_ranking_aggregate_metrics(
    query_metrics: pl.DataFrame, config: ResolvedConfig
) -> pl.DataFrame:
    """Aggregate overall and named slices with normalized-query group bootstrap."""
    rows: list[dict[str, object]] = []
    base_keys = ["protocol", "stage", "threshold_id", "metric", "cutoff"]
    dimensions: tuple[tuple[str, str | None], ...] = (
        ("all", None),
        ("query_length", "query_length_bucket"),
        ("lexical_specificity", "lexical_specificity_bucket"),
        ("brand_presence", "brand_presence"),
        ("color_presence", "color_presence"),
        ("model_presence", "model_presence"),
        ("compatibility_presence", "compatibility_presence"),
        ("source", "source"),
        ("project_split", "project_split"),
        ("judgment_composition", "judgment_composition"),
    )
    for dimension, column in dimensions:
        keys = [*base_keys, *([column] if column else [])]
        for partition in query_metrics.partition_by(keys, maintain_order=True):
            first = partition.row(0, named=True)
            slice_value = "all" if column is None else str(first[column])
            identity = (
                "|".join(str(first[key]) for key in (*base_keys,)) + f"|{dimension}|{slice_value}"
            )
            mean, median, lower, upper, group_count = _summary_values(
                partition, config=config, key=identity
            )
            rows.append(
                {
                    "profile": first["profile"],
                    "protocol": first["protocol"],
                    "population_id": first["population_id"],
                    "stage": first["stage"],
                    "threshold_id": first["threshold_id"],
                    "metric": first["metric"],
                    "cutoff": first["cutoff"],
                    "slice_dimension": dimension,
                    "slice_value": slice_value,
                    "mean": mean,
                    "median": median,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "query_count": partition.height,
                    "normalized_query_groups": group_count,
                    "empty_query_count": int(partition["empty_result"].sum()),
                    "returned_count": int(partition["returned_count"].sum()),
                    "judged_count": int(partition["judged_count"].sum()),
                    "unjudged_count": int(partition["unjudged_count"].sum()),
                    "relevant_judgment_count": int(partition["relevant_judgment_count"].sum()),
                    "bootstrap_replicates": config.config.evaluation.bootstrap_replicates,
                    "bootstrap_method": "normalized-query-group-v1",
                }
            )
    return pl.DataFrame(rows, schema=_AGGREGATE_SCHEMA).sort(
        "protocol", "stage", "threshold_id", "metric", "cutoff", "slice_dimension", "slice_value"
    )


def _paired_comparison(
    query_metrics: pl.DataFrame,
    config: ResolvedConfig,
    *,
    protocol: Protocol,
    ablation_id: str,
    treatment: EvaluationStage,
    baseline: EvaluationStage,
    threshold_id: str,
    metric: str,
    cutoff: int,
) -> dict[str, object]:
    keys = ["query_id", "normalized_query_sha256"]
    selected = query_metrics.filter(
        (pl.col("protocol") == protocol)
        & (pl.col("threshold_id") == threshold_id)
        & (pl.col("metric") == metric)
        & (pl.col("cutoff") == cutoff)
    )
    treatment_frame = selected.filter(pl.col("stage") == treatment).select(
        *keys, pl.col("value").alias("treatment_value")
    )
    baseline_frame = selected.filter(pl.col("stage") == baseline).select(
        *keys, pl.col("value").alias("baseline_value")
    )
    paired = treatment_frame.join(baseline_frame, on=keys, how="inner", validate="1:1")
    if paired.height != treatment_frame.height or paired.height != baseline_frame.height:
        raise RankingEvaluationBuildError("ranking comparison cohorts are not identical")
    higher = metric in _HIGHER_IS_BETTER
    paired = paired.with_columns(
        (
            pl.col("treatment_value") - pl.col("baseline_value")
            if higher
            else pl.col("baseline_value") - pl.col("treatment_value")
        ).alias("value")
    )
    mean, median, lower, upper, group_count = _summary_values(
        paired,
        config=config,
        key=f"{protocol}|{ablation_id}|{threshold_id}|{metric}|{cutoff}",
    )
    values = paired["value"].to_numpy()
    tolerance = config.config.ranking_evaluation.selection_tie_tolerance
    return {
        "profile": selected.item(0, "profile"),
        "protocol": protocol,
        "ablation_id": ablation_id,
        "treatment_stage": treatment,
        "baseline_stage": baseline,
        "metric_direction": "higher" if higher else "lower",
        "threshold_id": threshold_id,
        "metric": metric,
        "cutoff": cutoff,
        "mean_improvement": mean,
        "median_improvement": median,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "query_count": paired.height,
        "normalized_query_groups": group_count,
        "win_count": int(np.count_nonzero(values > tolerance)),
        "tie_count": int(np.count_nonzero(np.abs(values) <= tolerance)),
        "loss_count": int(np.count_nonzero(values < -tolerance)),
        "bootstrap_replicates": config.config.evaluation.bootstrap_replicates,
        "bootstrap_method": "normalized-query-group-v1",
    }


def build_ranking_comparisons(query_metrics: pl.DataFrame, config: ResolvedConfig) -> pl.DataFrame:
    """Build ABL-04/05 and explicitly diagnostic end-to-end paired comparisons."""
    rows: list[dict[str, object]] = []
    plans: tuple[tuple[Protocol, str, EvaluationStage, EvaluationStage], ...] = (
        (CLOSED_POOL_PROTOCOL, "ABL-04", "lambdamart", "rrf"),
        (CLOSED_POOL_PROTOCOL, "ABL-05", "lambdamart", "pointwise"),
        (END_TO_END_PROTOCOL, "E2E-01", "pointwise", "rrf"),
        (END_TO_END_PROTOCOL, "E2E-02", "lambdamart", "rrf"),
    )
    combinations = query_metrics.select("protocol", "threshold_id", "metric", "cutoff").unique()
    for protocol, ablation_id, treatment, baseline in plans:
        for combination in combinations.filter(pl.col("protocol") == protocol).iter_rows(
            named=True
        ):
            rows.append(
                _paired_comparison(
                    query_metrics,
                    config,
                    protocol=protocol,
                    ablation_id=ablation_id,
                    treatment=treatment,
                    baseline=baseline,
                    threshold_id=str(combination["threshold_id"]),
                    metric=str(combination["metric"]),
                    cutoff=int(combination["cutoff"]),
                )
            )
    return pl.DataFrame(rows, schema=_COMPARISON_SCHEMA).sort(
        "protocol", "ablation_id", "threshold_id", "metric", "cutoff"
    )


def _champion_candidates(
    means: dict[tuple[EvaluationStage, int], float],
    ranking: RankingEvaluationConfig,
) -> tuple[EvaluationStage, tuple[ChampionCandidate, ChampionCandidate, ChampionCandidate]]:
    baseline = tuple((cutoff, means[("rrf", cutoff)]) for cutoff in ranking.closed_cutoffs)
    candidates: list[ChampionCandidate] = [
        ChampionCandidate(
            stage="rrf",
            eligible=True,
            selection_score=float(np.mean([value for _, value in baseline])),
            ndcg_by_cutoff=baseline,
            delta_from_rrf=tuple((cutoff, 0.0) for cutoff in ranking.closed_cutoffs),
            decision_reason="always-eligible deterministic direct-score RRF baseline",
        )
    ]
    for stage in ("pointwise", "lambdamart"):
        values = tuple((cutoff, means[(stage, cutoff)]) for cutoff in ranking.closed_cutoffs)
        deltas = tuple(
            (cutoff, means[(stage, cutoff)] - means[("rrf", cutoff)])
            for cutoff in ranking.closed_cutoffs
        )
        improves = any(delta > ranking.minimum_model_improvement for _, delta in deltas)
        safe = all(delta >= -ranking.material_regression_tolerance for _, delta in deltas)
        eligible = improves and safe
        reason = (
            "improves at least one configured NDCG cutoff without material regression"
            if eligible
            else "does not clear improvement and no-material-regression validation guardrails"
        )
        candidates.append(
            ChampionCandidate(
                stage=stage,
                eligible=eligible,
                selection_score=float(np.mean([value for _, value in values])),
                ndcg_by_cutoff=values,
                delta_from_rrf=deltas,
                decision_reason=reason,
            )
        )
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate.eligible and (
            candidate.selection_score > selected.selection_score + ranking.selection_tie_tolerance
        ):
            selected = candidate
    return selected.stage, cast(Any, tuple(candidates))


def _select_champion(
    query_metrics: pl.DataFrame,
    config: ResolvedConfig,
    dependencies: _Dependencies,
) -> ActiveRelevanceContract:
    ranking = config.config.ranking_evaluation
    ndcg = query_metrics.filter(
        (pl.col("protocol") == CLOSED_POOL_PROTOCOL)
        & (pl.col("threshold_id") == "official_gain")
        & (pl.col("metric") == "ndcg_official_gain")
    )
    means = {
        (cast(EvaluationStage, stage), int(cutoff)): float(value)
        for stage, cutoff, value in ndcg.group_by("stage", "cutoff")
        .agg(pl.col("value").mean().alias("mean"))
        .iter_rows()
    }
    selected_stage, candidates = _champion_candidates(means, ranking)
    selected_model = None if selected_stage == "rrf" else cast(Any, selected_stage)
    score_field = {
        "rrf": "hybrid_rrf_score",
        "pointwise": "pointwise_score",
        "lambdamart": "lambdamart_score",
    }[selected_stage]
    return ActiveRelevanceContract(
        selected_stage=selected_stage,
        selected_model_id=selected_model,
        selection_cutoffs=ranking.closed_cutoffs,
        selection_query_count=inputs_query_count(query_metrics),
        active_score_field=cast(Any, score_field),
        ranking_models_artifact_id=dependencies.models.manifest.artifact_id,
        ranking_models_manifest_sha256=dependencies.models.manifest_sha256,
        feature_artifact_id=dependencies.features.manifest.artifact_id,
        feature_registry_sha256=dependencies.models_manifest.feature_registry_sha256,
        candidates=candidates,
    )


def inputs_query_count(query_metrics: pl.DataFrame) -> int:
    return query_metrics.filter(pl.col("protocol") == CLOSED_POOL_PROTOCOL)["query_id"].n_unique()


def _failure_analysis(query_metrics: pl.DataFrame, config: ResolvedConfig) -> pl.DataFrame:
    ranking = config.config.ranking_evaluation
    selected = query_metrics.filter(
        (pl.col("protocol") == CLOSED_POOL_PROTOCOL)
        & (pl.col("threshold_id") == "official_gain")
        & (pl.col("metric") == "ndcg_official_gain")
        & (pl.col("cutoff") == ranking.closed_cutoffs[0])
    )
    rows: list[dict[str, object]] = []
    for ablation_id, treatment, baseline in (
        ("ABL-04", "lambdamart", "rrf"),
        ("ABL-05", "lambdamart", "pointwise"),
    ):
        context_columns = [
            "query_id",
            "normalized_query_sha256",
            "source",
            "query_length_bucket",
            "lexical_specificity_bucket",
            "brand_presence",
            "color_presence",
            "model_presence",
            "compatibility_presence",
            "judgment_composition",
        ]
        treatment_frame = selected.filter(pl.col("stage") == treatment).select(
            *context_columns, pl.col("value").alias("treatment_value")
        )
        baseline_frame = selected.filter(pl.col("stage") == baseline).select(
            "query_id", pl.col("value").alias("baseline_value")
        )
        paired = (
            treatment_frame.join(baseline_frame, on="query_id", how="inner", validate="1:1")
            .with_columns((pl.col("treatment_value") - pl.col("baseline_value")).alias("delta"))
            .with_columns(pl.col("delta").abs().alias("absolute_delta"))
            .sort("absolute_delta", "query_id", descending=[True, False])
            .head(ranking.failure_analysis_queries)
        )
        tolerance = ranking.selection_tie_tolerance
        for row in paired.iter_rows(named=True):
            delta = float(row["delta"])
            rows.append(
                {
                    "profile": selected.item(0, "profile"),
                    "protocol": CLOSED_POOL_PROTOCOL,
                    "ablation_id": ablation_id,
                    **{column: row[column] for column in context_columns},
                    "metric": "ndcg_official_gain",
                    "cutoff": ranking.closed_cutoffs[0],
                    "baseline_stage": baseline,
                    "treatment_stage": treatment,
                    "baseline_value": row["baseline_value"],
                    "treatment_value": row["treatment_value"],
                    "delta": delta,
                    "outcome": "win"
                    if delta > tolerance
                    else "loss"
                    if delta < -tolerance
                    else "tie",
                    "selection_method": "largest-absolute-paired-delta-v1",
                }
            )
    return pl.DataFrame(rows, schema=_FAILURE_SCHEMA).sort("ablation_id", "outcome", "query_id")


def _reuse(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    dependencies: _Dependencies,
    store: ArtifactStore,
) -> RankingEvaluationBuildResult:
    artifact = store.load(ranking_evaluation_artifact_id(release, config.sha256, profile))
    if artifact.manifest.dependencies != _artifact_dependency(dependencies):
        raise RankingEvaluationValidationError("ranking-evaluation parent model is incompatible")
    manifest = load_ranking_evaluation_manifest(artifact.path / RANKING_EVALUATION_FILENAME)
    active = load_active_relevance_contract(artifact.path / ACTIVE_RELEVANCE_FILENAME)
    if (
        manifest.artifact_id != artifact.manifest.artifact_id
        or manifest.config_sha256 != config.sha256
        or manifest.profile != profile
        or manifest.active_relevance != active
    ):
        raise RankingEvaluationValidationError(
            "ranking-evaluation metadata identity is incompatible"
        )
    return RankingEvaluationBuildResult(artifact, manifest, True)


def evaluate_frozen_ranking_test(
    store: ArtifactStore,
    config: ResolvedConfig,
    manifest: RankingEvaluationManifest,
) -> FrozenRankingEvaluation:
    """Evaluate project test once using the validation-frozen active contract."""
    if manifest.profile != "portfolio" or manifest.config_sha256 != config.sha256:
        raise RankingEvaluationValidationError(
            "frozen test evaluation requires the compatible portfolio validation manifest"
        )
    try:
        models = store.load(manifest.ranking_models_artifact_id)
        models_manifest = load_ranking_models_manifest(models.path / RANKING_MODELS_FILENAME)
        features = store.load(manifest.ranking_features_artifact_id)
        features_manifest = load_ranking_feature_manifest(features.path / FEATURE_ARTIFACT_FILENAME)
        retrieval = store.load(manifest.retrieval_evaluation_artifact_id)
        retrieval_manifest = load_retrieval_evaluation_manifest(
            retrieval.path / RETRIEVAL_EVALUATION_FILENAME
        )
        rankers = load_rankers(store, models.manifest.artifact_id)
    except (OSError, RuntimeError) as exc:
        raise RankingEvaluationBuildError("cannot load frozen portfolio test dependencies") from exc
    if (
        models.manifest_sha256 != manifest.ranking_models_manifest_sha256
        or models_manifest.feature_artifact_id != features.manifest.artifact_id
        or features.manifest_sha256 != manifest.ranking_features_manifest_sha256
        or features_manifest.retrieval_evaluation_artifact_id != retrieval.manifest.artifact_id
        or retrieval.manifest_sha256 != manifest.retrieval_evaluation_manifest_sha256
        or retrieval_manifest.config_sha256 != config.sha256
        or retrieval_manifest.profile != "portfolio"
    ):
        raise RankingEvaluationValidationError(
            "frozen portfolio test dependency lineage is incompatible"
        )
    dependencies = _Dependencies(
        models=models,
        models_manifest=models_manifest,
        features=features,
        features_manifest=features_manifest,
        retrieval=retrieval,
        retrieval_manifest=retrieval_manifest,
    )
    inputs = _read_evaluation_inputs(dependencies, config, split="test")
    closed = _score_columns(inputs.closed, rankers)
    candidates = _score_columns(inputs.candidates, rankers)
    prediction_rows = _prediction_rows(
        closed,
        profile="portfolio",
        protocol=CLOSED_POOL_PROTOCOL,
        population_id="esci_task1_us_judged_pool_test_v1",
    ) + _prediction_rows(
        candidates,
        profile="portfolio",
        protocol=END_TO_END_PROTOCOL,
        population_id="end_to_end_hybrid_union_test_v1",
    )
    predictions = (
        pl.DataFrame(prediction_rows, schema=_PREDICTION_SCHEMA)
        .with_columns(
            (pl.col("stage") == manifest.active_relevance.selected_stage).alias("active_relevance")
        )
        .sort("protocol", "query_id", "stage", "rank")
    )
    query_metrics = _query_metric_rows(
        predictions,
        inputs,
        config,
        "portfolio",
        split="test",
    )
    return FrozenRankingEvaluation(
        split="test",
        predictions=predictions,
        query_metrics=query_metrics,
        aggregate_metrics=build_ranking_aggregate_metrics(query_metrics, config),
        comparisons=build_ranking_comparisons(query_metrics, config),
        failure_analysis=_failure_analysis(query_metrics, config),
    )


def build_ranking_evaluation(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    profile: Profile | None = None,
    artifact_store: ArtifactStore | None = None,
) -> RankingEvaluationBuildResult:
    """Evaluate validation ranking protocols and promote one active relevance contract."""
    selected_profile: Profile = profile or config.config.evaluation.default_profile
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    dependencies = _load_dependencies(release, config, selected_profile, store)
    artifact_id = ranking_evaluation_artifact_id(release, config.sha256, selected_profile)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse(release, config, selected_profile, dependencies, store)

    try:
        rankers = load_rankers(store, dependencies.models.manifest.artifact_id)
        inputs = _read_evaluation_inputs(dependencies, config, split="validation")
    except RuntimeError as exc:
        raise RankingEvaluationBuildError(f"cannot load ranking evaluation inputs: {exc}") from exc
    rss_limit = config.config.runtime.rss_limit_mb * 1024 * 1024
    load_peak = _peak_rss_bytes()
    initial = RankingEvaluationResourceMeasurement(
        load_peak_rss_bytes=load_peak,
        evaluation_peak_rss_bytes=load_peak,
        promotion_peak_rss_bytes=load_peak,
        peak_rss_bytes=load_peak,
        rss_limit_bytes=rss_limit,
        artifact_payload_bytes=0,
        passed=load_peak <= rss_limit,
    )
    if not initial.passed:
        raise RankingEvaluationResourceError(initial)

    closed = _score_columns(inputs.closed, rankers)
    candidates = _score_columns(inputs.candidates, rankers)
    prediction_rows = _prediction_rows(
        closed,
        profile=selected_profile,
        protocol=CLOSED_POOL_PROTOCOL,
        population_id="esci_task1_us_judged_pool_validation_v1",
    ) + _prediction_rows(
        candidates,
        profile=selected_profile,
        protocol=END_TO_END_PROTOCOL,
        population_id="end_to_end_hybrid_union_validation_v1",
    )
    predictions = pl.DataFrame(prediction_rows, schema=_PREDICTION_SCHEMA).sort(
        "protocol", "query_id", "stage", "rank"
    )
    query_metrics = _query_metric_rows(
        predictions, inputs, config, selected_profile, split="validation"
    )
    aggregate = build_ranking_aggregate_metrics(query_metrics, config)
    comparisons = build_ranking_comparisons(query_metrics, config)
    active = _select_champion(query_metrics, config, dependencies)
    predictions = predictions.with_columns(
        (pl.col("stage") == active.selected_stage).alias("active_relevance")
    )
    failures = _failure_analysis(query_metrics, config)
    evaluation_peak = _peak_rss_bytes()

    transaction = store.stage(
        artifact_type="ranking-evaluation",
        dataset_version=release.manifest.dataset_version,
        profile=selected_profile,
        component_version=config.config.ranking_evaluation.component_version,
        config_sha256=config.sha256,
        code_revision=code_revision,
        dependencies=_artifact_dependency(dependencies),
    )
    try:
        with transaction:
            root = transaction.path(RANKING_EVALUATION_FILENAME).parent
            predictions.write_parquet(
                root / PREDICTIONS_FILENAME, compression="zstd", statistics=True
            )
            query_metrics.write_parquet(
                root / QUERY_METRICS_FILENAME, compression="zstd", statistics=True
            )
            aggregate.write_parquet(root / METRICS_FILENAME, compression="zstd", statistics=True)
            comparisons.write_parquet(
                root / COMPARISONS_FILENAME, compression="zstd", statistics=True
            )
            failures.write_parquet(
                root / FAILURE_ANALYSIS_FILENAME, compression="zstd", statistics=True
            )
            (root / ACTIVE_RELEVANCE_FILENAME).write_text(_canonical_json(active), encoding="utf-8")
            promotion_peak = _peak_rss_bytes()
            payload_bytes = _directory_bytes(root)
            measurement = RankingEvaluationResourceMeasurement(
                load_peak_rss_bytes=load_peak,
                evaluation_peak_rss_bytes=evaluation_peak,
                promotion_peak_rss_bytes=promotion_peak,
                peak_rss_bytes=max(load_peak, evaluation_peak, promotion_peak),
                rss_limit_bytes=rss_limit,
                artifact_payload_bytes=payload_bytes,
                passed=max(load_peak, evaluation_peak, promotion_peak) <= rss_limit,
            )
            if not measurement.passed:
                raise RankingEvaluationResourceError(measurement)
            validation_queries = inputs.contexts.height
            ablations = (
                AblationRecord(
                    ablation_id="ABL-01",
                    status="inherited",
                    protocol="retrieval_catalog_task1_us_v1",
                    evidence_artifact_id=dependencies.retrieval.manifest.artifact_id,
                    query_count=dependencies.retrieval_manifest.query_count,
                ),
                AblationRecord(
                    ablation_id="ABL-02",
                    status="inherited",
                    protocol="retrieval_catalog_task1_us_v1",
                    evidence_artifact_id=dependencies.retrieval.manifest.artifact_id,
                    query_count=dependencies.retrieval_manifest.query_count,
                ),
                AblationRecord(
                    ablation_id="ABL-03",
                    status="inherited",
                    protocol="retrieval_catalog_task1_us_v1",
                    evidence_artifact_id=dependencies.retrieval.manifest.artifact_id,
                    query_count=dependencies.retrieval_manifest.query_count,
                ),
                AblationRecord(
                    ablation_id="ABL-04",
                    status="evaluated",
                    protocol=CLOSED_POOL_PROTOCOL,
                    evidence_artifact_id=artifact_id,
                    query_count=validation_queries,
                ),
                AblationRecord(
                    ablation_id="ABL-05",
                    status="evaluated",
                    protocol=CLOSED_POOL_PROTOCOL,
                    evidence_artifact_id=artifact_id,
                    query_count=validation_queries,
                ),
            )
            run = RankingExperimentRun(
                run_id=artifact_id,
                config_sha256=config.sha256,
                profile=selected_profile,
                ranking_models_artifact_id=dependencies.models.manifest.artifact_id,
                ranking_models_manifest_sha256=dependencies.models.manifest_sha256,
                ranking_features_artifact_id=dependencies.features.manifest.artifact_id,
                ranking_features_manifest_sha256=dependencies.features.manifest_sha256,
                retrieval_evaluation_artifact_id=dependencies.retrieval.manifest.artifact_id,
                retrieval_evaluation_manifest_sha256=dependencies.retrieval.manifest_sha256,
                active_relevance=active,
                ablations=ablations,
                metric_rows=aggregate.height,
                comparison_rows=comparisons.height,
                failure_analysis_rows=failures.height,
                resource=measurement,
            )
            (root / RUN_FILENAME).write_text(_canonical_json(run), encoding="utf-8")
            checks = tuple(
                sorted(
                    (
                        RankingEvaluationCheck(
                            check_id="active_relevance_unique",
                            detail=f"one validation-selected active stage: {active.selected_stage}",
                        ),
                        RankingEvaluationCheck(
                            check_id="fixed_closed_cohort",
                            detail="RRF, pointwise, and LambdaMART use identical judged pools",
                        ),
                        RankingEvaluationCheck(
                            check_id="grouped_bootstrap",
                            detail=(
                                f"{config.config.evaluation.bootstrap_replicates} fixed-seed "
                                "normalized-query group replicates"
                            ),
                        ),
                        RankingEvaluationCheck(
                            check_id="protocol_separation",
                            detail="end-to-end diagnostics expose no Precision, MAP, or NDCG",
                        ),
                        RankingEvaluationCheck(
                            check_id="required_ablations",
                            detail="ABL-01 through ABL-05 have inherited or evaluated evidence",
                        ),
                        RankingEvaluationCheck(
                            check_id="resource_gate",
                            detail=(
                                f"peak RSS {measurement.peak_rss_bytes} <= "
                                f"{measurement.rss_limit_bytes} bytes"
                            ),
                        ),
                        RankingEvaluationCheck(
                            check_id="test_quarantined",
                            detail="project test was filtered before evaluation materialization",
                        ),
                    ),
                    key=lambda check: check.check_id,
                )
            )
            manifest = RankingEvaluationManifest(
                artifact_id=artifact_id,
                dataset_version=release.manifest.dataset_version,
                config_sha256=config.sha256,
                profile=selected_profile,
                ranking_models_artifact_id=dependencies.models.manifest.artifact_id,
                ranking_models_manifest_sha256=dependencies.models.manifest_sha256,
                ranking_features_artifact_id=dependencies.features.manifest.artifact_id,
                ranking_features_manifest_sha256=dependencies.features.manifest_sha256,
                retrieval_evaluation_artifact_id=dependencies.retrieval.manifest.artifact_id,
                retrieval_evaluation_manifest_sha256=dependencies.retrieval.manifest_sha256,
                closed_cutoffs=config.config.ranking_evaluation.closed_cutoffs,
                diagnostic_cutoffs=config.config.ranking_evaluation.diagnostic_cutoffs,
                bootstrap_replicates=config.config.evaluation.bootstrap_replicates,
                bootstrap_seed=config.config.runtime.seed,
                validation_queries=validation_queries,
                normalized_query_groups=inputs.contexts["normalized_query_sha256"].n_unique(),
                closed_rows=inputs.closed.height,
                candidate_rows=inputs.candidates.height,
                prediction_rows=predictions.height,
                query_metric_rows=query_metrics.height,
                aggregate_metric_rows=aggregate.height,
                comparison_rows=comparisons.height,
                failure_analysis_rows=failures.height,
                active_relevance=active,
                resource=measurement,
                checks=checks,
            )
            (root / RANKING_EVALUATION_FILENAME).write_text(
                _canonical_json(manifest), encoding="utf-8"
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse(release, config, selected_profile, dependencies, store)
    return RankingEvaluationBuildResult(artifact, manifest, False)


__all__ = [
    "ACTIVE_RELEVANCE_FILENAME",
    "COMPARISONS_FILENAME",
    "FAILURE_ANALYSIS_FILENAME",
    "METRICS_FILENAME",
    "PREDICTIONS_FILENAME",
    "QUERY_METRICS_FILENAME",
    "RANKING_EVALUATION_FILENAME",
    "RUN_FILENAME",
    "ActiveRelevanceContract",
    "FrozenRankingEvaluation",
    "RankingEvaluationBuildError",
    "RankingEvaluationBuildResult",
    "RankingEvaluationError",
    "RankingEvaluationManifest",
    "RankingEvaluationResourceError",
    "RankingEvaluationValidationError",
    "build_ranking_aggregate_metrics",
    "build_ranking_comparisons",
    "build_ranking_evaluation",
    "evaluate_frozen_ranking_test",
    "load_active_relevance_contract",
    "load_ranking_evaluation_manifest",
    "ranking_evaluation_artifact_id",
]
