"""CPU-bounded Pointwise LightGBM and LambdaMART training with immutable lineage."""

from __future__ import annotations

import gc
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import lightgbm as lgb
import numpy as np
import polars as pl
from lightgbm.basic import LightGBMError
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
from market_rank.config import RankerTrainingConfig, ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.features.artifact import (
    CLOSED_MATRIX_DIRECTORY,
    FEATURE_ARTIFACT_FILENAME,
    FEATURE_REGISTRY_FILENAME,
    FEATURE_STATE_FILENAME,
    FeatureState,
    RankingFeatureManifest,
    load_feature_state,
    load_ranking_feature_manifest,
    ranking_feature_artifact_id,
)
from market_rank.features.registry import FEATURE_NAMES, FeatureRegistry
from market_rank.ranking.population import (
    OFFICIAL_GAIN_MAPPING,
    PreparedPopulation,
    PreparedSplit,
    TrainingPopulationError,
    TrainingPopulationManifest,
    prepare_training_population,
    stable_grouped_prediction_ranks,
    stable_rank_predictions,
)

RANKING_MODELS_FILENAME = "ranking-models.json"
TRAINING_POPULATION_FILENAME = "training-population.json"
POPULATION_AUDIT_FILENAME = "population-audit.parquet"
POINTWISE_MODEL_FILENAME: Literal["pointwise-lightgbm.txt"] = "pointwise-lightgbm.txt"
LAMBDAMART_MODEL_FILENAME: Literal["lambdamart.txt"] = "lambdamart.txt"
VALIDATION_HISTORY_FILENAME = "validation-history.parquet"
FEATURE_IMPORTANCE_FILENAME = "feature-importance.parquet"
RELOAD_PARITY_FILENAME = "reload-parity.parquet"
EXPLANATION_SAMPLE_FILENAME = "explanation-contributions.parquet"

Profile = Literal["development", "portfolio"]
ModelId = Literal["pointwise", "lambdamart"]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]


class RankingTrainingError(RuntimeError):
    """Base exception for training-population, model-build, and reload failures."""


class RankingTrainingBuildError(RankingTrainingError):
    """Raised when compatible features cannot produce both required rankers."""


class RankingTrainingValidationError(RankingTrainingError):
    """Raised when a persisted ranker artifact violates its strict contract."""


class RankingTrainingResourceError(RankingTrainingBuildError):
    """Raised when matrix/model training exceeds the configured RSS limit."""

    def __init__(self, measurement: TrainingResourceMeasurement) -> None:
        super().__init__(
            f"ranker training peak RSS {measurement.peak_rss_bytes} exceeds "
            f"the {measurement.rss_limit_bytes}-byte limit"
        )
        self.measurement = measurement


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingResourceMeasurement(_StrictModel):
    matrix_peak_rss_bytes: int = Field(strict=True, ge=0)
    training_peak_rss_bytes: int = Field(strict=True, ge=0)
    reload_peak_rss_bytes: int = Field(strict=True, ge=0)
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    artifact_payload_bytes: int = Field(strict=True, ge=0)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        observed = max(
            self.matrix_peak_rss_bytes,
            self.training_peak_rss_bytes,
            self.reload_peak_rss_bytes,
        )
        if self.peak_rss_bytes != observed:
            raise ValueError("peak RSS must equal the maximum observed training phase")
        if self.passed != (self.peak_rss_bytes <= self.rss_limit_bytes):
            raise ValueError("training resource status does not match the RSS gate")
        return self


class RankerModelSummary(_StrictModel):
    model_id: ModelId
    objective: Literal["regression_l2", "lambdarank"]
    model_filename: Literal["pointwise-lightgbm.txt", "lambdamart.txt"]
    model_sha256: Sha256Digest
    model_bytes: int = Field(strict=True, ge=1)
    best_iteration: int = Field(strict=True, ge=1)
    trained_iterations: int = Field(strict=True, ge=1)
    training_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    parameters_json: str = Field(strict=True, min_length=2)
    validation_best_ndcg: tuple[tuple[int, float], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        cutoffs = tuple(cutoff for cutoff, _ in self.validation_best_ndcg)
        if cutoffs != tuple(sorted(set(cutoffs))):
            raise ValueError("validation NDCG cutoffs must be unique and sorted")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for _, value in self.validation_best_ndcg
        ):
            raise ValueError("validation NDCG values must be finite in [0,1]")
        return self


class RankingTrainingCheck(_StrictModel):
    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class RankingModelsManifest(_StrictModel):
    """Exact feature dependency, population, two-model, parity, and resource contract."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    profile: Profile
    component_version: Literal["lightgbm-rankers-v1"] = "lightgbm-rankers-v1"
    lightgbm_version: str = Field(strict=True, min_length=1)
    feature_artifact_id: str = Field(strict=True, min_length=1)
    feature_manifest_sha256: Sha256Digest
    feature_registry_sha256: Sha256Digest
    feature_state_sha256: Sha256Digest
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    feature_names: tuple[str, ...] = Field(min_length=1)
    feature_dtypes: tuple[tuple[str, str], ...] = Field(min_length=1)
    categorical_features: tuple[str, ...]
    fallback_contract: Literal["rrf-on-model-failure-v1"] = "rrf-on-model-failure-v1"
    official_gain_mapping: tuple[tuple[int, float], ...] = OFFICIAL_GAIN_MAPPING
    population: TrainingPopulationManifest
    models: tuple[RankerModelSummary, RankerModelSummary]
    reload_parity_rows: int = Field(strict=True, ge=1)
    maximum_reload_prediction_delta: float = Field(ge=0.0, allow_inf_nan=False)
    explanation_rows: int = Field(strict=True, ge=1)
    resource: TrainingResourceMeasurement
    checks: tuple[RankingTrainingCheck, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("ranker feature order differs from ltr_core_v1")
        if tuple(name for name, _ in self.feature_dtypes) != self.feature_names:
            raise ValueError("ranker feature dtype order differs from feature names")
        if not set(self.categorical_features) <= set(self.feature_names):
            raise ValueError("ranker categorical features are not registered inputs")
        if tuple(model.model_id for model in self.models) != ("pointwise", "lambdamart"):
            raise ValueError("ranker models must be ordered pointwise then LambdaMART")
        check_ids = tuple(check.check_id for check in self.checks)
        if check_ids != tuple(sorted(set(check_ids))):
            raise ValueError("ranker checks must be unique and sorted")
        return self


@dataclass(frozen=True, slots=True)
class RankingTrainingBuildResult:
    artifact: LoadedArtifact
    manifest: RankingModelsManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class RankPrediction:
    product_id: str
    score: float
    one_based_rank: int


@dataclass(frozen=True, slots=True)
class LoadedRankers:
    artifact: LoadedArtifact
    manifest: RankingModelsManifest
    pointwise: lgb.Booster
    lambdamart: lgb.Booster

    def predict(
        self,
        model_id: ModelId,
        features: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise RankingTrainingValidationError("prediction matrix has incompatible dimensions")
        if not np.isfinite(features).all():
            raise RankingTrainingValidationError("prediction matrix contains non-finite values")
        booster = self.pointwise if model_id == "pointwise" else self.lambdamart
        summary = next(model for model in self.manifest.models if model.model_id == model_id)
        return _predict(booster, features, summary.best_iteration)

    def rank(
        self,
        model_id: ModelId,
        product_ids: tuple[str, ...],
        features: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ) -> tuple[RankPrediction, ...]:
        predictions = self.predict(model_id, features)
        ranks = stable_rank_predictions(product_ids, predictions)
        return tuple(
            RankPrediction(product_id, float(score), rank)
            for product_id, score, rank in zip(product_ids, predictions, ranks, strict=True)
        )


@dataclass(frozen=True, slots=True)
class _FeatureDependency:
    artifact: LoadedArtifact
    manifest: RankingFeatureManifest
    state: FeatureState
    registry: FeatureRegistry


@dataclass(frozen=True, slots=True)
class _TrainedModel:
    summary: RankerModelSummary
    validation_predictions: np.ndarray[tuple[int], np.dtype[np.float64]]
    history_rows: list[dict[str, object]]
    importance_rows: list[dict[str, object]]
    explanation_rows: list[dict[str, object]]


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_mapping(mapping: dict[str, object]) -> str:
    return json.dumps(
        mapping,
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


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _load_feature_dependency(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    store: ArtifactStore,
) -> _FeatureDependency:
    try:
        artifact = store.load(ranking_feature_artifact_id(release, config.sha256, profile))
        manifest = load_ranking_feature_manifest(artifact.path / FEATURE_ARTIFACT_FILENAME)
        state = load_feature_state(artifact.path / FEATURE_STATE_FILENAME)
        registry = FeatureRegistry.model_validate(
            json.loads((artifact.path / FEATURE_REGISTRY_FILENAME).read_text(encoding="utf-8"))
        )
    except (OSError, RuntimeError, json.JSONDecodeError, ValidationError) as exc:
        raise RankingTrainingBuildError(
            "a compatible Goldfish 010 ranking-feature artifact is required before training"
        ) from exc
    if (
        manifest.artifact_id != artifact.manifest.artifact_id
        or manifest.config_sha256 != config.sha256
        or manifest.profile != profile
        or manifest.registry_sha256 != state.registry_sha256
        or tuple(feature.name for feature in registry.features) != FEATURE_NAMES
        or sha256(_canonical_json(registry).encode("utf-8")).hexdigest() != state.registry_sha256
    ):
        raise RankingTrainingBuildError("ranking-feature dependency identity is incompatible")
    return _FeatureDependency(artifact, manifest, state, registry)


def ranking_models_artifact_id(
    release: ResolvedReleaseManifest, config_sha256: str, profile: Profile
) -> str:
    return "/".join(
        (
            "ranking-models",
            release.manifest.dataset_version,
            profile,
            "lightgbm-rankers-v1",
            config_sha256,
        )
    )


def load_ranking_models_manifest(path: Path) -> RankingModelsManifest:
    try:
        return RankingModelsManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise RankingTrainingValidationError(
            f"cannot load ranking-model manifest {path}: {exc}"
        ) from exc


def _artifact_dependency(feature: _FeatureDependency) -> tuple[ArtifactDependency, ...]:
    return (
        ArtifactDependency(
            artifact_id=feature.artifact.manifest.artifact_id,
            manifest_sha256=feature.artifact.manifest_sha256,
        ),
    )


def _read_training_rows(feature: _FeatureDependency) -> pl.DataFrame:
    path_glob = str(feature.artifact.path / CLOSED_MATRIX_DIRECTORY / "*.parquet")
    try:
        return (
            pl.scan_parquet(path_glob)
            .filter(pl.col("project_split").is_in(("train", "validation")))
            .collect()
        )
    except pl.exceptions.PolarsError as exc:
        raise RankingTrainingBuildError(f"cannot read closed feature matrices: {exc}") from exc


def _lightgbm_parameters(
    model_id: ModelId,
    config: RankerTrainingConfig,
    runtime_seed: int,
    max_threads: int,
) -> dict[str, object]:
    objective = (
        config.pointwise_objective if model_id == "pointwise" else config.lambdamart_objective
    )
    return {
        "objective": objective,
        "metric": "ndcg",
        "ndcg_eval_at": list(config.ndcg_eval_at),
        "label_gain": [gain for _, gain in OFFICIAL_GAIN_MAPPING],
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_data_in_leaf": config.min_data_in_leaf,
        "max_bin": config.max_bin,
        "lambda_l1": config.lambda_l1,
        "lambda_l2": config.lambda_l2,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": max_threads,
        "seed": runtime_seed,
        "feature_fraction_seed": runtime_seed,
        "bagging_seed": runtime_seed,
        "data_random_seed": runtime_seed,
        "verbosity": -1,
    }


def _predict(
    booster: lgb.Booster,
    features: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    best_iteration: int,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    predictions = np.asarray(
        booster.predict(features, num_iteration=best_iteration), dtype=np.float64
    ).reshape(-1)
    if len(predictions) != features.shape[0] or not np.isfinite(predictions).all():
        raise RankingTrainingBuildError("LightGBM produced invalid predictions")
    return predictions


def _bounded_indices(
    split: PreparedSplit, limit: int
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    selected: list[int] = []
    offset = 0
    for group_size in split.group_sizes:
        if selected and len(selected) + group_size > limit:
            break
        selected.extend(range(offset, offset + group_size))
        offset += group_size
        if len(selected) >= limit:
            break
    if not selected:
        selected.extend(range(min(limit, split.features.shape[0])))
    return np.asarray(selected, dtype=np.int64)


def _history_rows(
    model_id: ModelId, history: dict[str, dict[str, list[float]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    validation = history.get("validation", {})
    for metric_name in sorted(validation):
        if not metric_name.startswith("ndcg@"):
            continue
        cutoff = int(metric_name.split("@", maxsplit=1)[1])
        rows.extend(
            {
                "model_id": model_id,
                "iteration": iteration,
                "metric": "ndcg",
                "cutoff": cutoff,
                "value": float(value),
            }
            for iteration, value in enumerate(validation[metric_name], start=1)
        )
    return rows


def _importance_rows(model_id: ModelId, booster: lgb.Booster) -> list[dict[str, object]]:
    split_values = booster.feature_importance(importance_type="split")
    gain_values = booster.feature_importance(importance_type="gain")
    return [
        {
            "model_id": model_id,
            "feature": feature,
            "split_importance": int(split_value),
            "gain_importance": float(gain_value),
        }
        for feature, split_value, gain_value in zip(
            FEATURE_NAMES, split_values, gain_values, strict=True
        )
    ]


def _explanation_rows(
    model_id: ModelId,
    booster: lgb.Booster,
    split: PreparedSplit,
    indices: np.ndarray[tuple[int], np.dtype[np.int64]],
    best_iteration: int,
) -> list[dict[str, object]]:
    contributions = np.asarray(
        booster.predict(
            split.features[indices],
            num_iteration=best_iteration,
            pred_contrib=True,
        ),
        dtype=np.float64,
    )
    if contributions.shape != (len(indices), len(FEATURE_NAMES) + 1):
        raise RankingTrainingBuildError("LightGBM contribution matrix has incompatible shape")
    names = (*FEATURE_NAMES, "__bias__")
    rows: list[dict[str, object]] = []
    selected_frame = split.frame[indices.tolist()]
    for row_offset, feature_values in enumerate(contributions):
        query_id = int(selected_frame["query_id"][row_offset])
        product_id = str(selected_frame["product_id"][row_offset])
        rows.extend(
            {
                "model_id": model_id,
                "query_id": query_id,
                "product_id": product_id,
                "feature": feature_name,
                "contribution": float(value),
            }
            for feature_name, value in zip(names, feature_values, strict=True)
        )
    return rows


def _train_one(
    model_id: ModelId,
    population: PreparedPopulation,
    config: ResolvedConfig,
    model_path: Path,
) -> _TrainedModel:
    ranker_config = config.config.ranker_training
    parameters = _lightgbm_parameters(
        model_id,
        ranker_config,
        config.config.runtime.seed,
        config.config.runtime.max_threads,
    )
    train_set = lgb.Dataset(
        population.train.features,
        label=population.train.labels,
        group=list(population.train.group_sizes),
        feature_name=list(FEATURE_NAMES),
        categorical_feature=list(population.manifest.categorical_features),
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        population.validation.features,
        label=population.validation.labels,
        group=list(population.validation.group_sizes),
        feature_name=list(FEATURE_NAMES),
        categorical_feature=list(population.manifest.categorical_features),
        reference=train_set,
        free_raw_data=True,
    )
    history: dict[str, dict[str, list[float]]] = {}
    started = time.perf_counter()
    try:
        booster = lgb.train(
            parameters,
            train_set,
            num_boost_round=ranker_config.max_boost_rounds,
            valid_sets=[validation_set],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(ranker_config.early_stopping_rounds, verbose=False),
                lgb.record_evaluation(history),
            ],
        )
    except LightGBMError as exc:
        raise RankingTrainingBuildError(f"{model_id} LightGBM training failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    best_iteration = booster.best_iteration or booster.current_iteration()
    if best_iteration < 1:
        raise RankingTrainingBuildError(f"{model_id} produced no boosting iterations")
    booster.save_model(str(model_path), num_iteration=best_iteration)
    model_bytes, model_sha256 = _sha256_file(model_path)
    validation_predictions = _predict(booster, population.validation.features, best_iteration)
    importance = _importance_rows(model_id, booster)
    explanation_indices = _bounded_indices(population.validation, ranker_config.explanation_rows)
    explanations = _explanation_rows(
        model_id,
        booster,
        population.validation,
        explanation_indices,
        best_iteration,
    )
    best_metrics = tuple(
        (
            cutoff,
            float(history["validation"][f"ndcg@{cutoff}"][best_iteration - 1]),
        )
        for cutoff in ranker_config.ndcg_eval_at
    )
    filename: Literal["pointwise-lightgbm.txt", "lambdamart.txt"] = (
        POINTWISE_MODEL_FILENAME if model_id == "pointwise" else LAMBDAMART_MODEL_FILENAME
    )
    objective: Literal["regression_l2", "lambdarank"] = (
        ranker_config.pointwise_objective
        if model_id == "pointwise"
        else ranker_config.lambdamart_objective
    )
    summary = RankerModelSummary(
        model_id=model_id,
        objective=objective,
        model_filename=filename,
        model_sha256=model_sha256,
        model_bytes=model_bytes,
        best_iteration=best_iteration,
        trained_iterations=max(len(values) for values in history.get("validation", {}).values()),
        training_seconds=elapsed,
        parameters_json=_canonical_mapping(parameters),
        validation_best_ndcg=best_metrics,
    )
    return _TrainedModel(
        summary=summary,
        validation_predictions=validation_predictions,
        history_rows=_history_rows(model_id, history),
        importance_rows=importance,
        explanation_rows=explanations,
    )


def _parity_rows(
    population: PreparedPopulation,
    trained: dict[ModelId, _TrainedModel],
    reloaded: dict[ModelId, lgb.Booster],
    limit: int,
) -> tuple[list[dict[str, object]], float]:
    indices = _bounded_indices(population.validation, limit)
    selected = population.validation.frame[indices.tolist()]
    rows: list[dict[str, object]] = []
    maximum_delta = 0.0
    model_ids: tuple[ModelId, ModelId] = ("pointwise", "lambdamart")
    for model_id in model_ids:
        summary = trained[model_id].summary
        before = trained[model_id].validation_predictions[indices]
        after = _predict(
            reloaded[model_id], population.validation.features[indices], summary.best_iteration
        )
        deltas = np.abs(before - after)
        maximum_delta = max(maximum_delta, float(deltas.max(initial=0.0)))
        product_ids = tuple(cast(list[str], selected["product_id"].to_list()))
        query_ids = tuple(cast(list[int], selected["query_id"].to_list()))
        before_ranks = stable_grouped_prediction_ranks(query_ids, product_ids, before)
        after_ranks = stable_grouped_prediction_ranks(query_ids, product_ids, after)
        rows.extend(
            {
                "model_id": model_id,
                "query_id": int(selected["query_id"][offset]),
                "product_id": product_id,
                "trained_prediction": float(before[offset]),
                "reloaded_prediction": float(after[offset]),
                "absolute_delta": float(deltas[offset]),
                "trained_rank": before_ranks[offset],
                "reloaded_rank": after_ranks[offset],
            }
            for offset, product_id in enumerate(product_ids)
        )
    if maximum_delta > 1e-12 or any(row["trained_rank"] != row["reloaded_rank"] for row in rows):
        raise RankingTrainingBuildError("serialized model reload changed predictions or ranks")
    return rows, maximum_delta


def _reuse(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    feature: _FeatureDependency,
    store: ArtifactStore,
) -> RankingTrainingBuildResult:
    artifact = store.load(ranking_models_artifact_id(release, config.sha256, profile))
    if artifact.manifest.dependencies != _artifact_dependency(feature):
        raise RankingTrainingValidationError("ranking-model feature dependency is incompatible")
    manifest = load_ranking_models_manifest(artifact.path / RANKING_MODELS_FILENAME)
    if (
        manifest.artifact_id != artifact.manifest.artifact_id
        or manifest.config_sha256 != config.sha256
        or manifest.profile != profile
        or manifest.feature_manifest_sha256 != feature.artifact.manifest_sha256
    ):
        raise RankingTrainingValidationError("ranking-model metadata identity is incompatible")
    return RankingTrainingBuildResult(artifact, manifest, True)


def build_rankers(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    profile: Profile | None = None,
    artifact_store: ArtifactStore | None = None,
) -> RankingTrainingBuildResult:
    """Build/reuse identical-population pointwise and LambdaMART model artifacts."""
    selected_profile: Profile = profile or config.config.evaluation.default_profile
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    feature = _load_feature_dependency(release, config, selected_profile, store)
    artifact_id = ranking_models_artifact_id(release, config.sha256, selected_profile)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse(release, config, selected_profile, feature, store)

    try:
        population = prepare_training_population(
            _read_training_rows(feature),
            feature.registry,
            config.config.ranker_training,
        )
    except TrainingPopulationError as exc:
        raise RankingTrainingBuildError(f"invalid training population: {exc}") from exc
    rss_limit = config.config.runtime.rss_limit_mb * 1024 * 1024
    matrix_peak = _peak_rss_bytes()
    initial_resource = TrainingResourceMeasurement(
        matrix_peak_rss_bytes=matrix_peak,
        training_peak_rss_bytes=matrix_peak,
        reload_peak_rss_bytes=matrix_peak,
        peak_rss_bytes=matrix_peak,
        rss_limit_bytes=rss_limit,
        artifact_payload_bytes=0,
        passed=matrix_peak <= rss_limit,
    )
    if not initial_resource.passed:
        raise RankingTrainingResourceError(initial_resource)

    transaction = store.stage(
        artifact_type="ranking-models",
        dataset_version=release.manifest.dataset_version,
        profile=selected_profile,
        component_version=config.config.ranker_training.component_version,
        config_sha256=config.sha256,
        code_revision=code_revision,
        dependencies=_artifact_dependency(feature),
    )
    try:
        with transaction:
            staging_root = transaction.path(RANKING_MODELS_FILENAME).parent
            population.audit.write_parquet(
                transaction.path(POPULATION_AUDIT_FILENAME),
                compression="zstd",
                statistics=True,
            )
            transaction.path(TRAINING_POPULATION_FILENAME).write_text(
                _canonical_json(population.manifest), encoding="utf-8"
            )
            trained: dict[ModelId, _TrainedModel] = {}
            trained["pointwise"] = _train_one(
                "pointwise",
                population,
                config,
                transaction.path(POINTWISE_MODEL_FILENAME),
            )
            gc.collect()
            trained["lambdamart"] = _train_one(
                "lambdamart",
                population,
                config,
                transaction.path(LAMBDAMART_MODEL_FILENAME),
            )
            training_peak = _peak_rss_bytes()
            history_rows = trained["pointwise"].history_rows + trained["lambdamart"].history_rows
            pl.DataFrame(history_rows).sort("model_id", "cutoff", "iteration").write_parquet(
                transaction.path(VALIDATION_HISTORY_FILENAME),
                compression="zstd",
                statistics=True,
            )
            importance_rows = (
                trained["pointwise"].importance_rows + trained["lambdamart"].importance_rows
            )
            pl.DataFrame(importance_rows).sort("model_id", "feature").write_parquet(
                transaction.path(FEATURE_IMPORTANCE_FILENAME),
                compression="zstd",
                statistics=True,
            )
            explanation_rows = (
                trained["pointwise"].explanation_rows + trained["lambdamart"].explanation_rows
            )
            pl.DataFrame(explanation_rows).sort(
                "model_id", "query_id", "product_id", "feature"
            ).write_parquet(
                transaction.path(EXPLANATION_SAMPLE_FILENAME),
                compression="zstd",
                statistics=True,
            )
            reloaded: dict[ModelId, lgb.Booster] = {
                "pointwise": lgb.Booster(model_file=str(staging_root / POINTWISE_MODEL_FILENAME)),
                "lambdamart": lgb.Booster(model_file=str(staging_root / LAMBDAMART_MODEL_FILENAME)),
            }
            for model_id, booster in reloaded.items():
                if tuple(booster.feature_name()) != FEATURE_NAMES:
                    raise RankingTrainingBuildError(
                        f"{model_id} reload changed feature names or order"
                    )
            parity_rows, maximum_delta = _parity_rows(
                population,
                trained,
                reloaded,
                config.config.ranker_training.reload_parity_rows,
            )
            pl.DataFrame(parity_rows).sort("model_id", "query_id", "product_id").write_parquet(
                transaction.path(RELOAD_PARITY_FILENAME),
                compression="zstd",
                statistics=True,
            )
            reload_peak = _peak_rss_bytes()
            payload_bytes = _directory_bytes(staging_root)
            measurement = TrainingResourceMeasurement(
                matrix_peak_rss_bytes=matrix_peak,
                training_peak_rss_bytes=training_peak,
                reload_peak_rss_bytes=reload_peak,
                peak_rss_bytes=max(matrix_peak, training_peak, reload_peak),
                rss_limit_bytes=rss_limit,
                artifact_payload_bytes=payload_bytes,
                passed=max(matrix_peak, training_peak, reload_peak) <= rss_limit,
            )
            if not measurement.passed:
                raise RankingTrainingResourceError(measurement)
            checks = tuple(
                sorted(
                    (
                        RankingTrainingCheck(
                            check_id="identical_model_population",
                            detail=(
                                "pointwise and LambdaMART use identical rows, features, and groups"
                            ),
                        ),
                        RankingTrainingCheck(
                            check_id="official_gain_mapping",
                            detail="labels 0/1/2/3 retain gains 0/0.01/0.1/1 separately",
                        ),
                        RankingTrainingCheck(
                            check_id="reload_prediction_parity",
                            detail=f"maximum serialized prediction delta {maximum_delta:.3g}",
                        ),
                        RankingTrainingCheck(
                            check_id="resource_gate",
                            detail=(
                                f"peak RSS {measurement.peak_rss_bytes} <= "
                                f"{measurement.rss_limit_bytes} bytes"
                            ),
                        ),
                        RankingTrainingCheck(
                            check_id="test_split_quarantined",
                            detail="only project train and validation rows were read for fitting",
                        ),
                        RankingTrainingCheck(
                            check_id="validation_early_stopping",
                            detail="both objectives select iterations only from validation NDCG",
                        ),
                    ),
                    key=lambda check: check.check_id,
                )
            )
            manifest = RankingModelsManifest(
                artifact_id=artifact_id,
                dataset_version=release.manifest.dataset_version,
                config_sha256=config.sha256,
                profile=selected_profile,
                lightgbm_version=lgb.__version__,
                feature_artifact_id=feature.artifact.manifest.artifact_id,
                feature_manifest_sha256=feature.artifact.manifest_sha256,
                feature_registry_sha256=feature.manifest.registry_sha256,
                feature_state_sha256=feature.manifest.state_sha256,
                feature_names=FEATURE_NAMES,
                feature_dtypes=tuple(
                    (registered.name, registered.dtype) for registered in feature.registry.features
                ),
                categorical_features=population.manifest.categorical_features,
                population=population.manifest,
                models=(trained["pointwise"].summary, trained["lambdamart"].summary),
                reload_parity_rows=len(parity_rows),
                maximum_reload_prediction_delta=maximum_delta,
                explanation_rows=len(
                    {
                        (
                            cast(str, row["model_id"]),
                            cast(int, row["query_id"]),
                            cast(str, row["product_id"]),
                        )
                        for row in explanation_rows
                    }
                ),
                resource=measurement,
                checks=checks,
            )
            transaction.path(RANKING_MODELS_FILENAME).write_text(
                _canonical_json(manifest), encoding="utf-8"
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse(release, config, selected_profile, feature, store)
    except LightGBMError as exc:
        raise RankingTrainingBuildError(f"cannot load serialized LightGBM models: {exc}") from exc
    return RankingTrainingBuildResult(artifact, manifest, False)


def load_rankers(store: ArtifactStore, artifact_id: str) -> LoadedRankers:
    """Recursively verify and load both persisted rankers without training."""
    try:
        artifact = store.load(artifact_id)
        manifest = load_ranking_models_manifest(artifact.path / RANKING_MODELS_FILENAME)
        pointwise = lgb.Booster(model_file=str(artifact.path / POINTWISE_MODEL_FILENAME))
        lambdamart = lgb.Booster(model_file=str(artifact.path / LAMBDAMART_MODEL_FILENAME))
    except (OSError, RuntimeError, LightGBMError) as exc:
        raise RankingTrainingValidationError(
            f"cannot load ranking models {artifact_id}: {exc}"
        ) from exc
    if manifest.artifact_id != artifact.manifest.artifact_id:
        raise RankingTrainingValidationError("ranking-model manifest artifact ID is incompatible")
    for model_id, booster in (("pointwise", pointwise), ("lambdamart", lambdamart)):
        if tuple(booster.feature_name()) != manifest.feature_names:
            raise RankingTrainingValidationError(
                f"{model_id} feature names/order differ from the manifest"
            )
    return LoadedRankers(artifact, manifest, pointwise, lambdamart)


__all__ = [
    "EXPLANATION_SAMPLE_FILENAME",
    "FEATURE_IMPORTANCE_FILENAME",
    "LAMBDAMART_MODEL_FILENAME",
    "POINTWISE_MODEL_FILENAME",
    "POPULATION_AUDIT_FILENAME",
    "RANKING_MODELS_FILENAME",
    "RELOAD_PARITY_FILENAME",
    "TRAINING_POPULATION_FILENAME",
    "VALIDATION_HISTORY_FILENAME",
    "LoadedRankers",
    "RankPrediction",
    "RankerModelSummary",
    "RankingModelsManifest",
    "RankingTrainingBuildError",
    "RankingTrainingBuildResult",
    "RankingTrainingError",
    "RankingTrainingResourceError",
    "RankingTrainingValidationError",
    "TrainingResourceMeasurement",
    "build_rankers",
    "load_rankers",
    "load_ranking_models_manifest",
    "ranking_models_artifact_id",
]
