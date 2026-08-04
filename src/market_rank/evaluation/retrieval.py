"""Fixed-catalog sparse/dense/RRF evaluation with grouped confidence intervals."""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from contextlib import ExitStack
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
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.data.foundation import (
    FOUNDATION_MANIFEST_FILENAME,
    JUDGED_POOLS_FILENAME,
    QUERIES_FILENAME,
    CatalogId,
    DataFoundationManifest,
    foundation_artifact_id,
    load_foundation_manifest,
)
from market_rank.evaluation.metrics import (
    RETRIEVAL_PROTOCOL,
    EsciLabel,
    Judgment,
    evaluate_ranked_products,
)
from market_rank.retrieval.dense import (
    DENSE_METADATA_FILENAME,
    DenseEncoder,
    DenseIndexMetadata,
    SentenceTransformerEncoder,
    dense_artifact_id,
    load_dense_index,
    load_dense_metadata,
)
from market_rank.retrieval.hybrid import HybridResult, fuse_rrf
from market_rank.retrieval.sparse import (
    SPARSE_METADATA_FILENAME,
    SparseIndexMetadata,
    load_sparse_index,
    load_sparse_metadata,
    sparse_artifact_id,
)

RETRIEVAL_EVALUATION_FILENAME = "retrieval-evaluation.json"
AGGREGATE_METRICS_FILENAME = "aggregate-metrics.parquet"
COMPARISON_METRICS_FILENAME = "comparison-metrics.parquet"
CANDIDATE_DIRECTORY = "candidates"
QUERY_METRIC_DIRECTORY = "query-metrics"

Profile = Literal["development", "portfolio"]
Stage = Literal["sparse", "dense", "hybrid"]

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]

_STAGES: tuple[Stage, ...] = ("sparse", "dense", "hybrid")
_THRESHOLDS: tuple[tuple[str, frozenset[EsciLabel]], ...] = (
    ("exact", frozenset({"E"})),
    ("exact_substitute", frozenset({"E", "S"})),
)
_HIGHER_IS_BETTER = frozenset(
    {"judged_recall", "exact_hit", "judged_mrr", "known_judgment_coverage"}
)
_OFFICIAL_GAINS: dict[EsciLabel, float] = {"I": 0.0, "C": 0.01, "S": 0.1, "E": 1.0}


class RetrievalEvaluationError(RuntimeError):
    """Base exception for Goldfish 009 evaluation construction and loading."""


class RetrievalEvaluationBuildError(RetrievalEvaluationError):
    """Raised when fixed-cohort retrieval evaluation cannot be constructed."""


class RetrievalEvaluationValidationError(RetrievalEvaluationError):
    """Raised when a promoted retrieval evaluation is incompatible or corrupt."""


class HybridResourceError(RetrievalEvaluationBuildError):
    """Raised when the combined sparse+dense process exceeds its RSS gate."""

    def __init__(self, measurement: CombinedResourceMeasurement) -> None:
        super().__init__(
            f"combined retrieval peak RSS {measurement.peak_rss_bytes} exceeds "
            f"the {measurement.rss_limit_bytes}-byte limit"
        )
        self.measurement = measurement


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CombinedResourceMeasurement(_StrictModel):
    """Observed simultaneous sparse+dense load and evaluation process facts."""

    load_peak_rss_bytes: int = Field(strict=True, ge=0)
    evaluation_peak_rss_bytes: int = Field(strict=True, ge=0)
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    sparse_artifact_bytes: int = Field(strict=True, ge=0)
    dense_artifact_bytes: int = Field(strict=True, ge=0)
    evaluation_artifact_bytes: int = Field(strict=True, ge=0)
    query_count: int = Field(strict=True, ge=0)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if self.peak_rss_bytes != max(self.load_peak_rss_bytes, self.evaluation_peak_rss_bytes):
            raise ValueError("combined peak RSS does not match phase observations")
        if self.passed != (self.peak_rss_bytes <= self.rss_limit_bytes):
            raise ValueError("combined resource status does not match RSS gate")
        return self


class StageSummary(_StrictModel):
    """Candidate volume, empty-result, and warm stage-latency facts."""

    stage: Stage
    query_count: int = Field(strict=True, ge=1)
    candidate_rows: int = Field(strict=True, ge=0)
    empty_queries: int = Field(strict=True, ge=0)
    latency_p50_ms: float = Field(ge=0.0, allow_inf_nan=False)
    latency_p95_ms: float = Field(ge=0.0, allow_inf_nan=False)
    latency_maximum_ms: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.empty_queries > self.query_count:
            raise ValueError("stage empty-query count exceeds query cohort")
        if not self.latency_p50_ms <= self.latency_p95_ms <= self.latency_maximum_ms:
            raise ValueError("stage latency percentiles are not ordered")
        return self


class EvaluationCheck(_StrictModel):
    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class RetrievalEvaluationManifest(_StrictModel):
    """Strict lineage, cohort, protocol, fusion, output, and resource contract."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    profile: Profile
    protocol: Literal["retrieval_catalog_task1_us_v1"] = RETRIEVAL_PROTOCOL
    population_id: str = Field(strict=True, min_length=1)
    catalog_id: CatalogId
    catalog_membership_sha256: Sha256Digest
    foundation_artifact_id: str = Field(strict=True, min_length=1)
    foundation_manifest_sha256: Sha256Digest
    sparse_artifact_id: str = Field(strict=True, min_length=1)
    sparse_manifest_sha256: Sha256Digest
    dense_artifact_id: str = Field(strict=True, min_length=1)
    dense_manifest_sha256: Sha256Digest
    component_version: Literal["retrieval-eval-v1"]
    fusion_version: Literal["rrf-v1"]
    rrf_constant: int = Field(strict=True, ge=1)
    sparse_top_k: int = Field(strict=True, ge=1)
    dense_top_k: int = Field(strict=True, ge=1)
    union_top_k: int = Field(strict=True, ge=1)
    cutoffs: tuple[int, ...]
    relevant_thresholds: tuple[tuple[str, tuple[str, ...]], ...]
    bootstrap_method: Literal["normalized-query-group-v1"] = "normalized-query-group-v1"
    bootstrap_replicates: int = Field(strict=True, ge=100)
    bootstrap_seed: int = Field(strict=True, ge=0)
    query_count: int = Field(strict=True, ge=1)
    normalized_query_groups: int = Field(strict=True, ge=1)
    candidate_partitions: int = Field(strict=True, ge=1)
    query_metric_partitions: int = Field(strict=True, ge=1)
    aggregate_metric_rows: int = Field(strict=True, ge=1)
    comparison_metric_rows: int = Field(strict=True, ge=1)
    stages: tuple[StageSummary, ...]
    resource: CombinedResourceMeasurement
    checks: tuple[EvaluationCheck, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if tuple(stage.stage for stage in self.stages) != _STAGES:
            raise ValueError("stage summaries must use sparse, dense, hybrid order")
        if self.cutoffs != tuple(sorted(set(self.cutoffs))):
            raise ValueError("evaluation cutoffs must be unique and sorted")
        if self.relevant_thresholds != (
            ("exact", ("E",)),
            ("exact_substitute", ("E", "S")),
        ):
            raise ValueError("retrieval relevance thresholds are incompatible")
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(check_ids) != len(set(check_ids)):
            raise ValueError("evaluation checks must be unique and sorted")
        return self


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationBuildResult:
    artifact: LoadedArtifact
    manifest: RetrievalEvaluationManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class _Dependencies:
    foundation: LoadedArtifact
    foundation_manifest: DataFoundationManifest
    sparse: LoadedArtifact
    sparse_metadata: SparseIndexMetadata
    dense: LoadedArtifact
    dense_metadata: DenseIndexMetadata


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


def _artifact_payload_bytes(artifact: LoadedArtifact) -> int:
    return sum(item.size_bytes for item in artifact.manifest.files)


def retrieval_evaluation_artifact_id(
    release: ResolvedReleaseManifest,
    config_sha256: str,
    profile: Profile,
) -> str:
    """Return deterministic Goldfish 009 evaluation coordinates."""
    return "/".join(
        (
            "retrieval-evaluation",
            release.manifest.dataset_version,
            profile,
            "retrieval-eval-v1",
            config_sha256,
        )
    )


def _load_dependencies(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    store: ArtifactStore,
) -> _Dependencies:
    try:
        foundation = store.load(foundation_artifact_id(release, config.sha256))
        foundation_manifest = load_foundation_manifest(
            foundation.path / FOUNDATION_MANIFEST_FILENAME
        )
        sparse = store.load(sparse_artifact_id(release, config.sha256))
        sparse_metadata = load_sparse_metadata(sparse.path / SPARSE_METADATA_FILENAME)
        dense = store.load(dense_artifact_id(release, config.sha256))
        dense_metadata = load_dense_metadata(dense.path / DENSE_METADATA_FILENAME)
    except (OSError, RuntimeError) as exc:
        raise RetrievalEvaluationBuildError(
            "compatible Goldfish 006, 007, and 008 artifacts are required before evaluation"
        ) from exc
    if (
        foundation_manifest.config_sha256 != config.sha256
        or sparse_metadata.config_sha256 != config.sha256
        or dense_metadata.config_sha256 != config.sha256
        or sparse_metadata.foundation_manifest_sha256 != foundation.manifest_sha256
        or dense_metadata.foundation_manifest_sha256 != foundation.manifest_sha256
        or sparse_metadata.catalog_id != foundation_manifest.catalog_id
        or dense_metadata.catalog_id != foundation_manifest.catalog_id
        or sparse_metadata.catalog_membership_sha256 != dense_metadata.catalog_membership_sha256
        or sparse_metadata.catalog_membership_sha256
        != _catalog_membership_hash(foundation_manifest)
        or sparse_metadata.document_count != dense_metadata.document_count
        or sparse_metadata.document_count != foundation_manifest.catalog_products
    ):
        raise RetrievalEvaluationBuildError(
            "sparse, dense, and foundation catalog identity or lineage is incompatible"
        )
    return _Dependencies(
        foundation=foundation,
        foundation_manifest=foundation_manifest,
        sparse=sparse,
        sparse_metadata=sparse_metadata,
        dense=dense,
        dense_metadata=dense_metadata,
    )


def _catalog_membership_hash(manifest: DataFoundationManifest) -> str:
    for table in manifest.tables:
        if table.filename == "catalog-membership.parquet":
            return table.sha256
    raise RetrievalEvaluationBuildError("foundation omits catalog membership integrity")


def _dependencies_as_manifest(dependencies: _Dependencies) -> tuple[ArtifactDependency, ...]:
    result = (
        ArtifactDependency(
            artifact_id=dependencies.foundation.manifest.artifact_id,
            manifest_sha256=dependencies.foundation.manifest_sha256,
        ),
        ArtifactDependency(
            artifact_id=dependencies.sparse.manifest.artifact_id,
            manifest_sha256=dependencies.sparse.manifest_sha256,
        ),
        ArtifactDependency(
            artifact_id=dependencies.dense.manifest.artifact_id,
            manifest_sha256=dependencies.dense.manifest_sha256,
        ),
    )
    return tuple(sorted(result, key=lambda item: item.artifact_id))


def load_retrieval_evaluation_manifest(path: Path) -> RetrievalEvaluationManifest:
    """Load the strict Goldfish 009 report manifest."""
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        return RetrievalEvaluationManifest.model_validate(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise RetrievalEvaluationValidationError(
            f"cannot load retrieval evaluation manifest {path}: {exc}"
        ) from exc


class _PartitionWriter:
    def __init__(self, root: Path, schema: pl.Schema, row_limit: int) -> None:
        self.root = root
        self.schema = schema
        self.row_limit = row_limit
        self.root.mkdir(parents=True, exist_ok=False)
        self._rows: list[dict[str, object]] = []
        self._paths: list[Path] = []

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._paths)

    def append(self, row: dict[str, object]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.row_limit:
            self._flush()

    def _flush(self) -> None:
        if not self._rows and self._paths:
            return
        path = self.root / f"part-{len(self._paths):05d}.parquet"
        frame = pl.DataFrame(self._rows, schema=self.schema)
        frame.write_parquet(path, compression="zstd", statistics=True)
        self._paths.append(path)
        self._rows.clear()

    def close(self) -> None:
        if self._rows or not self._paths:
            self._flush()


_CANDIDATE_SCHEMA = pl.Schema(
    {
        "profile": pl.String,
        "protocol": pl.String,
        "query_id": pl.Int64,
        "normalized_query_sha256": pl.String,
        "locale": pl.String,
        "project_split": pl.String,
        "stage": pl.String,
        "product_id": pl.String,
        "rank": pl.UInt32,
        "raw_score": pl.Float32,
        "rrf_score": pl.Float32,
        "sparse_score": pl.Float32,
        "sparse_rank": pl.UInt32,
        "sparse_retriever_id": pl.String,
        "sparse_index_id": pl.String,
        "dense_score": pl.Float32,
        "dense_rank": pl.UInt32,
        "dense_retriever_id": pl.String,
        "dense_index_id": pl.String,
        "source_count": pl.UInt8,
        "retriever_id": pl.String,
        "index_id": pl.String,
        "latency_ms": pl.Float64,
        "degraded": pl.Boolean,
    }
)

_QUERY_METRIC_SCHEMA = pl.Schema(
    {
        "profile": pl.String,
        "protocol": pl.String,
        "population_id": pl.String,
        "catalog_id": pl.String,
        "query_id": pl.Int64,
        "normalized_query_sha256": pl.String,
        "source": pl.String,
        "project_split": pl.String,
        "query_length_bucket": pl.String,
        "exact_presence": pl.String,
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
        "catalog_id": pl.String,
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
        "comparison_id": pl.String,
        "selected_baseline_stage": pl.String,
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


def _query_length_bucket(query: str) -> str:
    count = len(query.split())
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def _stage_candidate_rows(
    *,
    profile: Profile,
    query: dict[str, object],
    stage: Stage,
    candidates: tuple[Any, ...],
    latency_ms: float,
    hybrid_artifact_id: str,
    degraded: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if stage == "sparse":
            sparse_score, sparse_rank = candidate.raw_score, candidate.one_based_rank
            sparse_retriever_id, sparse_index_id = candidate.retriever_id, candidate.index_id
            dense_score, dense_rank = None, None
            dense_retriever_id, dense_index_id = None, None
            raw_score, rrf_score = candidate.raw_score, None
            source_count = 1
            index_id = candidate.index_id
        elif stage == "dense":
            sparse_score, sparse_rank = None, None
            sparse_retriever_id, sparse_index_id = None, None
            dense_score, dense_rank = candidate.raw_score, candidate.one_based_rank
            dense_retriever_id, dense_index_id = candidate.retriever_id, candidate.index_id
            raw_score, rrf_score = candidate.raw_score, None
            source_count = 1
            index_id = candidate.index_id
        else:
            sparse_score, sparse_rank = candidate.sparse_score, candidate.sparse_rank
            sparse_retriever_id = candidate.sparse_retriever_id
            sparse_index_id = candidate.sparse_index_id
            dense_score, dense_rank = candidate.dense_score, candidate.dense_rank
            dense_retriever_id = candidate.dense_retriever_id
            dense_index_id = candidate.dense_index_id
            raw_score, rrf_score = None, candidate.rrf_score
            source_count = candidate.source_count
            index_id = hybrid_artifact_id
        rows.append(
            {
                "profile": profile,
                "protocol": RETRIEVAL_PROTOCOL,
                "query_id": query["query_id"],
                "normalized_query_sha256": query["normalized_query_sha256"],
                "locale": query["locale"],
                "project_split": query["project_split"],
                "stage": stage,
                "product_id": candidate.product_id,
                "rank": candidate.one_based_rank,
                "raw_score": raw_score,
                "rrf_score": rrf_score,
                "sparse_score": sparse_score,
                "sparse_rank": sparse_rank,
                "sparse_retriever_id": sparse_retriever_id,
                "sparse_index_id": sparse_index_id,
                "dense_score": dense_score,
                "dense_rank": dense_rank,
                "dense_retriever_id": dense_retriever_id,
                "dense_index_id": dense_index_id,
                "source_count": source_count,
                "retriever_id": candidate.retriever_id,
                "index_id": index_id,
                "latency_ms": latency_ms,
                "degraded": degraded,
            }
        )
    return rows


def _metric_rows(
    *,
    profile: Profile,
    population_id: str,
    catalog_id: str,
    query: dict[str, object],
    stage: Stage,
    ranked_ids: tuple[str, ...],
    judgments: tuple[Judgment, ...],
    cutoffs: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    exact_presence = "present" if any(item.label == "E" for item in judgments) else "absent"
    for threshold_id, labels in _THRESHOLDS:
        for cutoff in cutoffs:
            for record in evaluate_ranked_products(
                RETRIEVAL_PROTOCOL,
                ranked_ids,
                judgments,
                k=cutoff,
                relevant_labels=labels,
            ):
                rows.append(
                    {
                        "profile": profile,
                        "protocol": record.protocol,
                        "population_id": population_id,
                        "catalog_id": catalog_id,
                        "query_id": query["query_id"],
                        "normalized_query_sha256": query["normalized_query_sha256"],
                        "source": query["source"],
                        "project_split": query["project_split"],
                        "query_length_bucket": _query_length_bucket(str(query["normalized_query"])),
                        "exact_presence": exact_presence,
                        "stage": stage,
                        "threshold_id": threshold_id,
                        "metric": record.metric,
                        "cutoff": cutoff,
                        "value": record.value,
                        "returned_count": record.returned_count,
                        "judged_count": record.judged_count,
                        "unjudged_count": record.unjudged_count,
                        "relevant_judgment_count": record.relevant_judgment_count,
                        "empty_result": not ranked_ids,
                    }
                )
    return rows


def _stable_seed(seed: int, key: str) -> int:
    digest = sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _bootstrap_interval(
    frame: pl.DataFrame,
    *,
    seed: int,
    key: str,
    replicates: int,
    batch_replicates: int,
) -> tuple[float, float, int]:
    grouped = (
        frame.group_by("normalized_query_sha256")
        .agg(pl.col("value").sum().alias("value_sum"), pl.len().alias("query_count"))
        .sort("normalized_query_sha256")
    )
    sums = grouped["value_sum"].to_numpy()
    counts = grouped["query_count"].to_numpy()
    group_count = grouped.height
    rng = np.random.default_rng(_stable_seed(seed, key))
    samples = np.empty(replicates, dtype=np.float64)
    completed = 0
    while completed < replicates:
        size = min(batch_replicates, replicates - completed)
        indices = rng.integers(0, group_count, size=(size, group_count))
        sample_sums = sums[indices].sum(axis=1)
        sample_counts = counts[indices].sum(axis=1)
        samples[completed : completed + size] = sample_sums / sample_counts
        completed += size
    return (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
        group_count,
    )


def _summary_values(
    frame: pl.DataFrame,
    *,
    seed: int,
    key: str,
    replicates: int,
    batch_replicates: int,
) -> tuple[float, float, float, float, int]:
    values = frame["value"].to_numpy()
    lower, upper, groups = _bootstrap_interval(
        frame,
        seed=seed,
        key=key,
        replicates=replicates,
        batch_replicates=batch_replicates,
    )
    return float(values.mean()), float(np.median(values)), lower, upper, groups


def build_aggregate_metrics(
    query_metrics: pl.DataFrame,
    config: ResolvedConfig,
) -> pl.DataFrame:
    """Build overall and named-slice summaries with grouped bootstrap intervals."""
    evaluation = config.config.evaluation
    rows: list[dict[str, object]] = []
    base_keys = ["stage", "threshold_id", "metric", "cutoff"]
    dimensions: tuple[tuple[str, str | None], ...] = (
        ("all", None),
        ("query_length", "query_length_bucket"),
        ("source", "source"),
        ("project_split", "project_split"),
        ("exact_presence", "exact_presence"),
    )
    for dimension, column in dimensions:
        group_keys = [*base_keys, *([column] if column else [])]
        for partition in query_metrics.partition_by(group_keys, maintain_order=True):
            first = partition.row(0, named=True)
            slice_value = "all" if column is None else str(first[column])
            identity = (
                "|".join(str(first[key]) for key in (*base_keys,)) + f"|{dimension}|{slice_value}"
            )
            mean, median, lower, upper, groups = _summary_values(
                partition,
                seed=config.config.runtime.seed,
                key=identity,
                replicates=evaluation.bootstrap_replicates,
                batch_replicates=evaluation.bootstrap_batch_replicates,
            )
            rows.append(
                {
                    "profile": first["profile"],
                    "protocol": first["protocol"],
                    "population_id": first["population_id"],
                    "catalog_id": first["catalog_id"],
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
                    "normalized_query_groups": groups,
                    "empty_query_count": int(partition["empty_result"].sum()),
                    "returned_count": int(partition["returned_count"].sum()),
                    "judged_count": int(partition["judged_count"].sum()),
                    "unjudged_count": int(partition["unjudged_count"].sum()),
                    "relevant_judgment_count": int(partition["relevant_judgment_count"].sum()),
                    "bootstrap_replicates": evaluation.bootstrap_replicates,
                    "bootstrap_method": "normalized-query-group-v1",
                }
            )
    return pl.DataFrame(rows, schema=_AGGREGATE_SCHEMA).sort(
        "stage", "threshold_id", "metric", "cutoff", "slice_dimension", "slice_value"
    )


def _paired_comparison(
    query_metrics: pl.DataFrame,
    config: ResolvedConfig,
    *,
    threshold_id: str,
    metric: str,
    cutoff: int,
    baseline_stage: Literal["sparse", "dense"],
    comparison_id: str,
) -> dict[str, object]:
    keys = ["query_id", "normalized_query_sha256"]
    selected = query_metrics.filter(
        (pl.col("threshold_id") == threshold_id)
        & (pl.col("metric") == metric)
        & (pl.col("cutoff") == cutoff)
    )
    hybrid = selected.filter(pl.col("stage") == "hybrid").select(
        *keys, pl.col("value").alias("hybrid_value")
    )
    baseline = selected.filter(pl.col("stage") == baseline_stage).select(
        *keys, pl.col("value").alias("baseline_value")
    )
    paired = hybrid.join(baseline, on=keys, how="inner", validate="1:1")
    expected = selected.filter(pl.col("stage") == "hybrid").height
    if paired.height != expected:
        raise RetrievalEvaluationBuildError("paired comparison query cohorts are not identical")
    higher = metric in _HIGHER_IS_BETTER
    paired = paired.with_columns(
        (
            pl.col("hybrid_value") - pl.col("baseline_value")
            if higher
            else pl.col("baseline_value") - pl.col("hybrid_value")
        ).alias("value")
    )
    identity = f"{comparison_id}|{threshold_id}|{metric}|{cutoff}"
    mean, median, lower, upper, groups = _summary_values(
        paired,
        seed=config.config.runtime.seed,
        key=identity,
        replicates=config.config.evaluation.bootstrap_replicates,
        batch_replicates=config.config.evaluation.bootstrap_batch_replicates,
    )
    values = paired["value"].to_numpy()
    tolerance = 1e-12
    return {
        "profile": selected.item(0, "profile"),
        "protocol": RETRIEVAL_PROTOCOL,
        "comparison_id": comparison_id,
        "selected_baseline_stage": baseline_stage,
        "metric_direction": "higher" if higher else "lower",
        "threshold_id": threshold_id,
        "metric": metric,
        "cutoff": cutoff,
        "mean_improvement": mean,
        "median_improvement": median,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "query_count": paired.height,
        "normalized_query_groups": groups,
        "win_count": int(np.count_nonzero(values > tolerance)),
        "tie_count": int(np.count_nonzero(np.abs(values) <= tolerance)),
        "loss_count": int(np.count_nonzero(values < -tolerance)),
        "bootstrap_replicates": config.config.evaluation.bootstrap_replicates,
        "bootstrap_method": "normalized-query-group-v1",
    }


def build_comparison_metrics(
    query_metrics: pl.DataFrame,
    config: ResolvedConfig,
) -> pl.DataFrame:
    """Build paired hybrid-vs-source and hybrid-vs-best-single improvement intervals."""
    combinations = (
        query_metrics.select("threshold_id", "metric", "cutoff")
        .unique()
        .sort("threshold_id", "metric", "cutoff")
    )
    rows: list[dict[str, object]] = []
    for combination in combinations.iter_rows(named=True):
        threshold_id = str(combination["threshold_id"])
        metric = str(combination["metric"])
        cutoff = int(combination["cutoff"])
        source_means: dict[Literal["sparse", "dense"], float] = {}
        for source_stage in ("sparse", "dense"):
            source_means[source_stage] = cast(
                float,
                query_metrics.filter(
                    (pl.col("stage") == source_stage)
                    & (pl.col("threshold_id") == threshold_id)
                    & (pl.col("metric") == metric)
                    & (pl.col("cutoff") == cutoff)
                )["value"].mean(),
            )
            rows.append(
                _paired_comparison(
                    query_metrics,
                    config,
                    threshold_id=threshold_id,
                    metric=metric,
                    cutoff=cutoff,
                    baseline_stage=source_stage,
                    comparison_id=f"hybrid_minus_{source_stage}",
                )
            )
        if metric in _HIGHER_IS_BETTER:
            best: Literal["sparse", "dense"] = (
                "dense" if source_means["dense"] > source_means["sparse"] else "sparse"
            )
        else:
            best = "dense" if source_means["dense"] < source_means["sparse"] else "sparse"
        rows.append(
            _paired_comparison(
                query_metrics,
                config,
                threshold_id=threshold_id,
                metric=metric,
                cutoff=cutoff,
                baseline_stage=best,
                comparison_id="hybrid_minus_best_single",
            )
        )
    return pl.DataFrame(rows, schema=_COMPARISON_SCHEMA).sort(
        "comparison_id", "threshold_id", "metric", "cutoff"
    )


def _stage_summary(
    stage: Stage,
    latencies: list[float],
    candidate_rows: int,
    empty_queries: int,
) -> StageSummary:
    values = np.asarray(latencies, dtype=np.float64)
    return StageSummary(
        stage=stage,
        query_count=len(latencies),
        candidate_rows=candidate_rows,
        empty_queries=empty_queries,
        latency_p50_ms=float(np.percentile(values, 50)),
        latency_p95_ms=float(np.percentile(values, 95)),
        latency_maximum_ms=float(values.max()),
    )


def _load_evaluation_cohort(
    dependencies: _Dependencies,
    profile: Profile,
) -> tuple[pl.DataFrame, dict[int, tuple[Judgment, ...]]]:
    try:
        queries = pl.read_parquet(dependencies.foundation.path / QUERIES_FILENAME)
        if profile == "development":
            queries = queries.filter(pl.col("in_development"))
        queries = queries.sort("query_id")
        pools = pl.read_parquet(dependencies.foundation.path / JUDGED_POOLS_FILENAME).filter(
            pl.col("profile") == profile
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise RetrievalEvaluationBuildError(f"cannot load evaluation cohort: {exc}") from exc
    query_ids = set(int(value) for value in queries["query_id"].to_list())
    pool_ids = set(int(value) for value in pools["query_id"].to_list())
    if not query_ids or query_ids != pool_ids:
        raise RetrievalEvaluationBuildError("query and judged-pool cohorts are not identical")
    judgment_map: dict[int, tuple[Judgment, ...]] = {}
    for query_id in sorted(query_ids):
        rows = pools.filter(pl.col("query_id") == query_id).sort("product_id")
        query_judgments: list[Judgment] = []
        for row in rows.iter_rows(named=True):
            label = cast(EsciLabel, str(row["esci_label"]))
            expected_gain = _OFFICIAL_GAINS[label]
            if not math.isclose(float(row["gain"]), expected_gain, abs_tol=1e-6):
                raise RetrievalEvaluationBuildError(
                    "persisted judgment gain does not match the official label mapping"
                )
            query_judgments.append(
                Judgment(
                    product_id=str(row["product_id"]),
                    label=label,
                    gain=expected_gain,
                )
            )
        judgment_map[query_id] = tuple(query_judgments)
    return queries, judgment_map


def _reuse_evaluation(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    dependencies: _Dependencies,
    store: ArtifactStore,
) -> RetrievalEvaluationBuildResult:
    artifact = store.load(retrieval_evaluation_artifact_id(release, config.sha256, profile))
    if artifact.manifest.dependencies != _dependencies_as_manifest(dependencies):
        raise RetrievalEvaluationValidationError(
            "existing retrieval evaluation has incompatible parent artifacts"
        )
    manifest = load_retrieval_evaluation_manifest(artifact.path / RETRIEVAL_EVALUATION_FILENAME)
    if (
        manifest.artifact_id != artifact.manifest.artifact_id
        or manifest.config_sha256 != config.sha256
        or manifest.profile != profile
    ):
        raise RetrievalEvaluationValidationError(
            "existing retrieval evaluation metadata has incompatible lineage"
        )
    return RetrievalEvaluationBuildResult(artifact=artifact, manifest=manifest, reused=True)


def build_retrieval_evaluation(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    profile: Profile | None = None,
    artifact_store: ArtifactStore | None = None,
    dense_encoder: DenseEncoder | None = None,
) -> RetrievalEvaluationBuildResult:
    """Generate candidates and a protocol-safe sparse/dense/RRF comparison artifact."""
    selected_profile: Profile = profile or config.config.evaluation.default_profile
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    dependencies = _load_dependencies(release, config, store)
    artifact_id = retrieval_evaluation_artifact_id(release, config.sha256, selected_profile)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse_evaluation(release, config, selected_profile, dependencies, store)
    queries, judgments = _load_evaluation_cohort(dependencies, selected_profile)
    population_id = f"esci_task1_us_{selected_profile}_retrieval_queries_v1"
    hybrid_config = config.config.retrieval.hybrid
    evaluation = config.config.evaluation
    active_encoder = (
        dense_encoder
        if dense_encoder is not None
        else cast(DenseEncoder, SentenceTransformerEncoder(config))
    )
    parent_dependencies = _dependencies_as_manifest(dependencies)

    with ExitStack() as stack:
        sparse_index = stack.enter_context(
            load_sparse_index(store, dependencies.sparse.manifest.artifact_id)
        )
        dense_index = stack.enter_context(
            load_dense_index(
                store,
                dependencies.dense.manifest.artifact_id,
                encoder=active_encoder,
                max_threads=config.config.runtime.max_threads,
            )
        )
        load_peak = _peak_rss_bytes()
        rss_limit = config.config.runtime.rss_limit_mb * 1024 * 1024
        initial_resource = CombinedResourceMeasurement(
            load_peak_rss_bytes=load_peak,
            evaluation_peak_rss_bytes=load_peak,
            peak_rss_bytes=load_peak,
            rss_limit_bytes=rss_limit,
            sparse_artifact_bytes=_artifact_payload_bytes(dependencies.sparse),
            dense_artifact_bytes=_artifact_payload_bytes(dependencies.dense),
            evaluation_artifact_bytes=0,
            query_count=0,
            passed=load_peak <= rss_limit,
        )
        if not initial_resource.passed:
            raise HybridResourceError(initial_resource)
        try:
            with store.stage(
                artifact_type="retrieval-evaluation",
                dataset_version=release.manifest.dataset_version,
                profile=selected_profile,
                component_version=evaluation.component_version,
                config_sha256=config.sha256,
                code_revision=code_revision,
                dependencies=parent_dependencies,
            ) as transaction:
                root = transaction.path(RETRIEVAL_EVALUATION_FILENAME).parent
                candidate_writer = _PartitionWriter(
                    root / CANDIDATE_DIRECTORY,
                    _CANDIDATE_SCHEMA,
                    evaluation.candidate_partition_rows,
                )
                metric_writer = _PartitionWriter(
                    root / QUERY_METRIC_DIRECTORY,
                    _QUERY_METRIC_SCHEMA,
                    evaluation.metric_partition_rows,
                )
                latencies: dict[Stage, list[float]] = {
                    "sparse": [],
                    "dense": [],
                    "hybrid": [],
                }
                candidate_counts: dict[Stage, int] = {
                    "sparse": 0,
                    "dense": 0,
                    "hybrid": 0,
                }
                empty_counts: dict[Stage, int] = {
                    "sparse": 0,
                    "dense": 0,
                    "hybrid": 0,
                }
                for query in queries.iter_rows(named=True):
                    query_id = int(query["query_id"])
                    text = str(query["normalized_query"])
                    started = time.perf_counter()
                    sparse_candidates = sparse_index.search(text, hybrid_config.sparse_top_k)
                    sparse_latency = (time.perf_counter() - started) * 1000.0
                    started = time.perf_counter()
                    dense_candidates = dense_index.search(text, hybrid_config.dense_top_k)
                    dense_latency = (time.perf_counter() - started) * 1000.0
                    started = time.perf_counter()
                    hybrid_result: HybridResult = fuse_rrf(
                        sparse_candidates,
                        dense_candidates,
                        rrf_constant=hybrid_config.rrf_constant,
                        top_k=hybrid_config.union_top_k,
                        max_top_k=hybrid_config.max_union_top_k,
                    )
                    hybrid_latency = (time.perf_counter() - started) * 1000.0
                    stage_candidates: tuple[tuple[Stage, tuple[Any, ...], float], ...] = (
                        ("sparse", sparse_candidates, sparse_latency),
                        ("dense", dense_candidates, dense_latency),
                        ("hybrid", hybrid_result.candidates, hybrid_latency),
                    )
                    for stage, candidates, latency in stage_candidates:
                        latencies[stage].append(latency)
                        candidate_counts[stage] += len(candidates)
                        empty_counts[stage] += not candidates
                        for row in _stage_candidate_rows(
                            profile=selected_profile,
                            query=query,
                            stage=stage,
                            candidates=candidates,
                            latency_ms=latency,
                            hybrid_artifact_id=artifact_id,
                            degraded=bool(hybrid_result.degraded_sources)
                            if stage == "hybrid"
                            else False,
                        ):
                            candidate_writer.append(row)
                        ranked_ids = tuple(candidate.product_id for candidate in candidates)
                        for row in _metric_rows(
                            profile=selected_profile,
                            population_id=population_id,
                            catalog_id=dependencies.foundation_manifest.catalog_id,
                            query=query,
                            stage=stage,
                            ranked_ids=ranked_ids,
                            judgments=judgments[query_id],
                            cutoffs=evaluation.cutoffs,
                        ):
                            metric_writer.append(row)
                candidate_writer.close()
                metric_writer.close()
                query_metrics = pl.read_parquet([str(path) for path in metric_writer.paths]).sort(
                    "query_id", "stage", "threshold_id", "metric", "cutoff"
                )
                aggregate = build_aggregate_metrics(query_metrics, config)
                comparisons = build_comparison_metrics(query_metrics, config)
                aggregate.write_parquet(
                    root / AGGREGATE_METRICS_FILENAME,
                    compression="zstd",
                    statistics=True,
                )
                comparisons.write_parquet(
                    root / COMPARISON_METRICS_FILENAME,
                    compression="zstd",
                    statistics=True,
                )
                evaluation_bytes = sum(
                    path.stat().st_size for path in root.rglob("*") if path.is_file()
                )
                final_peak = _peak_rss_bytes()
                combined_resource = CombinedResourceMeasurement(
                    load_peak_rss_bytes=load_peak,
                    evaluation_peak_rss_bytes=final_peak,
                    peak_rss_bytes=max(load_peak, final_peak),
                    rss_limit_bytes=rss_limit,
                    sparse_artifact_bytes=_artifact_payload_bytes(dependencies.sparse),
                    dense_artifact_bytes=_artifact_payload_bytes(dependencies.dense),
                    evaluation_artifact_bytes=evaluation_bytes,
                    query_count=queries.height,
                    passed=max(load_peak, final_peak) <= rss_limit,
                )
                if not combined_resource.passed:
                    raise HybridResourceError(combined_resource)
                stage_summaries = tuple(
                    _stage_summary(
                        stage,
                        latencies[stage],
                        candidate_counts[stage],
                        empty_counts[stage],
                    )
                    for stage in _STAGES
                )
                checks = tuple(
                    sorted(
                        (
                            EvaluationCheck(
                                check_id="catalog_lineage",
                                detail="foundation, sparse, and dense use one exact catalog hash",
                            ),
                            EvaluationCheck(
                                check_id="combined_resource_gate",
                                detail=(
                                    f"peak RSS {combined_resource.peak_rss_bytes} <= "
                                    f"{combined_resource.rss_limit_bytes} bytes"
                                ),
                            ),
                            EvaluationCheck(
                                check_id="fixed_query_cohort",
                                detail=(
                                    f"all three stages evaluate the same {queries.height} queries"
                                ),
                            ),
                            EvaluationCheck(
                                check_id="grouped_bootstrap",
                                detail=(
                                    f"{evaluation.bootstrap_replicates} fixed-seed normalized-"
                                    "query group replicates"
                                ),
                            ),
                            EvaluationCheck(
                                check_id="protocol_separation",
                                detail="catalog reports expose retrieval-safe metrics only",
                            ),
                            EvaluationCheck(
                                check_id="rrf_provenance",
                                detail="union rows retain nullable sparse/dense ranks and scores",
                            ),
                        ),
                        key=lambda item: item.check_id,
                    )
                )
                manifest = RetrievalEvaluationManifest(
                    artifact_id=artifact_id,
                    dataset_version=release.manifest.dataset_version,
                    config_sha256=config.sha256,
                    profile=selected_profile,
                    population_id=population_id,
                    catalog_id=dependencies.foundation_manifest.catalog_id,
                    catalog_membership_sha256=dependencies.sparse_metadata.catalog_membership_sha256,
                    foundation_artifact_id=dependencies.foundation.manifest.artifact_id,
                    foundation_manifest_sha256=dependencies.foundation.manifest_sha256,
                    sparse_artifact_id=dependencies.sparse.manifest.artifact_id,
                    sparse_manifest_sha256=dependencies.sparse.manifest_sha256,
                    dense_artifact_id=dependencies.dense.manifest.artifact_id,
                    dense_manifest_sha256=dependencies.dense.manifest_sha256,
                    component_version=evaluation.component_version,
                    fusion_version=hybrid_config.component_version,
                    rrf_constant=hybrid_config.rrf_constant,
                    sparse_top_k=hybrid_config.sparse_top_k,
                    dense_top_k=hybrid_config.dense_top_k,
                    union_top_k=hybrid_config.union_top_k,
                    cutoffs=evaluation.cutoffs,
                    relevant_thresholds=(
                        ("exact", ("E",)),
                        ("exact_substitute", ("E", "S")),
                    ),
                    bootstrap_replicates=evaluation.bootstrap_replicates,
                    bootstrap_seed=config.config.runtime.seed,
                    query_count=queries.height,
                    normalized_query_groups=queries["normalized_query_sha256"].n_unique(),
                    candidate_partitions=len(candidate_writer.paths),
                    query_metric_partitions=len(metric_writer.paths),
                    aggregate_metric_rows=aggregate.height,
                    comparison_metric_rows=comparisons.height,
                    stages=stage_summaries,
                    resource=combined_resource,
                    checks=checks,
                )
                transaction.path(RETRIEVAL_EVALUATION_FILENAME).write_text(
                    _canonical_json(manifest), encoding="utf-8"
                )
                artifact = transaction.commit()
        except ArtifactExistsError:
            return _reuse_evaluation(release, config, selected_profile, dependencies, store)
    return RetrievalEvaluationBuildResult(artifact=artifact, manifest=manifest, reused=False)


__all__ = [
    "AGGREGATE_METRICS_FILENAME",
    "CANDIDATE_DIRECTORY",
    "COMPARISON_METRICS_FILENAME",
    "QUERY_METRIC_DIRECTORY",
    "RETRIEVAL_EVALUATION_FILENAME",
    "CombinedResourceMeasurement",
    "HybridResourceError",
    "RetrievalEvaluationBuildError",
    "RetrievalEvaluationBuildResult",
    "RetrievalEvaluationError",
    "RetrievalEvaluationManifest",
    "RetrievalEvaluationValidationError",
    "StageSummary",
    "build_aggregate_metrics",
    "build_comparison_metrics",
    "build_retrieval_evaluation",
    "load_retrieval_evaluation_manifest",
    "retrieval_evaluation_artifact_id",
]
