"""Strict, deterministic configuration loading for MarketRank."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver


class ConfigError(ValueError):
    """Base exception for configuration loading and validation failures."""


class ConfigFileError(ConfigError):
    """Raised when a configuration file cannot be read as a strict YAML mapping."""


class ConfigOverrideError(ConfigError):
    """Raised when a dotted override cannot be applied to the configuration tree."""


class ConfigValidationError(ConfigError):
    """Raised when the resolved configuration violates the typed schema."""


class _StrictModel(BaseModel):
    """Shared immutable, unknown-key-rejecting model behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_StrictModel):
    """Stable project identity fields."""

    name: Literal["market-rank"] = "market-rank"
    locale: Literal["us"] = "us"


class PathsConfig(_StrictModel):
    """Repository-relative lifecycle paths."""

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    experiments_dir: Path = Path("experiments")
    reports_dir: Path = Path("reports")


class RuntimeConfig(_StrictModel):
    """Local deterministic and resource controls."""

    seed: int = Field(default=20260801, strict=True, ge=0, le=2**32 - 1)
    max_threads: int = Field(default=4, strict=True, ge=1, le=16)
    rss_limit_mb: int = Field(default=5632, strict=True, ge=512, le=8192)
    offline: bool = Field(default=True, strict=True)


class DatasetConfig(_StrictModel):
    """Deterministic Task-1 cohort, split, and nested-profile controls."""

    query_normalization_version: Literal["nfkc-casefold-ws-v1"] = "nfkc-casefold-ws-v1"
    split_version: Literal["normalized-query-sha256-v1"] = "normalized-query-sha256-v1"
    profile_version: Literal["nested-query-sha256-v1"] = "nested-query-sha256-v1"
    train_basis_points: int = Field(default=8500, strict=True, ge=1, le=9999)
    development_query_groups: int = Field(default=5000, strict=True, ge=1)
    portfolio_query_groups: int = Field(default=20000, strict=True, ge=1)
    catalog_mode: Literal["full", "compact"] = "compact"
    catalog_selection_version: Literal["portfolio-judged-plus-sha256-v1"] = (
        "portfolio-judged-plus-sha256-v1"
    )
    compact_catalog_distractor_products: int = Field(default=100000, strict=True, ge=0, le=2000000)
    product_document_version: Literal["product-document-v1"] = "product-document-v1"
    title_max_chars: int = Field(default=512, strict=True, ge=32, le=4096)
    brand_max_chars: int = Field(default=128, strict=True, ge=16, le=1024)
    color_max_chars: int = Field(default=128, strict=True, ge=16, le=1024)
    bullets_max_chars: int = Field(default=2048, strict=True, ge=128, le=16384)
    description_max_chars: int = Field(default=4096, strict=True, ge=256, le=32768)
    m2_runtime_reserve_mb: int = Field(default=512, strict=True, ge=64, le=4096)

    @model_validator(mode="after")
    def validate_nested_targets(self) -> Self:
        if self.development_query_groups > self.portfolio_query_groups:
            raise ValueError("development_query_groups must not exceed portfolio_query_groups")
        return self


class SparseRetrievalConfig(_StrictModel):
    """Deterministic local BM25 build and query controls."""

    tokenizer_version: Literal["unicode-word-v1"] = "unicode-word-v1"
    component_version: Literal["bm25-v1"] = "bm25-v1"
    k1: float = Field(default=1.2, strict=True, gt=0.0, le=5.0)
    b: float = Field(default=0.75, strict=True, ge=0.0, le=1.0)
    default_top_k: int = Field(default=150, strict=True, ge=1, le=1000)
    max_top_k: int = Field(default=1000, strict=True, ge=1, le=5000)
    sqlite_batch_rows: int = Field(default=10000, strict=True, ge=100, le=100000)

    @model_validator(mode="after")
    def validate_top_k(self) -> Self:
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k must not exceed max_top_k")
        return self


class DenseRetrievalConfig(_StrictModel):
    """Pinned, offline-first MiniLM and exact FAISS controls."""

    model_id: Literal["sentence-transformers/all-MiniLM-L6-v2"] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_revision: Literal["c9745ed1d9f207416be6d2e6f8de32d1f16199bf"] = (
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
    component_version: Literal["minilm-l6-v2-flatip-v1"] = "minilm-l6-v2-flatip-v1"
    embedding_dimension: Literal[384] = 384
    embedding_batch_size: int = Field(default=16, strict=True, ge=1, le=64)
    default_top_k: int = Field(default=150, strict=True, ge=1, le=1000)
    max_top_k: int = Field(default=1000, strict=True, ge=1, le=5000)
    latency_sample_queries: int = Field(default=20, strict=True, ge=1, le=100)
    model_cache_dir: Path = Path("models/huggingface")

    @model_validator(mode="after")
    def validate_top_k(self) -> Self:
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k must not exceed max_top_k")
        return self


class HybridRetrievalConfig(_StrictModel):
    """Deterministic reciprocal-rank fusion and candidate bounds."""

    component_version: Literal["rrf-v1"] = "rrf-v1"
    rrf_constant: int = Field(default=60, strict=True, ge=1, le=1000)
    sparse_top_k: int = Field(default=150, strict=True, ge=1, le=5000)
    dense_top_k: int = Field(default=150, strict=True, ge=1, le=5000)
    union_top_k: int = Field(default=200, strict=True, ge=1, le=5000)
    max_union_top_k: int = Field(default=1000, strict=True, ge=1, le=5000)

    @model_validator(mode="after")
    def validate_union(self) -> Self:
        if self.union_top_k > self.max_union_top_k:
            raise ValueError("union_top_k must not exceed max_union_top_k")
        return self


class RetrievalConfig(_StrictModel):
    """Versioned retrieval subsystem configuration."""

    sparse: SparseRetrievalConfig = SparseRetrievalConfig()
    dense: DenseRetrievalConfig = DenseRetrievalConfig()
    hybrid: HybridRetrievalConfig = HybridRetrievalConfig()

    @model_validator(mode="after")
    def validate_source_limits(self) -> Self:
        if self.hybrid.sparse_top_k > self.sparse.max_top_k:
            raise ValueError("hybrid sparse_top_k exceeds sparse max_top_k")
        if self.hybrid.dense_top_k > self.dense.max_top_k:
            raise ValueError("hybrid dense_top_k exceeds dense max_top_k")
        return self


class EvaluationConfig(_StrictModel):
    """Fixed-cohort retrieval summaries, slices, and bounded grouped bootstrap."""

    component_version: Literal["retrieval-eval-v1"] = "retrieval-eval-v1"
    default_profile: Literal["development", "portfolio"] = "development"
    cutoffs: tuple[int, ...] = (10, 100)
    bootstrap_replicates: int = Field(default=1000, strict=True, ge=100, le=10000)
    bootstrap_batch_replicates: int = Field(default=100, strict=True, ge=10, le=1000)
    candidate_partition_rows: int = Field(default=100000, strict=True, ge=1, le=1000000)
    metric_partition_rows: int = Field(default=100000, strict=True, ge=1, le=1000000)

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        if not self.cutoffs or any(
            not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 1
            for cutoff in self.cutoffs
        ):
            raise ValueError("evaluation cutoffs must be positive integers")
        if self.cutoffs != tuple(sorted(set(self.cutoffs))):
            raise ValueError("evaluation cutoffs must be unique and sorted")
        if self.bootstrap_batch_replicates > self.bootstrap_replicates:
            raise ValueError("bootstrap batch size exceeds total replicates")
        return self


class QueryUnderstandingConfig(_StrictModel):
    """Bounded deterministic query parsing and dictionary state."""

    parser_version: Literal["query-parser-v1"] = "query-parser-v1"
    max_query_chars: int = Field(default=512, strict=True, ge=1, le=4096)
    max_query_bytes: int = Field(default=2048, strict=True, ge=4, le=16384)
    max_query_tokens: int = Field(default=64, strict=True, ge=1, le=512)
    brand_min_chars: int = Field(default=2, strict=True, ge=1, le=64)


class RankingFeatureConfig(_StrictModel):
    """Versioned, candidate-aligned ranking feature materialization controls."""

    component_version: Literal["ranking-features-v1"] = "ranking-features-v1"
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    registry_version: Literal["feature-registry-v1"] = "feature-registry-v1"
    state_version: Literal["feature-state-v1"] = "feature-state-v1"
    matrix_partition_rows: int = Field(default=100000, strict=True, ge=1, le=1000000)
    query_batch_size: int = Field(default=64, strict=True, ge=1, le=1000)
    parity_fixture_rows: int = Field(default=128, strict=True, ge=1, le=10000)
    max_rows_per_query: int = Field(default=200, strict=True, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_partitioning(self) -> Self:
        if self.max_rows_per_query > self.matrix_partition_rows:
            raise ValueError("max_rows_per_query must not exceed matrix_partition_rows")
        return self


class RankerTrainingConfig(_StrictModel):
    """Deterministic, CPU-bounded pointwise and LambdaMART training controls."""

    component_version: Literal["lightgbm-rankers-v1"] = "lightgbm-rankers-v1"
    population_version: Literal["training-population-v1"] = "training-population-v1"
    pointwise_objective: Literal["regression_l2"] = "regression_l2"
    lambdamart_objective: Literal["lambdarank"] = "lambdarank"
    learning_rate: float = Field(default=0.05, strict=True, gt=0.0, le=1.0)
    num_leaves: int = Field(default=31, strict=True, ge=2, le=255)
    max_depth: int = Field(default=-1, strict=True, ge=-1, le=32)
    min_data_in_leaf: int = Field(default=20, strict=True, ge=1, le=10000)
    max_bin: int = Field(default=63, strict=True, ge=16, le=255)
    lambda_l1: float = Field(default=0.0, strict=True, ge=0.0, le=100.0)
    lambda_l2: float = Field(default=1.0, strict=True, ge=0.0, le=100.0)
    max_boost_rounds: int = Field(default=300, strict=True, ge=10, le=5000)
    early_stopping_rounds: int = Field(default=30, strict=True, ge=1, le=500)
    ndcg_eval_at: tuple[int, ...] = (10, 20)
    min_group_rows: int = Field(default=2, strict=True, ge=2, le=1000)
    min_distinct_labels: int = Field(default=2, strict=True, ge=2, le=4)
    max_train_rows: int = Field(default=500000, strict=True, ge=1, le=1000000)
    max_validation_rows: int = Field(default=200000, strict=True, ge=1, le=500000)
    reload_parity_rows: int = Field(default=128, strict=True, ge=1, le=10000)
    explanation_rows: int = Field(default=64, strict=True, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_training(self) -> Self:
        if self.early_stopping_rounds >= self.max_boost_rounds:
            raise ValueError("early stopping rounds must be smaller than maximum boost rounds")
        if not self.ndcg_eval_at or any(
            not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 1
            for cutoff in self.ndcg_eval_at
        ):
            raise ValueError("ranker NDCG cutoffs must be positive integers")
        if self.ndcg_eval_at != tuple(sorted(set(self.ndcg_eval_at))):
            raise ValueError("ranker NDCG cutoffs must be unique and sorted")
        return self


class RankingEvaluationConfig(_StrictModel):
    """Protocol-safe ranking evaluation and validation-only promotion controls."""

    component_version: Literal["ranking-eval-v1"] = "ranking-eval-v1"
    selection_split: Literal["validation"] = "validation"
    closed_cutoffs: tuple[int, ...] = (10, 20)
    diagnostic_cutoffs: tuple[int, ...] = (10, 100)
    minimum_model_improvement: float = Field(default=0.0, strict=True, ge=0.0, le=1.0)
    material_regression_tolerance: float = Field(default=0.005, strict=True, ge=0.0, le=0.25)
    selection_tie_tolerance: float = Field(default=1e-12, strict=True, ge=0.0, le=0.01)
    failure_analysis_queries: int = Field(default=20, strict=True, ge=1, le=1000)
    max_closed_rows: int = Field(default=200000, strict=True, ge=1, le=500000)
    max_candidate_rows: int = Field(default=200000, strict=True, ge=1, le=500000)

    @model_validator(mode="after")
    def validate_ranking_evaluation(self) -> Self:
        for name, cutoffs in (
            ("closed", self.closed_cutoffs),
            ("diagnostic", self.diagnostic_cutoffs),
        ):
            if not cutoffs or any(
                not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 1
                for cutoff in cutoffs
            ):
                raise ValueError(f"{name} ranking-evaluation cutoffs must be positive integers")
            if cutoffs != tuple(sorted(set(cutoffs))):
                raise ValueError(f"{name} ranking-evaluation cutoffs must be unique and sorted")
        return self


class ServingConfig(_StrictModel):
    """Explicit-bundle, local-only, bounded FastAPI serving controls."""

    component_version: Literal["serving-bundle-v1"] = "serving-bundle-v1"
    bind_host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8000, strict=True, ge=1024, le=65535)
    default_top_k: int = Field(default=10, strict=True, ge=1, le=50)
    max_response_top_k: int = Field(default=50, strict=True, ge=1, le=100)
    default_deadline_ms: int = Field(default=1000, strict=True, ge=100, le=10000)
    max_deadline_ms: int = Field(default=3000, strict=True, ge=100, le=30000)
    max_concurrency: int = Field(default=2, strict=True, ge=1, le=16)
    max_request_body_bytes: int = Field(default=16384, strict=True, ge=1024, le=1048576)
    max_debug_candidates: int = Field(default=20, strict=True, ge=1, le=50)
    description_snippet_chars: int = Field(default=240, strict=True, ge=32, le=1000)
    product_store_batch_rows: int = Field(default=10000, strict=True, ge=100, le=100000)
    allow_degraded_retrieval: bool = Field(default=True, strict=True)
    allow_ranker_fallback: bool = Field(default=True, strict=True)
    debug_enabled: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def validate_serving(self) -> Self:
        if self.default_top_k > self.max_response_top_k:
            raise ValueError("serving default_top_k must not exceed max_response_top_k")
        if self.default_deadline_ms > self.max_deadline_ms:
            raise ValueError("serving default deadline must not exceed maximum deadline")
        if self.max_debug_candidates > self.max_response_top_k:
            raise ValueError("serving debug candidate bound exceeds response top-K")
        return self


class DemoConfig(_StrictModel):
    """Local-only, API-backed Streamlit portfolio-demo controls."""

    component_version: Literal["streamlit-demo-v1"] = "streamlit-demo-v1"
    api_base_url: Literal["http://127.0.0.1:8000"] = "http://127.0.0.1:8000"
    bind_host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8501, strict=True, ge=1024, le=65535)
    request_timeout_seconds: float = Field(default=5.0, strict=True, ge=0.1, le=30.0)
    default_top_k: int = Field(default=10, strict=True, ge=1, le=50)
    max_comparison_modes: int = Field(default=4, strict=True, ge=1, le=6)
    max_product_cards: int = Field(default=12, strict=True, ge=1, le=50)
    example_queries: tuple[str, ...] = (
        "wireless mouse",
        "running shoes men",
        "iphone 13 case",
        "coffee maker",
        "blue dress",
    )

    @model_validator(mode="after")
    def validate_demo(self) -> Self:
        normalized = tuple(query.strip() for query in self.example_queries)
        if not normalized or any(not query or len(query) > 512 for query in normalized):
            raise ValueError("demo example queries must be nonempty and at most 512 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("demo example queries must be unique")
        if normalized != self.example_queries:
            raise ValueError("demo example queries must not contain surrounding whitespace")
        return self


class QualificationConfig(_StrictModel):
    """Fail-closed release qualification on the reference Apple host."""

    component_version: Literal["release-qualification-v1"] = "release-qualification-v1"
    generation: Literal["rc1"] = "rc1"
    required_system: Literal["Darwin"] = "Darwin"
    required_machine: Literal["arm64"] = "arm64"
    required_chip: Literal["Apple M3"] = "Apple M3"
    required_memory_bytes: int = Field(default=8 * 1024**3, strict=True, ge=1)
    required_power_source: Literal["AC Power"] = "AC Power"
    require_clean_revision: bool = Field(default=True, strict=True)
    warmup_rounds: int = Field(default=2, strict=True, ge=1, le=20)
    measured_rounds: int = Field(default=5, strict=True, ge=3, le=100)
    concurrency_workers: int = Field(default=2, strict=True, ge=1, le=4)
    concurrency_rounds: int = Field(default=3, strict=True, ge=1, le=20)
    top_k: int = Field(default=10, strict=True, ge=1, le=50)
    cold_startup_target_ms: float = Field(default=30000.0, strict=True, gt=0.0)
    request_p95_target_ms: float = Field(default=1000.0, strict=True, gt=0.0)
    concurrency_p95_target_ms: float = Field(default=1500.0, strict=True, gt=0.0)
    parse_p95_target_ms: float = Field(default=50.0, strict=True, gt=0.0)
    sparse_p95_target_ms: float = Field(default=200.0, strict=True, gt=0.0)
    dense_p95_target_ms: float = Field(default=200.0, strict=True, gt=0.0)
    fusion_p95_target_ms: float = Field(default=50.0, strict=True, gt=0.0)
    features_p95_target_ms: float = Field(default=250.0, strict=True, gt=0.0)
    ranker_p95_target_ms: float = Field(default=50.0, strict=True, gt=0.0)
    modes: tuple[Literal["active", "bm25", "dense", "hybrid", "pointwise", "lambdamart"], ...] = (
        "active",
        "bm25",
        "dense",
        "hybrid",
        "pointwise",
        "lambdamart",
    )
    queries: tuple[str, ...] = (
        "wireless mouse",
        "running shoes men",
        "iphone 13 case",
        "coffee maker",
        "blue dress",
    )

    @model_validator(mode="after")
    def validate_qualification(self) -> Self:
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("qualification modes must be nonempty and unique")
        normalized = tuple(query.strip() for query in self.queries)
        if not normalized or any(not query or len(query) > 512 for query in normalized):
            raise ValueError("qualification queries must be nonempty and at most 512 characters")
        if normalized != self.queries or len(set(normalized)) != len(normalized):
            raise ValueError("qualification queries must be stripped and unique")
        return self


class PortfolioReportConfig(_StrictModel):
    """Frozen core portfolio release-package requirements."""

    component_version: Literal["portfolio-release-v1"] = "portfolio-release-v1"
    generation: Literal["final-v1"] = "final-v1"
    required_profile: Literal["portfolio"] = "portfolio"
    final_evaluation_split: Literal["test"] = "test"
    screenshot_filenames: tuple[str, ...] = (
        "ranking-comparison.png",
        "product-provenance.png",
        "dataset-limitations.png",
    )
    minimum_screenshot_width: int = Field(default=1000, strict=True, ge=640, le=7680)
    minimum_screenshot_height: int = Field(default=600, strict=True, ge=480, le=4320)
    maximum_screenshot_bytes: int = Field(default=10 * 1024**2, strict=True, ge=1024)
    require_clean_reproduction: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def validate_portfolio_report(self) -> Self:
        expected = (
            "ranking-comparison.png",
            "product-provenance.png",
            "dataset-limitations.png",
        )
        if self.screenshot_filenames != expected:
            raise ValueError("portfolio screenshots must use the canonical ordered filenames")
        return self


class LoggingConfig(_StrictModel):
    """Safe local logging defaults."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    log_raw_queries: bool = Field(default=False, strict=True)


class AppConfig(_StrictModel):
    """Versioned root configuration model."""

    schema_version: Literal[1] = 1
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    dataset: DatasetConfig = DatasetConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    query_understanding: QueryUnderstandingConfig = QueryUnderstandingConfig()
    ranking_features: RankingFeatureConfig = RankingFeatureConfig()
    ranker_training: RankerTrainingConfig = RankerTrainingConfig()
    ranking_evaluation: RankingEvaluationConfig = RankingEvaluationConfig()
    serving: ServingConfig = ServingConfig()
    demo: DemoConfig = DemoConfig()
    qualification: QualificationConfig = QualificationConfig()
    portfolio_report: PortfolioReportConfig = PortfolioReportConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def validate_evaluation_retrieval_bounds(self) -> Self:
        largest_cutoff = max(self.evaluation.cutoffs)
        if largest_cutoff > min(
            self.retrieval.hybrid.sparse_top_k,
            self.retrieval.hybrid.dense_top_k,
            self.retrieval.hybrid.union_top_k,
        ):
            raise ValueError("evaluation cutoffs exceed a configured retrieval depth")
        if self.ranking_features.max_rows_per_query < self.retrieval.hybrid.union_top_k:
            raise ValueError("ranking feature row bound is smaller than the hybrid union depth")
        if self.serving.max_response_top_k > self.retrieval.hybrid.union_top_k:
            raise ValueError("serving response top-K exceeds the hybrid union depth")
        if self.demo.default_top_k > self.serving.max_response_top_k:
            raise ValueError("demo default top-K exceeds the serving response bound")
        if self.demo.max_product_cards > self.serving.max_response_top_k:
            raise ValueError("demo product-card bound exceeds the serving response bound")
        if self.qualification.top_k > self.serving.max_response_top_k:
            raise ValueError("qualification top-K exceeds the serving response bound")
        if self.qualification.concurrency_workers > self.serving.max_concurrency:
            raise ValueError("qualification concurrency exceeds the serving concurrency bound")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Validated configuration plus deterministic identity and source lineage."""

    config: AppConfig
    canonical_json: str
    sha256: str
    source_paths: tuple[Path, ...]

    @property
    def short_hash(self) -> str:
        """Return a human-readable, non-authoritative hash prefix."""
        return self.sha256[:12]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and non-string mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    mapping: dict[str, Any] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConstructorError(
                "while constructing a configuration mapping",
                node.start_mark,
                "configuration keys must be strings",
                key_node.start_mark,
            )
        if key in mapping:
            raise ConstructorError(
                "while constructing a configuration mapping",
                node.start_mark,
                f"duplicate key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(f"cannot read configuration file {path}: {exc}") from exc

    try:
        document = yaml.load(raw_text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"invalid configuration YAML in {path}: {exc}") from exc

    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ConfigFileError(f"configuration root must be a mapping: {path}")
    return document


def _deep_merge(base: Mapping[str, Any], later: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, later_value in later.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(later_value, Mapping):
            merged[key] = _deep_merge(base_value, later_value)
        else:
            merged[key] = deepcopy(later_value)
    return merged


def _apply_dotted_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, object],
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if not parts or any(not part for part in parts):
            raise ConfigOverrideError(f"invalid dotted override key: {dotted_key!r}")

        cursor: dict[str, Any] = resolved
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                child: dict[str, Any] = {}
                cursor[part] = child
                cursor = child
            elif isinstance(existing, dict):
                cursor = existing
            else:
                raise ConfigOverrideError(f"cannot apply {dotted_key!r}: {part!r} is not a mapping")
        cursor[parts[-1]] = deepcopy(value)

    return resolved


def _canonicalize(config: AppConfig) -> str:
    semantic_document = config.model_dump(mode="json")
    return json.dumps(
        semantic_document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def load_config(
    paths: Sequence[Path],
    overrides: Mapping[str, object] | None = None,
) -> ResolvedConfig:
    """Load ordered YAML layers, apply overrides, validate, and hash semantics."""
    if not paths:
        raise ConfigFileError("at least one configuration path is required")

    merged: dict[str, Any] = {}
    source_paths: list[Path] = []
    for input_path in paths:
        path = Path(input_path)
        merged = _deep_merge(merged, _load_yaml_mapping(path))
        source_paths.append(path.resolve(strict=False))

    if overrides:
        merged = _apply_dotted_overrides(merged, overrides)

    try:
        config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigValidationError(f"resolved configuration is invalid: {exc}") from exc

    canonical_json = _canonicalize(config)
    config_sha256 = sha256(canonical_json.encode("utf-8")).hexdigest()
    return ResolvedConfig(
        config=config,
        canonical_json=canonical_json,
        sha256=config_sha256,
        source_paths=tuple(source_paths),
    )


__all__ = [
    "AppConfig",
    "ConfigError",
    "ConfigFileError",
    "ConfigOverrideError",
    "ConfigValidationError",
    "DatasetConfig",
    "DemoConfig",
    "DenseRetrievalConfig",
    "EvaluationConfig",
    "HybridRetrievalConfig",
    "LoggingConfig",
    "PathsConfig",
    "PortfolioReportConfig",
    "ProjectConfig",
    "QualificationConfig",
    "QueryUnderstandingConfig",
    "RankerTrainingConfig",
    "RankingEvaluationConfig",
    "RankingFeatureConfig",
    "ResolvedConfig",
    "RetrievalConfig",
    "RuntimeConfig",
    "ServingConfig",
    "SparseRetrievalConfig",
    "load_config",
]
