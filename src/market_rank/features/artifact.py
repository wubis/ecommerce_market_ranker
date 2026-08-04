"""Immutable, bounded Goldfish 010 query-state and ranking-feature materialization."""

from __future__ import annotations

import json
import resource
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

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
    CATALOG_MEMBERSHIP_FILENAME,
    FOUNDATION_MANIFEST_FILENAME,
    JUDGED_POOLS_FILENAME,
    PRODUCTS_FILENAME,
    QUERIES_FILENAME,
    DataFoundationManifest,
    foundation_artifact_id,
    load_foundation_manifest,
)
from market_rank.evaluation.retrieval import (
    CANDIDATE_DIRECTORY,
    RETRIEVAL_EVALUATION_FILENAME,
    RetrievalEvaluationManifest,
    load_retrieval_evaluation_manifest,
    retrieval_evaluation_artifact_id,
)
from market_rank.features.core import ProductFeatureView, compute_core_features, rank_fractions
from market_rank.features.registry import FEATURE_NAMES, FeatureRegistry, ltr_core_v1_registry
from market_rank.query.parser import ParsedQuery, QueryParser, QueryParserState, build_parser_state
from market_rank.retrieval.dense import (
    DENSE_METADATA_FILENAME,
    DenseEncoder,
    DenseIndexMetadata,
    SentenceTransformerEncoder,
    dense_artifact_id,
    load_dense_index,
    load_dense_metadata,
)
from market_rank.retrieval.sparse import (
    SPARSE_METADATA_FILENAME,
    SparseIndexMetadata,
    load_sparse_index,
    load_sparse_metadata,
    sparse_artifact_id,
)

FEATURE_ARTIFACT_FILENAME = "ranking-features.json"
FEATURE_REGISTRY_FILENAME = "feature-registry.json"
FEATURE_STATE_FILENAME = "feature-state.json"
PARSED_QUERIES_FILENAME = "parsed-queries.parquet"
LEAKAGE_REPORT_FILENAME = "leakage-report.json"
DISTRIBUTION_REPORT_FILENAME = "distribution-report.parquet"
PARITY_FIXTURES_FILENAME = "parity-fixtures.parquet"
CLOSED_MATRIX_DIRECTORY = "closed-matrix"
CANDIDATE_MATRIX_DIRECTORY = "candidate-matrix"

Profile = Literal["development", "portfolio"]
Population = Literal["closed_judged", "retrieved_union"]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]


class RankingFeatureError(RuntimeError):
    """Base exception for feature-state construction, materialization, and validation."""


class RankingFeatureBuildError(RankingFeatureError):
    """Raised when compatible parents cannot produce the feature artifact."""


class RankingFeatureValidationError(RankingFeatureError):
    """Raised when a persisted feature artifact violates its strict contract."""


class RankingFeatureResourceError(RankingFeatureBuildError):
    """Raised when combined feature materialization exceeds the configured RSS limit."""

    def __init__(self, measurement: FeatureResourceMeasurement) -> None:
        super().__init__(
            f"feature materialization peak RSS {measurement.peak_rss_bytes} exceeds "
            f"the {measurement.rss_limit_bytes}-byte limit"
        )
        self.measurement = measurement


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureState(_StrictModel):
    """Persisted parser, train-only categorical, and scorer identities."""

    schema_version: Literal[1] = 1
    state_version: Literal["feature-state-v1"] = "feature-state-v1"
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    registry_sha256: Sha256Digest
    parser_state: QueryParserState
    categorical_fit_project_split: Literal["train"] = "train"
    brand_codes: tuple[tuple[str, int], ...]
    color_codes: tuple[tuple[str, int], ...]
    missing_category_code: Literal[0] = 0
    unknown_category_code: Literal[1] = 1
    sparse_index_id: str = Field(strict=True, min_length=1)
    sparse_retriever_id: str = Field(strict=True, min_length=1)
    dense_index_id: str = Field(strict=True, min_length=1)
    dense_retriever_id: str = Field(strict=True, min_length=1)
    rrf_constant: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def validate_codes(self) -> Self:
        for name, entries in (("brand", self.brand_codes), ("color", self.color_codes)):
            values = tuple(value for value, _ in entries)
            codes = tuple(code for _, code in entries)
            if values != tuple(sorted(set(values))) or codes != tuple(range(2, len(entries) + 2)):
                raise ValueError(f"{name} category entries must be sorted with contiguous codes")
        return self


class LeakageCheck(_StrictModel):
    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class LeakageReport(_StrictModel):
    schema_version: Literal[1] = 1
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    prohibited_feature_names: tuple[str, ...]
    fit_project_splits: tuple[Literal["train"], ...] = ("train",)
    registry_feature_count: int = Field(strict=True, ge=1)
    checks: tuple[LeakageCheck, ...]

    @model_validator(mode="after")
    def validate_checks(self) -> Self:
        ids = tuple(check.check_id for check in self.checks)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("leakage checks must be unique and sorted")
        return self


class FeatureResourceMeasurement(_StrictModel):
    load_peak_rss_bytes: int = Field(strict=True, ge=0)
    materialization_peak_rss_bytes: int = Field(strict=True, ge=0)
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    artifact_payload_bytes: int = Field(strict=True, ge=0)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        if self.peak_rss_bytes != max(
            self.load_peak_rss_bytes, self.materialization_peak_rss_bytes
        ):
            raise ValueError("peak RSS must be the maximum observed phase")
        if self.passed != (self.peak_rss_bytes <= self.rss_limit_bytes):
            raise ValueError("resource status does not match the RSS gate")
        return self


class FeatureCheck(_StrictModel):
    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class RankingFeatureManifest(_StrictModel):
    """Strict identity, lineage, matrix, leakage, parity, and resource summary."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    profile: Profile
    component_version: Literal["ranking-features-v1"] = "ranking-features-v1"
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    registry_sha256: Sha256Digest
    state_sha256: Sha256Digest
    foundation_artifact_id: str = Field(strict=True, min_length=1)
    foundation_manifest_sha256: Sha256Digest
    sparse_artifact_id: str = Field(strict=True, min_length=1)
    sparse_manifest_sha256: Sha256Digest
    dense_artifact_id: str = Field(strict=True, min_length=1)
    dense_manifest_sha256: Sha256Digest
    retrieval_evaluation_artifact_id: str = Field(strict=True, min_length=1)
    retrieval_evaluation_manifest_sha256: Sha256Digest
    query_count: int = Field(strict=True, ge=1)
    normalized_query_groups: int = Field(strict=True, ge=1)
    feature_count: int = Field(strict=True, ge=1)
    closed_rows: int = Field(strict=True, ge=1)
    closed_excluded_outside_catalog: int = Field(strict=True, ge=0)
    candidate_rows: int = Field(strict=True, ge=0)
    closed_partitions: int = Field(strict=True, ge=1)
    candidate_partitions: int = Field(strict=True, ge=1)
    parity_fixture_rows: int = Field(strict=True, ge=1)
    resource: FeatureResourceMeasurement
    checks: tuple[FeatureCheck, ...]

    @model_validator(mode="after")
    def validate_checks(self) -> Self:
        ids = tuple(check.check_id for check in self.checks)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("feature checks must be unique and sorted")
        return self


@dataclass(frozen=True, slots=True)
class RankingFeatureBuildResult:
    artifact: LoadedArtifact
    manifest: RankingFeatureManifest
    state: FeatureState
    reused: bool


@dataclass(frozen=True, slots=True)
class _Dependencies:
    foundation: LoadedArtifact
    foundation_manifest: DataFoundationManifest
    sparse: LoadedArtifact
    sparse_metadata: SparseIndexMetadata
    dense: LoadedArtifact
    dense_metadata: DenseIndexMetadata
    evaluation: LoadedArtifact
    evaluation_manifest: RetrievalEvaluationManifest


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

    def write_group(self, rows: list[dict[str, object]]) -> None:
        if len(rows) > self.row_limit:
            raise RankingFeatureBuildError("one query group exceeds the matrix partition bound")
        if self._rows and len(self._rows) + len(rows) > self.row_limit:
            self._flush()
        self._rows.extend(rows)

    def _flush(self) -> None:
        if not self._rows and self._paths:
            return
        path = self.root / f"part-{len(self._paths):05d}.parquet"
        pl.DataFrame(self._rows, schema=self.schema).write_parquet(
            path, compression="zstd", statistics=True
        )
        self._paths.append(path)
        self._rows.clear()

    def close(self) -> None:
        if self._rows or not self._paths:
            self._flush()


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _model_sha256(model: BaseModel) -> str:
    return sha256(_canonical_json(model).encode("utf-8")).hexdigest()


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _feature_schema(*, labels: bool) -> pl.Schema:
    columns: dict[str, Any] = {
        "population": pl.String,
        "profile": pl.String,
        "query_id": pl.Int64,
        "normalized_query_sha256": pl.String,
        "project_split": pl.String,
        "locale": pl.String,
        "product_id": pl.String,
        "feature_set_id": pl.String,
    }
    if labels:
        columns.update({"label_id": pl.UInt8, "gain": pl.Float32})
    registry = ltr_core_v1_registry()
    type_map = {"float32": pl.Float32, "uint8": pl.UInt8, "uint32": pl.UInt32}
    columns.update({feature.name: type_map[feature.dtype] for feature in registry.features})
    return pl.Schema(columns)


_PARSED_SCHEMA = pl.Schema(
    cast(
        Any,
        {
            "profile": pl.String,
            "query_id": pl.Int64,
            "normalized_query_sha256": pl.String,
            "raw_text": pl.String,
            "normalized_text": pl.String,
            "tokens": pl.List(pl.String),
            "reduced_tokens": pl.List(pl.String),
            "numbers": pl.List(pl.String),
            "units": pl.List(pl.String),
            "measurements": pl.List(pl.String),
            "model_tokens": pl.List(pl.String),
            "compatibility_tokens": pl.List(pl.String),
            "compatibility_phrases": pl.List(pl.String),
            "brand": pl.String,
            "brand_confidence": pl.Float32,
            "color": pl.String,
            "color_confidence": pl.Float32,
            "parser_version": pl.String,
            "parser_state_sha256": pl.String,
            "query_sha256": pl.String,
            "warnings": pl.List(pl.String),
        },
    )
)


def ranking_feature_artifact_id(
    release: ResolvedReleaseManifest, config_sha256: str, profile: Profile
) -> str:
    return "/".join(
        (
            "ranking-features",
            release.manifest.dataset_version,
            profile,
            "ranking-features-v1",
            config_sha256,
        )
    )


def load_ranking_feature_manifest(path: Path) -> RankingFeatureManifest:
    try:
        return RankingFeatureManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise RankingFeatureValidationError(
            f"cannot load ranking feature manifest {path}: {exc}"
        ) from exc


def load_feature_state(path: Path) -> FeatureState:
    try:
        return FeatureState.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise RankingFeatureValidationError(f"cannot load feature state {path}: {exc}") from exc


def _load_dependencies(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
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
        evaluation = store.load(retrieval_evaluation_artifact_id(release, config.sha256, profile))
        evaluation_manifest = load_retrieval_evaluation_manifest(
            evaluation.path / RETRIEVAL_EVALUATION_FILENAME
        )
    except (OSError, RuntimeError, pl.exceptions.PolarsError) as exc:
        raise RankingFeatureBuildError(
            "compatible Goldfish 006-009 artifacts are required before ranking features"
        ) from exc
    if (
        foundation_manifest.config_sha256 != config.sha256
        or sparse_metadata.config_sha256 != config.sha256
        or dense_metadata.config_sha256 != config.sha256
        or evaluation_manifest.config_sha256 != config.sha256
        or evaluation_manifest.profile != profile
        or sparse_metadata.foundation_manifest_sha256 != foundation.manifest_sha256
        or dense_metadata.foundation_manifest_sha256 != foundation.manifest_sha256
        or evaluation_manifest.foundation_manifest_sha256 != foundation.manifest_sha256
        or evaluation_manifest.sparse_manifest_sha256 != sparse.manifest_sha256
        or evaluation_manifest.dense_manifest_sha256 != dense.manifest_sha256
        or sparse_metadata.catalog_membership_sha256 != dense_metadata.catalog_membership_sha256
    ):
        raise RankingFeatureBuildError(
            "feature parent lineage, profile, or catalog is incompatible"
        )
    return _Dependencies(
        foundation,
        foundation_manifest,
        sparse,
        sparse_metadata,
        dense,
        dense_metadata,
        evaluation,
        evaluation_manifest,
    )


def _artifact_dependencies(dependencies: _Dependencies) -> tuple[ArtifactDependency, ...]:
    return tuple(
        sorted(
            (
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
                ArtifactDependency(
                    artifact_id=dependencies.evaluation.manifest.artifact_id,
                    manifest_sha256=dependencies.evaluation.manifest_sha256,
                ),
            ),
            key=lambda dependency: dependency.artifact_id,
        )
    )


def _load_queries(dependencies: _Dependencies, profile: Profile) -> pl.DataFrame:
    profile_column = "in_development" if profile == "development" else "in_portfolio"
    queries = (
        pl.read_parquet(dependencies.foundation.path / QUERIES_FILENAME)
        .filter(pl.col(profile_column))
        .sort("query_id")
    )
    if queries.is_empty() or queries.height != dependencies.evaluation_manifest.query_count:
        raise RankingFeatureBuildError("feature query cohort differs from retrieval evaluation")
    return queries


def _fit_state(
    dependencies: _Dependencies,
    config: ResolvedConfig,
    registry: FeatureRegistry,
) -> FeatureState:
    products_path = dependencies.foundation.path / PRODUCTS_FILENAME
    catalog_attributes = (
        pl.scan_parquet(products_path)
        .filter(pl.col("has_usable_text"))
        .select("normalized_brand", "normalized_color")
        .unique()
        .collect()
    )
    parser_state = build_parser_state(
        tuple(catalog_attributes["normalized_brand"].drop_nulls().to_list()),
        tuple(catalog_attributes["normalized_color"].drop_nulls().to_list()),
        config.config.query_understanding,
    )
    train_products = (
        pl.scan_parquet(dependencies.foundation.path / JUDGED_POOLS_FILENAME)
        .filter((pl.col("profile") == "portfolio") & (pl.col("project_split") == "train"))
        .select("locale", "product_id")
        .unique()
    )
    fitted = (
        pl.scan_parquet(products_path)
        .select(
            pl.col("product_locale").alias("locale"),
            "product_id",
            "normalized_brand",
            "normalized_color",
        )
        .join(train_products, on=("locale", "product_id"), how="inner")
        .collect()
    )
    if fitted.is_empty():
        raise RankingFeatureBuildError("train-only category fitting has no product rows")

    def codes(column: str) -> tuple[tuple[str, int], ...]:
        values = sorted(set(cast(list[str], fitted[column].drop_nulls().to_list())) - {""})
        return tuple((value, code) for code, value in enumerate(values, start=2))

    sparse_retriever = (
        f"bm25:{dependencies.sparse_metadata.component_version}:"
        f"{dependencies.sparse_metadata.tokenizer_version}:"
        f"k1={dependencies.sparse_metadata.k1}:b={dependencies.sparse_metadata.b}"
    )
    dense_retriever = (
        f"dense:{dependencies.dense_metadata.component_version}:"
        f"{dependencies.dense_metadata.model_id}@{dependencies.dense_metadata.model_revision}"
    )
    return FeatureState(
        state_version=config.config.ranking_features.state_version,
        registry_sha256=_model_sha256(registry),
        parser_state=parser_state,
        brand_codes=codes("normalized_brand"),
        color_codes=codes("normalized_color"),
        sparse_index_id=dependencies.sparse.manifest.artifact_id,
        sparse_retriever_id=sparse_retriever,
        dense_index_id=dependencies.dense.manifest.artifact_id,
        dense_retriever_id=dense_retriever,
        rrf_constant=config.config.retrieval.hybrid.rrf_constant,
    )


def _parsed_row(profile: Profile, query: dict[str, Any], parsed: ParsedQuery) -> dict[str, object]:
    return {
        "profile": profile,
        "query_id": query["query_id"],
        "normalized_query_sha256": query["normalized_query_sha256"],
        "raw_text": parsed.raw_text,
        "normalized_text": parsed.normalized_text,
        "tokens": list(parsed.tokens),
        "reduced_tokens": list(parsed.reduced_tokens),
        "numbers": list(parsed.numbers),
        "units": list(parsed.units),
        "measurements": list(parsed.measurements),
        "model_tokens": list(parsed.model_tokens),
        "compatibility_tokens": list(parsed.compatibility_tokens),
        "compatibility_phrases": list(parsed.compatibility_phrases),
        "brand": parsed.brand.value if parsed.brand else None,
        "brand_confidence": parsed.brand.confidence if parsed.brand else None,
        "color": parsed.color.value if parsed.color else None,
        "color_confidence": parsed.color.confidence if parsed.color else None,
        "parser_version": parsed.parser_version,
        "parser_state_sha256": parsed.parser_state_sha256,
        "query_sha256": parsed.query_sha256,
        "warnings": list(parsed.warnings),
    }


def _product_views(frame: pl.DataFrame) -> dict[str, ProductFeatureView]:
    views: dict[str, ProductFeatureView] = {}
    for row in frame.iter_rows(named=True):
        product_id = cast(str, row["product_id"])
        views[product_id] = ProductFeatureView(
            locale=cast(str, row["product_locale"]),
            title=cast(str, row["clean_title"]),
            brand=cast(str, row["clean_brand"]),
            color=cast(str, row["clean_color"]),
            bullets=cast(str, row["clean_bullets"]),
            description=cast(str, row["clean_description"]),
            normalized_brand=cast(str, row["normalized_brand"]),
            normalized_color=cast(str, row["normalized_color"]),
            title_missing=cast(bool, row["title_missing"]),
            brand_missing=cast(bool, row["brand_missing"]),
            color_missing=cast(bool, row["color_missing"]),
            bullets_missing=cast(bool, row["bullets_missing"]),
            description_missing=cast(bool, row["description_missing"]),
        )
    return views


def _matrix_group(
    *,
    population: Population,
    profile: Profile,
    query: dict[str, Any],
    parsed: ParsedQuery,
    pairs: pl.DataFrame,
    products: dict[str, ProductFeatureView],
    sparse_index: Any,
    dense_index: Any,
    state: FeatureState,
    labels: bool,
) -> list[dict[str, object]]:
    product_ids = tuple(cast(list[str], pairs.sort("product_id")["product_id"].to_list()))
    if not product_ids:
        return []
    sparse_scores = {
        item.product_id: item.raw_score
        for item in sparse_index.score_pairs(parsed.normalized_text, product_ids)
    }
    dense_scores = {
        item.product_id: item.raw_score
        for item in dense_index.score_pairs(parsed.normalized_text, product_ids)
    }
    sparse_ranks, sparse_fractions = rank_fractions(sparse_scores)
    dense_ranks, dense_fractions = rank_fractions(dense_scores)
    rrf_scores = {
        product_id: 1.0 / (state.rrf_constant + sparse_ranks[product_id])
        + 1.0 / (state.rrf_constant + dense_ranks[product_id])
        for product_id in product_ids
    }
    _, rrf_fractions = rank_fractions(rrf_scores)
    idfs = sparse_index.query_idf_values(parsed.normalized_text)
    lexical_specificity = sum(idfs) / len(idfs) if idfs else 0.0
    brand_codes = dict(state.brand_codes)
    color_codes = dict(state.color_codes)
    pair_rows = {cast(str, row["product_id"]): row for row in pairs.iter_rows(named=True)}
    rows: list[dict[str, object]] = []
    for product_id in product_ids:
        if product_id not in products:
            raise RankingFeatureBuildError(
                f"feature product is missing official fields: {product_id}"
            )
        row: dict[str, object] = {
            "population": population,
            "profile": profile,
            "query_id": query["query_id"],
            "normalized_query_sha256": query["normalized_query_sha256"],
            "project_split": query["project_split"],
            "locale": query["locale"],
            "product_id": product_id,
            "feature_set_id": state.feature_set_id,
        }
        if labels:
            row["label_id"] = pair_rows[product_id]["label_id"]
            row["gain"] = pair_rows[product_id]["gain"]
        row.update(
            compute_core_features(
                parsed,
                products[product_id],
                lexical_specificity=lexical_specificity,
                brand_codes=brand_codes,
                color_codes=color_codes,
                bm25_score=sparse_scores[product_id],
                bm25_rank_fraction=sparse_fractions[product_id],
                dense_score=dense_scores[product_id],
                dense_rank_fraction=dense_fractions[product_id],
                rrf_score=rrf_scores[product_id],
                rrf_rank_fraction=rrf_fractions[product_id],
            )
        )
        rows.append(row)
    return rows


def _distribution_rows(
    population: Population, paths: tuple[Path, ...], registry: FeatureRegistry
) -> list[dict[str, object]]:
    lazy = pl.scan_parquet([str(path) for path in paths])
    rows: list[dict[str, object]] = []
    for feature in registry.features:
        summary = (
            lazy.select(
                pl.len().alias("rows"),
                pl.col(feature.name).null_count().alias("nulls"),
                pl.col(feature.name).cast(pl.Float64).mean().alias("mean"),
                pl.col(feature.name).cast(pl.Float64).std(ddof=0).fill_null(0.0).alias("std"),
                pl.col(feature.name).cast(pl.Float64).min().alias("min"),
                pl.col(feature.name).cast(pl.Float64).max().alias("max"),
            )
            .collect()
            .row(0, named=True)
        )
        rows.append(
            {
                "population": population,
                "feature_set_id": registry.feature_set_id,
                "feature": feature.name,
                "dtype": feature.dtype,
                "rows": summary["rows"],
                "nulls": summary["nulls"],
                "mean": summary["mean"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
            }
        )
    return rows


def _leakage_report(registry: FeatureRegistry) -> LeakageReport:
    prohibited = (
        "absolute_candidate_count",
        "absolute_candidate_rank",
        "esci_label",
        "gain",
        "label_id",
        "product_id",
        "source_count",
        "source_topk_rank",
        "target_history",
    )
    names = set(FEATURE_NAMES)
    if names & set(prohibited):
        raise RankingFeatureBuildError("the primary feature registry contains prohibited inputs")
    checks = tuple(
        sorted(
            (
                LeakageCheck(
                    check_id="candidate_payload_label_free",
                    detail="retrieved-union matrix schema has no label or gain columns",
                ),
                LeakageCheck(
                    check_id="categorical_state_train_only",
                    detail=(
                        "brand/color codes fit only official source fields from project train rows"
                    ),
                ),
                LeakageCheck(
                    check_id="no_product_identity_feature",
                    detail="product_id is a row key and absent from the ordered model feature list",
                ),
                LeakageCheck(
                    check_id="no_source_topk_features",
                    detail="generator membership/scores/ranks are excluded from ltr_core_v1",
                ),
                LeakageCheck(
                    check_id="no_target_derived_features",
                    detail=(
                        "registry formulas use no labels, gains, or relevance-history aggregates"
                    ),
                ),
                LeakageCheck(
                    check_id="online_formula_availability",
                    detail="every registry formula is available through compute_core_features",
                ),
            ),
            key=lambda check: check.check_id,
        )
    )
    return LeakageReport(
        prohibited_feature_names=prohibited,
        registry_feature_count=len(registry.features),
        checks=checks,
    )


def _parity_frame(paths: tuple[Path, ...], limit: int) -> pl.DataFrame:
    columns = [
        "population",
        "profile",
        "query_id",
        "normalized_query_sha256",
        "locale",
        "product_id",
        "feature_set_id",
        *FEATURE_NAMES,
    ]
    frame = pl.read_parquet([str(path) for path in paths], columns=columns).head(limit)
    hashes = []
    for row in frame.select(FEATURE_NAMES).iter_rows(named=False):
        payload = json.dumps(row, separators=(",", ":"), allow_nan=False).encode("utf-8")
        hashes.append(sha256(payload).hexdigest())
    return frame.with_columns(pl.Series("feature_vector_sha256", hashes, dtype=pl.String))


def _reuse(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    dependencies: _Dependencies,
    store: ArtifactStore,
) -> RankingFeatureBuildResult:
    artifact = store.load(ranking_feature_artifact_id(release, config.sha256, profile))
    if artifact.manifest.dependencies != _artifact_dependencies(dependencies):
        raise RankingFeatureValidationError("ranking feature dependencies are incompatible")
    manifest = load_ranking_feature_manifest(artifact.path / FEATURE_ARTIFACT_FILENAME)
    state = load_feature_state(artifact.path / FEATURE_STATE_FILENAME)
    if (
        manifest.artifact_id != artifact.manifest.artifact_id
        or manifest.state_sha256 != _model_sha256(state)
        or manifest.registry_sha256 != state.registry_sha256
    ):
        raise RankingFeatureValidationError("ranking feature payload identity is incompatible")
    return RankingFeatureBuildResult(artifact, manifest, state, True)


def build_ranking_features(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    profile: Profile | None = None,
    artifact_store: ArtifactStore | None = None,
    dense_encoder: DenseEncoder | None = None,
) -> RankingFeatureBuildResult:
    """Build/reuse parser state and bounded closed/candidate `ltr_core_v1` matrices."""
    selected_profile: Profile = profile or config.config.evaluation.default_profile
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    dependencies = _load_dependencies(release, config, selected_profile, store)
    artifact_id = ranking_feature_artifact_id(release, config.sha256, selected_profile)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse(release, config, selected_profile, dependencies, store)

    registry = ltr_core_v1_registry()
    state = _fit_state(dependencies, config, registry)
    queries = _load_queries(dependencies, selected_profile)
    parser = QueryParser(state.parser_state, config.config.query_understanding)
    parsed_by_id: dict[int, ParsedQuery] = {}
    parsed_rows: list[dict[str, object]] = []
    query_rows = queries.to_dicts()
    for query in query_rows:
        parsed = parser.parse(cast(str, query["raw_query"]))
        if parsed.normalized_text != query["normalized_query"]:
            raise RankingFeatureBuildError("parser normalization differs from canonical query text")
        query_id = cast(int, query["query_id"])
        parsed_by_id[query_id] = parsed
        parsed_rows.append(_parsed_row(selected_profile, query, parsed))

    rss_limit = config.config.runtime.rss_limit_mb * 1024 * 1024
    encoder = (
        dense_encoder
        if dense_encoder is not None
        else cast(DenseEncoder, SentenceTransformerEncoder(config))
    )
    with ExitStack() as stack:
        sparse_index = stack.enter_context(
            load_sparse_index(store, dependencies.sparse.manifest.artifact_id)
        )
        dense_index = stack.enter_context(
            load_dense_index(store, dependencies.dense.manifest.artifact_id, encoder=encoder)
        )
        load_peak = _peak_rss_bytes()
        initial_resource = FeatureResourceMeasurement(
            load_peak_rss_bytes=load_peak,
            materialization_peak_rss_bytes=load_peak,
            peak_rss_bytes=load_peak,
            rss_limit_bytes=rss_limit,
            artifact_payload_bytes=0,
            passed=load_peak <= rss_limit,
        )
        if not initial_resource.passed:
            raise RankingFeatureResourceError(initial_resource)

        transaction = store.stage(
            artifact_type="ranking-features",
            dataset_version=release.manifest.dataset_version,
            profile=selected_profile,
            component_version=config.config.ranking_features.component_version,
            config_sha256=config.sha256,
            code_revision=code_revision,
            dependencies=_artifact_dependencies(dependencies),
        )
        try:
            with transaction:
                staging_root = transaction.path(FEATURE_ARTIFACT_FILENAME).parent
                closed_writer = _PartitionWriter(
                    transaction.path(CLOSED_MATRIX_DIRECTORY),
                    _feature_schema(labels=True),
                    config.config.ranking_features.matrix_partition_rows,
                )
                candidate_writer = _PartitionWriter(
                    transaction.path(CANDIDATE_MATRIX_DIRECTORY),
                    _feature_schema(labels=False),
                    config.config.ranking_features.matrix_partition_rows,
                )
                closed_rows = 0
                closed_excluded_outside_catalog = 0
                candidate_rows = 0
                batch_size = config.config.ranking_features.query_batch_size
                products_path = dependencies.foundation.path / PRODUCTS_FILENAME
                pools_path = dependencies.foundation.path / JUDGED_POOLS_FILENAME
                membership_path = dependencies.foundation.path / CATALOG_MEMBERSHIP_FILENAME
                candidate_glob = str(
                    dependencies.evaluation.path / CANDIDATE_DIRECTORY / "*.parquet"
                )
                for start in range(0, len(query_rows), batch_size):
                    batch = query_rows[start : start + batch_size]
                    query_ids = [cast(int, row["query_id"]) for row in batch]
                    selected_pool = (
                        pl.scan_parquet(pools_path)
                        .filter(
                            (pl.col("profile") == selected_profile)
                            & pl.col("query_id").is_in(query_ids)
                        )
                        .select("query_id", "locale", "product_id", "label_id", "gain")
                    )
                    selected_pool_rows = selected_pool.select(pl.len()).collect().item()
                    closed = selected_pool.join(
                        pl.scan_parquet(membership_path).select("locale", "product_id"),
                        on=("locale", "product_id"),
                        how="inner",
                    ).collect()
                    closed_excluded_outside_catalog += selected_pool_rows - closed.height
                    candidates = (
                        pl.scan_parquet(candidate_glob)
                        .filter((pl.col("stage") == "hybrid") & pl.col("query_id").is_in(query_ids))
                        .select("query_id", "locale", "product_id")
                        .collect()
                    )
                    keys = pl.concat(
                        (
                            closed.select("locale", "product_id"),
                            candidates.select("locale", "product_id"),
                        )
                    ).unique()
                    products = (
                        pl.scan_parquet(products_path)
                        .join(
                            keys.lazy(),
                            left_on=("product_locale", "product_id"),
                            right_on=("locale", "product_id"),
                            how="inner",
                        )
                        .select(
                            "product_id",
                            "product_locale",
                            "clean_title",
                            "clean_brand",
                            "clean_color",
                            "clean_bullets",
                            "clean_description",
                            "normalized_brand",
                            "normalized_color",
                            "title_missing",
                            "brand_missing",
                            "color_missing",
                            "bullets_missing",
                            "description_missing",
                        )
                        .collect()
                    )
                    product_views = _product_views(products)
                    for query in batch:
                        query_id = cast(int, query["query_id"])
                        closed_pairs = closed.filter(pl.col("query_id") == query_id)
                        candidate_pairs = candidates.filter(pl.col("query_id") == query_id)
                        if closed_pairs.height > config.config.ranking_features.max_rows_per_query:
                            raise RankingFeatureBuildError(
                                "closed judged group exceeds configured row bound"
                            )
                        if (
                            candidate_pairs.height
                            > config.config.ranking_features.max_rows_per_query
                        ):
                            raise RankingFeatureBuildError(
                                "retrieved union exceeds configured row bound"
                            )
                        closed_group = _matrix_group(
                            population="closed_judged",
                            profile=selected_profile,
                            query=query,
                            parsed=parsed_by_id[query_id],
                            pairs=closed_pairs,
                            products=product_views,
                            sparse_index=sparse_index,
                            dense_index=dense_index,
                            state=state,
                            labels=True,
                        )
                        candidate_group = _matrix_group(
                            population="retrieved_union",
                            profile=selected_profile,
                            query=query,
                            parsed=parsed_by_id[query_id],
                            pairs=candidate_pairs,
                            products=product_views,
                            sparse_index=sparse_index,
                            dense_index=dense_index,
                            state=state,
                            labels=False,
                        )
                        closed_writer.write_group(closed_group)
                        candidate_writer.write_group(candidate_group)
                        closed_rows += len(closed_group)
                        candidate_rows += len(candidate_group)
                closed_writer.close()
                candidate_writer.close()
                if closed_rows == 0:
                    raise RankingFeatureBuildError("closed feature matrix cannot be empty")

                pl.DataFrame(parsed_rows, schema=_PARSED_SCHEMA).write_parquet(
                    transaction.path(PARSED_QUERIES_FILENAME), compression="zstd", statistics=True
                )
                transaction.path(FEATURE_REGISTRY_FILENAME).write_text(
                    _canonical_json(registry), encoding="utf-8"
                )
                transaction.path(FEATURE_STATE_FILENAME).write_text(
                    _canonical_json(state), encoding="utf-8"
                )
                leakage = _leakage_report(registry)
                transaction.path(LEAKAGE_REPORT_FILENAME).write_text(
                    _canonical_json(leakage), encoding="utf-8"
                )
                distribution = pl.DataFrame(
                    _distribution_rows("closed_judged", closed_writer.paths, registry)
                    + _distribution_rows("retrieved_union", candidate_writer.paths, registry)
                ).sort("population", "feature")
                distribution.write_parquet(
                    transaction.path(DISTRIBUTION_REPORT_FILENAME),
                    compression="zstd",
                    statistics=True,
                )
                parity_limit = config.config.ranking_features.parity_fixture_rows
                parity = pl.concat(
                    (
                        _parity_frame(closed_writer.paths, parity_limit),
                        _parity_frame(candidate_writer.paths, parity_limit),
                    ),
                    how="vertical",
                ).head(parity_limit)
                parity.write_parquet(
                    transaction.path(PARITY_FIXTURES_FILENAME), compression="zstd", statistics=True
                )
                materialization_peak = _peak_rss_bytes()
                payload_bytes = _directory_bytes(staging_root)
                measurement = FeatureResourceMeasurement(
                    load_peak_rss_bytes=load_peak,
                    materialization_peak_rss_bytes=materialization_peak,
                    peak_rss_bytes=max(load_peak, materialization_peak),
                    rss_limit_bytes=rss_limit,
                    artifact_payload_bytes=payload_bytes,
                    passed=max(load_peak, materialization_peak) <= rss_limit,
                )
                if not measurement.passed:
                    raise RankingFeatureResourceError(measurement)
                checks = tuple(
                    sorted(
                        (
                            FeatureCheck(
                                check_id="bounded_candidate_alignment",
                                detail=(
                                    "features are materialized only for closed or retrieved "
                                    "query-product pairs"
                                ),
                            ),
                            FeatureCheck(
                                check_id="catalog_eligible_closed_population",
                                detail=(
                                    f"excluded {closed_excluded_outside_catalog} judged rows "
                                    "without fixed-catalog documents"
                                ),
                            ),
                            FeatureCheck(
                                check_id="closed_pair_score_coverage",
                                detail=(
                                    "direct sparse and dense scores exist for all "
                                    f"{closed_rows} judged rows"
                                ),
                            ),
                            FeatureCheck(
                                check_id="distribution_report",
                                detail=(
                                    "closed and retrieved populations report every "
                                    "registered feature"
                                ),
                            ),
                            FeatureCheck(
                                check_id="leakage_review",
                                detail=(
                                    "target, identity, absolute-rank, and source-provenance "
                                    "inputs are excluded"
                                ),
                            ),
                            FeatureCheck(
                                check_id="offline_online_parity",
                                detail=(
                                    f"{parity.height} shared-formula feature vectors persisted "
                                    "as parity fixtures"
                                ),
                            ),
                            FeatureCheck(
                                check_id="resource_gate",
                                detail=(
                                    f"peak RSS {measurement.peak_rss_bytes} <= "
                                    f"{measurement.rss_limit_bytes} bytes"
                                ),
                            ),
                        ),
                        key=lambda check: check.check_id,
                    )
                )
                manifest = RankingFeatureManifest(
                    artifact_id=artifact_id,
                    dataset_version=release.manifest.dataset_version,
                    config_sha256=config.sha256,
                    profile=selected_profile,
                    registry_sha256=state.registry_sha256,
                    state_sha256=_model_sha256(state),
                    foundation_artifact_id=dependencies.foundation.manifest.artifact_id,
                    foundation_manifest_sha256=dependencies.foundation.manifest_sha256,
                    sparse_artifact_id=dependencies.sparse.manifest.artifact_id,
                    sparse_manifest_sha256=dependencies.sparse.manifest_sha256,
                    dense_artifact_id=dependencies.dense.manifest.artifact_id,
                    dense_manifest_sha256=dependencies.dense.manifest_sha256,
                    retrieval_evaluation_artifact_id=dependencies.evaluation.manifest.artifact_id,
                    retrieval_evaluation_manifest_sha256=dependencies.evaluation.manifest_sha256,
                    query_count=queries.height,
                    normalized_query_groups=queries["normalized_query_sha256"].n_unique(),
                    feature_count=len(registry.features),
                    closed_rows=closed_rows,
                    closed_excluded_outside_catalog=closed_excluded_outside_catalog,
                    candidate_rows=candidate_rows,
                    closed_partitions=len(closed_writer.paths),
                    candidate_partitions=len(candidate_writer.paths),
                    parity_fixture_rows=parity.height,
                    resource=measurement,
                    checks=checks,
                )
                transaction.path(FEATURE_ARTIFACT_FILENAME).write_text(
                    _canonical_json(manifest), encoding="utf-8"
                )
                artifact = transaction.commit()
        except ArtifactExistsError:
            return _reuse(release, config, selected_profile, dependencies, store)
    return RankingFeatureBuildResult(artifact, manifest, state, False)


__all__ = [
    "CANDIDATE_MATRIX_DIRECTORY",
    "CLOSED_MATRIX_DIRECTORY",
    "DISTRIBUTION_REPORT_FILENAME",
    "FEATURE_ARTIFACT_FILENAME",
    "FEATURE_REGISTRY_FILENAME",
    "FEATURE_STATE_FILENAME",
    "LEAKAGE_REPORT_FILENAME",
    "PARITY_FIXTURES_FILENAME",
    "PARSED_QUERIES_FILENAME",
    "FeatureResourceMeasurement",
    "FeatureState",
    "LeakageReport",
    "RankingFeatureBuildError",
    "RankingFeatureBuildResult",
    "RankingFeatureError",
    "RankingFeatureManifest",
    "RankingFeatureResourceError",
    "RankingFeatureValidationError",
    "build_ranking_features",
    "load_feature_state",
    "load_ranking_feature_manifest",
    "ranking_feature_artifact_id",
]
