"""Canonical ESCI tables, catalog, judged pools, documents, and M2 resource gate."""

from __future__ import annotations

import gc
import json
import unicodedata
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Any, Literal, Self

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
from market_rank.data.profiles import (
    ASSIGNMENTS_FILENAME,
    EsciProfileManifest,
    load_profile_manifest,
    normalize_query_group,
    profile_artifact_id,
)

FOUNDATION_COMPONENT_VERSION = "data-foundation-v2"
FOUNDATION_MANIFEST_FILENAME = "foundation-manifest.json"
FULL_CATALOG_ID: Literal["esci_task1_us_catalog_v1"] = "esci_task1_us_catalog_v1"
COMPACT_CATALOG_ID: Literal["esci_task1_us_compact_catalog_v1"] = "esci_task1_us_compact_catalog_v1"
CatalogId = Literal["esci_task1_us_catalog_v1", "esci_task1_us_compact_catalog_v1"]
JUDGED_POOL_ID: Literal["esci_task1_us_judged_pool_v1"] = "esci_task1_us_judged_pool_v1"
LABEL_MAPPING_ID: Literal["esci-label-id-v1"] = "esci-label-id-v1"
GAIN_MAPPING_ID: Literal["esci-task1-gain-v1"] = "esci-task1-gain-v1"

QUERIES_FILENAME = "queries.parquet"
SOURCES_FILENAME = "sources.parquet"
JUDGMENTS_FILENAME = "judgments.parquet"
PRODUCTS_FILENAME = "products.parquet"
PRODUCT_DOCUMENTS_FILENAME = "product-documents.parquet"
CATALOG_MEMBERSHIP_FILENAME = "catalog-membership.parquet"
CATALOG_EXCLUSIONS_FILENAME = "catalog-exclusions.parquet"
JUDGED_POOLS_FILENAME = "judged-pools.parquet"

_TABLE_ORDER = (
    QUERIES_FILENAME,
    SOURCES_FILENAME,
    JUDGMENTS_FILENAME,
    PRODUCTS_FILENAME,
    PRODUCT_DOCUMENTS_FILENAME,
    CATALOG_MEMBERSHIP_FILENAME,
    CATALOG_EXCLUSIONS_FILENAME,
    JUDGED_POOLS_FILENAME,
)
_LABEL_IDS = {"I": 0, "C": 1, "S": 2, "E": 3}
_LABEL_GAINS = {"I": 0.0, "C": 0.01, "S": 0.1, "E": 1.0}

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class DataFoundationError(RuntimeError):
    """Base exception for Goldfish 006 construction and validation failures."""


class FoundationDependencyError(DataFoundationError):
    """Raised when the Goldfish 005 parent is missing or incompatible."""


class FoundationInvariantError(DataFoundationError):
    """Raised when canonical data violates keys, joins, mappings, or populations."""


class FoundationManifestError(DataFoundationError):
    """Raised when persisted Goldfish 006 metadata is invalid."""


class ResourceGateError(DataFoundationError):
    """Raised when the preliminary M3/8 GB serving estimate exceeds its limit."""

    def __init__(self, estimate: ResourceEstimate) -> None:
        super().__init__(
            f"M2 resource estimate {estimate.projected_runtime_bytes} bytes exceeds "
            f"the configured {estimate.rss_limit_bytes}-byte RSS limit"
        )
        self.estimate = estimate


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TableSummary(_StrictModel):
    """Integrity and schema facts for one canonical output table."""

    filename: str = Field(strict=True, min_length=1)
    row_count: int = Field(strict=True, ge=0)
    size_bytes: int = Field(strict=True, ge=0)
    sha256: Sha256Digest
    primary_key: tuple[str, ...]
    columns: tuple[str, ...]


class ProfilePoolSummary(_StrictModel):
    """Observed complete judged-pool size for one nested profile."""

    profile: Literal["development", "portfolio"]
    query_ids: int = Field(strict=True, ge=0)
    judgments: int = Field(strict=True, ge=0)
    project_train_query_ids: int = Field(strict=True, ge=0)
    project_validation_query_ids: int = Field(strict=True, ge=0)
    project_test_query_ids: int = Field(strict=True, ge=0)


class CatalogSelectionSummary(_StrictModel):
    """Label-blind fixed-catalog selection and audit counts."""

    mode: Literal["full", "compact"]
    method: Literal["full-task1-us-v1", "portfolio-judged-plus-sha256-v1"]
    source_products: int = Field(strict=True, ge=1)
    required_judged_products: int = Field(strict=True, ge=1)
    selected_distractor_products: int = Field(strict=True, ge=0)
    configured_distractor_products: int = Field(strict=True, ge=0)
    selected_candidate_products: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        available_distractors = self.source_products - self.required_judged_products
        if available_distractors < 0:
            raise ValueError("required judged products exceed the source catalog")
        if self.selected_candidate_products != (
            self.required_judged_products + self.selected_distractor_products
        ):
            raise ValueError("catalog candidates do not equal required plus distractor products")
        if self.mode == "full":
            if (
                self.method != "full-task1-us-v1"
                or self.selected_candidate_products != self.source_products
                or self.selected_distractor_products != available_distractors
            ):
                raise ValueError("full catalog selection does not retain every source product")
        elif (
            self.method != "portfolio-judged-plus-sha256-v1"
            or self.selected_distractor_products
            != min(self.configured_distractor_products, available_distractors)
        ):
            raise ValueError("compact catalog selection does not match its deterministic target")
        return self


class ResourceEstimate(_StrictModel):
    """Transparent M2 preliminary serving-memory estimate and proceed/block decision."""

    method: Literal["m2-catalog-linear-v1"] = "m2-catalog-linear-v1"
    catalog_products: int = Field(strict=True, ge=0)
    document_utf8_bytes: int = Field(strict=True, ge=0)
    compact_display_utf8_bytes: int = Field(strict=True, ge=0)
    projected_sparse_index_bytes: int = Field(strict=True, ge=0)
    projected_dense_vector_bytes: int = Field(strict=True, ge=0)
    projected_id_overhead_bytes: int = Field(strict=True, ge=0)
    fixed_runtime_reserve_bytes: int = Field(strict=True, ge=0)
    projected_runtime_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    proceed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        expected = (
            self.projected_sparse_index_bytes
            + self.projected_dense_vector_bytes
            + self.projected_id_overhead_bytes
            + self.compact_display_utf8_bytes
            + self.fixed_runtime_reserve_bytes
        )
        if self.projected_runtime_bytes != expected:
            raise ValueError("projected_runtime_bytes does not equal its components")
        if self.proceed != (expected <= self.rss_limit_bytes):
            raise ValueError("resource proceed decision does not match the RSS limit")
        return self


class FoundationCheck(_StrictModel):
    """One successful hard validation retained with the artifact."""

    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class DataFoundationManifest(_StrictModel):
    """Lineage, populations, mappings, tables, and resource gate for Goldfish 006."""

    schema_version: Literal[2] = 2
    dataset_version: str = Field(strict=True, min_length=1)
    source_revision: str = Field(strict=True, min_length=1)
    release_manifest_sha256: Sha256Digest
    config_sha256: Sha256Digest
    profile_artifact_id: str = Field(strict=True, min_length=1)
    profile_manifest_sha256: Sha256Digest
    benchmark_predicate: Literal["product_locale == 'us' and small_version == 1"] = (
        "product_locale == 'us' and small_version == 1"
    )
    catalog_id: CatalogId
    catalog_selection: CatalogSelectionSummary
    catalog_candidate_products: int = Field(strict=True, ge=1)
    catalog_products: int = Field(strict=True, ge=1)
    catalog_excluded_no_text: int = Field(strict=True, ge=0)
    judged_pool_id: Literal["esci_task1_us_judged_pool_v1"] = JUDGED_POOL_ID
    label_mapping_id: Literal["esci-label-id-v1"] = LABEL_MAPPING_ID
    gain_mapping_id: Literal["esci-task1-gain-v1"] = GAIN_MAPPING_ID
    label_mapping: tuple[tuple[str, int, float], ...]
    product_document_version: Literal["product-document-v1"]
    product_document_template: Literal[
        "[TITLE] title [BRAND] brand [COLOR] color [BULLETS] bullets [DESCRIPTION] description"
    ] = "[TITLE] title [BRAND] brand [COLOR] color [BULLETS] bullets [DESCRIPTION] description"
    tables: tuple[TableSummary, ...]
    pools: tuple[ProfilePoolSummary, ...]
    resource_estimate: ResourceEstimate
    checks: tuple[FoundationCheck, ...]

    @model_validator(mode="after")
    def validate_collections(self) -> Self:
        expected_catalog_id = (
            COMPACT_CATALOG_ID if self.catalog_selection.mode == "compact" else FULL_CATALOG_ID
        )
        if self.catalog_id != expected_catalog_id:
            raise ValueError("catalog ID does not match the selected catalog mode")
        if self.catalog_candidate_products != self.catalog_selection.selected_candidate_products:
            raise ValueError("catalog candidate count does not match the selection audit")
        if tuple(table.filename for table in self.tables) != _TABLE_ORDER:
            raise ValueError("tables are not in the canonical Goldfish 006 order")
        if tuple(pool.profile for pool in self.pools) != ("development", "portfolio"):
            raise ValueError("pools must be ordered development, portfolio")
        if self.label_mapping != (
            ("I", 0, 0.0),
            ("C", 1, 0.01),
            ("S", 2, 0.1),
            ("E", 3, 1.0),
        ):
            raise ValueError("label mapping must use official Task-1 IDs and gains")
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(check_ids) != len(set(check_ids)):
            raise ValueError("checks must be unique and sorted")
        return self


@dataclass(frozen=True, slots=True)
class FoundationBuildResult:
    """Published or reused M2 data foundation."""

    artifact: LoadedArtifact
    manifest: DataFoundationManifest
    reused: bool


class _PlainTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _clean_text(value: str | None, *, max_chars: int) -> str:
    if value is None:
        return ""
    parser = _PlainTextExtractor()
    try:
        parser.feed(value)
        parser.close()
        text = " ".join(parser.parts)
    except (ValueError, AssertionError):
        text = value
    normalized = unicodedata.normalize("NFKC", text)
    without_controls = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    return " ".join(without_controls.split())[:max_chars].rstrip()


def _normalize_attribute(value: str | None, *, max_chars: int) -> str | None:
    cleaned = _clean_text(value, max_chars=max_chars)
    return cleaned.casefold() if cleaned else None


def _document_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def catalog_id_for_config(config: ResolvedConfig) -> CatalogId:
    """Return the explicit full or compact fixed-catalog identity."""
    return (
        COMPACT_CATALOG_ID if config.config.dataset.catalog_mode == "compact" else FULL_CATALOG_ID
    )


def _catalog_priority(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size_bytes = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
    except OSError as exc:
        raise DataFoundationError(f"cannot hash {path}: {exc}") from exc
    return size_bytes, digest.hexdigest()


def _verify_raw_files(release: ResolvedReleaseManifest, raw_root: Path) -> dict[str, Path]:
    if raw_root.is_symlink():
        raise DataFoundationError(f"raw root cannot be a symbolic link: {raw_root}")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise DataFoundationError(f"raw root is unavailable: {raw_root}: {exc}") from exc
    paths: dict[str, Path] = {}
    for source in release.manifest.files:
        path = root / source.filename
        if path.is_symlink() or not path.is_file():
            raise DataFoundationError(f"raw source is not a regular file: {path}")
        size_bytes, file_sha256 = _sha256_file(path)
        if size_bytes != source.size_bytes or file_sha256 != source.sha256:
            raise DataFoundationError(
                f"raw {source.role} source no longer matches the validated pinned release"
            )
        paths[source.role] = path
    return paths


def _load_profile_dependency(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    store: ArtifactStore,
) -> tuple[LoadedArtifact, EsciProfileManifest, pl.DataFrame]:
    expected_id = profile_artifact_id(release, config.sha256)
    try:
        artifact = store.load(expected_id)
        manifest = load_profile_manifest(artifact.path / "profile-manifest.json")
        assignments = pl.read_parquet(artifact.path / ASSIGNMENTS_FILENAME)
    except (OSError, RuntimeError, pl.exceptions.PolarsError) as exc:
        raise FoundationDependencyError(
            "a compatible Goldfish 005 profile artifact is required; run "
            "`market-rank data build-esci-profiles` first"
        ) from exc
    expected_columns = (
        "query_id",
        "official_split",
        "project_split",
        "normalized_query_sha256",
        "judgment_count",
        "in_development",
        "in_portfolio",
        "quarantine_reason",
    )
    if tuple(assignments.columns) != expected_columns:
        raise FoundationDependencyError("profile assignments have an incompatible schema")
    if (
        manifest.release_manifest_sha256 != release.sha256
        or manifest.config_sha256 != config.sha256
        or assignments.height != manifest.task1_us_query_ids
    ):
        raise FoundationDependencyError("profile artifact lineage or row count is incompatible")
    if assignments["query_id"].n_unique() != assignments.height:
        raise FoundationDependencyError("profile assignments contain duplicate query IDs")
    return artifact, manifest, assignments


def _collect_scalar(frame: pl.LazyFrame, column: str = "value") -> int:
    return int(frame.collect(engine="streaming").item(0, column))


def _sink(frame: pl.LazyFrame, path: Path) -> None:
    try:
        frame.sink_parquet(path, compression="zstd", statistics=True, maintain_order=True)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise DataFoundationError(f"cannot write canonical table {path.name}: {exc}") from exc


def _table_summary(path: Path, primary_key: tuple[str, ...]) -> TableSummary:
    try:
        lazy = pl.scan_parquet(path)
        row_count = _collect_scalar(lazy.select(pl.len().alias("value")))
        columns = tuple(lazy.collect_schema().names())
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise FoundationInvariantError(f"cannot validate table {path.name}: {exc}") from exc
    if primary_key and row_count:
        unique_count = _collect_scalar(
            lazy.select(pl.struct(primary_key).n_unique().alias("value"))
        )
        if unique_count != row_count:
            raise FoundationInvariantError(f"{path.name} violates primary key {primary_key}")
    size_bytes, file_sha256 = _sha256_file(path)
    return TableSummary(
        filename=path.name,
        row_count=row_count,
        size_bytes=size_bytes,
        sha256=file_sha256,
        primary_key=primary_key,
        columns=columns,
    )


def _profile_pool_summary(
    path: Path, profile: Literal["development", "portfolio"]
) -> ProfilePoolSummary:
    pool = pl.scan_parquet(path).filter(pl.col("profile") == profile)
    query_ids = _collect_scalar(pool.select(pl.col("query_id").n_unique().alias("value")))
    judgments = _collect_scalar(pool.select(pl.len().alias("value")))

    def split_count(project_split: str) -> int:
        return _collect_scalar(
            pool.filter(pl.col("project_split") == project_split).select(
                pl.col("query_id").n_unique().alias("value")
            )
        )

    return ProfilePoolSummary(
        profile=profile,
        query_ids=query_ids,
        judgments=judgments,
        project_train_query_ids=split_count("train"),
        project_validation_query_ids=split_count("validation"),
        project_test_query_ids=split_count("test"),
    )


def _resource_estimate(
    transaction_root: Path,
    config: ResolvedConfig,
) -> ResourceEstimate:
    documents = pl.scan_parquet(transaction_root / PRODUCT_DOCUMENTS_FILENAME)
    products = pl.scan_parquet(transaction_root / PRODUCTS_FILENAME).filter(
        pl.col("has_usable_text")
    )
    catalog_products = _collect_scalar(documents.select(pl.len().alias("value")))
    document_bytes = _collect_scalar(
        documents.select(pl.col("document").str.len_bytes().sum().alias("value"))
    )
    compact_display_bytes = _collect_scalar(
        products.select(
            pl.sum_horizontal(
                pl.col("product_title").fill_null("").str.len_bytes(),
                pl.col("product_brand").fill_null("").str.len_bytes(),
                pl.col("product_color").fill_null("").str.len_bytes(),
            )
            .sum()
            .alias("value")
        )
    )
    sparse_bytes = (document_bytes * 3) // 2
    dense_bytes = catalog_products * 384 * 4
    id_overhead_bytes = catalog_products * 256
    reserve_bytes = config.config.dataset.m2_runtime_reserve_mb * 1024 * 1024
    projected = (
        sparse_bytes + dense_bytes + id_overhead_bytes + compact_display_bytes + reserve_bytes
    )
    limit = config.config.runtime.rss_limit_mb * 1024 * 1024
    return ResourceEstimate(
        catalog_products=catalog_products,
        document_utf8_bytes=document_bytes,
        compact_display_utf8_bytes=compact_display_bytes,
        projected_sparse_index_bytes=sparse_bytes,
        projected_dense_vector_bytes=dense_bytes,
        projected_id_overhead_bytes=id_overhead_bytes,
        fixed_runtime_reserve_bytes=reserve_bytes,
        projected_runtime_bytes=projected,
        rss_limit_bytes=limit,
        proceed=projected <= limit,
    )


def _write_tables(
    root: Path,
    raw_paths: dict[str, Path],
    assignments: pl.DataFrame,
    config: ResolvedConfig,
) -> CatalogSelectionSummary:
    task_examples = pl.scan_parquet(raw_paths["examples"]).filter(
        (pl.col("product_locale") == "us") & (pl.col("small_version") == 1)
    )
    included_assignments = assignments.filter(pl.col("in_portfolio")).lazy()
    selected_examples = task_examples.join(
        included_assignments.select(
            "query_id",
            "project_split",
            "normalized_query_sha256",
            "in_development",
            "in_portfolio",
        ),
        on="query_id",
        how="inner",
    )

    raw_sources = pl.scan_csv(raw_paths["sources"])
    queries = (
        selected_examples.group_by("query_id")
        .agg(
            pl.col("query").first().alias("raw_query"),
            pl.col("product_locale").first().alias("locale"),
            pl.col("split").first().alias("official_split"),
            pl.col("project_split").first(),
            pl.col("normalized_query_sha256").first(),
            pl.col("in_development").first(),
            pl.col("in_portfolio").first(),
        )
        .with_columns(
            pl.col("raw_query")
            .map_elements(normalize_query_group, return_dtype=pl.String)
            .alias("normalized_query")
        )
        .join(raw_sources, on="query_id", how="left")
        .select(
            "query_id",
            "raw_query",
            "normalized_query",
            "normalized_query_sha256",
            "locale",
            "official_split",
            "project_split",
            "source",
            "in_development",
            "in_portfolio",
        )
        .sort("query_id")
    )
    missing_sources = _collect_scalar(
        queries.filter(pl.col("source").is_null()).select(pl.len().alias("value"))
    )
    if missing_sources:
        raise FoundationInvariantError(f"{missing_sources} canonical queries lack a source row")
    _sink(queries, root / QUERIES_FILENAME)
    _sink(queries.select("query_id", "source"), root / SOURCES_FILENAME)

    grouped_judgments = selected_examples.group_by("query_id", "product_locale", "product_id").agg(
        pl.col("example_id").min().alias("example_id"),
        pl.len().cast(pl.UInt32).alias("duplicate_count"),
        pl.col("esci_label").first().alias("esci_label"),
        pl.col("esci_label").n_unique().alias("_label_variants"),
        pl.col("small_version").first(),
        pl.col("large_version").first(),
        pl.col("split").first().alias("official_split"),
        pl.col("split").n_unique().alias("_split_variants"),
        pl.col("project_split").first(),
        pl.col("in_development").first(),
        pl.col("in_portfolio").first(),
    )
    conflicts = _collect_scalar(
        grouped_judgments.filter(
            (pl.col("_label_variants") != 1) | (pl.col("_split_variants") != 1)
        ).select(pl.len().alias("value"))
    )
    if conflicts:
        raise FoundationInvariantError(
            f"{conflicts} judgment keys contain conflicting labels or official splits"
        )
    judgments = (
        grouped_judgments.with_columns(
            pl.col("esci_label")
            .replace_strict(_LABEL_IDS, return_dtype=pl.UInt8)
            .alias("label_id"),
            pl.col("esci_label")
            .replace_strict(_LABEL_GAINS, return_dtype=pl.Float32)
            .alias("gain"),
        )
        .sort("query_id", "product_locale", "product_id")
        .with_columns(
            pl.col("product_id")
            .rank(method="ordinal")
            .over("query_id")
            .cast(pl.UInt32)
            .alias("stable_ordinal")
        )
        .select(
            "query_id",
            pl.col("product_locale").alias("locale"),
            "product_id",
            "example_id",
            "duplicate_count",
            "esci_label",
            "label_id",
            "gain",
            "small_version",
            "large_version",
            "official_split",
            "project_split",
            "in_development",
            "in_portfolio",
            "stable_ordinal",
        )
    )
    _sink(judgments, root / JUDGMENTS_FILENAME)

    source_catalog_keys = task_examples.select("product_locale", "product_id").unique()
    required_catalog_keys = selected_examples.select("product_locale", "product_id").unique()
    dataset = config.config.dataset
    source_products = _collect_scalar(source_catalog_keys.select(pl.len().alias("value")))
    required_products = _collect_scalar(required_catalog_keys.select(pl.len().alias("value")))
    if dataset.catalog_mode == "compact":
        distractors = (
            source_catalog_keys.join(
                required_catalog_keys,
                on=("product_locale", "product_id"),
                how="anti",
            )
            .with_columns(
                pl.concat_str(
                    pl.lit(dataset.catalog_selection_version),
                    pl.lit(str(config.config.runtime.seed)),
                    pl.col("product_locale"),
                    pl.col("product_id"),
                    separator="\0",
                )
                .map_elements(_catalog_priority, return_dtype=pl.String)
                .alias("_selection_priority")
            )
            .sort("_selection_priority", "product_locale", "product_id")
            .head(dataset.compact_catalog_distractor_products)
            .select("product_locale", "product_id")
        )
        catalog_keys = (
            pl.concat((required_catalog_keys, distractors))
            .unique()
            .sort("product_locale", "product_id")
        )
        selection_method: Literal["full-task1-us-v1", "portfolio-judged-plus-sha256-v1"] = (
            "portfolio-judged-plus-sha256-v1"
        )
    else:
        catalog_keys = source_catalog_keys.sort("product_locale", "product_id")
        selection_method = "full-task1-us-v1"
    try:
        selected_catalog_keys = catalog_keys.collect(engine="streaming")
    except pl.exceptions.PolarsError as exc:
        raise FoundationInvariantError(f"cannot select the fixed catalog: {exc}") from exc
    selected_products = selected_catalog_keys.height
    catalog_keys = selected_catalog_keys.lazy()
    selected_distractors = selected_products - required_products
    selection = CatalogSelectionSummary(
        mode=dataset.catalog_mode,
        method=selection_method,
        source_products=source_products,
        required_judged_products=required_products,
        selected_distractor_products=selected_distractors,
        configured_distractor_products=dataset.compact_catalog_distractor_products,
        selected_candidate_products=selected_products,
    )
    raw_products = pl.scan_parquet(raw_paths["products"])
    missing_products = _collect_scalar(
        catalog_keys.join(
            raw_products.select("product_locale", "product_id"),
            on=("product_locale", "product_id"),
            how="anti",
        ).select(pl.len().alias("value"))
    )
    if missing_products:
        raise FoundationInvariantError(
            f"{missing_products} Task-1 catalog products lack official product rows"
        )

    products = (
        raw_products.join(catalog_keys, on=("product_locale", "product_id"), how="inner")
        .with_columns(
            pl.col("product_title")
            .map_elements(
                partial(_clean_text, max_chars=dataset.title_max_chars),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias("clean_title"),
            pl.col("product_brand")
            .map_elements(
                partial(_clean_text, max_chars=dataset.brand_max_chars),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias("clean_brand"),
            pl.col("product_color")
            .map_elements(
                partial(_clean_text, max_chars=dataset.color_max_chars),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias("clean_color"),
            pl.col("product_bullet_point")
            .map_elements(
                partial(_clean_text, max_chars=dataset.bullets_max_chars),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias("clean_bullets"),
            pl.col("product_description")
            .map_elements(
                partial(_clean_text, max_chars=dataset.description_max_chars),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias("clean_description"),
            pl.col("product_brand")
            .map_elements(
                partial(_normalize_attribute, max_chars=dataset.brand_max_chars),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias("normalized_brand"),
            pl.col("product_color")
            .map_elements(
                partial(_normalize_attribute, max_chars=dataset.color_max_chars),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias("normalized_color"),
            pl.col("product_title").is_null().alias("title_missing"),
            pl.col("product_brand").is_null().alias("brand_missing"),
            pl.col("product_color").is_null().alias("color_missing"),
            pl.col("product_bullet_point").is_null().alias("bullets_missing"),
            pl.col("product_description").is_null().alias("description_missing"),
        )
        .with_columns(
            pl.any_horizontal(
                pl.col("clean_title") != "",
                pl.col("clean_brand") != "",
                pl.col("clean_color") != "",
                pl.col("clean_bullets") != "",
                pl.col("clean_description") != "",
            ).alias("has_usable_text")
        )
        .sort("product_locale", "product_id")
    )
    _sink(products, root / PRODUCTS_FILENAME)
    del products
    gc.collect()

    canonical_products = pl.scan_parquet(root / PRODUCTS_FILENAME)
    documents = (
        canonical_products.filter(pl.col("has_usable_text"))
        .with_columns(
            pl.concat_str(
                pl.lit("[TITLE] "),
                pl.col("clean_title"),
                pl.lit(" [BRAND] "),
                pl.col("clean_brand"),
                pl.lit(" [COLOR] "),
                pl.col("clean_color"),
                pl.lit(" [BULLETS] "),
                pl.col("clean_bullets"),
                pl.lit(" [DESCRIPTION] "),
                pl.col("clean_description"),
            ).alias("document")
        )
        .with_columns(
            pl.col("document")
            .map_elements(_document_sha256, return_dtype=pl.String)
            .alias("document_sha256"),
            pl.col("document").str.len_chars().cast(pl.UInt32).alias("document_chars"),
            pl.col("document").str.len_bytes().cast(pl.UInt32).alias("document_bytes"),
            pl.lit(dataset.product_document_version).alias("document_version"),
        )
        .select(
            pl.col("product_locale").alias("locale"),
            "product_id",
            "document",
            "document_sha256",
            "document_chars",
            "document_bytes",
            "document_version",
        )
        .sort("locale", "product_id")
    )
    _sink(documents, root / PRODUCT_DOCUMENTS_FILENAME)
    catalog_id = catalog_id_for_config(config)
    membership = (
        pl.scan_parquet(root / PRODUCT_DOCUMENTS_FILENAME)
        .select("locale", "product_id", "document_sha256")
        .with_row_index("catalog_ordinal")
        .with_columns(pl.lit(catalog_id).alias("catalog_id"))
        .select("catalog_id", "catalog_ordinal", "locale", "product_id", "document_sha256")
    )
    _sink(membership, root / CATALOG_MEMBERSHIP_FILENAME)
    exclusions = (
        canonical_products.filter(~pl.col("has_usable_text"))
        .select(
            pl.lit(catalog_id).alias("catalog_id"),
            pl.col("product_locale").alias("locale"),
            "product_id",
            pl.lit("no_usable_source_text").alias("reason"),
        )
        .sort("locale", "product_id")
    )
    _sink(exclusions, root / CATALOG_EXCLUSIONS_FILENAME)

    portfolio_pool = judgments.with_columns(
        pl.lit("portfolio").alias("profile"), pl.lit(JUDGED_POOL_ID).alias("pool_id")
    )
    development_pool = judgments.filter(pl.col("in_development")).with_columns(
        pl.lit("development").alias("profile"), pl.lit(JUDGED_POOL_ID).alias("pool_id")
    )
    pools = (
        pl.concat((development_pool, portfolio_pool))
        .select(
            "pool_id",
            "profile",
            "project_split",
            "query_id",
            "locale",
            "product_id",
            "esci_label",
            "label_id",
            "gain",
            "stable_ordinal",
        )
        .sort("profile", "project_split", "query_id", "stable_ordinal", "product_id")
    )
    _sink(pools, root / JUDGED_POOLS_FILENAME)
    return selection


_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    QUERIES_FILENAME: ("query_id",),
    SOURCES_FILENAME: ("query_id",),
    JUDGMENTS_FILENAME: ("query_id", "locale", "product_id"),
    PRODUCTS_FILENAME: ("product_locale", "product_id"),
    PRODUCT_DOCUMENTS_FILENAME: ("locale", "product_id"),
    CATALOG_MEMBERSHIP_FILENAME: ("catalog_id", "locale", "product_id"),
    CATALOG_EXCLUSIONS_FILENAME: ("catalog_id", "locale", "product_id"),
    JUDGED_POOLS_FILENAME: ("profile", "query_id", "locale", "product_id"),
}


def _build_manifest(
    root: Path,
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile_artifact: LoadedArtifact,
    catalog_selection: CatalogSelectionSummary,
) -> DataFoundationManifest:
    tables = tuple(
        _table_summary(root / filename, _PRIMARY_KEYS[filename]) for filename in _TABLE_ORDER
    )
    by_name = {table.filename: table for table in tables}
    candidate_products = by_name[PRODUCTS_FILENAME].row_count
    catalog_products = by_name[CATALOG_MEMBERSHIP_FILENAME].row_count
    excluded_products = by_name[CATALOG_EXCLUSIONS_FILENAME].row_count
    if candidate_products != catalog_selection.selected_candidate_products:
        raise FoundationInvariantError(
            "persisted catalog candidates differ from the selection audit"
        )
    if candidate_products != catalog_products + excluded_products:
        raise FoundationInvariantError(
            "catalog candidates do not partition into membership and no-text exclusions"
        )
    resource = _resource_estimate(root, config)
    if not resource.proceed:
        raise ResourceGateError(resource)
    pools = (
        _profile_pool_summary(root / JUDGED_POOLS_FILENAME, "development"),
        _profile_pool_summary(root / JUDGED_POOLS_FILENAME, "portfolio"),
    )
    queries = by_name[QUERIES_FILENAME].row_count
    judgments = by_name[JUDGMENTS_FILENAME].row_count
    required_judged_products = _collect_scalar(
        pl.scan_parquet(root / JUDGMENTS_FILENAME)
        .select("locale", "product_id")
        .unique()
        .select(pl.len().alias("value"))
    )
    if required_judged_products != catalog_selection.required_judged_products:
        raise FoundationInvariantError(
            "compact catalog does not retain every portfolio judged product"
        )
    if pools[1].query_ids != queries or pools[1].judgments != judgments:
        raise FoundationInvariantError("portfolio pool is not complete for canonical tables")
    checks = tuple(
        sorted(
            (
                FoundationCheck(
                    check_id="catalog_partition",
                    detail=(
                        f"{candidate_products} candidates partition into {catalog_products} "
                        f"documents and {excluded_products} exclusions"
                    ),
                ),
                FoundationCheck(
                    check_id="catalog_selection",
                    detail=(
                        f"{catalog_selection.required_judged_products} required judged products "
                        f"plus {catalog_selection.selected_distractor_products} deterministic "
                        f"distractors selected from {catalog_selection.source_products} products"
                    ),
                ),
                FoundationCheck(
                    check_id="official_gain_mapping",
                    detail="I/C/S/E map separately to IDs 0/1/2/3 and gains 0/0.01/0.1/1",
                ),
                FoundationCheck(
                    check_id="portfolio_pool_complete",
                    detail=f"{queries} queries and {judgments} judgments are retained",
                ),
                FoundationCheck(
                    check_id="primary_keys_unique",
                    detail="all eight canonical table primary keys are unique",
                ),
                FoundationCheck(
                    check_id="resource_gate",
                    detail=(
                        f"projected {resource.projected_runtime_bytes} bytes <= "
                        f"{resource.rss_limit_bytes}-byte limit"
                    ),
                ),
            ),
            key=lambda item: item.check_id,
        )
    )
    return DataFoundationManifest(
        dataset_version=release.manifest.dataset_version,
        source_revision=release.manifest.source_revision,
        release_manifest_sha256=release.sha256,
        config_sha256=config.sha256,
        profile_artifact_id=profile_artifact.manifest.artifact_id,
        profile_manifest_sha256=profile_artifact.manifest_sha256,
        catalog_id=catalog_id_for_config(config),
        catalog_selection=catalog_selection,
        catalog_candidate_products=candidate_products,
        catalog_products=catalog_products,
        catalog_excluded_no_text=excluded_products,
        label_mapping=(("I", 0, 0.0), ("C", 1, 0.01), ("S", 2, 0.1), ("E", 3, 1.0)),
        product_document_version=config.config.dataset.product_document_version,
        tables=tables,
        pools=pools,
        resource_estimate=resource,
        checks=checks,
    )


def load_foundation_manifest(path: Path) -> DataFoundationManifest:
    """Load one strict Goldfish 006 foundation manifest."""
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        return DataFoundationManifest.model_validate(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise FoundationManifestError(f"cannot load foundation manifest {path}: {exc}") from exc


def foundation_artifact_id(release: ResolvedReleaseManifest, config_sha256: str) -> str:
    """Return deterministic Goldfish 006 artifact coordinates."""
    return "/".join(
        (
            "data-foundation",
            release.manifest.dataset_version,
            "portfolio",
            FOUNDATION_COMPONENT_VERSION,
            config_sha256,
        )
    )


def _reuse_foundation(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile_artifact: LoadedArtifact,
    store: ArtifactStore,
) -> FoundationBuildResult:
    artifact = store.load(foundation_artifact_id(release, config.sha256))
    dependency = ArtifactDependency(
        artifact_id=profile_artifact.manifest.artifact_id,
        manifest_sha256=profile_artifact.manifest_sha256,
    )
    if artifact.manifest.dependencies != (dependency,):
        raise FoundationDependencyError("existing foundation has an incompatible profile parent")
    manifest = load_foundation_manifest(artifact.path / FOUNDATION_MANIFEST_FILENAME)
    if (
        manifest.release_manifest_sha256 != release.sha256
        or manifest.config_sha256 != config.sha256
        or manifest.profile_manifest_sha256 != profile_artifact.manifest_sha256
    ):
        raise FoundationDependencyError("existing foundation has incompatible lineage")
    return FoundationBuildResult(artifact=artifact, manifest=manifest, reused=True)


def build_esci_foundation(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    raw_root: Path | None = None,
    artifact_store: ArtifactStore | None = None,
) -> FoundationBuildResult:
    """Build or reuse the complete validated M2 data foundation."""
    selected_raw_root = raw_root or config.config.paths.data_dir / "raw" / "esci"
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    profile_artifact, _, assignments = _load_profile_dependency(release, config, store)
    raw_paths = _verify_raw_files(release, selected_raw_root)
    artifact_id = foundation_artifact_id(release, config.sha256)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse_foundation(release, config, profile_artifact, store)

    dependency = ArtifactDependency(
        artifact_id=profile_artifact.manifest.artifact_id,
        manifest_sha256=profile_artifact.manifest_sha256,
    )
    try:
        with store.stage(
            artifact_type="data-foundation",
            dataset_version=release.manifest.dataset_version,
            profile="portfolio",
            component_version=FOUNDATION_COMPONENT_VERSION,
            config_sha256=config.sha256,
            code_revision=code_revision,
            dependencies=(dependency,),
        ) as transaction:
            first_output = transaction.path(QUERIES_FILENAME)
            root = first_output.parent
            catalog_selection = _write_tables(root, raw_paths, assignments, config)
            manifest = _build_manifest(root, release, config, profile_artifact, catalog_selection)
            transaction.path(FOUNDATION_MANIFEST_FILENAME).write_text(
                _canonical_json(manifest), encoding="utf-8"
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse_foundation(release, config, profile_artifact, store)
    return FoundationBuildResult(artifact=artifact, manifest=manifest, reused=False)


__all__ = [
    "CATALOG_EXCLUSIONS_FILENAME",
    "CATALOG_MEMBERSHIP_FILENAME",
    "COMPACT_CATALOG_ID",
    "FOUNDATION_COMPONENT_VERSION",
    "FOUNDATION_MANIFEST_FILENAME",
    "FULL_CATALOG_ID",
    "JUDGED_POOLS_FILENAME",
    "JUDGMENTS_FILENAME",
    "PRODUCTS_FILENAME",
    "PRODUCT_DOCUMENTS_FILENAME",
    "QUERIES_FILENAME",
    "SOURCES_FILENAME",
    "CatalogId",
    "CatalogSelectionSummary",
    "DataFoundationError",
    "DataFoundationManifest",
    "FoundationBuildResult",
    "FoundationDependencyError",
    "FoundationInvariantError",
    "FoundationManifestError",
    "ResourceEstimate",
    "ResourceGateError",
    "build_esci_foundation",
    "catalog_id_for_config",
    "foundation_artifact_id",
    "load_foundation_manifest",
]
