"""Pinned ESCI raw-source manifests and memory-bounded structural validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from market_rank.artifacts import ArtifactExistsError, ArtifactStore, LoadedArtifact

OFFICIAL_REPOSITORY = "https://github.com/amazon-science/esci-data"
OFFICIAL_PAPER = "https://arxiv.org/abs/2206.06588"
RAW_SCHEMA_VERSION = "schema-v1"

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
GitRevision = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$"),
]
DatasetVersion = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^esci-[0-9a-f]{12}$"),
]
RawRole = Literal["examples", "products", "sources"]
RawFormat = Literal["parquet", "csv"]
SemanticType = Literal["integer", "string", "flag"]

_EXPECTED_FILES: dict[RawRole, tuple[str, RawFormat]] = {
    "examples": ("shopping_queries_dataset_examples.parquet", "parquet"),
    "products": ("shopping_queries_dataset_products.parquet", "parquet"),
    "sources": ("shopping_queries_dataset_sources.csv", "csv"),
}


class RawDataError(RuntimeError):
    """Base exception for raw dataset manifest and validation failures."""


class RawManifestError(RawDataError):
    """Raised when a release manifest cannot be read or validated."""


class RawDataValidationError(RawDataError):
    """Raised when an invalid report is used as a promoted stage output."""

    def __init__(self, report: RawValidationReport) -> None:
        super().__init__("raw ESCI validation failed; inspect the attached report")
        self.report = report


class _StrictModel(BaseModel):
    """Shared immutable, unknown-key-rejecting data-contract behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


class RawFileSource(_StrictModel):
    """Pinned expected identity for one official ESCI source file."""

    role: RawRole
    filename: str = Field(strict=True, min_length=1)
    format: RawFormat
    source_url: HttpUrl
    size_bytes: int = Field(strict=True, gt=0)
    sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_role_contract(self) -> Self:
        expected_filename, expected_format = _EXPECTED_FILES[self.role]
        if self.filename != expected_filename or self.format != expected_format:
            raise ValueError(
                f"{self.role} must use {expected_filename!r} with format {expected_format!r}"
            )
        return self


class EsciReleaseManifest(_StrictModel):
    """Versioned official source, license, and file-integrity contract."""

    schema_version: Literal[1] = 1
    dataset_id: Literal["amazon-esci-shopping-queries"] = "amazon-esci-shopping-queries"
    dataset_version: DatasetVersion
    source_repository: HttpUrl
    source_revision: GitRevision
    source_commit_utc: datetime
    license_spdx: Literal["Apache-2.0"] = "Apache-2.0"
    license_url: HttpUrl
    paper_url: HttpUrl
    files: tuple[RawFileSource, ...] = Field(min_length=3, max_length=3)

    @field_validator("source_commit_utc")
    @classmethod
    def validate_source_commit_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, "source_commit_utc")

    @model_validator(mode="after")
    def validate_release_identity(self) -> Self:
        expected_version = f"esci-{self.source_revision[:12]}"
        if self.dataset_version != expected_version:
            raise ValueError(f"dataset_version must equal {expected_version!r}")
        if str(self.source_repository).rstrip("/") != OFFICIAL_REPOSITORY:
            raise ValueError("source_repository must be the official Amazon Science repository")
        if str(self.paper_url).rstrip("/") != OFFICIAL_PAPER:
            raise ValueError("paper_url must identify the Shopping Queries Dataset paper")

        roles = tuple(source.role for source in self.files)
        if roles != tuple(sorted(_EXPECTED_FILES)):
            raise ValueError("files must contain examples, products, and sources in role order")

        expected_license_url = f"{OFFICIAL_REPOSITORY}/blob/{self.source_revision}/LICENSE"
        if str(self.license_url) != expected_license_url:
            raise ValueError("license_url must pin the official LICENSE at source_revision")
        for source in self.files:
            expected_url = (
                f"{OFFICIAL_REPOSITORY}/raw/{self.source_revision}/"
                f"shopping_queries_dataset/{source.filename}"
            )
            if str(source.source_url) != expected_url:
                raise ValueError(f"source_url for {source.role} must pin source_revision")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedReleaseManifest:
    """Strict release manifest plus canonical semantic identity."""

    manifest: EsciReleaseManifest
    canonical_json: str
    sha256: str
    source_path: Path


@dataclass(frozen=True, slots=True)
class RawValidationPublication:
    """A verified validation artifact and whether it predated this call."""

    artifact: LoadedArtifact
    reused: bool


class ValidationCheck(_StrictModel):
    """One deterministic validation assertion and its bounded diagnostic."""

    check_id: str = Field(strict=True, min_length=1)
    passed: bool = Field(strict=True)
    detail: str = Field(strict=True, min_length=1)


class ObservedColumn(_StrictModel):
    """Physical source column observed by Polars."""

    name: str = Field(strict=True, min_length=1)
    dtype: str = Field(strict=True, min_length=1)


class RawFileValidation(_StrictModel):
    """Integrity and schema observations for one source file."""

    role: RawRole
    filename: str
    size_bytes: int | None
    sha256: Sha256Digest | None
    row_count: int | None = Field(default=None, ge=0)
    columns: tuple[ObservedColumn, ...] = ()
    checks: tuple[ValidationCheck, ...] = Field(min_length=1)
    valid: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.valid != all(check.passed for check in self.checks):
            raise ValueError("file valid status must equal the conjunction of its checks")
        return self


class RawValidationReport(_StrictModel):
    """Deterministic raw-data validation evidence for promotion decisions."""

    schema_version: Literal[1] = 1
    dataset_id: Literal["amazon-esci-shopping-queries"] = "amazon-esci-shopping-queries"
    dataset_version: DatasetVersion
    source_revision: GitRevision
    release_manifest_sha256: Sha256Digest
    retrieved_utc: datetime
    files: tuple[RawFileValidation, ...] = Field(min_length=3, max_length=3)
    dataset_checks: tuple[ValidationCheck, ...] = Field(min_length=1)
    valid: bool = Field(strict=True)

    @field_validator("retrieved_utc")
    @classmethod
    def validate_retrieved_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, "retrieved_utc")

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        roles = tuple(file.role for file in self.files)
        if roles != tuple(sorted(_EXPECTED_FILES)):
            raise ValueError("files must contain examples, products, and sources in role order")
        expected = all(file.valid for file in self.files) and all(
            check.passed for check in self.dataset_checks
        )
        if self.valid != expected:
            raise ValueError("report valid status must equal every file and dataset check")
        return self

    def require_valid(self) -> None:
        """Fail before downstream use while preserving the complete report."""
        if not self.valid:
            raise RawDataValidationError(self)


@dataclass(frozen=True, slots=True)
class _SchemaSpec:
    columns: tuple[str, ...]
    semantic_types: tuple[tuple[str, SemanticType], ...]
    required_non_null: tuple[str, ...]
    primary_key: tuple[str, ...]
    domains: tuple[tuple[str, tuple[str, ...]], ...] = ()
    flags: tuple[str, ...] = ()
    optional_ignored_columns: tuple[str, ...] = ()


_SCHEMAS: dict[RawRole, _SchemaSpec] = {
    "examples": _SchemaSpec(
        columns=(
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "large_version",
            "split",
        ),
        semantic_types=(
            ("example_id", "integer"),
            ("query", "string"),
            ("query_id", "integer"),
            ("product_id", "string"),
            ("product_locale", "string"),
            ("esci_label", "string"),
            ("small_version", "flag"),
            ("large_version", "flag"),
            ("split", "string"),
        ),
        required_non_null=(
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "large_version",
            "split",
        ),
        primary_key=("example_id",),
        domains=(
            ("product_locale", ("us", "es", "jp")),
            ("esci_label", ("E", "S", "C", "I")),
            ("split", ("train", "test")),
        ),
        flags=("small_version", "large_version"),
        optional_ignored_columns=("__index_level_0__",),
    ),
    "products": _SchemaSpec(
        columns=(
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_locale",
        ),
        semantic_types=(
            ("product_id", "string"),
            ("product_title", "string"),
            ("product_description", "string"),
            ("product_bullet_point", "string"),
            ("product_brand", "string"),
            ("product_color", "string"),
            ("product_locale", "string"),
        ),
        required_non_null=("product_id", "product_locale"),
        primary_key=("product_locale", "product_id"),
        domains=(("product_locale", ("us", "es", "jp")),),
    ),
    "sources": _SchemaSpec(
        columns=("query_id", "source"),
        semantic_types=(("query_id", "integer"), ("source", "string")),
        required_non_null=("query_id", "source"),
        primary_key=("query_id",),
    ),
}


class _DuplicateJsonKeyError(ValueError):
    """Internal signal for duplicate release-manifest keys."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        document[key] = value
    return document


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def load_release_manifest(path: Path) -> ResolvedReleaseManifest:
    """Read, strictly validate, canonicalize, and hash a pinned release manifest."""
    try:
        raw_bytes = path.read_bytes()
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise RawManifestError(f"cannot read strict release manifest {path}: {exc}") from exc

    try:
        manifest = EsciReleaseManifest.model_validate(document)
    except ValidationError as exc:
        raise RawManifestError(f"invalid ESCI release manifest {path}: {exc}") from exc

    canonical_json = _canonical_json(manifest)
    return ResolvedReleaseManifest(
        manifest=manifest,
        canonical_json=canonical_json,
        sha256=sha256(canonical_json.encode("utf-8")).hexdigest(),
        source_path=path.resolve(strict=False),
    )


def load_validation_report(path: Path) -> RawValidationReport:
    """Read a strict persisted raw-validation report."""
    try:
        raw_bytes = path.read_bytes()
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise RawManifestError(f"cannot read strict validation report {path}: {exc}") from exc

    try:
        return RawValidationReport.model_validate(document)
    except ValidationError as exc:
        raise RawManifestError(f"invalid raw-validation report {path}: {exc}") from exc


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()


def _check(check_id: str, passed: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(check_id=check_id, passed=passed, detail=detail)


def _type_matches(dtype: pl.DataType, semantic_type: SemanticType) -> bool:
    if semantic_type == "integer":
        return dtype.is_integer()
    if semantic_type == "string":
        return dtype == pl.String
    return dtype == pl.Boolean or dtype.is_integer()


def _collect_scalar(query: pl.LazyFrame) -> int:
    return int(query.collect(engine="streaming").item())


def _scan_file(source: RawFileSource, path: Path) -> pl.LazyFrame:
    if source.format == "parquet":
        return pl.scan_parquet(path)
    return pl.scan_csv(path, infer_schema_length=10_000, try_parse_dates=False)


def _validate_file(
    source: RawFileSource,
    raw_root: Path,
) -> tuple[RawFileValidation, pl.LazyFrame | None]:
    checks: list[ValidationCheck] = []
    observed_size: int | None = None
    observed_sha256: str | None = None
    row_count: int | None = None
    columns: tuple[ObservedColumn, ...] = ()

    candidate = raw_root / source.filename
    try:
        if candidate.is_symlink():
            raise OSError("symbolic links are not allowed")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(raw_root)
        if not resolved.is_file():
            raise OSError("source path is not a regular file")
    except (OSError, ValueError) as exc:
        checks.append(_check("file_access", False, f"unavailable: {exc}"))
        return (
            RawFileValidation(
                role=source.role,
                filename=source.filename,
                size_bytes=None,
                sha256=None,
                checks=tuple(checks),
                valid=False,
            ),
            None,
        )

    checks.append(_check("file_access", True, "regular file below the raw root"))
    try:
        observed_size, observed_sha256 = _sha256_file(resolved)
    except OSError as exc:
        checks.append(_check("file_read", False, f"cannot read source bytes: {exc}"))
        return (
            RawFileValidation(
                role=source.role,
                filename=source.filename,
                size_bytes=None,
                sha256=None,
                checks=tuple(checks),
                valid=False,
            ),
            None,
        )
    size_matches = observed_size == source.size_bytes
    checksum_matches = observed_sha256 == source.sha256
    checks.append(
        _check(
            "size_bytes",
            size_matches,
            f"expected {source.size_bytes}, observed {observed_size}",
        )
    )
    checks.append(
        _check(
            "sha256",
            checksum_matches,
            f"expected {source.sha256}, observed {observed_sha256}",
        )
    )
    if not size_matches or not checksum_matches:
        checks.append(_check("schema_readable", False, "skipped because integrity failed"))
        return (
            RawFileValidation(
                role=source.role,
                filename=source.filename,
                size_bytes=observed_size,
                sha256=observed_sha256,
                checks=tuple(checks),
                valid=False,
            ),
            None,
        )

    try:
        lazy_frame = _scan_file(source, resolved)
        observed_schema = lazy_frame.collect_schema()
        columns = tuple(
            ObservedColumn(name=name, dtype=str(observed_schema[name]))
            for name in observed_schema.names()
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        checks.append(_check("schema_readable", False, f"cannot scan source: {exc}"))
        return (
            RawFileValidation(
                role=source.role,
                filename=source.filename,
                size_bytes=observed_size,
                sha256=observed_sha256,
                columns=columns,
                checks=tuple(checks),
                valid=False,
            ),
            None,
        )

    checks.append(_check("schema_readable", True, "source metadata is readable"))
    spec = _SCHEMAS[source.role]
    observed_names = tuple(column.name for column in columns)
    permitted_columns = (spec.columns, spec.columns + spec.optional_ignored_columns)
    columns_match = observed_names in permitted_columns
    checks.append(
        _check(
            "exact_columns",
            columns_match,
            (
                f"expected {spec.columns} with optional ignored "
                f"{spec.optional_ignored_columns}, observed {observed_names}"
            ),
        )
    )
    if not columns_match:
        return (
            RawFileValidation(
                role=source.role,
                filename=source.filename,
                size_bytes=observed_size,
                sha256=observed_sha256,
                columns=columns,
                checks=tuple(checks),
                valid=False,
            ),
            None,
        )
    lazy_frame = lazy_frame.select(spec.columns)

    type_failures = [
        f"{name}={observed_schema[name]}"
        for name, semantic_type in spec.semantic_types
        if not _type_matches(observed_schema[name], semantic_type)
    ]
    checks.append(
        _check(
            "semantic_types",
            not type_failures,
            "all compatible" if not type_failures else f"incompatible: {type_failures}",
        )
    )

    try:
        row_count = _collect_scalar(lazy_frame.select(pl.len()))
        checks.append(_check("non_empty", row_count > 0, f"observed {row_count} rows"))

        null_counts = lazy_frame.select(
            *(pl.col(name).null_count().alias(name) for name in spec.required_non_null)
        ).collect(engine="streaming")
        null_failures = {
            name: int(null_counts.item(0, name))
            for name in spec.required_non_null
            if int(null_counts.item(0, name)) > 0
        }
        checks.append(
            _check(
                "required_non_null",
                not null_failures,
                "no required nulls" if not null_failures else f"null counts: {null_failures}",
            )
        )

        for column_name, allowed_values in spec.domains:
            invalid_count = _collect_scalar(
                lazy_frame.filter(
                    pl.col(column_name).is_null() | ~pl.col(column_name).is_in(allowed_values)
                ).select(pl.len())
            )
            checks.append(
                _check(
                    f"domain:{column_name}",
                    invalid_count == 0,
                    f"{invalid_count} values outside {allowed_values}",
                )
            )

        for column_name in spec.flags:
            dtype = observed_schema[column_name]
            allowed_flags: Sequence[bool | int]
            allowed_flags = (False, True) if dtype == pl.Boolean else (0, 1)
            invalid_count = _collect_scalar(
                lazy_frame.filter(
                    pl.col(column_name).is_null() | ~pl.col(column_name).is_in(allowed_flags)
                ).select(pl.len())
            )
            checks.append(
                _check(
                    f"binary_flag:{column_name}",
                    invalid_count == 0,
                    f"{invalid_count} values outside 0/1",
                )
            )

        unique_keys = _collect_scalar(
            lazy_frame.select(pl.struct(spec.primary_key).n_unique().alias("unique_keys"))
        )
        duplicate_keys = row_count - unique_keys
        checks.append(
            _check(
                "primary_key_unique",
                duplicate_keys == 0,
                f"observed {duplicate_keys} duplicate key rows",
            )
        )

        if source.role == "examples":
            inconsistent_queries = _collect_scalar(
                lazy_frame.group_by("query_id")
                .agg(pl.col("query").n_unique().alias("query_values"))
                .filter(pl.col("query_values") != 1)
                .select(pl.len())
            )
            checks.append(
                _check(
                    "query_consistency",
                    inconsistent_queries == 0,
                    f"observed {inconsistent_queries} inconsistent query IDs",
                )
            )
    except (OSError, pl.exceptions.PolarsError) as exc:
        checks.append(_check("data_scan", False, f"scan failed: {exc}"))
        return (
            RawFileValidation(
                role=source.role,
                filename=source.filename,
                size_bytes=observed_size,
                sha256=observed_sha256,
                row_count=row_count,
                columns=columns,
                checks=tuple(checks),
                valid=False,
            ),
            None,
        )

    valid = all(check.passed for check in checks)
    return (
        RawFileValidation(
            role=source.role,
            filename=source.filename,
            size_bytes=observed_size,
            sha256=observed_sha256,
            row_count=row_count,
            columns=columns,
            checks=tuple(checks),
            valid=valid,
        ),
        lazy_frame if valid else None,
    )


def _cross_file_checks(frames: dict[RawRole, pl.LazyFrame]) -> tuple[ValidationCheck, ...]:
    if set(frames) != set(_EXPECTED_FILES):
        return (_check("cross_files_ready", False, "one or more source files are invalid"),)

    examples = frames["examples"]
    products = frames["products"]
    sources = frames["sources"]
    try:
        missing_products = _collect_scalar(
            examples.select("product_locale", "product_id")
            .unique()
            .join(
                products.select("product_locale", "product_id").unique(),
                on=("product_locale", "product_id"),
                how="anti",
            )
            .select(pl.len())
        )
        missing_sources = _collect_scalar(
            examples.select("query_id")
            .unique()
            .join(sources.select("query_id").unique(), on="query_id", how="anti")
            .select(pl.len())
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        return (_check("cross_file_joins", False, f"join validation failed: {exc}"),)

    return (
        _check(
            "examples_join_products",
            missing_products == 0,
            f"observed {missing_products} product keys without product rows",
        ),
        _check(
            "examples_join_sources",
            missing_sources == 0,
            f"observed {missing_sources} query IDs without source rows",
        ),
    )


def validate_raw_dataset(
    release: ResolvedReleaseManifest,
    raw_root: Path,
    *,
    retrieved_utc: datetime,
) -> RawValidationReport:
    """Validate pinned raw files without loading the full dataset into Python memory."""
    try:
        _require_utc(retrieved_utc, "retrieved_utc")
    except ValueError as exc:
        raise RawDataError(str(exc)) from exc

    expected_canonical_json = _canonical_json(release.manifest)
    expected_manifest_sha256 = sha256(expected_canonical_json.encode("utf-8")).hexdigest()
    if (
        release.canonical_json != expected_canonical_json
        or release.sha256 != expected_manifest_sha256
    ):
        raise RawDataError("resolved release manifest identity is inconsistent")

    root = raw_root.resolve(strict=False)
    file_reports: list[RawFileValidation] = []
    frames: dict[RawRole, pl.LazyFrame] = {}
    for source in release.manifest.files:
        file_report, frame = _validate_file(source, root)
        file_reports.append(file_report)
        if frame is not None:
            frames[source.role] = frame

    dataset_checks = _cross_file_checks(frames)
    valid = all(report.valid for report in file_reports) and all(
        check.passed for check in dataset_checks
    )
    return RawValidationReport(
        dataset_version=release.manifest.dataset_version,
        source_revision=release.manifest.source_revision,
        release_manifest_sha256=release.sha256,
        retrieved_utc=retrieved_utc,
        files=tuple(file_reports),
        dataset_checks=dataset_checks,
        valid=valid,
    )


def raw_validation_artifact_id(
    release: ResolvedReleaseManifest,
    config_sha256: str,
) -> str:
    """Return the deterministic Goldfish 003 identity for raw validation."""
    return (
        f"raw-validation/{release.manifest.dataset_version}/all-locales/"
        f"{RAW_SCHEMA_VERSION}/{config_sha256}"
    )


def _validate_report_release_identity(
    release: ResolvedReleaseManifest,
    report: RawValidationReport,
) -> None:
    report.require_valid()
    if (
        report.release_manifest_sha256 != release.sha256
        or report.dataset_version != release.manifest.dataset_version
        or report.source_revision != release.manifest.source_revision
    ):
        raise RawDataError("report does not belong to the supplied release manifest")


def _reuse_validation_artifact(
    release: ResolvedReleaseManifest,
    report: RawValidationReport,
    store: ArtifactStore,
    artifact_id: str,
) -> RawValidationPublication:
    artifact = store.load(artifact_id)
    release_path = artifact.path / "release-manifest.json"
    report_path = artifact.path / "validation-report.json"
    try:
        existing_release_json = release_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RawDataError(f"cannot read existing validation artifact: {exc}") from exc
    if existing_release_json != release.canonical_json:
        raise RawDataError("existing validation artifact pins different release metadata")

    existing_report = load_validation_report(report_path)
    existing_report.require_valid()
    current_semantics = report.model_dump(mode="json", exclude={"retrieved_utc"})
    existing_semantics = existing_report.model_dump(mode="json", exclude={"retrieved_utc"})
    if existing_semantics != current_semantics:
        raise RawDataError("existing validation artifact is incompatible with current validation")
    return RawValidationPublication(artifact=artifact, reused=True)


def ensure_raw_validation_artifact(
    release: ResolvedReleaseManifest,
    report: RawValidationReport,
    store: ArtifactStore,
    *,
    config_sha256: str,
    code_revision: str,
) -> RawValidationPublication:
    """Publish valid evidence or safely reuse its compatible immutable artifact."""
    _validate_report_release_identity(release, report)
    artifact_id = raw_validation_artifact_id(release, config_sha256)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse_validation_artifact(release, report, store, artifact_id)

    try:
        with store.stage(
            artifact_type="raw-validation",
            dataset_version=release.manifest.dataset_version,
            profile="all-locales",
            component_version=RAW_SCHEMA_VERSION,
            config_sha256=config_sha256,
            code_revision=code_revision,
        ) as transaction:
            transaction.path("release-manifest.json").write_text(
                release.canonical_json,
                encoding="utf-8",
            )
            transaction.path("validation-report.json").write_text(
                _canonical_json(report),
                encoding="utf-8",
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse_validation_artifact(release, report, store, artifact_id)
    return RawValidationPublication(artifact=artifact, reused=False)


def publish_raw_validation(
    release: ResolvedReleaseManifest,
    report: RawValidationReport,
    store: ArtifactStore,
    *,
    config_sha256: str,
    code_revision: str,
) -> LoadedArtifact:
    """Compatibility wrapper returning the published or reused artifact."""
    return ensure_raw_validation_artifact(
        release,
        report,
        store,
        config_sha256=config_sha256,
        code_revision=code_revision,
    ).artifact


__all__ = [
    "OFFICIAL_PAPER",
    "OFFICIAL_REPOSITORY",
    "RAW_SCHEMA_VERSION",
    "EsciReleaseManifest",
    "RawDataError",
    "RawDataValidationError",
    "RawFileSource",
    "RawManifestError",
    "RawValidationPublication",
    "RawValidationReport",
    "ResolvedReleaseManifest",
    "ensure_raw_validation_artifact",
    "load_release_manifest",
    "load_validation_report",
    "publish_raw_validation",
    "raw_validation_artifact_id",
    "validate_raw_dataset",
]
