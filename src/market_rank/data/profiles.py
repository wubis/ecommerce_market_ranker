"""Leakage-safe Task-1 US cohort splits and deterministic nested profiles."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
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
from market_rank.data.esci_raw import (
    ResolvedReleaseManifest,
    load_validation_report,
    raw_validation_artifact_id,
)

SPLIT_PROFILE_COMPONENT_VERSION = "split-profile-v1"
ASSIGNMENTS_FILENAME = "query-assignments.parquet"
PROFILE_MANIFEST_FILENAME = "profile-manifest.json"
TASK1_LOCALE = "us"
TASK1_MEMBERSHIP_COLUMN = "small_version"
TASK1_MEMBERSHIP_VALUE = 1

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
ProjectSplit = Literal["train", "validation", "test", "quarantine"]
ProfileName = Literal["development", "portfolio"]


class ProfileBuildError(RuntimeError):
    """Base exception for cohort, split, profile, and publication failures."""


class RawValidationDependencyError(ProfileBuildError):
    """Raised when the required validated-raw parent is absent or incompatible."""


class CohortInvariantError(ProfileBuildError):
    """Raised when source groups cannot satisfy the split/profile contract."""


class ProfileManifestError(ProfileBuildError):
    """Raised when persisted profile metadata cannot be read strictly."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitSummary(_StrictModel):
    """Counts for one project split within a cohort or profile."""

    project_split: ProjectSplit
    normalized_query_groups: int = Field(strict=True, ge=0)
    query_ids: int = Field(strict=True, ge=0)
    judgments: int = Field(strict=True, ge=0)


class ProfileSummary(_StrictModel):
    """Identity and observed size of one nested complete-query profile."""

    profile: ProfileName
    target_normalized_query_groups: int = Field(strict=True, ge=1)
    selected_normalized_query_groups: int = Field(strict=True, ge=0)
    query_ids: int = Field(strict=True, ge=0)
    judgments: int = Field(strict=True, ge=0)
    query_ids_sha256: Sha256Digest
    normalized_query_groups_sha256: Sha256Digest
    splits: tuple[SplitSummary, ...]


class CohortCheck(_StrictModel):
    """A compact hard-invariant result emitted with the profile artifact."""

    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class EsciProfileManifest(_StrictModel):
    """Complete lineage and audit summary for Task-1 cohort assignments."""

    schema_version: Literal[1] = 1
    dataset_version: str = Field(strict=True, min_length=1)
    source_revision: str = Field(strict=True, min_length=1)
    release_manifest_sha256: Sha256Digest
    raw_validation_artifact_id: str = Field(strict=True, min_length=1)
    raw_validation_manifest_sha256: Sha256Digest
    config_sha256: Sha256Digest
    predicate: Literal["product_locale == 'us' and small_version == 1"] = (
        "product_locale == 'us' and small_version == 1"
    )
    query_normalization_version: Literal["nfkc-casefold-ws-v1"]
    split_version: Literal["normalized-query-sha256-v1"]
    profile_version: Literal["nested-query-sha256-v1"]
    seed: int = Field(strict=True, ge=0)
    train_basis_points: int = Field(strict=True, ge=1, le=9999)
    task1_us_judgments: int = Field(strict=True, ge=1)
    task1_us_query_ids: int = Field(strict=True, ge=1)
    task1_us_normalized_query_groups: int = Field(strict=True, ge=1)
    eligible_normalized_query_groups: int = Field(strict=True, ge=1)
    quarantined_train_groups: int = Field(strict=True, ge=0)
    quarantined_train_query_ids: int = Field(strict=True, ge=0)
    quarantined_train_judgments: int = Field(strict=True, ge=0)
    assignment_sha256: Sha256Digest
    splits: tuple[SplitSummary, ...]
    profiles: tuple[ProfileSummary, ...]
    checks: tuple[CohortCheck, ...]

    @model_validator(mode="after")
    def validate_collections(self) -> Self:
        if tuple(item.project_split for item in self.splits) != (
            "train",
            "validation",
            "test",
            "quarantine",
        ):
            raise ValueError("splits must be ordered train, validation, test, quarantine")
        if tuple(item.profile for item in self.profiles) != ("development", "portfolio"):
            raise ValueError("profiles must be ordered development, portfolio")
        check_ids = tuple(check.check_id for check in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(check_ids) != len(set(check_ids)):
            raise ValueError("checks must have unique sorted check IDs")
        return self


@dataclass(frozen=True, slots=True)
class ProfileBuildResult:
    """Published or reused cohort assignments and their validated manifest."""

    artifact: LoadedArtifact
    manifest: EsciProfileManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class _QueryRecord:
    query_id: int
    query: str
    official_split: Literal["train", "test"]
    judgment_count: int
    normalized_query: str
    normalized_query_sha256: str


@dataclass(frozen=True, slots=True)
class _Assignment:
    query_id: int
    official_split: Literal["train", "test"]
    project_split: ProjectSplit
    normalized_query_sha256: str
    judgment_count: int
    in_development: bool
    in_portfolio: bool
    quarantine_reason: str | None


def _canonical_json(value: BaseModel | list[dict[str, Any]]) -> str:
    document = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def normalize_query_group(query: str) -> str:
    """Return the versioned, label-blind identity used only for leakage grouping."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(normalized.split())


def _hash_parts(*parts: object) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return sha256(encoded).hexdigest()


def _hash_values(values: Iterable[object]) -> str:
    return sha256(
        json.dumps(
            list(values),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size_bytes = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
    except OSError as exc:
        raise ProfileBuildError(f"cannot verify raw source {path}: {exc}") from exc
    return size_bytes, digest.hexdigest()


def _verify_examples_source(release: ResolvedReleaseManifest, raw_root: Path) -> Path:
    if raw_root.is_symlink():
        raise ProfileBuildError(f"raw root cannot be a symbolic link: {raw_root}")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise ProfileBuildError(f"raw root is unavailable: {raw_root}: {exc}") from exc
    source = next(item for item in release.manifest.files if item.role == "examples")
    path = root / source.filename
    if path.is_symlink() or not path.is_file():
        raise ProfileBuildError(f"raw examples source is not a regular file: {path}")
    size_bytes, file_sha256 = _sha256_file(path)
    if size_bytes != source.size_bytes or file_sha256 != source.sha256:
        raise ProfileBuildError(
            "raw examples source no longer matches the validated pinned release; "
            "rerun data download-esci after inspecting the file"
        )
    return path


def _verify_raw_validation_dependency(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    store: ArtifactStore,
) -> LoadedArtifact:
    artifact_id = raw_validation_artifact_id(release, config.sha256)
    try:
        artifact = store.load(artifact_id)
        release_json = (artifact.path / "release-manifest.json").read_text(encoding="utf-8")
        report = load_validation_report(artifact.path / "validation-report.json")
    except (OSError, UnicodeDecodeError, RuntimeError) as exc:
        raise RawValidationDependencyError(
            "a compatible validated-raw artifact is required; run "
            "`market-rank data download-esci` first"
        ) from exc
    report.require_valid()
    if release_json != release.canonical_json or (
        report.dataset_version != release.manifest.dataset_version
        or report.source_revision != release.manifest.source_revision
        or report.release_manifest_sha256 != release.sha256
    ):
        raise RawValidationDependencyError(
            "the validated-raw artifact is incompatible with the selected release"
        )
    return artifact


def _collect_query_records(examples_path: Path) -> tuple[_QueryRecord, ...]:
    try:
        grouped = (
            pl.scan_parquet(examples_path)
            .filter(
                (pl.col("product_locale") == TASK1_LOCALE)
                & (pl.col(TASK1_MEMBERSHIP_COLUMN) == TASK1_MEMBERSHIP_VALUE)
            )
            .group_by("query_id")
            .agg(
                pl.col("query").first().alias("query"),
                pl.col("query").n_unique().alias("query_variants"),
                pl.col("split").first().alias("official_split"),
                pl.col("split").n_unique().alias("split_variants"),
                pl.len().alias("judgment_count"),
            )
            .sort("query_id")
            .collect(engine="streaming")
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ProfileBuildError(f"cannot scan Task-1 US examples: {exc}") from exc
    if grouped.height == 0:
        raise CohortInvariantError("Task-1 US predicate selected no query groups")

    records: list[_QueryRecord] = []
    for row in grouped.iter_rows(named=True):
        if row["query_variants"] != 1 or row["split_variants"] != 1:
            raise CohortInvariantError(
                f"query_id {row['query_id']} has inconsistent query text or official split"
            )
        query = row["query"]
        official_split = row["official_split"]
        if not isinstance(query, str) or official_split not in {"train", "test"}:
            raise CohortInvariantError(f"query_id {row['query_id']} has invalid source values")
        normalized_query = normalize_query_group(query)
        if not normalized_query:
            raise CohortInvariantError(
                f"query_id {row['query_id']} normalizes to an empty group key"
            )
        records.append(
            _QueryRecord(
                query_id=int(row["query_id"]),
                query=query,
                official_split=official_split,
                judgment_count=int(row["judgment_count"]),
                normalized_query=normalized_query,
                normalized_query_sha256=_hash_parts(TASK1_LOCALE, normalized_query),
            )
        )
    return tuple(records)


def _choose_project_split(record: _QueryRecord, config: ResolvedConfig) -> ProjectSplit:
    digest = _hash_parts(
        config.config.dataset.split_version,
        config.config.runtime.seed,
        TASK1_LOCALE,
        record.normalized_query,
    )
    bucket = int(digest[:16], 16) % 10_000
    return "train" if bucket < config.config.dataset.train_basis_points else "validation"


def _select_profile_groups(
    group_keys: set[str],
    config: ResolvedConfig,
) -> tuple[set[str], set[str]]:
    ranked = sorted(
        group_keys,
        key=lambda key: (
            _hash_parts(
                config.config.dataset.profile_version,
                config.config.runtime.seed,
                key,
            ),
            key,
        ),
    )
    development_count = min(config.config.dataset.development_query_groups, len(ranked))
    portfolio_count = min(config.config.dataset.portfolio_query_groups, len(ranked))
    return set(ranked[:development_count]), set(ranked[:portfolio_count])


def _build_assignments(
    records: tuple[_QueryRecord, ...],
    config: ResolvedConfig,
) -> tuple[tuple[_Assignment, ...], int]:
    official_splits_by_group: dict[str, set[str]] = {}
    for record in records:
        official_splits_by_group.setdefault(record.normalized_query_sha256, set()).add(
            record.official_split
        )
    collision_groups = {
        key for key, splits in official_splits_by_group.items() if splits == {"train", "test"}
    }
    eligible_groups = set(official_splits_by_group)
    development_groups, portfolio_groups = _select_profile_groups(eligible_groups, config)

    assignments: list[_Assignment] = []
    for record in records:
        if record.official_split == "test":
            project_split: ProjectSplit = "test"
            quarantine_reason = None
        elif record.normalized_query_sha256 in collision_groups:
            project_split = "quarantine"
            quarantine_reason = "normalized_query_occurs_in_official_test"
        else:
            project_split = _choose_project_split(record, config)
            quarantine_reason = None
        included = project_split != "quarantine"
        assignments.append(
            _Assignment(
                query_id=record.query_id,
                official_split=record.official_split,
                project_split=project_split,
                normalized_query_sha256=record.normalized_query_sha256,
                judgment_count=record.judgment_count,
                in_development=(included and record.normalized_query_sha256 in development_groups),
                in_portfolio=included and record.normalized_query_sha256 in portfolio_groups,
                quarantine_reason=quarantine_reason,
            )
        )
    return tuple(assignments), len(collision_groups)


_SPLIT_ORDER: tuple[ProjectSplit, ...] = ("train", "validation", "test", "quarantine")


def _summarize_splits(assignments: Iterable[_Assignment]) -> tuple[SplitSummary, ...]:
    selected = tuple(assignments)
    return tuple(
        SplitSummary(
            project_split=project_split,
            normalized_query_groups=len(
                {
                    item.normalized_query_sha256
                    for item in selected
                    if item.project_split == project_split
                }
            ),
            query_ids=sum(item.project_split == project_split for item in selected),
            judgments=sum(
                item.judgment_count for item in selected if item.project_split == project_split
            ),
        )
        for project_split in _SPLIT_ORDER
    )


def _summarize_profile(
    name: ProfileName,
    assignments: tuple[_Assignment, ...],
    target: int,
) -> ProfileSummary:
    selected = tuple(
        item
        for item in assignments
        if (item.in_development if name == "development" else item.in_portfolio)
    )
    group_ids = sorted({item.normalized_query_sha256 for item in selected})
    query_ids = sorted(item.query_id for item in selected)
    return ProfileSummary(
        profile=name,
        target_normalized_query_groups=target,
        selected_normalized_query_groups=len(group_ids),
        query_ids=len(query_ids),
        judgments=sum(item.judgment_count for item in selected),
        query_ids_sha256=_hash_values(query_ids),
        normalized_query_groups_sha256=_hash_values(group_ids),
        splits=_summarize_splits(selected)[:3],
    )


def _validate_assignments(assignments: tuple[_Assignment, ...]) -> tuple[CohortCheck, ...]:
    if len({item.query_id for item in assignments}) != len(assignments):
        raise CohortInvariantError("query assignments contain duplicate query IDs")
    project_splits_by_group: dict[str, set[ProjectSplit]] = {}
    for item in assignments:
        if item.project_split != "quarantine":
            project_splits_by_group.setdefault(item.normalized_query_sha256, set()).add(
                item.project_split
            )
        if item.in_development and not item.in_portfolio:
            raise CohortInvariantError("development profile is not nested in portfolio")
        if item.project_split == "quarantine" and (item.in_development or item.in_portfolio):
            raise CohortInvariantError("quarantined query entered a profile")
    leakage_groups = sum(len(splits) > 1 for splits in project_splits_by_group.values())
    if leakage_groups:
        raise CohortInvariantError(f"{leakage_groups} normalized-query groups cross project splits")
    return (
        CohortCheck(
            check_id="complete_query_ids",
            detail=f"all {len(assignments)} query IDs have exactly one assignment",
        ),
        CohortCheck(
            check_id="development_nested_in_portfolio",
            detail="every development query ID is also in portfolio",
        ),
        CohortCheck(
            check_id="normalized_query_split_disjointness",
            detail="zero eligible normalized-query groups cross project splits",
        ),
        CohortCheck(
            check_id="quarantine_excluded",
            detail="all quarantined train queries are excluded from both profiles",
        ),
    )


def _manifest_for(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    parent: LoadedArtifact,
    assignments: tuple[_Assignment, ...],
    collision_group_count: int,
) -> EsciProfileManifest:
    checks = tuple(sorted(_validate_assignments(assignments), key=lambda item: item.check_id))
    assignment_documents = [
        {
            "in_development": item.in_development,
            "in_portfolio": item.in_portfolio,
            "judgment_count": item.judgment_count,
            "normalized_query_sha256": item.normalized_query_sha256,
            "official_split": item.official_split,
            "project_split": item.project_split,
            "quarantine_reason": item.quarantine_reason,
            "query_id": item.query_id,
        }
        for item in assignments
    ]
    quarantine = tuple(item for item in assignments if item.project_split == "quarantine")
    eligible_groups = {
        item.normalized_query_sha256 for item in assignments if item.project_split != "quarantine"
    }
    return EsciProfileManifest(
        dataset_version=release.manifest.dataset_version,
        source_revision=release.manifest.source_revision,
        release_manifest_sha256=release.sha256,
        raw_validation_artifact_id=parent.manifest.artifact_id,
        raw_validation_manifest_sha256=parent.manifest_sha256,
        config_sha256=config.sha256,
        query_normalization_version=config.config.dataset.query_normalization_version,
        split_version=config.config.dataset.split_version,
        profile_version=config.config.dataset.profile_version,
        seed=config.config.runtime.seed,
        train_basis_points=config.config.dataset.train_basis_points,
        task1_us_judgments=sum(item.judgment_count for item in assignments),
        task1_us_query_ids=len(assignments),
        task1_us_normalized_query_groups=len(
            {item.normalized_query_sha256 for item in assignments}
        ),
        eligible_normalized_query_groups=len(eligible_groups),
        quarantined_train_groups=collision_group_count,
        quarantined_train_query_ids=len(quarantine),
        quarantined_train_judgments=sum(item.judgment_count for item in quarantine),
        assignment_sha256=sha256(_canonical_json(assignment_documents).encode("utf-8")).hexdigest(),
        splits=_summarize_splits(assignments),
        profiles=(
            _summarize_profile(
                "development",
                assignments,
                config.config.dataset.development_query_groups,
            ),
            _summarize_profile(
                "portfolio",
                assignments,
                config.config.dataset.portfolio_query_groups,
            ),
        ),
        checks=checks,
    )


def _assignments_frame(assignments: tuple[_Assignment, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": [item.query_id for item in assignments],
            "official_split": [item.official_split for item in assignments],
            "project_split": [item.project_split for item in assignments],
            "normalized_query_sha256": [item.normalized_query_sha256 for item in assignments],
            "judgment_count": [item.judgment_count for item in assignments],
            "in_development": [item.in_development for item in assignments],
            "in_portfolio": [item.in_portfolio for item in assignments],
            "quarantine_reason": [item.quarantine_reason for item in assignments],
        },
        schema={
            "query_id": pl.Int64,
            "official_split": pl.String,
            "project_split": pl.String,
            "normalized_query_sha256": pl.String,
            "judgment_count": pl.UInt32,
            "in_development": pl.Boolean,
            "in_portfolio": pl.Boolean,
            "quarantine_reason": pl.String,
        },
    ).sort("query_id")


def load_profile_manifest(path: Path) -> EsciProfileManifest:
    """Load one strict Goldfish 005 manifest."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return EsciProfileManifest.model_validate(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ProfileManifestError(f"cannot load profile manifest {path}: {exc}") from exc


def profile_artifact_id(release: ResolvedReleaseManifest, config_sha256: str) -> str:
    """Return deterministic Goldfish 005 artifact coordinates."""
    return "/".join(
        (
            "dataset-profiles",
            release.manifest.dataset_version,
            "nested",
            SPLIT_PROFILE_COMPONENT_VERSION,
            config_sha256,
        )
    )


def _reuse_profile_artifact(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    parent: LoadedArtifact,
    store: ArtifactStore,
) -> ProfileBuildResult:
    artifact = store.load(profile_artifact_id(release, config.sha256))
    expected_dependency = ArtifactDependency(
        artifact_id=parent.manifest.artifact_id,
        manifest_sha256=parent.manifest_sha256,
    )
    if artifact.manifest.dependencies != (expected_dependency,):
        raise ProfileBuildError("existing profile artifact has an incompatible raw dependency")
    manifest = load_profile_manifest(artifact.path / PROFILE_MANIFEST_FILENAME)
    if (
        manifest.release_manifest_sha256 != release.sha256
        or manifest.config_sha256 != config.sha256
        or manifest.raw_validation_manifest_sha256 != parent.manifest_sha256
    ):
        raise ProfileBuildError("existing profile artifact has incompatible lineage")
    return ProfileBuildResult(artifact=artifact, manifest=manifest, reused=True)


def build_esci_profiles(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    raw_root: Path | None = None,
    artifact_store: ArtifactStore | None = None,
) -> ProfileBuildResult:
    """Build or reuse validated Task-1 US split and nested-profile assignments."""
    selected_raw_root = raw_root or config.config.paths.data_dir / "raw" / "esci"
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    parent = _verify_raw_validation_dependency(release, config, store)
    examples_path = _verify_examples_source(release, selected_raw_root)

    artifact_id = profile_artifact_id(release, config.sha256)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse_profile_artifact(release, config, parent, store)

    records = _collect_query_records(examples_path)
    assignments, collision_group_count = _build_assignments(records, config)
    manifest = _manifest_for(
        release,
        config,
        parent,
        assignments,
        collision_group_count,
    )
    dependency = ArtifactDependency(
        artifact_id=parent.manifest.artifact_id,
        manifest_sha256=parent.manifest_sha256,
    )
    try:
        with store.stage(
            artifact_type="dataset-profiles",
            dataset_version=release.manifest.dataset_version,
            profile="nested",
            component_version=SPLIT_PROFILE_COMPONENT_VERSION,
            config_sha256=config.sha256,
            code_revision=code_revision,
            dependencies=(dependency,),
        ) as transaction:
            _assignments_frame(assignments).write_parquet(
                transaction.path(ASSIGNMENTS_FILENAME),
                compression="zstd",
                statistics=True,
            )
            transaction.path(PROFILE_MANIFEST_FILENAME).write_text(
                _canonical_json(manifest),
                encoding="utf-8",
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse_profile_artifact(release, config, parent, store)
    return ProfileBuildResult(artifact=artifact, manifest=manifest, reused=False)


__all__ = [
    "ASSIGNMENTS_FILENAME",
    "PROFILE_MANIFEST_FILENAME",
    "SPLIT_PROFILE_COMPONENT_VERSION",
    "CohortInvariantError",
    "EsciProfileManifest",
    "ProfileBuildError",
    "ProfileBuildResult",
    "ProfileManifestError",
    "ProfileSummary",
    "RawValidationDependencyError",
    "SplitSummary",
    "build_esci_profiles",
    "load_profile_manifest",
    "normalize_query_group",
    "profile_artifact_id",
]
