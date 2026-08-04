"""Explicit immutable relevance-bundle promotion and safe product lookup."""

from __future__ import annotations

import json
import resource
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

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
    PRODUCTS_FILENAME,
    DataFoundationManifest,
    load_foundation_manifest,
)
from market_rank.evaluation.ranking import (
    ACTIVE_RELEVANCE_FILENAME,
    RANKING_EVALUATION_FILENAME,
    ActiveRelevanceContract,
    RankingEvaluationManifest,
    load_active_relevance_contract,
    load_ranking_evaluation_manifest,
    ranking_evaluation_artifact_id,
)
from market_rank.features.artifact import (
    FEATURE_ARTIFACT_FILENAME,
    FEATURE_REGISTRY_FILENAME,
    FEATURE_STATE_FILENAME,
    FeatureState,
    RankingFeatureManifest,
    load_feature_state,
    load_ranking_feature_manifest,
)
from market_rank.features.registry import FEATURE_NAMES, FeatureRegistry
from market_rank.ranking.training import (
    RANKING_MODELS_FILENAME,
    RankingModelsManifest,
    load_ranking_models_manifest,
)
from market_rank.retrieval.dense import (
    DENSE_METADATA_FILENAME,
    DenseIndexMetadata,
    load_dense_metadata,
)
from market_rank.retrieval.sparse import (
    SPARSE_METADATA_FILENAME,
    SparseIndexMetadata,
    load_sparse_metadata,
)

SERVING_BUNDLE_FILENAME: Literal["serving-bundle.json"] = "serving-bundle.json"
PRODUCT_STORE_FILENAME: Literal["products.sqlite3"] = "products.sqlite3"

Profile = Literal["development", "portfolio"]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]

_PRODUCT_COLUMNS = (
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


class ServingBundleError(RuntimeError):
    """Base error for bundle promotion, validation, and product lookup."""


class ServingBundleBuildError(ServingBundleError):
    """Raised when compatible Goldfish 006-012 artifacts cannot form a bundle."""


class ServingBundleValidationError(ServingBundleError):
    """Raised when a persisted serving bundle violates its compatibility contract."""


class ServingBundleResourceError(ServingBundleBuildError):
    """Raised when bundle promotion exceeds the configured RSS limit."""

    def __init__(self, measurement: ServingBundleResourceMeasurement) -> None:
        super().__init__(
            f"serving bundle peak RSS {measurement.peak_rss_bytes} exceeds "
            f"the {measurement.rss_limit_bytes}-byte limit"
        )
        self.measurement = measurement


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServingBundleResourceMeasurement(_StrictModel):
    dependency_peak_rss_bytes: int = Field(strict=True, ge=0)
    product_store_peak_rss_bytes: int = Field(strict=True, ge=0)
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    artifact_payload_bytes: int = Field(strict=True, ge=0)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        observed = max(self.dependency_peak_rss_bytes, self.product_store_peak_rss_bytes)
        if self.peak_rss_bytes != observed:
            raise ValueError("serving-bundle peak RSS differs from phase observations")
        if self.passed != (observed <= self.rss_limit_bytes):
            raise ValueError("serving-bundle resource status differs from the RSS gate")
        return self


class BundleComponent(_StrictModel):
    component: Literal[
        "foundation",
        "sparse",
        "dense",
        "features",
        "rankers",
        "ranking_evaluation",
    ]
    artifact_id: str = Field(strict=True, min_length=1)
    manifest_sha256: Sha256Digest
    required: Literal[True] = True


class ServingBundleCheck(_StrictModel):
    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class ServingBundleManifest(_StrictModel):
    """Complete explicit startup coordinates and compatibility facts."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    profile: Profile
    component_version: Literal["serving-bundle-v1"] = "serving-bundle-v1"
    bundle_id_policy: Literal["explicit-only-no-latest-v1"] = "explicit-only-no-latest-v1"
    catalog_id: Literal["esci_task1_us_catalog_v1"]
    catalog_membership_sha256: Sha256Digest
    product_document_version: Literal["product-document-v1"]
    product_count: int = Field(strict=True, ge=1)
    product_store_filename: Literal["products.sqlite3"] = PRODUCT_STORE_FILENAME
    product_store_schema_version: Literal[1] = 1
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    feature_names: tuple[str, ...] = Field(min_length=1)
    feature_registry_sha256: Sha256Digest
    feature_state_sha256: Sha256Digest
    parser_state_sha256: Sha256Digest
    sparse_retriever_id: str = Field(strict=True, min_length=1)
    dense_retriever_id: str = Field(strict=True, min_length=1)
    active_relevance: ActiveRelevanceContract
    fallback_contract: Literal["rrf-on-model-failure-v1"] = "rrf-on-model-failure-v1"
    readiness_policy: Literal["at-least-one-retriever-with-rrf-fallback-v1"] = (
        "at-least-one-retriever-with-rrf-fallback-v1"
    )
    offline_startup_required: Literal[True] = True
    components: tuple[
        BundleComponent,
        BundleComponent,
        BundleComponent,
        BundleComponent,
        BundleComponent,
        BundleComponent,
    ]
    resource: ServingBundleResourceMeasurement
    checks: tuple[ServingBundleCheck, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        expected = (
            "foundation",
            "sparse",
            "dense",
            "features",
            "rankers",
            "ranking_evaluation",
        )
        if tuple(component.component for component in self.components) != expected:
            raise ValueError("serving bundle components are not in canonical order")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("serving bundle feature order differs from ltr_core_v1")
        check_ids = tuple(check.check_id for check in self.checks)
        if check_ids != tuple(sorted(set(check_ids))):
            raise ValueError("serving-bundle checks must be unique and sorted")
        rankers = self.components[4]
        features = self.components[3]
        if (
            self.active_relevance.ranking_models_artifact_id != rankers.artifact_id
            or self.active_relevance.ranking_models_manifest_sha256 != rankers.manifest_sha256
            or self.active_relevance.feature_artifact_id != features.artifact_id
            or self.active_relevance.feature_registry_sha256 != self.feature_registry_sha256
        ):
            raise ValueError("active relevance differs from bundled model/feature lineage")
        return self


@dataclass(frozen=True, slots=True)
class ServingBundleBuildResult:
    artifact: LoadedArtifact
    manifest: ServingBundleManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class _Dependencies:
    evaluation: LoadedArtifact
    evaluation_manifest: RankingEvaluationManifest
    active: ActiveRelevanceContract
    models: LoadedArtifact
    models_manifest: RankingModelsManifest
    features: LoadedArtifact
    features_manifest: RankingFeatureManifest
    state: FeatureState
    registry: FeatureRegistry
    foundation: LoadedArtifact
    foundation_manifest: DataFoundationManifest
    sparse: LoadedArtifact
    sparse_metadata: SparseIndexMetadata
    dense: LoadedArtifact
    dense_metadata: DenseIndexMetadata


@dataclass(frozen=True, slots=True)
class ProductRecord:
    product_id: str
    locale: str
    title: str
    brand: str
    color: str
    bullets: str
    description: str
    normalized_brand: str
    normalized_color: str
    title_missing: bool
    brand_missing: bool
    color_missing: bool
    bullets_missing: bool
    description_missing: bool


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


def _table_sha256(manifest: DataFoundationManifest, filename: str) -> str:
    return next(table.sha256 for table in manifest.tables if table.filename == filename)


def serving_bundle_artifact_id(
    release: ResolvedReleaseManifest, config_sha256: str, profile: Profile
) -> str:
    return "/".join(
        (
            "serving-bundle",
            release.manifest.dataset_version,
            profile,
            "serving-bundle-v1",
            config_sha256,
        )
    )


def load_serving_bundle_manifest(path: Path) -> ServingBundleManifest:
    try:
        return ServingBundleManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ServingBundleValidationError(
            f"cannot load serving bundle manifest {path}: {exc}"
        ) from exc


def _load_dependencies(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    store: ArtifactStore,
) -> _Dependencies:
    try:
        evaluation = store.load(ranking_evaluation_artifact_id(release, config.sha256, profile))
        evaluation_manifest = load_ranking_evaluation_manifest(
            evaluation.path / RANKING_EVALUATION_FILENAME
        )
        active = load_active_relevance_contract(evaluation.path / ACTIVE_RELEVANCE_FILENAME)
        models = store.load(evaluation_manifest.ranking_models_artifact_id)
        models_manifest = load_ranking_models_manifest(models.path / RANKING_MODELS_FILENAME)
        features = store.load(evaluation_manifest.ranking_features_artifact_id)
        features_manifest = load_ranking_feature_manifest(features.path / FEATURE_ARTIFACT_FILENAME)
        state = load_feature_state(features.path / FEATURE_STATE_FILENAME)
        registry = FeatureRegistry.model_validate(
            json.loads((features.path / FEATURE_REGISTRY_FILENAME).read_text(encoding="utf-8"))
        )
        foundation = store.load(features_manifest.foundation_artifact_id)
        foundation_manifest = load_foundation_manifest(
            foundation.path / FOUNDATION_MANIFEST_FILENAME
        )
        sparse = store.load(features_manifest.sparse_artifact_id)
        sparse_metadata = load_sparse_metadata(sparse.path / SPARSE_METADATA_FILENAME)
        dense = store.load(features_manifest.dense_artifact_id)
        dense_metadata = load_dense_metadata(dense.path / DENSE_METADATA_FILENAME)
    except (OSError, RuntimeError, json.JSONDecodeError, ValidationError) as exc:
        raise ServingBundleBuildError(
            "compatible Goldfish 006-012 artifacts are required before bundle promotion"
        ) from exc
    catalog_hash = _table_sha256(foundation_manifest, CATALOG_MEMBERSHIP_FILENAME)
    if (
        evaluation_manifest.artifact_id != evaluation.manifest.artifact_id
        or evaluation_manifest.config_sha256 != config.sha256
        or evaluation_manifest.profile != profile
        or evaluation_manifest.active_relevance != active
        or evaluation_manifest.ranking_models_manifest_sha256 != models.manifest_sha256
        or evaluation_manifest.ranking_features_manifest_sha256 != features.manifest_sha256
        or models_manifest.artifact_id != models.manifest.artifact_id
        or models_manifest.feature_artifact_id != features.manifest.artifact_id
        or models_manifest.feature_manifest_sha256 != features.manifest_sha256
        or features_manifest.artifact_id != features.manifest.artifact_id
        or features_manifest.foundation_manifest_sha256 != foundation.manifest_sha256
        or features_manifest.sparse_manifest_sha256 != sparse.manifest_sha256
        or features_manifest.dense_manifest_sha256 != dense.manifest_sha256
        or state.registry_sha256 != models_manifest.feature_registry_sha256
        or state.sparse_index_id != sparse.manifest.artifact_id
        or state.dense_index_id != dense.manifest.artifact_id
        or sparse_metadata.catalog_membership_sha256 != catalog_hash
        or dense_metadata.catalog_membership_sha256 != catalog_hash
        or sparse_metadata.catalog_id != foundation_manifest.catalog_id
        or dense_metadata.catalog_id != foundation_manifest.catalog_id
        or sparse_metadata.document_count != foundation_manifest.catalog_products
        or dense_metadata.document_count != foundation_manifest.catalog_products
        or tuple(feature.name for feature in registry.features) != FEATURE_NAMES
    ):
        raise ServingBundleBuildError("serving-bundle artifact lineage is incompatible")
    return _Dependencies(
        evaluation,
        evaluation_manifest,
        active,
        models,
        models_manifest,
        features,
        features_manifest,
        state,
        registry,
        foundation,
        foundation_manifest,
        sparse,
        sparse_metadata,
        dense,
        dense_metadata,
    )


def _artifact_dependencies(dependencies: _Dependencies) -> tuple[ArtifactDependency, ...]:
    artifacts = (
        dependencies.foundation,
        dependencies.sparse,
        dependencies.dense,
        dependencies.features,
        dependencies.models,
        dependencies.evaluation,
    )
    return tuple(
        sorted(
            (
                ArtifactDependency(
                    artifact_id=artifact.manifest.artifact_id,
                    manifest_sha256=artifact.manifest_sha256,
                )
                for artifact in artifacts
            ),
            key=lambda item: item.artifact_id,
        )
    )


def _create_product_store(
    source: Path,
    membership_source: Path,
    destination: Path,
    *,
    expected_rows: int,
    batch_rows: int,
) -> int:
    connection = sqlite3.connect(destination)
    try:
        connection.execute("PRAGMA page_size = 4096")
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute(
            """
            CREATE TABLE products (
                product_id TEXT PRIMARY KEY NOT NULL,
                locale TEXT NOT NULL CHECK (locale = 'us'),
                title TEXT NOT NULL,
                brand TEXT NOT NULL,
                color TEXT NOT NULL,
                bullets TEXT NOT NULL,
                description TEXT NOT NULL,
                normalized_brand TEXT NOT NULL,
                normalized_color TEXT NOT NULL,
                title_missing INTEGER NOT NULL CHECK (title_missing IN (0,1)),
                brand_missing INTEGER NOT NULL CHECK (brand_missing IN (0,1)),
                color_missing INTEGER NOT NULL CHECK (color_missing IN (0,1)),
                bullets_missing INTEGER NOT NULL CHECK (bullets_missing IN (0,1)),
                description_missing INTEGER NOT NULL CHECK (description_missing IN (0,1))
            ) WITHOUT ROWID
            """
        )
        inserted = 0
        for offset in range(0, expected_rows, batch_rows):
            batch = (
                pl.scan_parquet(source)
                .join(
                    pl.scan_parquet(membership_source).select(
                        pl.col("locale").alias("product_locale"),
                        "product_id",
                        "catalog_ordinal",
                    ),
                    on=("product_locale", "product_id"),
                    how="inner",
                    validate="1:1",
                )
                .sort("catalog_ordinal")
                .select(*_PRODUCT_COLUMNS)
                .with_columns(
                    pl.col(column).fill_null("")
                    for column in (
                        "clean_title",
                        "clean_brand",
                        "clean_color",
                        "clean_bullets",
                        "clean_description",
                        "normalized_brand",
                        "normalized_color",
                    )
                )
                .slice(offset, batch_rows)
                .collect()
            )
            values = [
                (
                    row["product_id"],
                    row["product_locale"],
                    row["clean_title"],
                    row["clean_brand"],
                    row["clean_color"],
                    row["clean_bullets"],
                    row["clean_description"],
                    row["normalized_brand"],
                    row["normalized_color"],
                    int(row["title_missing"]),
                    int(row["brand_missing"]),
                    int(row["color_missing"]),
                    int(row["bullets_missing"]),
                    int(row["description_missing"]),
                )
                for row in batch.iter_rows(named=True)
            ]
            connection.executemany(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values
            )
            inserted += len(values)
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if inserted != expected_rows or integrity != ("ok",):
            raise ServingBundleBuildError("product store row count or integrity is incompatible")
        connection.execute("VACUUM")
        return inserted
    except (sqlite3.Error, pl.exceptions.PolarsError) as exc:
        raise ServingBundleBuildError(f"cannot build serving product store: {exc}") from exc
    finally:
        connection.close()


def _components(
    dependencies: _Dependencies,
) -> tuple[
    BundleComponent,
    BundleComponent,
    BundleComponent,
    BundleComponent,
    BundleComponent,
    BundleComponent,
]:
    return (
        BundleComponent(
            component="foundation",
            artifact_id=dependencies.foundation.manifest.artifact_id,
            manifest_sha256=dependencies.foundation.manifest_sha256,
        ),
        BundleComponent(
            component="sparse",
            artifact_id=dependencies.sparse.manifest.artifact_id,
            manifest_sha256=dependencies.sparse.manifest_sha256,
        ),
        BundleComponent(
            component="dense",
            artifact_id=dependencies.dense.manifest.artifact_id,
            manifest_sha256=dependencies.dense.manifest_sha256,
        ),
        BundleComponent(
            component="features",
            artifact_id=dependencies.features.manifest.artifact_id,
            manifest_sha256=dependencies.features.manifest_sha256,
        ),
        BundleComponent(
            component="rankers",
            artifact_id=dependencies.models.manifest.artifact_id,
            manifest_sha256=dependencies.models.manifest_sha256,
        ),
        BundleComponent(
            component="ranking_evaluation",
            artifact_id=dependencies.evaluation.manifest.artifact_id,
            manifest_sha256=dependencies.evaluation.manifest_sha256,
        ),
    )


def _reuse(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    profile: Profile,
    dependencies: _Dependencies,
    store: ArtifactStore,
) -> ServingBundleBuildResult:
    artifact = store.load(serving_bundle_artifact_id(release, config.sha256, profile))
    if artifact.manifest.dependencies != _artifact_dependencies(dependencies):
        raise ServingBundleValidationError("serving-bundle dependencies are incompatible")
    manifest = load_serving_bundle_manifest(artifact.path / SERVING_BUNDLE_FILENAME)
    if (
        manifest.artifact_id != artifact.manifest.artifact_id
        or manifest.config_sha256 != config.sha256
        or manifest.profile != profile
        or manifest.active_relevance != dependencies.active
    ):
        raise ServingBundleValidationError("serving-bundle metadata identity is incompatible")
    return ServingBundleBuildResult(artifact, manifest, True)


def build_serving_bundle(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    profile: Profile | None = None,
    artifact_store: ArtifactStore | None = None,
) -> ServingBundleBuildResult:
    """Promote one explicit, offline-loadable relevance bundle."""
    selected_profile: Profile = profile or config.config.evaluation.default_profile
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    dependencies = _load_dependencies(release, config, selected_profile, store)
    artifact_id = serving_bundle_artifact_id(release, config.sha256, selected_profile)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse(release, config, selected_profile, dependencies, store)

    rss_limit = config.config.runtime.rss_limit_mb * 1024 * 1024
    dependency_peak = _peak_rss_bytes()
    initial = ServingBundleResourceMeasurement(
        dependency_peak_rss_bytes=dependency_peak,
        product_store_peak_rss_bytes=dependency_peak,
        peak_rss_bytes=dependency_peak,
        rss_limit_bytes=rss_limit,
        artifact_payload_bytes=0,
        passed=dependency_peak <= rss_limit,
    )
    if not initial.passed:
        raise ServingBundleResourceError(initial)

    transaction = store.stage(
        artifact_type="serving-bundle",
        dataset_version=release.manifest.dataset_version,
        profile=selected_profile,
        component_version=config.config.serving.component_version,
        config_sha256=config.sha256,
        code_revision=code_revision,
        dependencies=_artifact_dependencies(dependencies),
    )
    try:
        with transaction:
            root = transaction.path(SERVING_BUNDLE_FILENAME).parent
            inserted = _create_product_store(
                dependencies.foundation.path / PRODUCTS_FILENAME,
                dependencies.foundation.path / CATALOG_MEMBERSHIP_FILENAME,
                root / PRODUCT_STORE_FILENAME,
                expected_rows=dependencies.foundation_manifest.catalog_products,
                batch_rows=config.config.serving.product_store_batch_rows,
            )
            product_store_peak = _peak_rss_bytes()
            payload_bytes = _directory_bytes(root)
            measurement = ServingBundleResourceMeasurement(
                dependency_peak_rss_bytes=dependency_peak,
                product_store_peak_rss_bytes=product_store_peak,
                peak_rss_bytes=max(dependency_peak, product_store_peak),
                rss_limit_bytes=rss_limit,
                artifact_payload_bytes=payload_bytes,
                passed=max(dependency_peak, product_store_peak) <= rss_limit,
            )
            if not measurement.passed:
                raise ServingBundleResourceError(measurement)
            checks = tuple(
                sorted(
                    (
                        ServingBundleCheck(
                            check_id="active_relevance",
                            detail=(
                                f"exact Goldfish 012 contract selects "
                                f"{dependencies.active.selected_stage}"
                            ),
                        ),
                        ServingBundleCheck(
                            check_id="catalog_alignment",
                            detail=f"all serving components align to {inserted} catalog products",
                        ),
                        ServingBundleCheck(
                            check_id="explicit_bundle",
                            detail=(
                                "startup requires this immutable bundle ID; latest is prohibited"
                            ),
                        ),
                        ServingBundleCheck(
                            check_id="offline_startup",
                            detail="bundle loading permits only local cached encoder assets",
                        ),
                        ServingBundleCheck(
                            check_id="resource_gate",
                            detail=(
                                f"peak RSS {measurement.peak_rss_bytes} <= "
                                f"{measurement.rss_limit_bytes} bytes"
                            ),
                        ),
                        ServingBundleCheck(
                            check_id="safe_product_store",
                            detail="bounded product projection uses indexed SQLite without pickle",
                        ),
                    ),
                    key=lambda check: check.check_id,
                )
            )
            manifest = ServingBundleManifest(
                artifact_id=artifact_id,
                dataset_version=release.manifest.dataset_version,
                config_sha256=config.sha256,
                profile=selected_profile,
                catalog_id=dependencies.foundation_manifest.catalog_id,
                catalog_membership_sha256=_table_sha256(
                    dependencies.foundation_manifest, CATALOG_MEMBERSHIP_FILENAME
                ),
                product_document_version=dependencies.foundation_manifest.product_document_version,
                product_count=inserted,
                feature_names=FEATURE_NAMES,
                feature_registry_sha256=dependencies.models_manifest.feature_registry_sha256,
                feature_state_sha256=dependencies.models_manifest.feature_state_sha256,
                parser_state_sha256=dependencies.state.parser_state.state_sha256,
                sparse_retriever_id=dependencies.state.sparse_retriever_id,
                dense_retriever_id=dependencies.state.dense_retriever_id,
                active_relevance=dependencies.active,
                components=_components(dependencies),
                resource=measurement,
                checks=checks,
            )
            (root / SERVING_BUNDLE_FILENAME).write_text(_canonical_json(manifest), encoding="utf-8")
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse(release, config, selected_profile, dependencies, store)
    return ServingBundleBuildResult(artifact, manifest, False)


class ProductStore:
    """Read-only indexed product projection with bounded key lookup."""

    def __init__(self, path: Path, expected_rows: int) -> None:
        try:
            self._connection = sqlite3.connect(
                f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
            )
            self._connection.execute("PRAGMA query_only = ON")
            observed = int(self._connection.execute("SELECT COUNT(*) FROM products").fetchone()[0])
            integrity = self._connection.execute("PRAGMA quick_check").fetchone()
        except sqlite3.Error as exc:
            raise ServingBundleValidationError(f"cannot open serving product store: {exc}") from exc
        if observed != expected_rows or integrity != ("ok",):
            self.close()
            raise ServingBundleValidationError("serving product store count or integrity changed")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(self, product_ids: tuple[str, ...]) -> dict[str, ProductRecord]:
        if len(product_ids) != len(set(product_ids)) or not product_ids:
            raise ServingBundleValidationError("product lookup keys must be nonempty and unique")
        placeholders = ",".join("?" for _ in product_ids)
        try:
            rows = self._connection.execute(
                f"SELECT * FROM products WHERE product_id IN ({placeholders})",
                product_ids,
            ).fetchall()
        except sqlite3.Error as exc:
            raise ServingBundleValidationError(f"serving product lookup failed: {exc}") from exc
        result = {
            str(row[0]): ProductRecord(
                product_id=str(row[0]),
                locale=str(row[1]),
                title=str(row[2]),
                brand=str(row[3]),
                color=str(row[4]),
                bullets=str(row[5]),
                description=str(row[6]),
                normalized_brand=str(row[7]),
                normalized_color=str(row[8]),
                title_missing=bool(row[9]),
                brand_missing=bool(row[10]),
                color_missing=bool(row[11]),
                bullets_missing=bool(row[12]),
                description_missing=bool(row[13]),
            )
            for row in rows
        }
        if set(result) != set(product_ids):
            raise ServingBundleValidationError("candidate products are missing from product store")
        return result


def load_product_store(artifact: LoadedArtifact, manifest: ServingBundleManifest) -> ProductStore:
    return ProductStore(artifact.path / manifest.product_store_filename, manifest.product_count)


__all__ = [
    "PRODUCT_STORE_FILENAME",
    "SERVING_BUNDLE_FILENAME",
    "ProductRecord",
    "ProductStore",
    "ServingBundleBuildError",
    "ServingBundleBuildResult",
    "ServingBundleError",
    "ServingBundleManifest",
    "ServingBundleResourceError",
    "ServingBundleValidationError",
    "build_serving_bundle",
    "load_product_store",
    "load_serving_bundle_manifest",
    "serving_bundle_artifact_id",
]
