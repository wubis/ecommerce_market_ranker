"""Fixture integration tests for the consolidated Goldfish 006 data foundation."""

from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

import market_rank.cli as cli_module
import market_rank.data.foundation as foundation_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig, load_config
from market_rank.data.esci_raw import (
    ResolvedReleaseManifest,
    ensure_raw_validation_artifact,
    validate_raw_dataset,
)
from market_rank.data.foundation import (
    CATALOG_EXCLUSIONS_FILENAME,
    CATALOG_MEMBERSHIP_FILENAME,
    FOUNDATION_MANIFEST_FILENAME,
    JUDGED_POOLS_FILENAME,
    JUDGMENTS_FILENAME,
    PRODUCT_DOCUMENTS_FILENAME,
    PRODUCTS_FILENAME,
    QUERIES_FILENAME,
    DataFoundationError,
    FoundationBuildResult,
    FoundationDependencyError,
    ResourceGateError,
    build_esci_foundation,
    load_foundation_manifest,
)
from market_rank.data.profiles import build_esci_profiles
from tests.unit.test_esci_profiles import (
    BASE_CONFIG,
    _config,
    _examples,
    _release,
    _write_raw,
)

RETRIEVED_UTC = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)


def _foundation_config(*, blocked: bool = False) -> ResolvedConfig:
    if not blocked:
        return _config()
    return load_config(
        [BASE_CONFIG],
        overrides={
            "runtime.seed": 42,
            "runtime.rss_limit_mb": 512,
            "dataset.train_basis_points": 5000,
            "dataset.development_query_groups": 3,
            "dataset.portfolio_query_groups": 6,
            "dataset.m2_runtime_reserve_mb": 512,
        },
    )


def _prepare_foundation(
    tmp_path: Path,
    *,
    reverse: bool = False,
    relabel: bool = False,
    conflicting_judgment: bool = False,
    blocked: bool = False,
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, Path, ArtifactStore]:
    config = _foundation_config(blocked=blocked)
    if conflicting_judgment:
        config = load_config(
            [BASE_CONFIG],
            overrides={
                "runtime.seed": 42,
                "dataset.train_basis_points": 5000,
                "dataset.development_query_groups": 3,
                "dataset.portfolio_query_groups": 7,
            },
        )
    examples = _examples(reverse=reverse, relabel=relabel)
    if conflicting_judgment:
        duplicate = examples.filter(pl.col("example_id") == 5).with_columns(
            pl.lit(10_000, dtype=pl.Int64).alias("example_id"),
            pl.lit("I").alias("esci_label"),
        )
        examples = pl.concat((examples, duplicate))
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, examples)
    products_path = raw_root / "shopping_queries_dataset_products.parquet"
    products = pl.read_parquet(products_path).with_columns(
        pl.when(pl.col("product_id") == "p3-0")
        .then(pl.lit("<b>Wireless</b>\u0000 Mouse"))
        .otherwise(pl.col("product_title"))
        .alias("product_title"),
        pl.when(pl.col("product_id") == "p3-0")
        .then(pl.lit("  A   compact <i>mouse</i>  "))
        .otherwise(pl.col("product_description"))
        .alias("product_description"),
    )
    for column in (
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ):
        products = products.with_columns(
            pl.when(pl.col("product_id") == "p4-0")
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col(column))
            .alias(column)
        )
    products.write_parquet(products_path)

    release = _release(raw_root, tmp_path / "release.json")
    report = validate_raw_dataset(release, raw_root, retrieved_utc=RETRIEVED_UTC)
    assert report.valid
    store = ArtifactStore(tmp_path / "artifacts")
    ensure_raw_validation_artifact(
        release,
        report,
        store,
        config_sha256=config.sha256,
        code_revision="fixture",
    )
    build_esci_profiles(
        release,
        config,
        code_revision="fixture",
        raw_root=raw_root,
        artifact_store=store,
    )
    return release, config, raw_root, store


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, Path, ArtifactStore],
) -> FoundationBuildResult:
    release, config, raw_root, store = prepared
    return build_esci_foundation(
        release,
        config,
        code_revision="fixture",
        raw_root=raw_root,
        artifact_store=store,
    )


def test_builds_verified_canonical_tables_with_profile_parent(tmp_path: Path) -> None:
    result = _build(_prepare_foundation(tmp_path))

    assert not result.reused
    assert result.artifact.manifest.artifact_type == "data-foundation"
    assert result.artifact.manifest.dependencies[0].artifact_id == (
        result.manifest.profile_artifact_id
    )
    assert tuple(table.filename for table in result.manifest.tables) == (
        "queries.parquet",
        "sources.parquet",
        "judgments.parquet",
        "products.parquet",
        "product-documents.parquet",
        "catalog-membership.parquet",
        "catalog-exclusions.parquet",
        "judged-pools.parquet",
    )
    assert load_foundation_manifest(result.artifact.path / FOUNDATION_MANIFEST_FILENAME) == (
        result.manifest
    )
    assert (
        ArtifactStore(tmp_path / "artifacts").load(result.artifact.manifest.artifact_id)
        == result.artifact
    )


def test_catalog_uses_full_label_blind_task1_population_not_portfolio(tmp_path: Path) -> None:
    result = _build(_prepare_foundation(tmp_path))
    judgments = pl.read_parquet(result.artifact.path / JUDGMENTS_FILENAME)

    assert result.manifest.catalog_candidate_products == 18
    assert result.manifest.catalog_products == 17
    assert result.manifest.catalog_excluded_no_text == 1
    assert result.manifest.catalog_candidate_products > judgments["product_id"].n_unique()


def test_catalog_membership_is_independent_of_label_values(tmp_path: Path) -> None:
    first = _build(_prepare_foundation(tmp_path / "first"))
    relabeled = _build(_prepare_foundation(tmp_path / "relabeled", relabel=True))
    first_membership = pl.read_parquet(first.artifact.path / CATALOG_MEMBERSHIP_FILENAME)
    relabeled_membership = pl.read_parquet(relabeled.artifact.path / CATALOG_MEMBERSHIP_FILENAME)

    assert first_membership.equals(relabeled_membership)


def test_canonical_outputs_are_stable_under_raw_row_reordering(tmp_path: Path) -> None:
    first = _build(_prepare_foundation(tmp_path / "first"))
    reordered = _build(_prepare_foundation(tmp_path / "reordered", reverse=True))

    for filename in (
        QUERIES_FILENAME,
        JUDGMENTS_FILENAME,
        PRODUCTS_FILENAME,
        PRODUCT_DOCUMENTS_FILENAME,
        CATALOG_MEMBERSHIP_FILENAME,
        JUDGED_POOLS_FILENAME,
    ):
        assert pl.read_parquet(first.artifact.path / filename).equals(
            pl.read_parquet(reordered.artifact.path / filename)
        )


def test_canonical_judgments_separate_label_ids_and_official_gains(tmp_path: Path) -> None:
    result = _build(_prepare_foundation(tmp_path))
    judgments = pl.read_parquet(result.artifact.path / JUDGMENTS_FILENAME)
    observed = {
        row["esci_label"]: (row["label_id"], row["gain"])
        for row in judgments.select("esci_label", "label_id", "gain").unique().iter_rows(named=True)
    }

    assert observed["E"] == pytest.approx((3, 1.0))
    assert observed["S"] == pytest.approx((2, 0.1))
    assert result.manifest.label_mapping == (
        ("I", 0, 0.0),
        ("C", 1, 0.01),
        ("S", 2, 0.1),
        ("E", 3, 1.0),
    )


def test_documents_strip_html_controls_and_audit_missing_text(tmp_path: Path) -> None:
    result = _build(_prepare_foundation(tmp_path))
    documents = pl.read_parquet(result.artifact.path / PRODUCT_DOCUMENTS_FILENAME)
    exclusions = pl.read_parquet(result.artifact.path / CATALOG_EXCLUSIONS_FILENAME)
    rich_document = documents.filter(pl.col("product_id") == "p3-0").item(0, "document")

    assert rich_document.startswith("[TITLE] Wireless Mouse [BRAND]")
    assert "<b>" not in rich_document
    assert "\u0000" not in rich_document
    assert "A compact mouse" in rich_document
    assert exclusions.filter(pl.col("product_id") == "p4-0").item(0, "reason") == (
        "no_usable_source_text"
    )
    assert documents.filter(pl.col("product_id") == "p4-0").is_empty()


def test_catalog_membership_and_products_have_unique_exact_keys(tmp_path: Path) -> None:
    result = _build(_prepare_foundation(tmp_path))
    products = pl.read_parquet(result.artifact.path / PRODUCTS_FILENAME)
    membership = pl.read_parquet(result.artifact.path / CATALOG_MEMBERSHIP_FILENAME)

    assert products.select(pl.struct("product_locale", "product_id").n_unique()).item() == (
        products.height
    )
    assert membership.select(pl.struct("locale", "product_id").n_unique()).item() == (
        membership.height
    )
    assert membership["catalog_ordinal"].to_list() == list(range(membership.height))


def test_nested_closed_pools_retain_complete_query_groups(tmp_path: Path) -> None:
    result = _build(_prepare_foundation(tmp_path))
    pools = pl.read_parquet(result.artifact.path / JUDGED_POOLS_FILENAME)
    development = pools.filter(pl.col("profile") == "development")
    portfolio = pools.filter(pl.col("profile") == "portfolio")
    development_keys = set(development.select("query_id", "product_id").iter_rows())
    portfolio_keys = set(portfolio.select("query_id", "product_id").iter_rows())

    assert development_keys <= portfolio_keys
    assert result.manifest.pools[1].judgments == portfolio.height
    assert (
        result.manifest.pools[1].query_ids
        == pl.read_parquet(result.artifact.path / QUERIES_FILENAME).height
    )
    ordinals = portfolio.group_by("query_id").agg(
        pl.col("stable_ordinal").sort().alias("ordinals"), pl.len().alias("count")
    )
    assert all(
        row["ordinals"] == list(range(1, row["count"] + 1))
        for row in ordinals.iter_rows(named=True)
    )


def test_preliminary_m3_resource_gate_passes_and_is_component_auditable(tmp_path: Path) -> None:
    estimate = _build(_prepare_foundation(tmp_path)).manifest.resource_estimate

    assert estimate.proceed
    assert estimate.projected_dense_vector_bytes == estimate.catalog_products * 384 * 4
    assert estimate.projected_sparse_index_bytes == estimate.document_utf8_bytes * 3 // 2
    assert estimate.projected_runtime_bytes <= estimate.rss_limit_bytes


def test_resource_gate_blocks_promotion_and_preserves_report(tmp_path: Path) -> None:
    prepared = _prepare_foundation(tmp_path, blocked=True)

    with pytest.raises(ResourceGateError) as caught:
        _build(prepared)

    assert not caught.value.estimate.proceed
    assert caught.value.estimate.projected_runtime_bytes > caught.value.estimate.rss_limit_bytes
    assert not any((tmp_path / "artifacts" / "data-foundation").rglob("_SUCCESS"))


def test_conflicting_duplicate_judgment_blocks_promotion(tmp_path: Path) -> None:
    prepared = _prepare_foundation(tmp_path, conflicting_judgment=True)

    with pytest.raises(DataFoundationError, match="conflicting labels"):
        _build(prepared)


def test_compatible_foundation_is_reused_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_foundation(tmp_path)
    first = _build(prepared)

    def forbid_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("compatible foundation must be reused")

    monkeypatch.setattr(foundation_module, "_write_tables", forbid_write)
    second = _build(prepared)

    assert second.reused
    assert second.artifact == first.artifact
    assert second.manifest == first.manifest


def test_missing_profile_dependency_fails_before_foundation_work(tmp_path: Path) -> None:
    config = _foundation_config()
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, _examples())
    release = _release(raw_root, tmp_path / "release.json")

    with pytest.raises(FoundationDependencyError, match="build-esci-profiles"):
        build_esci_foundation(
            release,
            config,
            code_revision="fixture",
            raw_root=raw_root,
            artifact_store=ArtifactStore(tmp_path / "empty-artifacts"),
        )


def test_cli_emits_bounded_foundation_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _build(_prepare_foundation(tmp_path))

    def fake_build(
        release: ResolvedReleaseManifest,
        config: ResolvedConfig,
        *,
        code_revision: str,
        raw_root: Path | None,
    ) -> FoundationBuildResult:
        del release, config, code_revision, raw_root
        return result

    monkeypatch.setattr(cli_module, "build_esci_foundation", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")

    exit_code = cli_module.main(["data", "build-esci-foundation"])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "canonical portfolio:" in output.out
    assert "fixed catalog: 17 products" in output.out
    assert "resource gate:" in output.out
    assert "data foundation:" in output.out
    assert output.err == ""


def test_cli_foundation_failure_is_concise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(arguments: Namespace) -> int:
        del arguments
        raise DataFoundationError("fixture foundation failed")

    monkeypatch.setattr(cli_module, "_run_build_esci_foundation", fail)
    exit_code = cli_module.main(["data", "build-esci-foundation"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err.strip() == "error: fixture foundation failed"
    assert "Traceback" not in output.err
