"""Tests for Goldfish 005 Task-1 US splits and nested query profiles."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import polars as pl
import pytest

import market_rank.cli as cli_module
import market_rank.data.profiles as profiles_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig, load_config
from market_rank.data.esci_raw import (
    OFFICIAL_PAPER,
    OFFICIAL_REPOSITORY,
    EsciReleaseManifest,
    RawFileSource,
    ResolvedReleaseManifest,
    ensure_raw_validation_artifact,
    load_release_manifest,
    validate_raw_dataset,
)
from market_rank.data.profiles import (
    ASSIGNMENTS_FILENAME,
    PROFILE_MANIFEST_FILENAME,
    ProfileBuildError,
    ProfileBuildResult,
    RawValidationDependencyError,
    build_esci_profiles,
    load_profile_manifest,
    normalize_query_group,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
BASE_CONFIG = REPOSITORY_ROOT / "configs" / "base.yaml"
REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
RETRIEVED_UTC = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def _config(*, seed: int = 42) -> ResolvedConfig:
    return load_config(
        [BASE_CONFIG],
        overrides={
            "runtime.seed": seed,
            "dataset.train_basis_points": 5000,
            "dataset.development_query_groups": 3,
            "dataset.portfolio_query_groups": 6,
        },
    )


def _examples(*, reverse: bool = False, relabel: bool = False) -> pl.DataFrame:
    queries = [
        (1, " Fancy\u3000Shoes ", "train", "us", 1),
        (2, "fancy shoes", "test", "us", 1),
        (3, "mouse", "train", "us", 1),
        (4, "keyboard", "train", "us", 1),
        (5, "monitor", "train", "us", 1),
        (6, "desk", "train", "us", 1),
        (7, "lamp", "test", "us", 1),
        (8, "chair", "test", "us", 1),
        (9, "excluded locale", "train", "es", 1),
        (10, "excluded large", "train", "us", 0),
        (11, " MOUSE ", "train", "us", 1),
    ]
    rows: list[dict[str, object]] = []
    example_id = 1
    labels = ("I", "E") if relabel else ("E", "S")
    for query_id, query, split, locale, small_version in queries:
        for product_index in range(2):
            rows.append(
                {
                    "example_id": example_id,
                    "query": query,
                    "query_id": query_id,
                    "product_id": f"p{query_id}-{product_index}",
                    "product_locale": locale,
                    "esci_label": labels[product_index],
                    "small_version": small_version,
                    "large_version": 1,
                    "split": split,
                }
            )
            example_id += 1
    if reverse:
        rows.reverse()
    return pl.DataFrame(
        rows,
        schema={
            "example_id": pl.Int64,
            "query": pl.String,
            "query_id": pl.Int64,
            "product_id": pl.String,
            "product_locale": pl.String,
            "esci_label": pl.String,
            "small_version": pl.Int8,
            "large_version": pl.Int8,
            "split": pl.String,
        },
    )


def _write_raw(raw_root: Path, examples: pl.DataFrame) -> None:
    raw_root.mkdir(parents=True)
    examples.write_parquet(raw_root / "shopping_queries_dataset_examples.parquet")
    products = (
        examples.select("product_id", "product_locale")
        .with_columns(
            pl.concat_str(pl.lit("Product "), pl.col("product_id")).alias("product_title"),
            pl.lit(None, dtype=pl.String).alias("product_description"),
            pl.lit(None, dtype=pl.String).alias("product_bullet_point"),
            pl.lit("Fixture").alias("product_brand"),
            pl.lit(None, dtype=pl.String).alias("product_color"),
        )
        .select(
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_locale",
        )
    )
    products.write_parquet(raw_root / "shopping_queries_dataset_products.parquet")
    examples.select("query_id").unique().sort("query_id").with_columns(
        pl.lit("fixture").alias("source")
    ).write_csv(raw_root / "shopping_queries_dataset_sources.csv")


def _digest(path: Path) -> tuple[int, str]:
    content = path.read_bytes()
    return len(content), sha256(content).hexdigest()


def _release(raw_root: Path, manifest_path: Path) -> ResolvedReleaseManifest:
    sources: list[RawFileSource] = []
    for role, filename, file_format in (
        ("examples", "shopping_queries_dataset_examples.parquet", "parquet"),
        ("products", "shopping_queries_dataset_products.parquet", "parquet"),
        ("sources", "shopping_queries_dataset_sources.csv", "csv"),
    ):
        size_bytes, file_sha256 = _digest(raw_root / filename)
        sources.append(
            RawFileSource.model_validate(
                {
                    "role": role,
                    "filename": filename,
                    "format": file_format,
                    "source_url": (
                        f"{OFFICIAL_REPOSITORY}/raw/{REVISION}/shopping_queries_dataset/{filename}"
                    ),
                    "size_bytes": size_bytes,
                    "sha256": file_sha256,
                }
            )
        )
    manifest = EsciReleaseManifest.model_validate(
        {
            "dataset_version": f"esci-{REVISION[:12]}",
            "source_repository": OFFICIAL_REPOSITORY,
            "source_revision": REVISION,
            "source_commit_utc": datetime(2024, 10, 7, 15, 52, 6, tzinfo=UTC),
            "license_url": f"{OFFICIAL_REPOSITORY}/blob/{REVISION}/LICENSE",
            "paper_url": OFFICIAL_PAPER,
            "files": tuple(sources),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return load_release_manifest(manifest_path)


def _prepare(
    tmp_path: Path,
    *,
    config: ResolvedConfig | None = None,
    reverse: bool = False,
    relabel: bool = False,
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, Path, ArtifactStore]:
    selected_config = config or _config()
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, _examples(reverse=reverse, relabel=relabel))
    release = _release(raw_root, tmp_path / "release.json")
    report = validate_raw_dataset(release, raw_root, retrieved_utc=RETRIEVED_UTC)
    assert report.valid
    store = ArtifactStore(tmp_path / "artifacts")
    ensure_raw_validation_artifact(
        release,
        report,
        store,
        config_sha256=selected_config.sha256,
        code_revision="fixture",
    )
    return release, selected_config, raw_root, store


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, Path, ArtifactStore],
) -> ProfileBuildResult:
    release, config, raw_root, store = prepared
    return build_esci_profiles(
        release,
        config,
        code_revision="fixture",
        raw_root=raw_root,
        artifact_store=store,
    )


def _assignments(result: ProfileBuildResult) -> pl.DataFrame:
    return pl.read_parquet(result.artifact.path / ASSIGNMENTS_FILENAME).sort("query_id")


def test_query_group_normalization_is_nfkc_casefolded_and_whitespace_collapsed() -> None:
    assert normalize_query_group("  FANCY\u3000Shoes\t") == "fancy shoes"
    assert normalize_query_group("Straße") == "strasse"


def test_build_filters_exact_task1_us_predicate_and_quarantines_train_collision(
    tmp_path: Path,
) -> None:
    result = _build(_prepare(tmp_path))
    assignments = _assignments(result)

    assert result.manifest.task1_us_query_ids == 9
    assert result.manifest.task1_us_judgments == 18
    assert set(assignments["query_id"].to_list()) == {1, 2, 3, 4, 5, 6, 7, 8, 11}
    train_collision = assignments.filter(pl.col("query_id") == 1).row(0, named=True)
    preserved_test = assignments.filter(pl.col("query_id") == 2).row(0, named=True)
    assert train_collision["project_split"] == "quarantine"
    assert train_collision["quarantine_reason"] == ("normalized_query_occurs_in_official_test")
    assert not train_collision["in_development"]
    assert not train_collision["in_portfolio"]
    assert preserved_test["project_split"] == "test"
    assert train_collision["normalized_query_sha256"] == preserved_test["normalized_query_sha256"]


def test_project_splits_are_group_disjoint_and_official_test_is_frozen(tmp_path: Path) -> None:
    assignments = _assignments(_build(_prepare(tmp_path)))
    eligible = assignments.filter(pl.col("project_split") != "quarantine")
    split_counts = eligible.group_by("normalized_query_sha256").agg(
        pl.col("project_split").n_unique().alias("split_count")
    )

    assert split_counts["split_count"].max() == 1
    assert set(
        assignments.filter(pl.col("official_split") == "test")["project_split"].to_list()
    ) == {"test"}
    mouse = assignments.filter(pl.col("query_id").is_in([3, 11]))
    assert mouse["normalized_query_sha256"].n_unique() == 1
    assert mouse["project_split"].n_unique() == 1


def test_development_is_an_exact_nested_subset_of_portfolio(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path))
    assignments = _assignments(result)
    development, portfolio = result.manifest.profiles
    development_ids = set(assignments.filter(pl.col("in_development"))["query_id"].to_list())
    portfolio_ids = set(assignments.filter(pl.col("in_portfolio"))["query_id"].to_list())

    assert development.selected_normalized_query_groups == 3
    assert portfolio.selected_normalized_query_groups == 6
    assert development_ids <= portfolio_ids
    assert assignments.filter(pl.col("in_development") & ~pl.col("in_portfolio")).is_empty()
    assert set(item.project_split for item in development.splits) == {
        "train",
        "validation",
        "test",
    }


def test_assignments_are_deterministic_under_input_reordering(tmp_path: Path) -> None:
    first = _build(_prepare(tmp_path / "first", reverse=False))
    second = _build(_prepare(tmp_path / "second", reverse=True))

    assert first.manifest.assignment_sha256 == second.manifest.assignment_sha256
    assert first.manifest.profiles == second.manifest.profiles
    assert _assignments(first).equals(_assignments(second))


def test_split_and_profile_membership_are_label_blind(tmp_path: Path) -> None:
    first = _build(_prepare(tmp_path / "first", relabel=False))
    relabeled = _build(_prepare(tmp_path / "relabeled", relabel=True))

    assert first.manifest.assignment_sha256 == relabeled.manifest.assignment_sha256
    assert first.manifest.profiles == relabeled.manifest.profiles
    assert _assignments(first).equals(_assignments(relabeled))


def test_seed_change_changes_deterministic_assignments(tmp_path: Path) -> None:
    first = _build(_prepare(tmp_path / "first", config=_config(seed=42)))
    second = _build(_prepare(tmp_path / "second", config=_config(seed=43)))

    assert first.manifest.assignment_sha256 != second.manifest.assignment_sha256


def test_profile_artifact_declares_validated_raw_parent_and_reloads(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path))
    dependency = result.artifact.manifest.dependencies[0]

    assert result.artifact.manifest.artifact_type == "dataset-profiles"
    assert dependency.artifact_id == result.manifest.raw_validation_artifact_id
    assert dependency.manifest_sha256 == result.manifest.raw_validation_manifest_sha256
    assert load_profile_manifest(result.artifact.path / PROFILE_MANIFEST_FILENAME) == (
        result.manifest
    )
    assert (
        ArtifactStore(tmp_path / "artifacts").load(result.artifact.manifest.artifact_id)
        == result.artifact
    )


def test_compatible_profile_artifact_is_reused_without_rescanning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    first = _build(prepared)

    def forbid_scan(path: Path) -> object:
        del path
        raise AssertionError("compatible artifact should be reused before cohort scan")

    monkeypatch.setattr(profiles_module, "_collect_query_records", forbid_scan)
    second = _build(prepared)

    assert second.reused
    assert second.artifact == first.artifact
    assert second.manifest == first.manifest


def test_missing_validated_raw_parent_fails_before_source_scan(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, _examples())
    release = _release(raw_root, tmp_path / "release.json")

    with pytest.raises(RawValidationDependencyError, match="download-esci"):
        build_esci_profiles(
            release,
            _config(),
            code_revision="fixture",
            raw_root=raw_root,
            artifact_store=ArtifactStore(tmp_path / "empty-artifacts"),
        )


def test_source_mutation_after_validation_is_rejected(tmp_path: Path) -> None:
    release, config, raw_root, store = _prepare(tmp_path)
    with (raw_root / "shopping_queries_dataset_examples.parquet").open("ab") as stream:
        stream.write(b"mutation")

    with pytest.raises(ProfileBuildError, match="no longer matches"):
        build_esci_profiles(
            release,
            config,
            code_revision="fixture",
            raw_root=raw_root,
            artifact_store=store,
        )


def test_cli_defaults_and_concise_profile_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _build(_prepare(tmp_path))

    def fake_build(
        release: ResolvedReleaseManifest,
        config: ResolvedConfig,
        *,
        code_revision: str,
        raw_root: Path | None,
    ) -> ProfileBuildResult:
        del release, config, code_revision, raw_root
        return result

    monkeypatch.setattr(cli_module, "build_esci_profiles", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")

    exit_code = cli_module.main(["data", "build-esci-profiles"])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "Task-1 US: 9 query IDs, 18 judgments" in output.out
    assert "quarantined: 1 train query IDs" in output.out
    assert "development: 3 normalized-query groups" in output.out
    assert "profile artifact:" in output.out
    assert output.err == ""


def test_cli_profile_failure_is_one_line_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(arguments: Namespace) -> int:
        del arguments
        raise ProfileBuildError("fixture cohort failed")

    monkeypatch.setattr(cli_module, "_run_build_esci_profiles", fail)

    exit_code = cli_module.main(["data", "build-esci-profiles"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err.strip() == "error: fixture cohort failed"
    assert "Traceback" not in output.err
