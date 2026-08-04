"""Persisted Goldfish 010 query-state and ranking-feature integration tests."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

import market_rank.cli as cli_module
import market_rank.features.artifact as feature_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.features.artifact import (
    CANDIDATE_MATRIX_DIRECTORY,
    CLOSED_MATRIX_DIRECTORY,
    DISTRIBUTION_REPORT_FILENAME,
    FEATURE_ARTIFACT_FILENAME,
    FEATURE_REGISTRY_FILENAME,
    FEATURE_STATE_FILENAME,
    LEAKAGE_REPORT_FILENAME,
    PARITY_FIXTURES_FILENAME,
    PARSED_QUERIES_FILENAME,
    RankingFeatureBuildResult,
    RankingFeatureResourceError,
    build_ranking_features,
    load_feature_state,
    load_ranking_feature_manifest,
)
from market_rank.features.registry import FEATURE_NAMES
from tests.integration.test_dense_retrieval import HashEncoder
from tests.integration.test_retrieval_evaluation import _build as _build_evaluation
from tests.integration.test_retrieval_evaluation import _prepare_hybrid


def _prepare(
    tmp_path: Path,
    *,
    profile: feature_module.Profile = "development",
    reverse: bool = False,
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore]:
    prepared = _prepare_hybrid(tmp_path, reverse=reverse)
    _build_evaluation(prepared, profile=profile)
    return prepared


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore],
    *,
    profile: feature_module.Profile = "development",
) -> RankingFeatureBuildResult:
    release, config, store = prepared
    return build_ranking_features(
        release,
        config,
        code_revision="fixture",
        profile=profile,
        artifact_store=store,
        dense_encoder=HashEncoder(),
    )


def _read_matrix(root: Path, directory: str) -> pl.DataFrame:
    return pl.read_parquet(str(root / directory / "*.parquet"))


def test_build_persists_four_parent_state_registry_matrices_and_reports(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path))
    paths = {item.relative_path for item in result.artifact.manifest.files}

    assert not result.reused
    assert result.artifact.manifest.artifact_type == "ranking-features"
    assert tuple(
        item.artifact_id.split("/")[0] for item in result.artifact.manifest.dependencies
    ) == (
        "data-foundation",
        "dense-index",
        "retrieval-evaluation",
        "sparse-index",
    )
    assert {
        FEATURE_ARTIFACT_FILENAME,
        FEATURE_REGISTRY_FILENAME,
        FEATURE_STATE_FILENAME,
        PARSED_QUERIES_FILENAME,
        LEAKAGE_REPORT_FILENAME,
        DISTRIBUTION_REPORT_FILENAME,
        PARITY_FIXTURES_FILENAME,
    } <= paths
    assert any(path.startswith(f"{CLOSED_MATRIX_DIRECTORY}/part-") for path in paths)
    assert any(path.startswith(f"{CANDIDATE_MATRIX_DIRECTORY}/part-") for path in paths)
    assert result.manifest.feature_count == len(FEATURE_NAMES)
    assert result.manifest.query_count == 4
    assert result.manifest.closed_rows > 0
    assert result.manifest.closed_excluded_outside_catalog == 0
    assert result.manifest.candidate_rows > 0
    assert (
        load_ranking_feature_manifest(result.artifact.path / FEATURE_ARTIFACT_FILENAME)
        == result.manifest
    )
    assert load_feature_state(result.artifact.path / FEATURE_STATE_FILENAME) == result.state


def test_closed_matrix_is_complete_labeled_and_candidate_matrix_is_label_free(
    tmp_path: Path,
) -> None:
    result = _build(_prepare(tmp_path))
    closed = _read_matrix(result.artifact.path, CLOSED_MATRIX_DIRECTORY)
    candidate = _read_matrix(result.artifact.path, CANDIDATE_MATRIX_DIRECTORY)

    assert {"label_id", "gain"} <= set(closed.columns)
    assert not {"label_id", "gain", "esci_label"} & set(candidate.columns)
    assert tuple(column for column in closed.columns if column in FEATURE_NAMES) == FEATURE_NAMES
    assert tuple(column for column in candidate.columns if column in FEATURE_NAMES) == FEATURE_NAMES
    assert closed.select(pl.struct("query_id", "product_id").n_unique()).item() == closed.height
    assert (
        candidate.select(pl.struct("query_id", "product_id").n_unique()).item() == candidate.height
    )
    assert closed["label_id"].null_count() == 0
    assert closed["gain"].null_count() == 0
    for column in (
        "bm25_rank_fraction",
        "dense_rank_fraction",
        "closed_rrf_rank_fraction",
    ):
        assert closed[column].is_between(0.0, 1.0).all()
        assert candidate[column].is_between(0.0, 1.0).all()


def test_parser_state_train_only_codes_leakage_and_distribution_evidence(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path))
    leakage = json.loads((result.artifact.path / LEAKAGE_REPORT_FILENAME).read_text())
    distribution = pl.read_parquet(result.artifact.path / DISTRIBUTION_REPORT_FILENAME)
    parsed = pl.read_parquet(result.artifact.path / PARSED_QUERIES_FILENAME)

    assert result.state.categorical_fit_project_split == "train"
    assert result.state.missing_category_code == 0
    assert result.state.unknown_category_code == 1
    assert leakage["fit_project_splits"] == ["train"]
    assert all(check["passed"] for check in leakage["checks"])
    assert set(distribution["population"]) == {"closed_judged", "retrieved_union"}
    assert distribution.height == 2 * len(FEATURE_NAMES)
    assert distribution["nulls"].sum() == 0
    assert parsed.height == result.manifest.query_count
    assert parsed["parser_state_sha256"].unique().to_list() == [
        result.state.parser_state.state_sha256
    ]


def test_parity_fixtures_have_complete_shared_formula_vectors_and_hashes(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path))
    parity = pl.read_parquet(result.artifact.path / PARITY_FIXTURES_FILENAME)

    assert parity.height == result.manifest.parity_fixture_rows
    assert parity.height > 0
    assert tuple(column for column in parity.columns if column in FEATURE_NAMES) == FEATURE_NAMES
    assert parity["feature_vector_sha256"].str.len_chars().unique().to_list() == [64]
    assert parity.select(FEATURE_NAMES).null_count().sum_horizontal().item() == 0


def test_feature_resource_overage_rolls_back_without_success_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    over = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024 + 1
    monkeypatch.setattr(feature_module, "_peak_rss_bytes", lambda: over)

    with pytest.raises(RankingFeatureResourceError):
        _build(prepared)

    assert not any((tmp_path / "artifacts" / "ranking-features").rglob("_SUCCESS"))


def test_final_resource_overage_discards_completed_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    limit = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024
    observations = iter((limit - 1, limit + 1))
    monkeypatch.setattr(feature_module, "_peak_rss_bytes", lambda: next(observations))

    with pytest.raises(RankingFeatureResourceError) as caught:
        _build(prepared)

    assert caught.value.measurement.artifact_payload_bytes > 0
    assert not any((tmp_path / "artifacts" / "ranking-features").rglob("_SUCCESS"))


def test_raw_input_reordering_preserves_state_matrices_and_reports(tmp_path: Path) -> None:
    first = _build(_prepare(tmp_path / "first"))
    reordered = _build(_prepare(tmp_path / "reordered", reverse=True))

    assert first.state == reordered.state
    for directory in (CLOSED_MATRIX_DIRECTORY, CANDIDATE_MATRIX_DIRECTORY):
        assert _read_matrix(first.artifact.path, directory).equals(
            _read_matrix(reordered.artifact.path, directory)
        )
    for filename in (
        PARSED_QUERIES_FILENAME,
        DISTRIBUTION_REPORT_FILENAME,
        PARITY_FIXTURES_FILENAME,
    ):
        assert pl.read_parquet(first.artifact.path / filename).equals(
            pl.read_parquet(reordered.artifact.path / filename)
        )


def test_portfolio_profile_materializes_its_complete_larger_cohort(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path, profile="portfolio"), profile="portfolio")
    closed = _read_matrix(result.artifact.path, CLOSED_MATRIX_DIRECTORY)

    assert result.manifest.profile == "portfolio"
    assert result.manifest.query_count == 7
    assert result.manifest.closed_excluded_outside_catalog == 1
    assert closed["query_id"].n_unique() == 7
    assert result.artifact.manifest.profile == "portfolio"


def test_compatible_feature_artifact_is_reused_before_encoder_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    first = _build(prepared)

    def forbid_encoder(config: ResolvedConfig) -> object:
        del config
        raise AssertionError("compatible feature artifact should be reused")

    monkeypatch.setattr(feature_module, "SentenceTransformerEncoder", forbid_encoder)
    second = build_ranking_features(
        prepared[0],
        prepared[1],
        code_revision="fixture",
        profile="development",
        artifact_store=prepared[2],
    )

    assert second.reused
    assert second.artifact == first.artifact
    assert second.manifest == first.manifest


def test_cli_outputs_bounded_feature_summary_and_failure(
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
        profile: feature_module.Profile | None,
    ) -> RankingFeatureBuildResult:
        del release, config, code_revision, profile
        return result

    monkeypatch.setattr(cli_module, "build_ranking_features", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")
    assert cli_module.main(["features", "build-ranking"]) == 0
    output = capsys.readouterr()
    assert "feature cohort: development, 4 queries" in output.out
    assert "label-free candidate rows" in output.out
    assert "resource: peak RSS" in output.out
    assert output.err == ""

    def fail(arguments: Namespace) -> int:
        del arguments
        raise feature_module.RankingFeatureBuildError("fixture feature failure")

    monkeypatch.setattr(cli_module, "_run_build_ranking_features", fail)
    assert cli_module.main(["features", "build-ranking"]) == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    assert failed.err.strip() == "error: fixture feature failure"
    assert "Traceback" not in failed.err
