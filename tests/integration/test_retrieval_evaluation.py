"""Persisted fixed-cohort sparse/dense/RRF evaluation tests for Goldfish 009."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

import market_rank.cli as cli_module
import market_rank.evaluation.retrieval as evaluation_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.evaluation.retrieval import (
    AGGREGATE_METRICS_FILENAME,
    CANDIDATE_DIRECTORY,
    COMPARISON_METRICS_FILENAME,
    QUERY_METRIC_DIRECTORY,
    RETRIEVAL_EVALUATION_FILENAME,
    HybridResourceError,
    RetrievalEvaluationBuildResult,
    build_retrieval_evaluation,
    load_retrieval_evaluation_manifest,
)
from market_rank.retrieval.dense import build_dense_index
from market_rank.retrieval.sparse import build_sparse_index
from tests.integration.test_dense_retrieval import HashEncoder
from tests.integration.test_esci_foundation import _build as _build_foundation
from tests.integration.test_esci_foundation import _prepare_foundation


def _prepare_hybrid(
    tmp_path: Path,
    *,
    reverse: bool = False,
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore]:
    prepared = _prepare_foundation(tmp_path, reverse=reverse)
    _build_foundation(prepared)
    release, config, _, store = prepared
    build_sparse_index(release, config, code_revision="fixture", artifact_store=store)
    build_dense_index(
        release,
        config,
        code_revision="fixture",
        artifact_store=store,
        encoder=HashEncoder(),
    )
    return release, config, store


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore],
    *,
    profile: evaluation_module.Profile = "development",
) -> RetrievalEvaluationBuildResult:
    release, config, store = prepared
    return build_retrieval_evaluation(
        release,
        config,
        code_revision="fixture",
        profile=profile,
        artifact_store=store,
        dense_encoder=HashEncoder(),
    )


def _read_partitions(root: Path, directory: str) -> pl.DataFrame:
    return pl.read_parquet(str(root / directory / "*.parquet"))


def test_build_persists_three_parent_lineage_candidates_metrics_cis_and_slices(
    tmp_path: Path,
) -> None:
    result = _build(_prepare_hybrid(tmp_path))
    artifact = result.artifact
    manifest = result.manifest
    payload_paths = {item.relative_path for item in artifact.manifest.files}

    assert not result.reused
    assert artifact.manifest.artifact_type == "retrieval-evaluation"
    assert tuple(item.artifact_id.split("/")[0] for item in artifact.manifest.dependencies) == (
        "data-foundation",
        "dense-index",
        "sparse-index",
    )
    assert RETRIEVAL_EVALUATION_FILENAME in payload_paths
    assert AGGREGATE_METRICS_FILENAME in payload_paths
    assert COMPARISON_METRICS_FILENAME in payload_paths
    assert any(path.startswith(f"{CANDIDATE_DIRECTORY}/part-") for path in payload_paths)
    assert any(path.startswith(f"{QUERY_METRIC_DIRECTORY}/part-") for path in payload_paths)
    assert manifest.query_count == 4
    assert manifest.normalized_query_groups == 3
    assert tuple(stage.stage for stage in manifest.stages) == ("sparse", "dense", "hybrid")
    assert (
        load_retrieval_evaluation_manifest(artifact.path / RETRIEVAL_EVALUATION_FILENAME)
        == manifest
    )


def test_candidate_union_is_deduplicated_bounded_and_preserves_source_provenance(
    tmp_path: Path,
) -> None:
    result = _build(_prepare_hybrid(tmp_path))
    candidates = _read_partitions(result.artifact.path, CANDIDATE_DIRECTORY)
    hybrid = candidates.filter(pl.col("stage") == "hybrid")

    assert not candidates.is_empty()
    assert candidates["protocol"].unique().to_list() == ["retrieval_catalog_task1_us_v1"]
    assert all(count <= 200 for count in hybrid.group_by("query_id").len()["len"].to_list())
    assert hybrid.select(pl.struct("query_id", "product_id").n_unique()).item() == hybrid.height
    for group in hybrid.partition_by("query_id"):
        assert group.sort("rank")["rank"].to_list() == list(range(1, group.height + 1))
    shared = hybrid.filter(pl.col("source_count") == 2)
    assert shared.height > 0
    assert shared["sparse_rank"].null_count() == 0
    assert shared["dense_rank"].null_count() == 0
    assert shared["sparse_index_id"].null_count() == 0
    assert shared["dense_index_id"].null_count() == 0
    one_source = hybrid.filter(pl.col("source_count") == 1)
    assert (one_source["sparse_rank"].is_null() ^ one_source["dense_rank"].is_null()).all()


def test_all_stages_use_identical_query_cohort_and_only_retrieval_safe_metrics(
    tmp_path: Path,
) -> None:
    result = _build(_prepare_hybrid(tmp_path))
    metrics = _read_partitions(result.artifact.path, QUERY_METRIC_DIRECTORY)

    expected_rows = result.manifest.query_count * 3 * 2 * 2 * 5
    assert metrics.height == expected_rows
    stage_queries = {
        stage: set(metrics.filter(pl.col("stage") == stage)["query_id"].to_list())
        for stage in ("sparse", "dense", "hybrid")
    }
    assert stage_queries["sparse"] == stage_queries["dense"] == stage_queries["hybrid"]
    assert set(metrics["metric"].unique()) == {
        "judged_recall",
        "exact_hit",
        "judged_mrr",
        "known_judgment_coverage",
        "unjudged_rate",
    }
    assert not set(metrics["metric"].unique()) & {"precision", "map", "ndcg"}
    assert set(metrics["threshold_id"].unique()) == {"exact", "exact_substitute"}
    assert set(metrics["cutoff"].unique()) == {10, 100}


def test_aggregate_slices_and_paired_best_single_comparisons_are_complete(
    tmp_path: Path,
) -> None:
    result = _build(_prepare_hybrid(tmp_path))
    aggregate = pl.read_parquet(result.artifact.path / AGGREGATE_METRICS_FILENAME)
    comparisons = pl.read_parquet(result.artifact.path / COMPARISON_METRICS_FILENAME)

    assert set(aggregate["slice_dimension"].unique()) == {
        "all",
        "query_length",
        "source",
        "project_split",
        "exact_presence",
    }
    assert (aggregate["ci95_lower"] <= aggregate["ci95_upper"]).all()
    assert aggregate["bootstrap_replicates"].unique().to_list() == [1000]
    assert aggregate["bootstrap_method"].unique().to_list() == ["normalized-query-group-v1"]
    assert set(comparisons["comparison_id"].unique()) == {
        "hybrid_minus_sparse",
        "hybrid_minus_dense",
        "hybrid_minus_best_single",
    }
    assert (
        comparisons["win_count"] + comparisons["tie_count"] + comparisons["loss_count"]
        == result.manifest.query_count
    ).all()
    assert set(comparisons["selected_baseline_stage"].unique()) <= {"sparse", "dense"}


def test_raw_row_reordering_preserves_candidate_scores_metrics_and_bootstrap_outputs(
    tmp_path: Path,
) -> None:
    first = _build(_prepare_hybrid(tmp_path / "first"))
    reordered = _build(_prepare_hybrid(tmp_path / "reordered", reverse=True))
    first_candidates = _read_partitions(first.artifact.path, CANDIDATE_DIRECTORY).drop("latency_ms")
    reordered_candidates = _read_partitions(reordered.artifact.path, CANDIDATE_DIRECTORY).drop(
        "latency_ms"
    )

    assert reordered_candidates.equals(first_candidates)
    assert _read_partitions(reordered.artifact.path, QUERY_METRIC_DIRECTORY).equals(
        _read_partitions(first.artifact.path, QUERY_METRIC_DIRECTORY)
    )
    assert pl.read_parquet(reordered.artifact.path / AGGREGATE_METRICS_FILENAME).equals(
        pl.read_parquet(first.artifact.path / AGGREGATE_METRICS_FILENAME)
    )
    assert pl.read_parquet(reordered.artifact.path / COMPARISON_METRICS_FILENAME).equals(
        pl.read_parquet(first.artifact.path / COMPARISON_METRICS_FILENAME)
    )


def test_portfolio_profile_uses_its_complete_larger_cohort(tmp_path: Path) -> None:
    result = _build(_prepare_hybrid(tmp_path), profile="portfolio")
    metrics = _read_partitions(result.artifact.path, QUERY_METRIC_DIRECTORY)

    assert result.manifest.profile == "portfolio"
    assert result.manifest.query_count == 7
    assert metrics["query_id"].n_unique() == 7
    assert result.artifact.manifest.profile == "portfolio"


def test_combined_sparse_dense_rss_over_limit_blocks_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_hybrid(tmp_path)
    observed = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024 + 1
    monkeypatch.setattr(evaluation_module, "_peak_rss_bytes", lambda: observed)

    with pytest.raises(HybridResourceError) as caught:
        _build(prepared)

    assert not caught.value.measurement.passed
    assert caught.value.measurement.query_count == 0
    assert not any((tmp_path / "artifacts" / "retrieval-evaluation").rglob("_SUCCESS"))


def test_evaluation_phase_rss_over_limit_rolls_back_completed_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_hybrid(tmp_path)
    limit = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024
    observations = iter((limit - 1, limit + 1))
    monkeypatch.setattr(evaluation_module, "_peak_rss_bytes", lambda: next(observations))

    with pytest.raises(HybridResourceError) as caught:
        _build(prepared)

    assert caught.value.measurement.query_count == 4
    assert caught.value.measurement.evaluation_artifact_bytes > 0
    assert not any((tmp_path / "artifacts" / "retrieval-evaluation").rglob("_SUCCESS"))


def test_compatible_evaluation_is_reused_without_loading_query_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_hybrid(tmp_path)
    first = _build(prepared)

    def forbid_encoder(config: ResolvedConfig) -> object:
        del config
        raise AssertionError("compatible evaluation should be reused before encoder load")

    monkeypatch.setattr(evaluation_module, "SentenceTransformerEncoder", forbid_encoder)
    second = build_retrieval_evaluation(
        prepared[0],
        prepared[1],
        code_revision="fixture",
        profile="development",
        artifact_store=prepared[2],
    )

    assert second.reused
    assert second.artifact == first.artifact
    assert second.manifest == first.manifest


def test_cli_outputs_concise_hybrid_summary_and_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _build(_prepare_hybrid(tmp_path))

    def fake_build(
        release: ResolvedReleaseManifest,
        config: ResolvedConfig,
        *,
        code_revision: str,
        profile: evaluation_module.Profile | None,
    ) -> RetrievalEvaluationBuildResult:
        del release, config, code_revision, profile
        return result

    monkeypatch.setattr(cli_module, "build_retrieval_evaluation", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")
    assert cli_module.main(["retrieval", "evaluate-hybrid"]) == 0
    output = capsys.readouterr()
    assert "retrieval cohort: development, 4 queries" in output.out
    assert "paired comparison rows" in output.out
    assert "combined resource:" in output.out
    assert output.err == ""

    def fail(arguments: Namespace) -> int:
        del arguments
        raise evaluation_module.RetrievalEvaluationBuildError("fixture hybrid failure")

    monkeypatch.setattr(cli_module, "_run_evaluate_hybrid", fail)
    assert cli_module.main(["retrieval", "evaluate-hybrid"]) == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    assert failed.err.strip() == "error: fixture hybrid failure"
    assert "Traceback" not in failed.err
