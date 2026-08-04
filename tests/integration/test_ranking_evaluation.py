"""Goldfish 012 protocol, ablation, experiment, and promotion integration tests."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

import market_rank.cli as cli_module
import market_rank.evaluation.ranking as ranking_module
from market_rank.artifacts import ArtifactStore, ArtifactValidationError
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.evaluation.metrics import CLOSED_POOL_PROTOCOL, END_TO_END_PROTOCOL
from market_rank.evaluation.ranking import (
    ACTIVE_RELEVANCE_FILENAME,
    COMPARISONS_FILENAME,
    FAILURE_ANALYSIS_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    QUERY_METRICS_FILENAME,
    RANKING_EVALUATION_FILENAME,
    RUN_FILENAME,
    RankingEvaluationBuildResult,
    RankingEvaluationResourceError,
    build_ranking_evaluation,
    load_active_relevance_contract,
    load_ranking_evaluation_manifest,
)
from tests.integration.test_ranker_training import _build as _build_rankers
from tests.integration.test_ranker_training import _prepare as _prepare_rankers


def _prepare(
    tmp_path: Path, *, reverse: bool = False
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore]:
    prepared = _prepare_rankers(tmp_path, reverse=reverse)
    _build_rankers(prepared)
    return prepared


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore],
) -> RankingEvaluationBuildResult:
    return build_ranking_evaluation(
        prepared[0],
        prepared[1],
        code_revision="fixture",
        profile="portfolio",
        artifact_store=prepared[2],
    )


def test_build_persists_protocol_reports_experiment_and_one_active_contract(
    tmp_path: Path,
) -> None:
    result = _build(_prepare(tmp_path))
    manifest = result.manifest
    paths = {item.relative_path for item in result.artifact.manifest.files}

    assert not result.reused
    assert result.artifact.manifest.artifact_type == "ranking-evaluation"
    assert tuple(
        dependency.artifact_id.split("/")[0] for dependency in result.artifact.manifest.dependencies
    ) == ("ranking-models",)
    assert paths == {
        ACTIVE_RELEVANCE_FILENAME,
        COMPARISONS_FILENAME,
        FAILURE_ANALYSIS_FILENAME,
        METRICS_FILENAME,
        PREDICTIONS_FILENAME,
        QUERY_METRICS_FILENAME,
        RANKING_EVALUATION_FILENAME,
        RUN_FILENAME,
    }
    assert manifest.protocols == (CLOSED_POOL_PROTOCOL, END_TO_END_PROTOCOL)
    assert manifest.selection_split == "validation"
    assert not manifest.test_evaluated
    assert tuple(candidate.stage for candidate in manifest.active_relevance.candidates) == (
        "rrf",
        "pointwise",
        "lambdamart",
    )
    assert manifest.active_relevance.selected_stage in {"rrf", "pointwise", "lambdamart"}
    assert (
        load_active_relevance_contract(result.artifact.path / ACTIVE_RELEVANCE_FILENAME)
        == manifest.active_relevance
    )
    assert (
        load_ranking_evaluation_manifest(result.artifact.path / RANKING_EVALUATION_FILENAME)
        == manifest
    )


def test_closed_and_end_to_end_protocols_keep_metrics_and_test_quarantined(
    tmp_path: Path,
) -> None:
    result = _build(_prepare(tmp_path))
    metrics = pl.read_parquet(result.artifact.path / QUERY_METRICS_FILENAME)
    predictions = pl.read_parquet(result.artifact.path / PREDICTIONS_FILENAME)

    assert metrics["project_split"].unique().to_list() == ["validation"]
    assert predictions["project_split"].unique().to_list() == ["validation"]
    closed = metrics.filter(pl.col("protocol") == CLOSED_POOL_PROTOCOL)
    diagnostic = metrics.filter(pl.col("protocol") == END_TO_END_PROTOCOL)
    assert {"ndcg_official_gain", "precision", "map", "mrr", "exact_hit"} == set(closed["metric"])
    assert set(diagnostic["metric"]) == {
        "judged_recall",
        "exact_hit",
        "judged_mrr",
        "known_judgment_coverage",
        "unjudged_rate",
    }
    assert not {"ndcg_official_gain", "precision", "map"} & set(diagnostic["metric"])
    for protocol in (CLOSED_POOL_PROTOCOL, END_TO_END_PROTOCOL):
        cohort = metrics.filter(pl.col("protocol") == protocol)
        stage_queries = {
            stage: set(cohort.filter(pl.col("stage") == stage)["query_id"].to_list())
            for stage in ("rrf", "pointwise", "lambdamart")
        }
        assert stage_queries["rrf"] == stage_queries["pointwise"] == stage_queries["lambdamart"]
    active_counts = predictions.group_by("protocol", "query_id", "product_id").agg(
        pl.col("active_relevance").sum().alias("active_count")
    )
    assert active_counts["active_count"].unique().to_list() == [1]


def test_grouped_intervals_slices_ablations_failures_and_run_lineage(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path))
    aggregate = pl.read_parquet(result.artifact.path / METRICS_FILENAME)
    comparisons = pl.read_parquet(result.artifact.path / COMPARISONS_FILENAME)
    failures = pl.read_parquet(result.artifact.path / FAILURE_ANALYSIS_FILENAME)
    run = json.loads((result.artifact.path / RUN_FILENAME).read_text(encoding="utf-8"))

    assert set(aggregate["slice_dimension"]) == {
        "all",
        "query_length",
        "lexical_specificity",
        "brand_presence",
        "color_presence",
        "model_presence",
        "compatibility_presence",
        "source",
        "project_split",
        "judgment_composition",
    }
    assert (aggregate["ci95_lower"] <= aggregate["ci95_upper"]).all()
    assert aggregate["bootstrap_method"].unique().to_list() == ["normalized-query-group-v1"]
    assert set(comparisons["ablation_id"]) == {"ABL-04", "ABL-05", "E2E-01", "E2E-02"}
    assert set(failures["ablation_id"]) == {"ABL-04", "ABL-05"}
    assert failures["selection_method"].unique().to_list() == ["largest-absolute-paired-delta-v1"]
    assert [item["ablation_id"] for item in run["ablations"]] == [
        "ABL-01",
        "ABL-02",
        "ABL-03",
        "ABL-04",
        "ABL-05",
    ]
    assert [item["status"] for item in run["ablations"]] == [
        "inherited",
        "inherited",
        "inherited",
        "evaluated",
        "evaluated",
    ]
    assert run["active_relevance"] == result.manifest.active_relevance.model_dump(mode="json")


def test_raw_input_order_preserves_numeric_reports_predictions_and_promotion(
    tmp_path: Path,
) -> None:
    first = _build(_prepare(tmp_path / "first"))
    reordered = _build(_prepare(tmp_path / "reordered", reverse=True))

    first_active = first.manifest.active_relevance.model_dump(
        exclude={"ranking_models_manifest_sha256"}
    )
    reordered_active = reordered.manifest.active_relevance.model_dump(
        exclude={"ranking_models_manifest_sha256"}
    )
    assert first_active == reordered_active
    for filename in (
        PREDICTIONS_FILENAME,
        QUERY_METRICS_FILENAME,
        METRICS_FILENAME,
        COMPARISONS_FILENAME,
        FAILURE_ANALYSIS_FILENAME,
    ):
        assert pl.read_parquet(first.artifact.path / filename).equals(
            pl.read_parquet(reordered.artifact.path / filename)
        )


def test_initial_resource_overage_blocks_before_evaluation_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepare(tmp_path)
    over = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024 + 1
    monkeypatch.setattr(ranking_module, "_peak_rss_bytes", lambda: over)

    with pytest.raises(RankingEvaluationResourceError):
        _build(prepared)

    assert not any((tmp_path / "artifacts" / "ranking-evaluation").rglob("_SUCCESS"))


def test_promotion_resource_overage_discards_completed_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepare(tmp_path)
    limit = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024
    observations = iter((limit - 1, limit - 1, limit + 1))
    monkeypatch.setattr(ranking_module, "_peak_rss_bytes", lambda: next(observations))

    with pytest.raises(RankingEvaluationResourceError) as caught:
        _build(prepared)

    assert caught.value.measurement.artifact_payload_bytes > 0
    assert not any((tmp_path / "artifacts" / "ranking-evaluation").rglob("_SUCCESS"))


def test_compatible_evaluation_reuses_before_matrix_read_or_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepare(tmp_path)
    first = _build(prepared)

    def forbid_inputs(dependencies: object, config: object) -> object:
        del dependencies, config
        raise AssertionError("compatible evaluation should reuse before matrix reads")

    monkeypatch.setattr(ranking_module, "_read_evaluation_inputs", forbid_inputs)
    second = _build(prepared)

    assert second.reused
    assert second.artifact == first.artifact
    assert second.manifest == first.manifest


def test_active_contract_payload_corruption_is_rejected_before_reuse(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    first = _build(prepared)
    active_path = first.artifact.path / ACTIVE_RELEVANCE_FILENAME
    active_path.write_text(active_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="integrity"):
        _build(prepared)


def test_cli_outputs_bounded_champion_summary_and_failure(
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
        profile: ranking_module.Profile | None,
    ) -> RankingEvaluationBuildResult:
        del release, config, code_revision, profile
        return result

    monkeypatch.setattr(cli_module, "build_ranking_evaluation", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")
    assert cli_module.main(["ranking", "evaluate", "--profile", "portfolio"]) == 0
    output = capsys.readouterr()
    assert "ranking evaluation: portfolio" in output.out
    assert "rrf: NDCG@10=" in output.out
    assert "active relevance:" in output.out
    assert "test evaluated=false" in output.out
    assert output.err == ""

    def fail(arguments: Namespace) -> int:
        del arguments
        raise ranking_module.RankingEvaluationBuildError("fixture evaluation failure")

    monkeypatch.setattr(cli_module, "_run_evaluate_rankers", fail)
    assert cli_module.main(["ranking", "evaluate"]) == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    assert failed.err.strip() == "error: fixture evaluation failure"
    assert "Traceback" not in failed.err
