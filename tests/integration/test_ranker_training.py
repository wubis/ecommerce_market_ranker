"""Goldfish 011 exact-population LightGBM/LambdaMART artifact integration tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import market_rank.cli as cli_module
import market_rank.ranking.training as training_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.features.artifact import CLOSED_MATRIX_DIRECTORY
from market_rank.features.registry import FEATURE_NAMES
from market_rank.ranking.training import (
    EXPLANATION_SAMPLE_FILENAME,
    FEATURE_IMPORTANCE_FILENAME,
    LAMBDAMART_MODEL_FILENAME,
    POINTWISE_MODEL_FILENAME,
    POPULATION_AUDIT_FILENAME,
    RANKING_MODELS_FILENAME,
    RELOAD_PARITY_FILENAME,
    TRAINING_POPULATION_FILENAME,
    VALIDATION_HISTORY_FILENAME,
    RankingTrainingBuildResult,
    RankingTrainingResourceError,
    RankingTrainingValidationError,
    build_rankers,
    load_rankers,
    load_ranking_models_manifest,
)
from tests.integration.test_ranking_features import _build as _build_features
from tests.integration.test_ranking_features import _prepare as _prepare_features


def _prepare(
    tmp_path: Path,
    *,
    reverse: bool = False,
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore]:
    prepared = _prepare_features(tmp_path, profile="portfolio", reverse=reverse)
    _build_features(prepared, profile="portfolio")
    return prepared


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore],
) -> RankingTrainingBuildResult:
    return build_rankers(
        prepared[0],
        prepared[1],
        code_revision="fixture",
        profile="portfolio",
        artifact_store=prepared[2],
    )


def test_build_persists_exact_feature_parent_population_and_two_models(tmp_path: Path) -> None:
    result = _build(_prepare(tmp_path))
    paths = {item.relative_path for item in result.artifact.manifest.files}
    manifest = result.manifest

    assert not result.reused
    assert result.artifact.manifest.artifact_type == "ranking-models"
    assert len(result.artifact.manifest.dependencies) == 1
    assert result.artifact.manifest.dependencies[0].artifact_id.startswith("ranking-features/")
    assert {
        RANKING_MODELS_FILENAME,
        TRAINING_POPULATION_FILENAME,
        POPULATION_AUDIT_FILENAME,
        POINTWISE_MODEL_FILENAME,
        LAMBDAMART_MODEL_FILENAME,
        VALIDATION_HISTORY_FILENAME,
        FEATURE_IMPORTANCE_FILENAME,
        RELOAD_PARITY_FILENAME,
        EXPLANATION_SAMPLE_FILENAME,
    } == paths
    assert tuple(model.model_id for model in manifest.models) == ("pointwise", "lambdamart")
    assert tuple(model.objective for model in manifest.models) == (
        "regression_l2",
        "lambdarank",
    )
    assert manifest.feature_names == FEATURE_NAMES
    assert tuple(name for name, _ in manifest.feature_dtypes) == FEATURE_NAMES
    assert manifest.categorical_features == ("brand_code", "color_code")
    assert manifest.fallback_contract == "rrf-on-model-failure-v1"
    assert manifest.population.train.eligible_query_ids == (6,)
    assert manifest.population.train.group_sizes == (2,)
    assert manifest.population.train.excluded_too_few_rows == 1
    assert manifest.population.validation.eligible_query_groups == 2
    assert manifest.population.validation.group_sizes == (2, 2)
    assert load_ranking_models_manifest(result.artifact.path / RANKING_MODELS_FILENAME) == manifest


def test_training_evidence_uses_no_test_rows_and_covers_history_importance_parity(
    tmp_path: Path,
) -> None:
    result = _build(_prepare(tmp_path))
    root = result.artifact.path
    audit = pl.read_parquet(root / POPULATION_AUDIT_FILENAME)
    history = pl.read_parquet(root / VALIDATION_HISTORY_FILENAME)
    importance = pl.read_parquet(root / FEATURE_IMPORTANCE_FILENAME)
    parity = pl.read_parquet(root / RELOAD_PARITY_FILENAME)
    explanations = pl.read_parquet(root / EXPLANATION_SAMPLE_FILENAME)

    assert set(audit["project_split"]) == {"train", "validation"}
    assert "test" not in set(audit["project_split"])
    assert set(audit["exclusion_reason"]) == {"eligible", "too_few_rows"}
    assert set(history["model_id"]) == {"pointwise", "lambdamart"}
    assert set(history["metric"]) == {"ndcg"}
    assert set(history["cutoff"]) == {10, 20}
    for model in result.manifest.models:
        assert history.filter(pl.col("model_id") == model.model_id)["iteration"].max() == (
            model.trained_iterations
        )
    assert importance.height == 2 * len(FEATURE_NAMES)
    assert set(importance["feature"]) == set(FEATURE_NAMES)
    assert parity.height == result.manifest.reload_parity_rows
    assert parity["absolute_delta"].max() == 0.0
    assert (parity["trained_rank"] == parity["reloaded_rank"]).all()
    assert set(explanations["feature"]) == {*FEATURE_NAMES, "__bias__"}
    assert result.manifest.maximum_reload_prediction_delta == 0.0


def test_cold_loaded_models_preserve_feature_order_predict_and_stably_rank(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    result = _build(prepared)
    loaded = load_rankers(prepared[2], result.artifact.manifest.artifact_id)
    feature_artifact = prepared[2].load(result.manifest.feature_artifact_id)
    closed = pl.read_parquet(str(feature_artifact.path / CLOSED_MATRIX_DIRECTORY / "*.parquet"))
    query = closed.filter(pl.col("query_id") == 3).sort("product_id")
    matrix = np.ascontiguousarray(query.select(FEATURE_NAMES).to_numpy(), dtype=np.float32)
    product_ids = tuple(query["product_id"].to_list())

    for model_id in ("pointwise", "lambdamart"):
        predictions = loaded.predict(model_id, matrix)
        ranked = loaded.rank(model_id, product_ids, matrix)
        assert predictions.shape == (query.height,)
        assert np.isfinite(predictions).all()
        assert {item.product_id for item in ranked} == set(product_ids)
        assert sorted(item.one_based_rank for item in ranked) == list(range(1, query.height + 1))
    with pytest.raises(RankingTrainingValidationError, match="dimensions"):
        loaded.predict("lambdamart", np.zeros((1, 2), dtype=np.float32))


def test_model_payload_corruption_is_rejected_before_lightgbm_load(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    result = _build(prepared)
    model_path = result.artifact.path / LAMBDAMART_MODEL_FILENAME
    model_path.write_text(model_path.read_text(encoding="utf-8") + "\ncorrupt\n", encoding="utf-8")

    with pytest.raises(RankingTrainingValidationError, match="cannot load"):
        load_rankers(prepared[2], result.artifact.manifest.artifact_id)


def test_raw_input_order_preserves_population_model_text_and_numeric_reports(
    tmp_path: Path,
) -> None:
    first = _build(_prepare(tmp_path / "first"))
    reordered = _build(_prepare(tmp_path / "reordered", reverse=True))

    assert first.manifest.population == reordered.manifest.population
    for filename in (POINTWISE_MODEL_FILENAME, LAMBDAMART_MODEL_FILENAME):
        assert (first.artifact.path / filename).read_bytes() == (
            reordered.artifact.path / filename
        ).read_bytes()
    for report_filename in (
        POPULATION_AUDIT_FILENAME,
        VALIDATION_HISTORY_FILENAME,
        FEATURE_IMPORTANCE_FILENAME,
        RELOAD_PARITY_FILENAME,
        EXPLANATION_SAMPLE_FILENAME,
    ):
        assert pl.read_parquet(first.artifact.path / report_filename).equals(
            pl.read_parquet(reordered.artifact.path / report_filename)
        )


def test_matrix_resource_overage_blocks_before_model_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    over = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024 + 1
    monkeypatch.setattr(training_module, "_peak_rss_bytes", lambda: over)

    with pytest.raises(RankingTrainingResourceError):
        _build(prepared)

    assert not any((tmp_path / "artifacts" / "ranking-models").rglob("_SUCCESS"))


def test_reload_resource_overage_discards_completed_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    limit = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024
    observations = iter((limit - 1, limit - 1, limit + 1))
    monkeypatch.setattr(training_module, "_peak_rss_bytes", lambda: next(observations))

    with pytest.raises(RankingTrainingResourceError) as caught:
        _build(prepared)

    assert caught.value.measurement.artifact_payload_bytes > 0
    assert not any((tmp_path / "artifacts" / "ranking-models").rglob("_SUCCESS"))


def test_compatible_model_artifact_reuses_before_matrix_read_or_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    first = _build(prepared)

    def forbid_matrix(feature: object) -> pl.DataFrame:
        del feature
        raise AssertionError("compatible rankers should reuse before reading matrices")

    monkeypatch.setattr(training_module, "_read_training_rows", forbid_matrix)
    second = _build(prepared)

    assert second.reused
    assert second.artifact == first.artifact
    assert second.manifest == first.manifest


def test_cli_outputs_bounded_training_summary_and_failure(
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
        profile: training_module.Profile | None,
    ) -> RankingTrainingBuildResult:
        del release, config, code_revision, profile
        return result

    monkeypatch.setattr(cli_module, "build_rankers", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")
    assert cli_module.main(["ranking", "train", "--profile", "portfolio"]) == 0
    output = capsys.readouterr()
    assert "training population: portfolio" in output.out
    assert "pointwise: iteration" in output.out
    assert "lambdamart: iteration" in output.out
    assert "resource: peak RSS" in output.out
    assert output.err == ""

    def fail(arguments: Namespace) -> int:
        del arguments
        raise training_module.RankingTrainingBuildError("fixture training failure")

    monkeypatch.setattr(cli_module, "_run_train_rankers", fail)
    assert cli_module.main(["ranking", "train"]) == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    assert failed.err.strip() == "error: fixture training failure"
    assert "Traceback" not in failed.err
