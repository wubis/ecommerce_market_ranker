"""Goldfish 016 frozen project-test and final portfolio-package integration tests."""

from __future__ import annotations

import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl
import pytest

from market_rank.artifacts import ArtifactValidationError
from market_rank.data.foundation import COMPACT_CATALOG_ID
from market_rank.evaluation.metrics import (
    CLOSED_POOL_PROTOCOL,
    END_TO_END_PROTOCOL,
    RETRIEVAL_PROTOCOL,
)
from market_rank.evaluation.ranking import METRICS_FILENAME, PREDICTIONS_FILENAME
from market_rank.portfolio import (
    FINAL_REPORT_FILENAME,
    PORTFOLIO_RELEASE_FILENAME,
    RANKING_TABLE_FILENAME,
    RETRIEVAL_TABLE_FILENAME,
    PortfolioValidationError,
    ReproductionCommand,
    ReproductionEvidence,
    build_portfolio_release,
    load_portfolio_manifest,
)
from market_rank.qualification import build_release_qualification
from tests.integration.test_qualification import _hardware, _loader
from tests.integration.test_serving import _build, _prepare


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _png(width: int = 1200, height: int = 800) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(pixels))
        + _chunk(b"IEND", b"")
    )


def _reproduction(config_sha256: str, code_revision: str) -> ReproductionEvidence:
    command_ids: tuple[Literal["lock", "format", "lint", "types", "tests"], ...] = (
        "lock",
        "format",
        "lint",
        "types",
        "tests",
    )
    commands = tuple(
        ReproductionCommand(
            command_id=command_id,
            command=f"fixture {command_id}",
            duration_seconds=0.01,
        )
        for command_id in command_ids
    )
    return ReproductionEvidence(
        config_sha256=config_sha256,
        code_revision=code_revision,
        created_utc=datetime(2026, 8, 3, tzinfo=UTC),
        commands=commands,  # type: ignore[arg-type]
        test_count=258,
    )


def test_final_portfolio_evaluates_test_once_packages_all_evidence_and_reuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    prepared = _prepare(tmp_path)
    bundle = _build(prepared)
    revision = "a" * 40
    qualification = build_release_qualification(
        prepared[1],
        bundle.artifact.manifest.artifact_id,
        code_revision=revision,
        background_conditions="AC power; fixture process only",
        artifact_store=prepared[2],
        hardware=_hardware(),
        runtime_loader=_loader,
    )
    reproduction_path = tmp_path / "reproduction.json"
    reproduction_path.write_text(
        _reproduction(prepared[1].sha256, revision).model_dump_json(), encoding="utf-8"
    )
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    for filename in prepared[1].config.portfolio_report.screenshot_filenames:
        (screenshots / filename).write_bytes(_png())

    missing = screenshots / "dataset-limitations.png"
    missing.rename(screenshots / "held.png")
    with pytest.raises(PortfolioValidationError, match="regular file"):
        build_portfolio_release(
            prepared[1],
            ranking_evaluation_id=bundle.manifest.components[5].artifact_id,
            serving_bundle_id=bundle.artifact.manifest.artifact_id,
            qualification_id=qualification.artifact.manifest.artifact_id,
            reproduction_evidence_path=reproduction_path,
            screenshots_dir=screenshots,
            code_revision=revision,
            artifact_store=prepared[2],
        )
    (screenshots / "held.png").rename(missing)

    result = build_portfolio_release(
        prepared[1],
        ranking_evaluation_id=bundle.manifest.components[5].artifact_id,
        serving_bundle_id=bundle.artifact.manifest.artifact_id,
        qualification_id=qualification.artifact.manifest.artifact_id,
        reproduction_evidence_path=reproduction_path,
        screenshots_dir=screenshots,
        code_revision=revision,
        artifact_store=prepared[2],
    )
    assert not result.reused
    assert result.artifact.manifest.artifact_type == "portfolio-release"
    assert len(result.artifact.manifest.dependencies) == 3
    assert result.manifest.final_evaluation_split == "test"
    assert result.manifest.selection_split == "validation"
    assert result.manifest.catalog_id == COMPACT_CATALOG_ID
    assert result.manifest.catalog_candidate_products == (
        result.manifest.catalog_required_judged_products
        + result.manifest.catalog_distractor_products
    )
    assert result.manifest.catalog_usable_products <= result.manifest.catalog_candidate_products
    assert result.manifest.test_query_count > 0
    assert len(result.manifest.screenshots) == 3
    assert (result.artifact.path / FINAL_REPORT_FILENAME).is_file()
    assert (
        load_portfolio_manifest(result.artifact.path / PORTFOLIO_RELEASE_FILENAME)
        == result.manifest
    )

    predictions = pl.read_parquet(result.artifact.path / PREDICTIONS_FILENAME)
    assert set(predictions["project_split"].unique()) == {"test"}
    assert predictions.filter(pl.col("active_relevance"))["stage"].unique().to_list() == [
        result.manifest.active_stage
    ]
    retrieval = pl.read_csv(result.artifact.path / RETRIEVAL_TABLE_FILENAME)
    ranking = pl.read_csv(result.artifact.path / RANKING_TABLE_FILENAME)
    assert set(retrieval["protocol"].unique()) == {RETRIEVAL_PROTOCOL}
    assert not set(retrieval["metric"].unique()) & {
        "precision",
        "map",
        "ndcg_official_gain",
    }
    assert set(ranking["protocol"].unique()) == {
        CLOSED_POOL_PROTOCOL,
        END_TO_END_PROTOCOL,
    }
    assert not ranking.filter(
        (pl.col("protocol") == CLOSED_POOL_PROTOCOL) & (pl.col("metric") == "ndcg_official_gain")
    ).is_empty()
    assert pl.read_parquet(result.artifact.path / METRICS_FILENAME).height >= ranking.height
    report = (result.artifact.path / FINAL_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "not catalog retrieval or live Amazon metrics" in report
    assert "it is not a full-catalog result" in report
    assert "Negative and inconclusive results" in report

    reused = build_portfolio_release(
        prepared[1],
        ranking_evaluation_id=bundle.manifest.components[5].artifact_id,
        serving_bundle_id=bundle.artifact.manifest.artifact_id,
        qualification_id=qualification.artifact.manifest.artifact_id,
        reproduction_evidence_path=tmp_path / "now-irrelevant",
        screenshots_dir=tmp_path / "now-irrelevant",
        code_revision=revision,
        artifact_store=prepared[2],
    )
    assert reused.reused
    assert reused.manifest == result.manifest

    manifest_path = result.artifact.path / PORTFOLIO_RELEASE_FILENAME
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="integrity"):
        prepared[2].load(result.artifact.manifest.artifact_id)
