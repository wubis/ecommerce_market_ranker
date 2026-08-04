"""Goldfish 015 release-qualification integration and failure-path tests."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

import market_rank.qualification as qualification_module
from market_rank.artifacts import ArtifactStore, ArtifactValidationError
from market_rank.config import ResolvedConfig
from market_rank.qualification import (
    QUALIFICATION_FILENAME,
    QUALIFICATION_MARKDOWN_FILENAME,
    HardwareSnapshot,
    QualificationExecutionError,
    QualificationGateError,
    QualificationOfflineError,
    build_release_qualification,
    load_qualification_report,
)
from market_rank.serving.orchestrator import ServingRuntime, load_serving_runtime
from tests.integration.test_dense_retrieval import HashEncoder
from tests.integration.test_serving import _build, _prepare


def _hardware(*, chip: str = "Apple M3") -> HardwareSnapshot:
    return HardwareSnapshot(
        system="Darwin",
        machine="arm64",
        chip=chip,
        machine_model="Mac15,3",
        physical_memory_bytes=8 * 1024**3,
        logical_cpus=8,
        os_version="fixture",
        python_version="3.11.fixture",
        power_source="AC Power",
        available_memory_bytes=4 * 1024**3,
    )


def _loader(
    store: ArtifactStore,
    bundle_id: str,
    config: ResolvedConfig,
) -> ServingRuntime:
    return load_serving_runtime(store, bundle_id, config, encoder=HashEncoder())


def test_qualification_fails_closed_then_publishes_and_reuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    prepared = _prepare(tmp_path)
    bundle = _build(prepared)

    def network_loader(
        store: ArtifactStore,
        bundle_id: str,
        config: ResolvedConfig,
    ) -> ServingRuntime:
        del store, bundle_id, config
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        raise AssertionError("offline guard should prevent this line")

    with pytest.raises(QualificationExecutionError, match="offline_network_attempt") as blocked:
        build_release_qualification(
            prepared[1],
            bundle.artifact.manifest.artifact_id,
            code_revision="a" * 40,
            background_conditions="AC power; fixture process only",
            artifact_store=prepared[2],
            hardware=_hardware(),
            runtime_loader=network_loader,
        )
    assert blocked.value.report_path.is_file()

    with pytest.raises(QualificationGateError, match="hardware_chip") as failed:
        build_release_qualification(
            prepared[1],
            bundle.artifact.manifest.artifact_id,
            code_revision="a" * 40,
            background_conditions="AC power; fixture process only",
            artifact_store=prepared[2],
            hardware=_hardware(chip="Apple M2"),
            runtime_loader=_loader,
        )
    assert failed.value.report_path.is_file()
    assert not any((tmp_path / "artifacts" / "release-qualification").rglob("_SUCCESS"))

    result = build_release_qualification(
        prepared[1],
        bundle.artifact.manifest.artifact_id,
        code_revision="a" * 40,
        background_conditions="AC power; fixture process only",
        artifact_store=prepared[2],
        hardware=_hardware(),
        runtime_loader=_loader,
    )
    assert not result.reused
    assert result.report.passed
    assert result.artifact.manifest.artifact_type == "release-qualification"
    assert result.artifact.manifest.dependencies[0].artifact_id == bundle.manifest.artifact_id
    assert {item.relative_path for item in result.artifact.manifest.files} == {
        QUALIFICATION_FILENAME,
        QUALIFICATION_MARKDOWN_FILENAME,
    }
    assert load_qualification_report(result.artifact.path / QUALIFICATION_FILENAME).passed
    assert (
        tuple(item.mode for item in result.report.mode_latencies)
        == prepared[1].config.qualification.modes
    )
    assert all(item.latency.samples == 25 for item in result.report.mode_latencies)
    assert result.report.concurrent_latency.samples == 6
    assert result.report.peak_rss_bytes <= result.report.rss_limit_bytes
    assert all(component.state == "ready" for component in result.report.components)
    assert not any("wireless mouse" in check.detail for check in result.report.checks)

    reused = build_release_qualification(
        prepared[1],
        bundle.artifact.manifest.artifact_id,
        code_revision="ignored-on-reuse",
        background_conditions="ignored on reuse",
        artifact_store=prepared[2],
    )
    assert reused.reused
    assert reused.report == result.report

    report_path = result.artifact.path / QUALIFICATION_FILENAME
    report_path.write_text(report_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="integrity"):
        prepared[2].load(result.artifact.manifest.artifact_id)


def test_offline_guard_blocks_and_restores_socket() -> None:
    original = socket.socket
    with (
        pytest.raises(QualificationOfflineError, match="network connection"),
        qualification_module._offline_network_guard(),
    ):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    assert socket.socket is original
