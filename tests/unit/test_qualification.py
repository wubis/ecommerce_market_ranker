"""Goldfish 015 hardware, configuration, report, and CLI unit tests."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

import market_rank.cli as cli_module
from market_rank.qualification import probe_hardware


def test_hardware_probe_keeps_only_safe_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware_payload = {
        "SPHardwareDataType": [
            {
                "chip_type": "Apple M3",
                "machine_model": "Mac15,3",
                "serial_number": "must-not-escape",
                "platform_UUID": "must-not-escape",
            }
        ]
    }

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, timeout
        if "system_profiler" in command:
            output = json.dumps(hardware_payload)
        elif "vm_stat" in command:
            output = "page size of 16384 bytes\nPages free: 10.\nPages inactive: 20.\n"
        else:
            output = "Now drawing from 'AC Power'\n"
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 524288, "SC_PAGE_SIZE": 16384}[name],
    )
    observed = probe_hardware()
    serialized = observed.model_dump_json()
    assert observed.chip == "Apple M3"
    assert observed.machine_model == "Mac15,3"
    assert observed.physical_memory_bytes == 8 * 1024**3
    assert "must-not-escape" not in serialized


def test_qualification_cli_requires_conditions_and_prints_bounded_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli_module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["qualification", "run", "--bundle-id", "bundle/fixture"])

    class Report:
        mode_latencies: tuple[object, ...] = ()
        startup_ms = 1.0
        peak_rss_bytes = 100
        rss_limit_bytes = 200

    artifact_id = "release-qualification/dataset/portfolio/release-qualification-v1/" + "a" * 64
    result = SimpleNamespace(
        artifact=SimpleNamespace(manifest=SimpleNamespace(artifact_id=artifact_id)),
        report=Report(),
        reused=False,
    )
    monkeypatch.setattr(cli_module, "build_release_qualification", lambda *args, **kwargs: result)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")
    assert (
        cli_module.main(
            [
                "qualification",
                "run",
                "--bundle-id",
                "serving-bundle/dataset/portfolio/serving-bundle-v1/" + "a" * 64,
                "--background-conditions",
                "AC power; idle",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "release qualification: pass" in output
    assert "published release qualification" in output
