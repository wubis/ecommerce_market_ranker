"""Goldfish 016 reproduction, screenshot, configuration, and CLI unit tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import market_rank.cli as cli_module
from market_rank.config import load_config
from market_rank.portfolio import (
    PortfolioReproductionError,
    load_reproduction_evidence,
    verify_clean_reproduction,
)
from tests.unit.test_esci_profiles import BASE_CONFIG


def test_clean_reproduction_runs_canonical_gates_without_storing_output(tmp_path: Path) -> None:
    config = load_config([BASE_CONFIG])
    observed: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        observed.append(command)
        output = "258 passed in 1.0s\n" if "pytest" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    output = tmp_path / "reproduction.json"
    evidence = verify_clean_reproduction(
        config,
        code_revision="a" * 40,
        repository_root=tmp_path,
        output_path=output,
        runner=runner,
    )
    assert evidence.test_count == 258
    assert tuple(item.command_id for item in evidence.commands) == (
        "lock",
        "format",
        "lint",
        "types",
        "tests",
    )
    assert load_reproduction_evidence(output) == evidence
    assert observed[0][:3] == ["git", "-C", str(tmp_path)]


def test_clean_reproduction_rejects_dirty_or_nonrevision_state(tmp_path: Path) -> None:
    config = load_config([BASE_CONFIG])

    def dirty(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout=" M local.py\n", stderr="")

    with pytest.raises(PortfolioReproductionError, match="empty Git worktree"):
        verify_clean_reproduction(
            config,
            code_revision="a" * 40,
            repository_root=tmp_path,
            output_path=tmp_path / "unused.json",
            runner=dirty,
        )
    with pytest.raises(PortfolioReproductionError, match="clean Git revision"):
        verify_clean_reproduction(
            config,
            code_revision="dirty",
            repository_root=tmp_path,
            output_path=tmp_path / "unused.json",
            runner=dirty,
        )


def test_portfolio_cli_dispatches_verification_and_finalization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    evidence = SimpleNamespace(test_count=258)
    monkeypatch.setattr(cli_module, "_find_repository_root", lambda path: tmp_path)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "a" * 40)
    monkeypatch.setattr(cli_module, "verify_clean_reproduction", lambda *args, **kwargs: evidence)
    assert (
        cli_module.main(
            [
                "portfolio",
                "verify-reproduction",
                "--output",
                str(tmp_path / "reproduction.json"),
            ]
        )
        == 0
    )
    assert "clean reproduction: 258 tests" in capsys.readouterr().out

    release = SimpleNamespace(
        reused=False,
        manifest=SimpleNamespace(test_query_count=10, active_stage="lambdamart"),
        artifact=SimpleNamespace(manifest=SimpleNamespace(artifact_id="portfolio-release/fixture")),
    )
    monkeypatch.setattr(cli_module, "build_portfolio_release", lambda *args, **kwargs: release)
    assert (
        cli_module.main(
            [
                "portfolio",
                "finalize",
                "--ranking-evaluation-id",
                "ranking/fixture",
                "--serving-bundle-id",
                "bundle/fixture",
                "--qualification-id",
                "qualification/fixture",
                "--reproduction-evidence",
                str(tmp_path / "reproduction.json"),
                "--screenshots-dir",
                str(tmp_path / "screenshots"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "10 project-test queries" in output
    assert "published portfolio release" in output
