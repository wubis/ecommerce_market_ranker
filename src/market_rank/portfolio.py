"""Frozen project-test evaluation and final core portfolio release packaging."""

from __future__ import annotations

import json
import re
import resource
import shutil
import struct
import subprocess
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Literal, Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from market_rank.artifacts import (
    ArtifactDependency,
    ArtifactExistsError,
    ArtifactStore,
    LoadedArtifact,
)
from market_rank.config import ResolvedConfig
from market_rank.evaluation.metrics import (
    CLOSED_POOL_PROTOCOL,
    END_TO_END_PROTOCOL,
    RETRIEVAL_PROTOCOL,
)
from market_rank.evaluation.ranking import (
    COMPARISONS_FILENAME,
    FAILURE_ANALYSIS_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    QUERY_METRICS_FILENAME,
    RANKING_EVALUATION_FILENAME,
    RankingEvaluationManifest,
    evaluate_frozen_ranking_test,
    load_ranking_evaluation_manifest,
)
from market_rank.evaluation.retrieval import (
    AGGREGATE_METRICS_FILENAME,
    RETRIEVAL_EVALUATION_FILENAME,
    RetrievalEvaluationManifest,
    load_retrieval_evaluation_manifest,
)
from market_rank.qualification import (
    QUALIFICATION_FILENAME,
    ReleaseQualificationReport,
    load_qualification_report,
)
from market_rank.serving.bundle import (
    SERVING_BUNDLE_FILENAME,
    ServingBundleManifest,
    load_serving_bundle_manifest,
)

PORTFOLIO_RELEASE_FILENAME: Literal["portfolio-release.json"] = "portfolio-release.json"
FINAL_REPORT_FILENAME: Literal["FINAL_REPORT.md"] = "FINAL_REPORT.md"
REPRODUCTION_FILENAME: Literal["reproduction.json"] = "reproduction.json"
LINEAGE_FILENAME: Literal["lineage.json"] = "lineage.json"
LIMITATIONS_FILENAME: Literal["LIMITATIONS.md"] = "LIMITATIONS.md"
RETRIEVAL_TABLE_FILENAME: Literal["retrieval-test.csv"] = "retrieval-test.csv"
RANKING_TABLE_FILENAME: Literal["ranking-test.csv"] = "ranking-test.csv"
SLICE_TABLE_FILENAME: Literal["ranking-test-slices.csv"] = "ranking-test-slices.csv"
ABLATION_TABLE_FILENAME: Literal["ablations-test.csv"] = "ablations-test.csv"
RESOURCE_TABLE_FILENAME: Literal["resources.csv"] = "resources.csv"
RECALL_PLOT_FILENAME: Literal["retrieval-recall.svg"] = "retrieval-recall.svg"
NDCG_PLOT_FILENAME: Literal["ranking-ndcg.svg"] = "ranking-ndcg.svg"
LATENCY_PLOT_FILENAME: Literal["serving-latency.svg"] = "serving-latency.svg"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_TEST_COUNT_RE = re.compile(r"(\d+) passed")


class PortfolioError(RuntimeError):
    """Base error for final portfolio verification and packaging."""


class PortfolioValidationError(PortfolioError):
    """Raised when final evidence is incomplete, incompatible, or misleading."""


class PortfolioReproductionError(PortfolioError):
    """Raised when clean-reproduction verification does not pass."""


class PortfolioResourceError(PortfolioError):
    """Raised when frozen final evaluation exceeds the process RSS limit."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReproductionCommand(_StrictModel):
    command_id: Literal["lock", "format", "lint", "types", "tests"]
    command: str = Field(strict=True, min_length=1, max_length=200)
    duration_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    exit_code: Literal[0] = 0


class ReproductionEvidence(_StrictModel):
    schema_version: Literal[1] = 1
    evidence_version: Literal["clean-reproduction-v1"] = "clean-reproduction-v1"
    config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(strict=True, pattern=r"^[0-9a-f]{40}$")
    created_utc: datetime
    workspace_clean: Literal[True] = True
    offline: Literal[True] = True
    commands: tuple[
        ReproductionCommand,
        ReproductionCommand,
        ReproductionCommand,
        ReproductionCommand,
        ReproductionCommand,
    ]
    test_count: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def validate_commands(self) -> Self:
        if tuple(item.command_id for item in self.commands) != (
            "lock",
            "format",
            "lint",
            "types",
            "tests",
        ):
            raise ValueError("reproduction commands must use the canonical order")
        return self


class ScreenshotEvidence(_StrictModel):
    filename: str = Field(strict=True, pattern=r"^[a-z0-9-]+\.png$")
    width: int = Field(strict=True, ge=1)
    height: int = Field(strict=True, ge=1)
    size_bytes: int = Field(strict=True, ge=1)
    sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")


class NegativeFinding(_StrictModel):
    finding_id: str = Field(strict=True, pattern=r"^[A-Z0-9-]+$")
    finding: str = Field(strict=True, min_length=1, max_length=300)


class PortfolioCheck(_StrictModel):
    check_id: str = Field(strict=True, pattern=r"^[a-z0-9_]+$")
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1, max_length=300)


class PortfolioReleaseManifest(_StrictModel):
    schema_version: Literal[1] = 1
    component_version: Literal["portfolio-release-v1"] = "portfolio-release-v1"
    generation: Literal["final-v1"] = "final-v1"
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    profile: Literal["portfolio"] = "portfolio"
    config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(strict=True, pattern=r"^[0-9a-f]{40}$")
    created_utc: datetime
    ranking_evaluation_artifact_id: str = Field(strict=True, min_length=1)
    serving_bundle_artifact_id: str = Field(strict=True, min_length=1)
    qualification_artifact_id: str = Field(strict=True, min_length=1)
    retrieval_evaluation_artifact_id: str = Field(strict=True, min_length=1)
    selection_split: Literal["validation"] = "validation"
    final_evaluation_split: Literal["test"] = "test"
    active_stage: Literal["rrf", "pointwise", "lambdamart"]
    test_query_count: int = Field(strict=True, ge=1)
    test_prediction_rows: int = Field(strict=True, ge=1)
    retrieval_table_rows: int = Field(strict=True, ge=1)
    ranking_table_rows: int = Field(strict=True, ge=1)
    slice_table_rows: int = Field(strict=True, ge=1)
    ablation_table_rows: int = Field(strict=True, ge=1)
    finalization_peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    screenshots: tuple[ScreenshotEvidence, ScreenshotEvidence, ScreenshotEvidence]
    reproduction_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    negative_findings: tuple[NegativeFinding, ...]
    checks: tuple[PortfolioCheck, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(set(check_ids))):
            raise ValueError("portfolio checks must be unique and sorted")
        finding_ids = tuple(item.finding_id for item in self.negative_findings)
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("portfolio negative findings must have unique identifiers")
        if self.finalization_peak_rss_bytes > self.rss_limit_bytes:
            raise ValueError("passing portfolio release exceeds the RSS limit")
        return self


@dataclass(frozen=True, slots=True)
class PortfolioBuildResult:
    artifact: LoadedArtifact
    manifest: PortfolioReleaseManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class _Inputs:
    ranking: LoadedArtifact
    ranking_manifest: RankingEvaluationManifest
    retrieval: LoadedArtifact
    retrieval_manifest: RetrievalEvaluationManifest
    serving: LoadedArtifact
    serving_manifest: ServingBundleManifest
    qualification: LoadedArtifact
    qualification_report: ReleaseQualificationReport


def _clean_revision(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def load_reproduction_evidence(path: Path) -> ReproductionEvidence:
    try:
        return ReproductionEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PortfolioValidationError(f"cannot load reproduction evidence {path}: {exc}") from exc


def _run_reproduction_command(
    command_id: Literal["lock", "format", "lint", "types", "tests"],
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[ReproductionCommand, str]:
    started = time.perf_counter()
    try:
        completed = runner(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PortfolioReproductionError(
            f"reproduction command {command_id} could not run: {type(exc).__name__}"
        ) from exc
    duration = time.perf_counter() - started
    if completed.returncode != 0:
        raise PortfolioReproductionError(
            f"reproduction command {command_id} exited with {completed.returncode}"
        )
    return (
        ReproductionCommand(
            command_id=command_id,
            command=" ".join(command),
            duration_seconds=duration,
        ),
        completed.stdout,
    )


def verify_clean_reproduction(
    config: ResolvedConfig,
    *,
    code_revision: str,
    repository_root: Path,
    output_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ReproductionEvidence:
    """Run canonical static/test gates and write evidence only from a clean revision."""
    if config.config.portfolio_report.require_clean_reproduction and not _clean_revision(
        code_revision
    ):
        raise PortfolioReproductionError("clean reproduction requires a clean Git revision")
    status = runner(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if status.returncode != 0 or status.stdout:
        raise PortfolioReproductionError("clean reproduction requires an empty Git worktree")
    plans: tuple[tuple[Literal["lock", "format", "lint", "types", "tests"], list[str]], ...] = (
        ("lock", ["uv", "lock", "--check"]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("lint", [sys.executable, "-m", "ruff", "check", "."]),
        ("types", [sys.executable, "-m", "mypy", "src", "tests"]),
        ("tests", [sys.executable, "-m", "pytest", "-q"]),
    )
    commands: list[ReproductionCommand] = []
    test_output = ""
    for command_id, command in plans:
        observed, output = _run_reproduction_command(
            command_id,
            command,
            cwd=repository_root,
            timeout_seconds=1800,
            runner=runner,
        )
        commands.append(observed)
        if command_id == "tests":
            test_output = output
    match = _TEST_COUNT_RE.search(test_output)
    if match is None:
        raise PortfolioReproductionError("pytest output did not contain a passing test count")
    evidence = ReproductionEvidence(
        config_sha256=config.sha256,
        code_revision=code_revision,
        created_utc=datetime.now(UTC),
        commands=tuple(commands),  # type: ignore[arg-type]
        test_count=int(match.group(1)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json(evidence))
    except FileExistsError as exc:
        raise PortfolioReproductionError(
            f"reproduction evidence already exists: {output_path}"
        ) from exc
    return evidence


def _load_inputs(
    store: ArtifactStore,
    config: ResolvedConfig,
    ranking_evaluation_id: str,
    serving_bundle_id: str,
    qualification_id: str,
) -> _Inputs:
    ranking = store.load(ranking_evaluation_id)
    serving = store.load(serving_bundle_id)
    qualification = store.load(qualification_id)
    if (
        ranking.manifest.artifact_type != "ranking-evaluation"
        or serving.manifest.artifact_type != "serving-bundle"
        or qualification.manifest.artifact_type != "release-qualification"
    ):
        raise PortfolioValidationError("portfolio inputs use incorrect artifact types")
    ranking_manifest = load_ranking_evaluation_manifest(ranking.path / RANKING_EVALUATION_FILENAME)
    retrieval = store.load(ranking_manifest.retrieval_evaluation_artifact_id)
    retrieval_manifest = load_retrieval_evaluation_manifest(
        retrieval.path / RETRIEVAL_EVALUATION_FILENAME
    )
    serving_manifest = load_serving_bundle_manifest(serving.path / SERVING_BUNDLE_FILENAME)
    qualification_report = load_qualification_report(qualification.path / QUALIFICATION_FILENAME)
    ranking_component = next(
        item for item in serving_manifest.components if item.component == "ranking_evaluation"
    )
    expected_qualification_dependency = ArtifactDependency(
        artifact_id=serving.manifest.artifact_id,
        manifest_sha256=serving.manifest_sha256,
    )
    if (
        ranking_manifest.profile != "portfolio"
        or retrieval_manifest.profile != "portfolio"
        or serving_manifest.profile != "portfolio"
        or ranking_manifest.config_sha256 != config.sha256
        or retrieval_manifest.config_sha256 != config.sha256
        or serving_manifest.config_sha256 != config.sha256
        or qualification_report.config_sha256 != config.sha256
        or ranking_component.artifact_id != ranking.manifest.artifact_id
        or ranking_component.manifest_sha256 != ranking.manifest_sha256
        or qualification.manifest.dependencies != (expected_qualification_dependency,)
        or qualification_report.bundle_id != serving.manifest.artifact_id
        or not qualification_report.passed
    ):
        raise PortfolioValidationError("portfolio artifact lineage is incompatible")
    return _Inputs(
        ranking,
        ranking_manifest,
        retrieval,
        retrieval_manifest,
        serving,
        serving_manifest,
        qualification,
        qualification_report,
    )


def _screenshot(path: Path, config: ResolvedConfig) -> ScreenshotEvidence:
    if path.is_symlink() or not path.is_file():
        raise PortfolioValidationError(f"screenshot must be a regular file: {path.name}")
    payload = path.read_bytes()
    if len(payload) < 45 or payload[:8] != _PNG_SIGNATURE:
        raise PortfolioValidationError(f"screenshot is not a valid PNG: {path.name}")
    offset = 8
    chunks: list[bytes] = []
    width = 0
    height = 0
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise PortfolioValidationError(f"screenshot PNG is truncated: {path.name}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise PortfolioValidationError(f"screenshot PNG chunk is truncated: {path.name}")
        chunk_data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != expected_crc:
            raise PortfolioValidationError(f"screenshot PNG checksum is invalid: {path.name}")
        chunks.append(chunk_type)
        if len(chunks) == 1:
            if chunk_type != b"IHDR" or length != 13:
                raise PortfolioValidationError(f"screenshot PNG omits IHDR: {path.name}")
            width, height = struct.unpack(">II", chunk_data[:8])
        offset = end
        if chunk_type == b"IEND":
            break
    if b"IDAT" not in chunks or not chunks or chunks[-1] != b"IEND" or offset != len(payload):
        raise PortfolioValidationError(f"screenshot PNG structure is incomplete: {path.name}")
    limits = config.config.portfolio_report
    if (
        width < limits.minimum_screenshot_width
        or height < limits.minimum_screenshot_height
        or len(payload) > limits.maximum_screenshot_bytes
    ):
        raise PortfolioValidationError(f"screenshot violates size bounds: {path.name}")
    return ScreenshotEvidence(
        filename=path.name,
        width=width,
        height=height,
        size_bytes=len(payload),
        sha256=sha256(payload).hexdigest(),
    )


def _validate_protocols(retrieval: pl.DataFrame, ranking: pl.DataFrame) -> None:
    if set(retrieval["protocol"].unique()) != {RETRIEVAL_PROTOCOL}:
        raise PortfolioValidationError("retrieval table has an incompatible protocol")
    prohibited_retrieval = {"precision", "map", "ndcg_official_gain"}
    if set(retrieval["metric"].unique()) & prohibited_retrieval:
        raise PortfolioValidationError("retrieval table contains prohibited ranking metrics")
    end_to_end = ranking.filter(pl.col("protocol") == END_TO_END_PROTOCOL)
    if set(end_to_end["metric"].unique()) & prohibited_retrieval:
        raise PortfolioValidationError("end-to-end table contains prohibited ranking metrics")
    closed = ranking.filter(pl.col("protocol") == CLOSED_POOL_PROTOCOL)
    if closed.filter(pl.col("metric") == "ndcg_official_gain").is_empty():
        raise PortfolioValidationError("closed-pool final table omits official-gain NDCG")


def _bar_svg(title: str, rows: Sequence[tuple[str, float]]) -> str:
    width = 960
    row_height = 34
    height = 90 + row_height * len(rows)
    maximum = max((value for _, value in rows), default=1.0) or 1.0
    body: list[str] = []
    for index, (label, value) in enumerate(rows):
        y = 60 + index * row_height
        bar_width = max(0.0, min(650.0, 650.0 * value / maximum))
        body.append(
            f'<text x="10" y="{y + 17}" font-size="13">{escape(label)}</text>'
            f'<rect x="250" y="{y}" width="{bar_width:.2f}" height="22" fill="#2563eb"/>'
            f'<text x="{260 + bar_width:.2f}" y="{y + 17}" font-size="13">{value:.4f}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="10" y="30" font-size="20" font-weight="bold">{escape(title)}</text>'
        + "".join(body)
        + "</svg>"
    )


def _negative_findings(
    comparisons: pl.DataFrame, manifest: RankingEvaluationManifest
) -> tuple[NegativeFinding, ...]:
    findings: list[NegativeFinding] = []
    for candidate in manifest.active_relevance.candidates:
        if not candidate.eligible:
            findings.append(
                NegativeFinding(
                    finding_id=f"CHAMPION-{candidate.stage.upper()}",
                    finding=f"{candidate.stage} was ineligible: {candidate.decision_reason}",
                )
            )
    for row in comparisons.iter_rows(named=True):
        if float(row["mean_improvement"]) <= 0.0 or float(row["ci95_lower"]) <= 0.0:
            findings.append(
                NegativeFinding(
                    finding_id=(
                        f"{row['ablation_id']}-{row['threshold_id']}-"
                        f"{row['metric']}-{row['cutoff']}"
                    )
                    .upper()
                    .replace("_", "-"),
                    finding=(
                        f"{row['ablation_id']} {row['metric']}@{row['cutoff']} was non-positive "
                        "or its grouped 95% interval included no improvement."
                    ),
                )
            )
    return tuple(findings)


def _headline_rows(
    ranking: pl.DataFrame, active_stage: str
) -> list[tuple[str, float, float, float]]:
    selected = ranking.filter(
        (pl.col("protocol") == CLOSED_POOL_PROTOCOL)
        & (pl.col("stage") == active_stage)
        & (pl.col("threshold_id") == "official_gain")
        & (pl.col("metric") == "ndcg_official_gain")
    ).sort("cutoff")
    return [
        (
            f"NDCG@{int(row['cutoff'])}",
            float(row["mean"]),
            float(row["ci95_lower"]),
            float(row["ci95_upper"]),
        )
        for row in selected.iter_rows(named=True)
    ]


def _report_markdown(
    manifest: PortfolioReleaseManifest,
    ranking: pl.DataFrame,
    qualification: ReleaseQualificationReport,
) -> str:
    headlines = _headline_rows(ranking, manifest.active_stage)
    headline_text = "\n".join(
        f"- {name}: {mean:.4f} (grouped 95% CI {lower:.4f}-{upper:.4f})"
        for name, mean, lower, upper in headlines
    )
    negatives = (
        "\n".join(f"- {item.finding}" for item in manifest.negative_findings)
        or "- No configured ablation had a non-positive mean or interval crossing zero."
    )
    return f"""# MarketRank Final Portfolio Report

## Measured headline

The validation-frozen `{manifest.active_stage}` stage was evaluated once on the project-test
closed judged pool under `{CLOSED_POOL_PROTOCOL}`. These are ranking-within-judged-pool results,
not catalog retrieval or live Amazon metrics.

{headline_text}

## System

MarketRank is a CPU-first multi-stage product-search reference: deterministic BM25 and pinned
MiniLM/FAISS retrieve candidates, RRF combines them, and validation-selected LightGBM relevance
may reorder the fixed union. Immutable artifacts connect data, features, models, evaluation,
serving, qualification, and this report.

## Protocol-separated evidence

- `retrieval-test.csv` contains fixed-catalog judged-retrieval diagnostics under
  `{RETRIEVAL_PROTOCOL}`. It contains no naive Precision, MAP, or NDCG.
- `ranking-test.csv` contains official-gain closed-pool ranking metrics and separately named
  end-to-end retrieval-aware diagnostics.
- `ablations-test.csv` contains paired grouped-bootstrap comparisons on the unchanged test cohort.
- `ranking-test-slices.csv` contains all configured slices, not selected favorable slices.

## Local qualification

- Host: {qualification.hardware.machine_model}, {qualification.hardware.chip}, 8 GiB
- Cold bundle startup: {qualification.startup_ms:.3f} ms
- Peak RSS: {qualification.peak_rss_bytes}/{qualification.rss_limit_bytes} bytes
- Qualification artifact: `{manifest.qualification_artifact_id}`

![Retrieval Recall]({RECALL_PLOT_FILENAME})

![Closed-pool NDCG]({NDCG_PLOT_FILENAME})

![Serving latency]({LATENCY_PLOT_FILENAME})

## Negative and inconclusive results

{negatives}

## Demo evidence

- [Ranking comparison](screenshots/ranking-comparison.png)
- [Product provenance](screenshots/product-provenance.png)
- [Dataset limitations](screenshots/dataset-limitations.png)

## Limitations

See [LIMITATIONS.md]({LIMITATIONS_FILENAME}). ESCI judgments are bounded; an unjudged product is
unknown, not irrelevant. The fixed research catalog is not live Amazon search and contains no
authoritative price, inventory, seller, shipping, review, sponsorship, conversion, margin, return,
or user-history fields. No business, causal, personalization, or fairness claim is made.

## Reproduction

Exact lineage is in `{LINEAGE_FILENAME}` and clean gate evidence is in
`{REPRODUCTION_FILENAME}`. Configuration: `{manifest.config_sha256}`; code:
`{manifest.code_revision}`; release artifact: `{manifest.artifact_id}`.
"""


def _limitations_markdown() -> str:
    return """# Limitations

- Amazon ESCI supplies bounded judged product lists, not exhaustive catalog judgments.
- Missing judgments mean unknown, never automatically irrelevant.
- Closed-pool NDCG measures ordering of supplied judged products; retrieval metrics use the fixed
  catalog and are reported separately.
- End-to-end ranking diagnostics are conditional on the retrieved union and are not official Task
  1 NDCG.
- The catalog is a fixed research snapshot, not current Amazon search behavior.
- Price, inventory, seller, fulfillment, shipping, reviews, sponsorship, conversion, margin,
  returns, product age, user histories, and canonical category fields are unavailable.
- Brand/list-composition diagnostics are descriptive and carry no fairness or business meaning.
- Optional neural reranking and diversity are not part of the core release.
- No online, causal, counterfactual, personalization, or production-scale claim is supported.
"""


def portfolio_artifact_id(ranking: LoadedArtifact, config: ResolvedConfig) -> str:
    return "/".join(
        (
            "portfolio-release",
            ranking.manifest.dataset_version,
            "portfolio",
            config.config.portfolio_report.component_version,
            config.sha256,
        )
    )


def load_portfolio_manifest(path: Path) -> PortfolioReleaseManifest:
    try:
        return PortfolioReleaseManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PortfolioValidationError(f"cannot load portfolio manifest {path}: {exc}") from exc


def _reuse(
    store: ArtifactStore,
    artifact_id: str,
    inputs: _Inputs,
    config: ResolvedConfig,
) -> PortfolioBuildResult:
    artifact = store.load(artifact_id)
    expected = tuple(
        sorted(
            (
                ArtifactDependency(
                    artifact_id=inputs.ranking.manifest.artifact_id,
                    manifest_sha256=inputs.ranking.manifest_sha256,
                ),
                ArtifactDependency(
                    artifact_id=inputs.serving.manifest.artifact_id,
                    manifest_sha256=inputs.serving.manifest_sha256,
                ),
                ArtifactDependency(
                    artifact_id=inputs.qualification.manifest.artifact_id,
                    manifest_sha256=inputs.qualification.manifest_sha256,
                ),
            ),
            key=lambda item: item.artifact_id,
        )
    )
    if artifact.manifest.dependencies != expected:
        raise PortfolioValidationError("portfolio release dependencies are incompatible")
    manifest = load_portfolio_manifest(artifact.path / PORTFOLIO_RELEASE_FILENAME)
    if manifest.config_sha256 != config.sha256 or manifest.artifact_id != artifact_id:
        raise PortfolioValidationError("portfolio release identity is incompatible")
    return PortfolioBuildResult(artifact, manifest, True)


def build_portfolio_release(
    config: ResolvedConfig,
    *,
    ranking_evaluation_id: str,
    serving_bundle_id: str,
    qualification_id: str,
    reproduction_evidence_path: Path,
    screenshots_dir: Path,
    code_revision: str,
    artifact_store: ArtifactStore | None = None,
) -> PortfolioBuildResult:
    """Evaluate frozen project test and publish one complete core portfolio package."""
    if not _clean_revision(code_revision):
        raise PortfolioValidationError("portfolio release requires a clean Git revision")
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    inputs = _load_inputs(
        store,
        config,
        ranking_evaluation_id,
        serving_bundle_id,
        qualification_id,
    )
    artifact_id = portfolio_artifact_id(inputs.ranking, config)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse(store, artifact_id, inputs, config)

    reproduction = load_reproduction_evidence(reproduction_evidence_path)
    reproduction_bytes = reproduction_evidence_path.read_bytes()
    if (
        reproduction.config_sha256 != config.sha256
        or reproduction.code_revision != code_revision
        or inputs.qualification_report.code_revision != code_revision
    ):
        raise PortfolioValidationError("reproduction, qualification, and release code differ")
    screenshot_paths = tuple(
        screenshots_dir / filename
        for filename in config.config.portfolio_report.screenshot_filenames
    )
    screenshots = tuple(_screenshot(path, config) for path in screenshot_paths)
    if (
        tuple(item.filename for item in screenshots)
        != config.config.portfolio_report.screenshot_filenames
    ):
        raise PortfolioValidationError("portfolio screenshots are incomplete or out of order")

    frozen = evaluate_frozen_ranking_test(store, config, inputs.ranking_manifest)
    retrieval_all = pl.read_parquet(inputs.retrieval.path / AGGREGATE_METRICS_FILENAME)
    retrieval_test = retrieval_all.filter(
        (pl.col("slice_dimension") == "project_split") & (pl.col("slice_value") == "test")
    ).sort("stage", "threshold_id", "metric", "cutoff")
    ranking_overall = frozen.aggregate_metrics.filter(pl.col("slice_dimension") == "all").sort(
        "protocol", "stage", "threshold_id", "metric", "cutoff"
    )
    ranking_slices = frozen.aggregate_metrics.filter(pl.col("slice_dimension") != "all")
    _validate_protocols(retrieval_test, ranking_overall)
    if (
        retrieval_test.is_empty()
        or ranking_overall.is_empty()
        or ranking_slices.is_empty()
        or frozen.comparisons.is_empty()
        or set(frozen.predictions["project_split"].unique()) != {"test"}
    ):
        raise PortfolioValidationError("final project-test evidence is incomplete")
    peak_rss = _peak_rss_bytes()
    rss_limit = config.config.runtime.rss_limit_mb * 1024 * 1024
    if peak_rss > rss_limit:
        raise PortfolioResourceError(
            f"portfolio finalization peak RSS {peak_rss} exceeds {rss_limit} bytes"
        )
    negative_findings = _negative_findings(frozen.comparisons, inputs.ranking_manifest)
    test_query_count = frozen.query_metrics["query_id"].n_unique()
    dependencies = tuple(
        sorted(
            (
                ArtifactDependency(
                    artifact_id=inputs.ranking.manifest.artifact_id,
                    manifest_sha256=inputs.ranking.manifest_sha256,
                ),
                ArtifactDependency(
                    artifact_id=inputs.serving.manifest.artifact_id,
                    manifest_sha256=inputs.serving.manifest_sha256,
                ),
                ArtifactDependency(
                    artifact_id=inputs.qualification.manifest.artifact_id,
                    manifest_sha256=inputs.qualification.manifest_sha256,
                ),
            ),
            key=lambda item: item.artifact_id,
        )
    )
    checks = tuple(
        sorted(
            (
                PortfolioCheck(
                    check_id="clean_reproduction",
                    detail=f"{reproduction.test_count} tests plus lock and static gates passed",
                ),
                PortfolioCheck(
                    check_id="demo_evidence",
                    detail=(
                        "three bounded PNG screenshots include comparison, provenance, limitations"
                    ),
                ),
                PortfolioCheck(
                    check_id="frozen_champion",
                    detail=(
                        "validation selected "
                        f"{inputs.ranking_manifest.active_relevance.selected_stage}; "
                        "project test performed no reselection"
                    ),
                ),
                PortfolioCheck(
                    check_id="negative_results",
                    detail=(
                        f"{len(negative_findings)} non-positive or inconclusive findings disclosed"
                    ),
                ),
                PortfolioCheck(
                    check_id="protocol_separation",
                    detail="catalog retrieval and closed-pool ranking tables remain distinct",
                ),
                PortfolioCheck(
                    check_id="qualification_passed",
                    detail="exact serving bundle passed M3/8 GB qualification",
                ),
                PortfolioCheck(
                    check_id="frozen_test_evaluation",
                    detail=(
                        f"fixed project-test cohort contains {test_query_count} queries and "
                        "performed no selection"
                    ),
                ),
            ),
            key=lambda item: item.check_id,
        )
    )
    manifest = PortfolioReleaseManifest(
        artifact_id=artifact_id,
        dataset_version=inputs.ranking.manifest.dataset_version,
        config_sha256=config.sha256,
        code_revision=code_revision,
        created_utc=datetime.now(UTC),
        ranking_evaluation_artifact_id=inputs.ranking.manifest.artifact_id,
        serving_bundle_artifact_id=inputs.serving.manifest.artifact_id,
        qualification_artifact_id=inputs.qualification.manifest.artifact_id,
        retrieval_evaluation_artifact_id=inputs.retrieval.manifest.artifact_id,
        active_stage=inputs.ranking_manifest.active_relevance.selected_stage,
        test_query_count=test_query_count,
        test_prediction_rows=frozen.predictions.height,
        retrieval_table_rows=retrieval_test.height,
        ranking_table_rows=ranking_overall.height,
        slice_table_rows=ranking_slices.height,
        ablation_table_rows=frozen.comparisons.height,
        finalization_peak_rss_bytes=peak_rss,
        rss_limit_bytes=rss_limit,
        screenshots=screenshots,  # type: ignore[arg-type]
        reproduction_sha256=sha256(reproduction_bytes).hexdigest(),
        negative_findings=negative_findings,
        checks=checks,
    )

    transaction = store.stage(
        artifact_type="portfolio-release",
        dataset_version=inputs.ranking.manifest.dataset_version,
        profile="portfolio",
        component_version=config.config.portfolio_report.component_version,
        config_sha256=config.sha256,
        code_revision=code_revision,
        dependencies=dependencies,
    )
    try:
        with transaction:
            root = transaction.path(PORTFOLIO_RELEASE_FILENAME).parent
            frozen.predictions.write_parquet(root / PREDICTIONS_FILENAME, compression="zstd")
            frozen.query_metrics.write_parquet(root / QUERY_METRICS_FILENAME, compression="zstd")
            frozen.aggregate_metrics.write_parquet(root / METRICS_FILENAME, compression="zstd")
            frozen.comparisons.write_parquet(root / COMPARISONS_FILENAME, compression="zstd")
            frozen.failure_analysis.write_parquet(
                root / FAILURE_ANALYSIS_FILENAME, compression="zstd"
            )
            retrieval_test.write_csv(root / RETRIEVAL_TABLE_FILENAME)
            ranking_overall.write_csv(root / RANKING_TABLE_FILENAME)
            ranking_slices.write_csv(root / SLICE_TABLE_FILENAME)
            frozen.comparisons.write_csv(root / ABLATION_TABLE_FILENAME)
            resource_rows = pl.DataFrame(
                {
                    "component": ("retrieval-evaluation", "ranking-evaluation", "qualification"),
                    "artifact_bytes": (
                        sum(item.size_bytes for item in inputs.retrieval.manifest.files),
                        sum(item.size_bytes for item in inputs.ranking.manifest.files),
                        sum(item.size_bytes for item in inputs.qualification.manifest.files),
                    ),
                    "peak_rss_bytes": (
                        inputs.retrieval_manifest.resource.peak_rss_bytes,
                        inputs.ranking_manifest.resource.peak_rss_bytes,
                        inputs.qualification_report.peak_rss_bytes,
                    ),
                }
            )
            resource_rows.write_csv(root / RESOURCE_TABLE_FILENAME)
            recall_rows = retrieval_test.filter(
                (pl.col("threshold_id") == "exact_substitute")
                & (pl.col("metric") == "judged_recall")
            )
            (root / RECALL_PLOT_FILENAME).write_text(
                _bar_svg(
                    "Fixed-catalog judged Recall (project test)",
                    [
                        (f"{row['stage']}@{row['cutoff']}", float(row["mean"]))
                        for row in recall_rows.iter_rows(named=True)
                    ],
                ),
                encoding="utf-8",
            )
            ndcg_rows = ranking_overall.filter(
                (pl.col("protocol") == CLOSED_POOL_PROTOCOL)
                & (pl.col("threshold_id") == "official_gain")
                & (pl.col("metric") == "ndcg_official_gain")
            )
            (root / NDCG_PLOT_FILENAME).write_text(
                _bar_svg(
                    "Closed-pool official-gain NDCG (project test)",
                    [
                        (f"{row['stage']}@{row['cutoff']}", float(row["mean"]))
                        for row in ndcg_rows.iter_rows(named=True)
                    ],
                ),
                encoding="utf-8",
            )
            (root / LATENCY_PLOT_FILENAME).write_text(
                _bar_svg(
                    "Serving stage p95 latency (ms)",
                    [
                        (item.stage, item.latency.p95_ms)
                        for item in inputs.qualification_report.stage_latencies
                    ],
                ),
                encoding="utf-8",
            )
            screenshots_root = root / "screenshots"
            screenshots_root.mkdir()
            for source in screenshot_paths:
                shutil.copyfile(source, screenshots_root / source.name)
            (root / REPRODUCTION_FILENAME).write_bytes(reproduction_bytes)
            lineage = {
                "config_sha256": config.sha256,
                "code_revision": code_revision,
                "ranking_evaluation": {
                    "artifact_id": inputs.ranking.manifest.artifact_id,
                    "manifest_sha256": inputs.ranking.manifest_sha256,
                },
                "retrieval_evaluation": {
                    "artifact_id": inputs.retrieval.manifest.artifact_id,
                    "manifest_sha256": inputs.retrieval.manifest_sha256,
                },
                "serving_bundle": {
                    "artifact_id": inputs.serving.manifest.artifact_id,
                    "manifest_sha256": inputs.serving.manifest_sha256,
                },
                "qualification": {
                    "artifact_id": inputs.qualification.manifest.artifact_id,
                    "manifest_sha256": inputs.qualification.manifest_sha256,
                },
            }
            (root / LINEAGE_FILENAME).write_text(
                json.dumps(lineage, sort_keys=True, separators=(",", ":")), encoding="utf-8"
            )
            (root / LIMITATIONS_FILENAME).write_text(_limitations_markdown(), encoding="utf-8")
            (root / FINAL_REPORT_FILENAME).write_text(
                _report_markdown(manifest, ranking_overall, inputs.qualification_report),
                encoding="utf-8",
            )
            (root / PORTFOLIO_RELEASE_FILENAME).write_text(
                _canonical_json(manifest), encoding="utf-8"
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse(store, artifact_id, inputs, config)
    return PortfolioBuildResult(artifact, manifest, False)


__all__ = [
    "FINAL_REPORT_FILENAME",
    "PORTFOLIO_RELEASE_FILENAME",
    "PortfolioBuildResult",
    "PortfolioError",
    "PortfolioReleaseManifest",
    "PortfolioReproductionError",
    "PortfolioResourceError",
    "PortfolioValidationError",
    "ReproductionEvidence",
    "build_portfolio_release",
    "load_portfolio_manifest",
    "load_reproduction_evidence",
    "portfolio_artifact_id",
    "verify_clean_reproduction",
]
