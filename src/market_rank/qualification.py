"""Fail-closed local release qualification for one explicit serving bundle."""

from __future__ import annotations

import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Self

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from market_rank.artifacts import (
    ArtifactDependency,
    ArtifactExistsError,
    ArtifactStore,
    LoadedArtifact,
)
from market_rank.config import QualificationConfig, ResolvedConfig
from market_rank.serving.api import create_app
from market_rank.serving.bundle import (
    SERVING_BUNDLE_FILENAME,
    ServingBundleManifest,
    load_serving_bundle_manifest,
)
from market_rank.serving.contracts import ComponentStatus, SearchMode, SearchRequest, SearchResponse
from market_rank.serving.orchestrator import ServingRuntime, load_serving_runtime

QUALIFICATION_FILENAME: Literal["release-qualification.json"] = "release-qualification.json"
QUALIFICATION_MARKDOWN_FILENAME: Literal["release-qualification.md"] = "release-qualification.md"
_MEMORY_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_MEMORY_COUNT_RE = re.compile(r"Pages (free|inactive|speculative):\s+(\d+)\.")


class QualificationError(RuntimeError):
    """Base error for release qualification."""


class QualificationValidationError(QualificationError):
    """Raised when persisted qualification evidence is invalid or incompatible."""


class QualificationOfflineError(QualificationError):
    """Raised if bundle startup or the benchmark attempts a network connection."""


class QualificationGateError(QualificationError):
    """Raised when measurements do not satisfy every release gate."""

    def __init__(self, report: ReleaseQualificationReport, report_path: Path) -> None:
        failed = tuple(check.check_id for check in report.checks if not check.passed)
        super().__init__(
            f"release qualification failed {len(failed)} gate(s): {', '.join(failed)}; "
            f"report retained at {report_path}"
        )
        self.report = report
        self.report_path = report_path


class QualificationExecutionError(QualificationError):
    """Raised when qualification cannot complete a valid measurement report."""

    def __init__(self, reason_code: str, report_path: Path) -> None:
        super().__init__(
            f"release qualification could not complete: {reason_code}; "
            f"failure record retained at {report_path}"
        )
        self.reason_code = reason_code
        self.report_path = report_path


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HardwareSnapshot(_StrictModel):
    system: str = Field(strict=True, min_length=1)
    machine: str = Field(strict=True, min_length=1)
    chip: str = Field(strict=True, min_length=1)
    machine_model: str = Field(strict=True, min_length=1)
    physical_memory_bytes: int = Field(strict=True, ge=1)
    logical_cpus: int = Field(strict=True, ge=1)
    os_version: str = Field(strict=True, min_length=1)
    python_version: str = Field(strict=True, min_length=1)
    power_source: str = Field(strict=True, min_length=1)
    available_memory_bytes: int = Field(strict=True, ge=0)


class DependencyVersion(_StrictModel):
    package: str = Field(strict=True, min_length=1)
    version: str = Field(strict=True, min_length=1)


class LatencyDistribution(_StrictModel):
    samples: int = Field(strict=True, ge=1)
    p50_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p95_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p99_ms: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_ms: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if not self.p50_ms <= self.p95_ms <= self.p99_ms <= self.maximum_ms:
            raise ValueError("latency percentiles must be ordered")
        return self


class ModeLatency(_StrictModel):
    mode: SearchMode
    latency: LatencyDistribution


class StageLatency(_StrictModel):
    stage: Literal["parse", "sparse", "dense", "fusion", "features", "ranker"]
    latency: LatencyDistribution
    p95_target_ms: float = Field(gt=0.0, allow_inf_nan=False)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        if self.passed != (self.latency.p95_ms <= self.p95_target_ms):
            raise ValueError("stage status differs from the p95 target")
        return self


class QualificationCheck(_StrictModel):
    check_id: str = Field(strict=True, pattern=r"^[a-z0-9_]+$")
    passed: bool = Field(strict=True)
    detail: str = Field(strict=True, min_length=1, max_length=300)


class ReleaseQualificationReport(_StrictModel):
    schema_version: Literal[1] = 1
    component_version: Literal["release-qualification-v1"] = "release-qualification-v1"
    generation: Literal["rc1"] = "rc1"
    bundle_id: str = Field(strict=True, min_length=1)
    bundle_manifest_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    dataset_version: str = Field(strict=True, min_length=1)
    profile: str = Field(strict=True, min_length=1)
    config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(strict=True, min_length=1)
    started_utc: datetime
    completed_utc: datetime
    background_conditions: str = Field(strict=True, min_length=1, max_length=240)
    hardware: HardwareSnapshot
    dependencies: tuple[DependencyVersion, ...]
    max_threads: int = Field(strict=True, ge=1)
    query_sha256s: tuple[str, ...]
    modes: tuple[SearchMode, ...]
    top_k: int = Field(strict=True, ge=1)
    warmup_rounds: int = Field(strict=True, ge=1)
    measured_rounds: int = Field(strict=True, ge=1)
    concurrency_workers: int = Field(strict=True, ge=1)
    concurrency_rounds: int = Field(strict=True, ge=1)
    startup_ms: float = Field(ge=0.0, allow_inf_nan=False)
    mode_latencies: tuple[ModeLatency, ...]
    concurrent_latency: LatencyDistribution
    stage_latencies: tuple[StageLatency, ...]
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    components: tuple[ComponentStatus, ...]
    checks: tuple[QualificationCheck, ...]
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.completed_utc < self.started_utc:
            raise ValueError("qualification completion precedes its start")
        if not self.query_sha256s or len(set(self.query_sha256s)) != len(self.query_sha256s):
            raise ValueError("qualification query hashes must be nonempty and unique")
        if tuple(item.mode for item in self.mode_latencies) != self.modes:
            raise ValueError("mode latency order differs from the workload")
        check_ids = tuple(check.check_id for check in self.checks)
        if check_ids != tuple(sorted(set(check_ids))):
            raise ValueError("qualification checks must be unique and sorted")
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("qualification status differs from its checks")
        return self


class QualificationFailureRecord(_StrictModel):
    schema_version: Literal[1] = 1
    component_version: Literal["release-qualification-v1"] = "release-qualification-v1"
    bundle_id: str = Field(strict=True, min_length=1)
    bundle_manifest_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(strict=True, min_length=1)
    background_conditions: str = Field(strict=True, min_length=1, max_length=240)
    failed_utc: datetime
    reason_code: Literal["offline_network_attempt", "qualification_execution_failed"]
    error_type: str = Field(strict=True, pattern=r"^[A-Za-z][A-Za-z0-9_]+$")
    promoted: Literal[False] = False


@dataclass(frozen=True, slots=True)
class QualificationBuildResult:
    artifact: LoadedArtifact
    report: ReleaseQualificationReport
    reused: bool


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip()


def _hardware_details() -> tuple[str, str]:
    output = _run_text(["system_profiler", "SPHardwareDataType", "-json"])
    try:
        row = json.loads(output)["SPHardwareDataType"][0]
        chip = str(row.get("chip_type") or "unavailable")
        machine_model = str(row.get("machine_model") or "unavailable")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return "unavailable", "unavailable"
    return chip, machine_model


def _available_memory_bytes() -> int:
    output = _run_text(["/usr/bin/vm_stat"])
    page_match = _MEMORY_PAGE_RE.search(output)
    if page_match is None:
        return 0
    page_size = int(page_match.group(1))
    counts = {name: int(value) for name, value in _MEMORY_COUNT_RE.findall(output)}
    return page_size * sum(counts.values())


def probe_hardware() -> HardwareSnapshot:
    """Capture safe release-machine facts without serial or device identifiers."""
    chip, machine_model = _hardware_details()
    physical_memory = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    power_output = _run_text(["/usr/bin/pmset", "-g", "batt"])
    first_line = power_output.splitlines()[0] if power_output else "unavailable"
    power_source = first_line.removeprefix("Now drawing from '").removesuffix("'")
    return HardwareSnapshot(
        system=platform.system() or "unavailable",
        machine=platform.machine() or "unavailable",
        chip=chip,
        machine_model=machine_model,
        physical_memory_bytes=physical_memory,
        logical_cpus=os.cpu_count() or 1,
        os_version=platform.mac_ver()[0] or platform.release() or "unavailable",
        python_version=platform.python_version(),
        power_source=power_source or "unavailable",
        available_memory_bytes=_available_memory_bytes(),
    )


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> LatencyDistribution:
    if not values:
        raise QualificationValidationError("latency distribution cannot be empty")
    return LatencyDistribution(
        samples=len(values),
        p50_ms=_percentile(values, 0.50),
        p95_ms=_percentile(values, 0.95),
        p99_ms=_percentile(values, 0.99),
        maximum_ms=max(values),
    )


def _dependency_versions() -> tuple[DependencyVersion, ...]:
    packages = (
        "faiss-cpu",
        "fastapi",
        "lightgbm",
        "market-rank",
        "sentence-transformers",
        "streamlit",
    )
    captured: list[DependencyVersion] = []
    for package in packages:
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "not-installed"
        captured.append(DependencyVersion(package=package, version=package_version))
    return tuple(captured)


@contextmanager
def _offline_network_guard() -> Iterator[list[str]]:
    attempts: list[str] = []
    original_socket = socket.socket
    previous = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }

    class GuardedSocket(original_socket):  # type: ignore[misc, valid-type]
        def connect(self, address: object) -> None:
            attempts.append(type(address).__name__)
            raise QualificationOfflineError("network connection attempted during qualification")

        def connect_ex(self, address: object) -> int:
            attempts.append(type(address).__name__)
            raise QualificationOfflineError("network connection attempted during qualification")

    for name in previous:
        os.environ[name] = "1"
    socket.socket = GuardedSocket  # type: ignore[misc]
    try:
        yield attempts
    finally:
        socket.socket = original_socket  # type: ignore[misc]
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _stage_targets(qualification: QualificationConfig) -> dict[str, float]:
    return {
        "parse": qualification.parse_p95_target_ms,
        "sparse": qualification.sparse_p95_target_ms,
        "dense": qualification.dense_p95_target_ms,
        "fusion": qualification.fusion_p95_target_ms,
        "features": qualification.features_p95_target_ms,
        "ranker": qualification.ranker_p95_target_ms,
    }


def _check(check_id: str, passed: bool, detail: str) -> QualificationCheck:
    return QualificationCheck(check_id=check_id, passed=passed, detail=detail)


def _clean_revision(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def measure_release_qualification(
    config: ResolvedConfig,
    bundle: LoadedArtifact,
    bundle_manifest: ServingBundleManifest,
    *,
    code_revision: str,
    background_conditions: str,
    artifact_store: ArtifactStore,
    hardware: HardwareSnapshot | None = None,
    runtime_loader: Callable[[ArtifactStore, str, ResolvedConfig], ServingRuntime] | None = None,
) -> ReleaseQualificationReport:
    """Measure a fresh-process bundle load plus sequential and modest-concurrency requests."""
    qualification = config.config.qualification
    snapshot = hardware or probe_hardware()
    started_utc = datetime.now(UTC)
    loader = runtime_loader or load_serving_runtime
    mode_values: dict[SearchMode, list[float]] = {mode: [] for mode in qualification.modes}
    stage_values: dict[str, list[float]] = {stage: [] for stage in _stage_targets(qualification)}
    concurrent_values: list[float] = []
    active_contract_valid = True
    runtime: ServingRuntime | None = None
    with _offline_network_guard() as network_attempts:
        startup_started = time.perf_counter()
        runtime = loader(artifact_store, bundle.manifest.artifact_id, config)
        startup_ms = (time.perf_counter() - startup_started) * 1000.0
        try:
            info = runtime.info()
            app = create_app(config, bundle.manifest.artifact_id, runtime=runtime)
            with TestClient(app, base_url="http://127.0.0.1") as client:
                readiness = client.get("/health/ready")
                if readiness.status_code != 200:
                    raise QualificationValidationError("qualified API did not become ready")
                for _ in range(qualification.warmup_rounds):
                    for query in qualification.queries:
                        for mode in qualification.modes:
                            response = client.post(
                                "/v1/search",
                                json={"query": query, "mode": mode, "top_k": qualification.top_k},
                            )
                            if response.status_code != 200:
                                raise QualificationValidationError(
                                    "warmup request failed safely with status "
                                    f"{response.status_code}"
                                )
                for _ in range(qualification.measured_rounds):
                    for query in qualification.queries:
                        for mode in qualification.modes:
                            request_started = time.perf_counter()
                            response = client.post(
                                "/v1/search",
                                json={"query": query, "mode": mode, "top_k": qualification.top_k},
                            )
                            elapsed_ms = (time.perf_counter() - request_started) * 1000.0
                            if response.status_code != 200:
                                raise QualificationValidationError(
                                    "measured request failed safely with status "
                                    f"{response.status_code}"
                                )
                            parsed = SearchResponse.model_validate(response.json())
                            mode_values[mode].append(elapsed_ms)
                            timings = parsed.timings
                            stage_values["parse"].append(timings.parse_ms)
                            stage_values["sparse"].append(timings.sparse_ms)
                            stage_values["dense"].append(timings.dense_ms)
                            stage_values["fusion"].append(timings.fusion_ms)
                            stage_values["features"].append(timings.features_ms)
                            stage_values["ranker"].append(timings.ranker_ms)
                            if mode == "active" and (
                                parsed.resolved_stage != parsed.promoted_stage or parsed.degraded
                            ):
                                active_contract_valid = False

            requests = tuple(
                SearchRequest(query=query, mode="active", top_k=qualification.top_k)
                for query in qualification.queries[: qualification.concurrency_workers]
            )
            for _ in range(qualification.concurrency_rounds):
                with ThreadPoolExecutor(max_workers=qualification.concurrency_workers) as executor:
                    futures = []
                    for request in requests:
                        request_started = time.perf_counter()
                        future = executor.submit(runtime.search, request)
                        futures.append((request_started, future))
                    for request_started, future in futures:
                        future.result()
                        concurrent_values.append((time.perf_counter() - request_started) * 1000.0)
        finally:
            runtime.close()

    mode_latencies = tuple(
        ModeLatency(mode=mode, latency=_distribution(mode_values[mode]))
        for mode in qualification.modes
    )
    stage_latencies = tuple(
        StageLatency(
            stage=stage,  # type: ignore[arg-type]
            latency=_distribution(stage_values[stage]),
            p95_target_ms=target,
            passed=_distribution(stage_values[stage]).p95_ms <= target,
        )
        for stage, target in _stage_targets(qualification).items()
    )
    concurrent_latency = _distribution(concurrent_values)
    peak_rss_bytes = _peak_rss_bytes()
    rss_limit_bytes = config.config.runtime.rss_limit_mb * 1024 * 1024
    components_ready = all(component.state == "ready" for component in info.components)
    mode_checks = tuple(
        _check(
            f"request_p95_{item.mode}",
            item.latency.p95_ms <= qualification.request_p95_target_ms,
            f"p95 {item.latency.p95_ms:.3f} ms <= {qualification.request_p95_target_ms:.3f} ms",
        )
        for item in mode_latencies
    )
    stage_checks = tuple(
        _check(
            f"stage_p95_{item.stage}",
            item.passed,
            f"p95 {item.latency.p95_ms:.3f} ms <= {item.p95_target_ms:.3f} ms",
        )
        for item in stage_latencies
    )
    checks = tuple(
        sorted(
            (
                _check(
                    "active_contract",
                    active_contract_valid,
                    "active requests resolve to the promoted non-degraded relevance stage",
                ),
                _check(
                    "code_revision_clean",
                    not qualification.require_clean_revision or _clean_revision(code_revision),
                    "release qualification requires a clean 40-character Git revision",
                ),
                _check(
                    "components_ready",
                    components_ready and info.ready and not info.degraded,
                    "all bundled serving components must be ready and non-degraded",
                ),
                _check(
                    "concurrency_p95",
                    concurrent_latency.p95_ms <= qualification.concurrency_p95_target_ms,
                    f"p95 {concurrent_latency.p95_ms:.3f} ms <= "
                    f"{qualification.concurrency_p95_target_ms:.3f} ms",
                ),
                _check(
                    "hardware_chip",
                    snapshot.chip == qualification.required_chip,
                    f"observed {snapshot.chip}; required {qualification.required_chip}",
                ),
                _check(
                    "hardware_machine",
                    snapshot.machine == qualification.required_machine,
                    f"observed {snapshot.machine}; required {qualification.required_machine}",
                ),
                _check(
                    "hardware_memory",
                    snapshot.physical_memory_bytes == qualification.required_memory_bytes,
                    f"observed {snapshot.physical_memory_bytes}; required "
                    f"{qualification.required_memory_bytes} bytes",
                ),
                _check(
                    "hardware_power",
                    snapshot.power_source == qualification.required_power_source,
                    f"observed {snapshot.power_source}; required "
                    f"{qualification.required_power_source}",
                ),
                _check(
                    "hardware_system",
                    snapshot.system == qualification.required_system,
                    f"observed {snapshot.system}; required {qualification.required_system}",
                ),
                _check(
                    "offline_config",
                    config.config.runtime.offline,
                    "runtime.offline must be true",
                ),
                _check(
                    "offline_network",
                    not network_attempts,
                    f"observed {len(network_attempts)} connection attempts",
                ),
                _check(
                    "peak_rss",
                    peak_rss_bytes <= rss_limit_bytes,
                    f"peak {peak_rss_bytes} bytes <= {rss_limit_bytes} bytes",
                ),
                _check(
                    "startup_latency",
                    startup_ms <= qualification.cold_startup_target_ms,
                    f"startup {startup_ms:.3f} ms <= {qualification.cold_startup_target_ms:.3f} ms",
                ),
                *mode_checks,
                *stage_checks,
            ),
            key=lambda check: check.check_id,
        )
    )
    return ReleaseQualificationReport(
        generation=qualification.generation,
        bundle_id=bundle.manifest.artifact_id,
        bundle_manifest_sha256=bundle.manifest_sha256,
        dataset_version=bundle.manifest.dataset_version,
        profile=bundle.manifest.profile,
        config_sha256=config.sha256,
        code_revision=code_revision,
        started_utc=started_utc,
        completed_utc=datetime.now(UTC),
        background_conditions=background_conditions,
        hardware=snapshot,
        dependencies=_dependency_versions(),
        max_threads=config.config.runtime.max_threads,
        query_sha256s=tuple(
            sha256(query.encode("utf-8")).hexdigest() for query in qualification.queries
        ),
        modes=qualification.modes,
        top_k=qualification.top_k,
        warmup_rounds=qualification.warmup_rounds,
        measured_rounds=qualification.measured_rounds,
        concurrency_workers=qualification.concurrency_workers,
        concurrency_rounds=qualification.concurrency_rounds,
        startup_ms=startup_ms,
        mode_latencies=mode_latencies,
        concurrent_latency=concurrent_latency,
        stage_latencies=stage_latencies,
        peak_rss_bytes=peak_rss_bytes,
        rss_limit_bytes=rss_limit_bytes,
        components=info.components,
        checks=checks,
        passed=all(check.passed for check in checks),
    )


def qualification_artifact_id(bundle: LoadedArtifact, config: ResolvedConfig) -> str:
    return "/".join(
        (
            "release-qualification",
            bundle.manifest.dataset_version,
            bundle.manifest.profile,
            config.config.qualification.component_version,
            config.sha256,
        )
    )


def _canonical_json(report: ReleaseQualificationReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _markdown(report: ReleaseQualificationReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    rows = "\n".join(
        f"| `{check.check_id}` | {'pass' if check.passed else 'fail'} | {check.detail} |"
        for check in report.checks
    )
    return (
        "# MarketRank Release Qualification\n\n"
        f"**Status:** {status}\n\n"
        f"- Bundle: `{report.bundle_id}`\n"
        f"- Config: `{report.config_sha256}`\n"
        f"- Host: {report.hardware.machine_model}, {report.hardware.chip}, "
        f"{report.hardware.physical_memory_bytes} bytes\n"
        f"- Peak RSS: {report.peak_rss_bytes}/{report.rss_limit_bytes} bytes\n"
        f"- Cold bundle startup: {report.startup_ms:.3f} ms\n\n"
        "| Check | Status | Evidence |\n|---|---|---|\n"
        f"{rows}\n"
    )


def load_qualification_report(path: Path) -> ReleaseQualificationReport:
    try:
        return ReleaseQualificationReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise QualificationValidationError(
            f"cannot load qualification report {path}: {exc}"
        ) from exc


def _failed_report_path(config: ResolvedConfig, observed_utc: datetime) -> Path:
    timestamp = observed_utc.strftime("%Y%m%dT%H%M%S%fZ")
    directory = config.config.paths.reports_dir / "generated" / "qualification" / "failed"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{timestamp}-{config.short_hash}.json"


def _write_new(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def _reuse(
    store: ArtifactStore,
    artifact_id: str,
    bundle: LoadedArtifact,
    config: ResolvedConfig,
) -> QualificationBuildResult:
    artifact = store.load(artifact_id)
    expected = (
        ArtifactDependency(
            artifact_id=bundle.manifest.artifact_id,
            manifest_sha256=bundle.manifest_sha256,
        ),
    )
    if artifact.manifest.dependencies != expected:
        raise QualificationValidationError("qualification bundle dependency is incompatible")
    report = load_qualification_report(artifact.path / QUALIFICATION_FILENAME)
    if (
        not report.passed
        or report.config_sha256 != config.sha256
        or report.bundle_id != bundle.manifest.artifact_id
    ):
        raise QualificationValidationError("qualification report identity is incompatible")
    return QualificationBuildResult(artifact=artifact, report=report, reused=True)


def build_release_qualification(
    config: ResolvedConfig,
    bundle_id: str,
    *,
    code_revision: str,
    background_conditions: str,
    artifact_store: ArtifactStore | None = None,
    hardware: HardwareSnapshot | None = None,
    runtime_loader: Callable[[ArtifactStore, str, ResolvedConfig], ServingRuntime] | None = None,
) -> QualificationBuildResult:
    """Measure and promote a passing release qualification for one explicit bundle."""
    if not background_conditions.strip():
        raise QualificationValidationError("background conditions must be explicitly recorded")
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    bundle = store.load(bundle_id)
    if bundle.manifest.artifact_type != "serving-bundle":
        raise QualificationValidationError("qualification requires a serving-bundle artifact")
    bundle_manifest = load_serving_bundle_manifest(bundle.path / SERVING_BUNDLE_FILENAME)
    if bundle_manifest.config_sha256 != config.sha256:
        raise QualificationValidationError("serving bundle differs from the resolved config")
    artifact_id = qualification_artifact_id(bundle, config)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse(store, artifact_id, bundle, config)

    try:
        report = measure_release_qualification(
            config,
            bundle,
            bundle_manifest,
            code_revision=code_revision,
            background_conditions=background_conditions.strip(),
            artifact_store=store,
            hardware=hardware,
            runtime_loader=runtime_loader,
        )
    except Exception as exc:
        failed_utc = datetime.now(UTC)
        reason_code: Literal["offline_network_attempt", "qualification_execution_failed"] = (
            "offline_network_attempt"
            if isinstance(exc, QualificationOfflineError)
            else "qualification_execution_failed"
        )
        failure = QualificationFailureRecord(
            bundle_id=bundle.manifest.artifact_id,
            bundle_manifest_sha256=bundle.manifest_sha256,
            config_sha256=config.sha256,
            code_revision=code_revision,
            background_conditions=background_conditions.strip(),
            failed_utc=failed_utc,
            reason_code=reason_code,
            error_type=type(exc).__name__,
        )
        failed_path = _failed_report_path(config, failed_utc)
        _write_new(
            failed_path,
            json.dumps(
                failure.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        raise QualificationExecutionError(reason_code, failed_path) from exc
    if not report.passed:
        failed_path = _failed_report_path(config, report.completed_utc)
        _write_new(failed_path, _canonical_json(report))
        raise QualificationGateError(report, failed_path)

    dependency = ArtifactDependency(
        artifact_id=bundle.manifest.artifact_id,
        manifest_sha256=bundle.manifest_sha256,
    )
    transaction = store.stage(
        artifact_type="release-qualification",
        dataset_version=bundle.manifest.dataset_version,
        profile=bundle.manifest.profile,
        component_version=config.config.qualification.component_version,
        config_sha256=config.sha256,
        code_revision=code_revision,
        dependencies=(dependency,),
    )
    try:
        with transaction:
            transaction.path(QUALIFICATION_FILENAME).write_text(
                _canonical_json(report), encoding="utf-8"
            )
            transaction.path(QUALIFICATION_MARKDOWN_FILENAME).write_text(
                _markdown(report), encoding="utf-8"
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse(store, artifact_id, bundle, config)
    return QualificationBuildResult(artifact=artifact, report=report, reused=False)


__all__ = [
    "QUALIFICATION_FILENAME",
    "HardwareSnapshot",
    "QualificationBuildResult",
    "QualificationError",
    "QualificationExecutionError",
    "QualificationFailureRecord",
    "QualificationGateError",
    "QualificationOfflineError",
    "QualificationValidationError",
    "ReleaseQualificationReport",
    "build_release_qualification",
    "load_qualification_report",
    "measure_release_qualification",
    "probe_hardware",
    "qualification_artifact_id",
]
