"""Offline tests for Goldfish 004A acquisition, orchestration, and CLI behavior."""

from __future__ import annotations

import importlib
import json
import socket
from argparse import Namespace
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Self

import pytest

import market_rank.cli as cli_module
import market_rank.data.download as download_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import load_config
from market_rank.data.download import (
    DownloadedFileMismatchError,
    DownloadPolicy,
    DownloadRetryExhaustedError,
    DownloadTransport,
    ExistingRawFileMismatchError,
    HttpDownloadTransport,
    PermanentDownloadError,
    ReadableStream,
    TransientDownloadError,
    acquire_esci_files,
    download_validate_esci,
)
from market_rank.data.esci_raw import (
    OFFICIAL_PAPER,
    OFFICIAL_REPOSITORY,
    EsciReleaseManifest,
    RawDataValidationError,
    RawFileSource,
    RawFileValidation,
    RawValidationPublication,
    RawValidationReport,
    ResolvedReleaseManifest,
    ValidationCheck,
    load_release_manifest,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
BASE_CONFIG = REPOSITORY_ROOT / "configs" / "base.yaml"
REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
CONFIG_SHA256 = load_config([BASE_CONFIG]).sha256
NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

PAYLOADS = {
    "shopping_queries_dataset_examples.parquet": b"examples-fixture-bytes",
    "shopping_queries_dataset_products.parquet": b"products-fixture-bytes",
    "shopping_queries_dataset_sources.csv": b"query_id,source\n1,other\n",
}


class _BytesResponse(AbstractContextManager[ReadableStream]):
    def __init__(self, content: bytes) -> None:
        self._content = content
        self._offset = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._content):
            return b""
        end = len(self._content) if size < 0 else self._offset + size
        chunk = self._content[self._offset : end]
        self._offset += len(chunk)
        return chunk


class _InterruptedResponse(_BytesResponse):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads > 1:
            raise TransientDownloadError("injected interrupted response")
        return super().read(4)


ResponseOutcome = bytes | TransientDownloadError | PermanentDownloadError | _InterruptedResponse


class _FakeTransport:
    def __init__(self, outcomes: dict[str, list[ResponseOutcome]]) -> None:
        self._outcomes = {url: list(values) for url, values in outcomes.items()}
        self.calls: list[str] = []

    def open(
        self,
        url: str,
        *,
        connect_timeout_s: float,
        read_timeout_s: float,
        user_agent: str,
    ) -> AbstractContextManager[ReadableStream]:
        assert connect_timeout_s > 0
        assert read_timeout_s > 0
        assert "MarketRank" in user_agent
        self.calls.append(url)
        outcome = self._outcomes[url].pop(0)
        if isinstance(outcome, (TransientDownloadError, PermanentDownloadError)):
            raise outcome
        if isinstance(outcome, bytes):
            return _BytesResponse(outcome)
        return outcome


def _release(
    tmp_path: Path,
    *,
    expected_payloads: dict[str, bytes] | None = None,
) -> ResolvedReleaseManifest:
    payloads = expected_payloads or PAYLOADS
    sources: list[RawFileSource] = []
    for role, filename, file_format in (
        ("examples", "shopping_queries_dataset_examples.parquet", "parquet"),
        ("products", "shopping_queries_dataset_products.parquet", "parquet"),
        ("sources", "shopping_queries_dataset_sources.csv", "csv"),
    ):
        content = payloads[filename]
        sources.append(
            RawFileSource.model_validate(
                {
                    "role": role,
                    "filename": filename,
                    "format": file_format,
                    "source_url": (
                        f"{OFFICIAL_REPOSITORY}/raw/{REVISION}/shopping_queries_dataset/{filename}"
                    ),
                    "size_bytes": len(content),
                    "sha256": sha256(content).hexdigest(),
                }
            )
        )

    manifest = EsciReleaseManifest.model_validate(
        {
            "dataset_version": f"esci-{REVISION[:12]}",
            "source_repository": OFFICIAL_REPOSITORY,
            "source_revision": REVISION,
            "source_commit_utc": datetime(2024, 10, 7, 15, 52, 6, tzinfo=UTC),
            "license_url": f"{OFFICIAL_REPOSITORY}/blob/{REVISION}/LICENSE",
            "paper_url": OFFICIAL_PAPER,
            "files": tuple(sources),
        }
    )
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return load_release_manifest(path)


def _transport_for(
    release: ResolvedReleaseManifest,
    payloads: dict[str, bytes] | None = None,
) -> _FakeTransport:
    selected = payloads or PAYLOADS
    return _FakeTransport(
        {str(source.source_url): [selected[source.filename]] for source in release.manifest.files}
    )


def _valid_report(
    release: ResolvedReleaseManifest,
    retrieved_utc: datetime,
    *,
    valid: bool = True,
) -> RawValidationReport:
    file_reports: list[RawFileValidation] = []
    for index, source in enumerate(release.manifest.files):
        passed = valid or index > 0
        check = ValidationCheck(
            check_id="fixture_integrity",
            passed=passed,
            detail="fixture passed" if passed else "fixture failed",
        )
        file_reports.append(
            RawFileValidation(
                role=source.role,
                filename=source.filename,
                size_bytes=source.size_bytes,
                sha256=source.sha256,
                row_count=1,
                checks=(check,),
                valid=passed,
            )
        )
    dataset_check = ValidationCheck(
        check_id="fixture_joins",
        passed=True,
        detail="fixture joins passed",
    )
    return RawValidationReport(
        dataset_version=release.manifest.dataset_version,
        source_revision=release.manifest.source_revision,
        release_manifest_sha256=release.sha256,
        retrieved_utc=retrieved_utc,
        files=tuple(file_reports),
        dataset_checks=(dataset_check,),
        valid=valid,
    )


def _fixture_validator(
    release: ResolvedReleaseManifest,
    raw_root: Path,
    *,
    retrieved_utc: datetime,
) -> RawValidationReport:
    assert raw_root.is_dir()
    return _valid_report(release, retrieved_utc)


def _policy(*, attempts: int = 1) -> DownloadPolicy:
    return DownloadPolicy(
        chunk_size=4096,
        connect_timeout_s=1.0,
        read_timeout_s=1.0,
        max_attempts=attempts,
        initial_backoff_s=0.0,
        max_backoff_s=0.0,
    )


def _partial_files(raw_root: Path) -> tuple[Path, ...]:
    return tuple(raw_root.glob(".*.partial-*.tmp")) if raw_root.exists() else ()


def test_successfully_streams_and_verifies_all_three_files(tmp_path: Path) -> None:
    release = _release(tmp_path)
    transport = _transport_for(release)
    progress: list[str] = []

    result = acquire_esci_files(
        release,
        tmp_path / "raw",
        transport=transport,
        policy=_policy(),
        progress=progress.append,
        clock=lambda: NOW,
    )

    assert tuple(file.status for file in result.files) == ("downloaded",) * 3
    assert tuple(file.size_bytes for file in result.files) == tuple(
        len(PAYLOADS[file.filename]) for file in result.files
    )
    assert result.completed_utc == NOW
    assert len(transport.calls) == 3
    assert not _partial_files(result.raw_root)
    for file in result.files:
        assert (result.raw_root / file.filename).read_bytes() == PAYLOADS[file.filename]
        assert file.sha256 == sha256(PAYLOADS[file.filename]).hexdigest()
    assert any(message.startswith("downloading") for message in progress)


class _FakeSocket:
    def settimeout(self, timeout: float) -> None:
        assert timeout == 1.0


class _FakeHttpResponse:
    def __init__(
        self,
        status: int,
        *,
        content: bytes = b"",
        location: str | None = None,
    ) -> None:
        self.status = status
        self.reason = "fixture"
        self._content = content
        self._location = location
        self._closed = False

    def getheader(self, name: str) -> str | None:
        return self._location if name == "Location" else None

    def read(self, size: int = -1) -> bytes:
        del size
        content = self._content
        self._content = b""
        return content

    def close(self) -> None:
        self._closed = True


class _FakeHttpConnection:
    def __init__(self, response: _FakeHttpResponse) -> None:
        self.sock = _FakeSocket()
        self._response = response
        self.request_target: str | None = None
        self.closed = False

    def request(self, method: str, target: str, headers: dict[str, str]) -> None:
        assert method == "GET"
        assert "User-Agent" in headers
        self.request_target = target

    def getresponse(self) -> _FakeHttpResponse:
        return self._response

    def close(self) -> None:
        self.closed = True


def test_standard_library_transport_follows_bounded_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"redirected-response"
    connections = [
        _FakeHttpConnection(_FakeHttpResponse(302, location="/payload")),
        _FakeHttpConnection(_FakeHttpResponse(200, content=payload)),
    ]
    opened_urls: list[str] = []

    def fake_open_connection(url: str, connect_timeout_s: float) -> _FakeHttpConnection:
        assert connect_timeout_s == 1.0
        opened_urls.append(url)
        return connections[len(opened_urls) - 1]

    monkeypatch.setattr(
        HttpDownloadTransport,
        "_open_connection",
        staticmethod(fake_open_connection),
    )

    with HttpDownloadTransport(max_redirects=2).open(
        "https://example.test/redirect",
        connect_timeout_s=1.0,
        read_timeout_s=1.0,
        user_agent="MarketRank-test",
    ) as response:
        assert response.read() == payload

    assert opened_urls == [
        "https://example.test/redirect",
        "https://example.test/payload",
    ]
    assert connections[0].closed


def test_matching_existing_files_are_reused_without_transport(tmp_path: Path) -> None:
    release = _release(tmp_path)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for filename, content in PAYLOADS.items():
        (raw_root / filename).write_bytes(content)
    transport = _transport_for(release)

    result = acquire_esci_files(
        release,
        raw_root,
        transport=transport,
        policy=_policy(),
        clock=lambda: NOW,
    )

    assert tuple(file.status for file in result.files) == ("reused",) * 3
    assert transport.calls == []


def test_mismatched_existing_file_is_left_untouched(tmp_path: Path) -> None:
    release = _release(tmp_path)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    target = raw_root / release.manifest.files[0].filename
    target.write_bytes(b"user-owned-mismatch")
    transport = _transport_for(release)

    with pytest.raises(ExistingRawFileMismatchError, match="left untouched"):
        acquire_esci_files(
            release,
            raw_root,
            transport=transport,
            policy=_policy(),
        )

    assert target.read_bytes() == b"user-owned-mismatch"
    assert transport.calls == []


def test_downloaded_checksum_mismatch_is_rejected_and_cleaned(tmp_path: Path) -> None:
    expected = dict(PAYLOADS)
    actual = dict(PAYLOADS)
    filename = "shopping_queries_dataset_examples.parquet"
    actual[filename] = b"X" * len(expected[filename])
    release = _release(tmp_path, expected_payloads=expected)

    with pytest.raises(DownloadedFileMismatchError, match="failed pinned integrity"):
        acquire_esci_files(
            release,
            tmp_path / "raw",
            transport=_transport_for(release, actual),
            policy=_policy(),
        )

    assert not (tmp_path / "raw" / filename).exists()
    assert not _partial_files(tmp_path / "raw")


def test_downloaded_size_mismatch_is_rejected_and_cleaned(tmp_path: Path) -> None:
    actual = dict(PAYLOADS)
    filename = "shopping_queries_dataset_examples.parquet"
    actual[filename] = PAYLOADS[filename][:-1]
    release = _release(tmp_path)

    with pytest.raises(DownloadedFileMismatchError, match="expected"):
        acquire_esci_files(
            release,
            tmp_path / "raw",
            transport=_transport_for(release, actual),
            policy=_policy(),
        )

    assert not (tmp_path / "raw" / filename).exists()
    assert not _partial_files(tmp_path / "raw")


def test_handled_interrupted_download_removes_partial_file(tmp_path: Path) -> None:
    release = _release(tmp_path)
    first = release.manifest.files[0]
    transport = _transport_for(release)
    transport = _FakeTransport(
        {str(first.source_url): [_InterruptedResponse(PAYLOADS[first.filename])]}
    )

    with pytest.raises(DownloadRetryExhaustedError, match="after 1 attempts"):
        acquire_esci_files(
            release,
            tmp_path / "raw",
            transport=transport,
            policy=_policy(),
        )

    assert not _partial_files(tmp_path / "raw")


def test_transient_failure_retries_within_bound(tmp_path: Path) -> None:
    release = _release(tmp_path)
    first = release.manifest.files[0]
    outcomes: dict[str, list[ResponseOutcome]] = {
        str(source.source_url): [PAYLOADS[source.filename]] for source in release.manifest.files
    }
    outcomes[str(first.source_url)] = [
        TransientDownloadError("temporary outage"),
        PAYLOADS[first.filename],
    ]
    transport = _FakeTransport(outcomes)
    sleeps: list[float] = []

    acquire_esci_files(
        release,
        tmp_path / "raw",
        transport=transport,
        policy=_policy(attempts=2),
        sleeper=sleeps.append,
        clock=lambda: NOW,
    )

    assert transport.calls.count(str(first.source_url)) == 2
    assert sleeps == [0.0]


def test_permanent_failure_is_not_retried(tmp_path: Path) -> None:
    release = _release(tmp_path)
    first = release.manifest.files[0]
    transport = _FakeTransport(
        {str(first.source_url): [PermanentDownloadError("HTTP 404 Not Found")]}
    )

    with pytest.raises(PermanentDownloadError, match="404"):
        acquire_esci_files(
            release,
            tmp_path / "raw",
            transport=transport,
            policy=_policy(attempts=3),
        )

    assert transport.calls == [str(first.source_url)]


def test_validation_runs_only_after_every_file_verifies(tmp_path: Path) -> None:
    release = _release(tmp_path)
    first, second, _ = release.manifest.files
    transport = _FakeTransport(
        {
            str(first.source_url): [PAYLOADS[first.filename]],
            str(second.source_url): [PermanentDownloadError("HTTP 403 Forbidden")],
        }
    )
    validation_calls = 0

    def validator(
        release: ResolvedReleaseManifest,
        raw_root: Path,
        *,
        retrieved_utc: datetime,
    ) -> RawValidationReport:
        del release, raw_root, retrieved_utc
        nonlocal validation_calls
        validation_calls += 1
        raise AssertionError("validation must not run")

    with pytest.raises(PermanentDownloadError, match="403"):
        download_validate_esci(
            release,
            load_config([BASE_CONFIG]),
            code_revision="abc123",
            raw_root=tmp_path / "raw",
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            transport=transport,
            policy=_policy(),
            validator=validator,
        )

    assert validation_calls == 0


def test_invalid_validation_report_is_never_published(tmp_path: Path) -> None:
    release = _release(tmp_path)
    publisher_calls = 0

    def invalid_validator(
        release: ResolvedReleaseManifest,
        raw_root: Path,
        *,
        retrieved_utc: datetime,
    ) -> RawValidationReport:
        assert raw_root.is_dir()
        return _valid_report(release, retrieved_utc, valid=False)

    def publisher(
        release: ResolvedReleaseManifest,
        report: RawValidationReport,
        store: ArtifactStore,
        *,
        config_sha256: str,
        code_revision: str,
    ) -> RawValidationPublication:
        del release, report, store, config_sha256, code_revision
        nonlocal publisher_calls
        publisher_calls += 1
        raise AssertionError("invalid report must not publish")

    with pytest.raises(RawDataValidationError) as caught:
        download_validate_esci(
            release,
            load_config([BASE_CONFIG]),
            code_revision="abc123",
            raw_root=tmp_path / "raw",
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            transport=_transport_for(release),
            policy=_policy(),
            clock=lambda: NOW,
            validator=invalid_validator,
            publisher=publisher,
        )

    assert not caught.value.report.valid
    assert publisher_calls == 0


def test_successful_workflow_publishes_validation_artifact(tmp_path: Path) -> None:
    release = _release(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    progress: list[str] = []

    result = download_validate_esci(
        release,
        load_config([BASE_CONFIG]),
        code_revision="abc123",
        raw_root=tmp_path / "raw",
        artifact_store=store,
        transport=_transport_for(release),
        policy=_policy(),
        progress=progress.append,
        clock=lambda: NOW,
        validator=_fixture_validator,
    )

    assert not result.publication.reused
    assert store.load(result.publication.artifact.manifest.artifact_id) == (
        result.publication.artifact
    )
    assert "validation passed" in progress
    assert progress[-1].startswith("published validation artifact:")


def test_rerun_reuses_files_and_compatible_artifact_without_network(tmp_path: Path) -> None:
    release = _release(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    config = load_config([BASE_CONFIG])
    raw_root = tmp_path / "raw"
    first = download_validate_esci(
        release,
        config,
        code_revision="first-revision",
        raw_root=raw_root,
        artifact_store=store,
        transport=_transport_for(release),
        policy=_policy(),
        clock=lambda: NOW,
        validator=_fixture_validator,
    )
    second_transport = _transport_for(release)
    later = datetime(2026, 8, 1, 13, 0, tzinfo=UTC)

    second = download_validate_esci(
        release,
        config,
        code_revision="second-revision",
        raw_root=raw_root,
        artifact_store=store,
        transport=second_transport,
        policy=_policy(),
        clock=lambda: later,
        validator=_fixture_validator,
    )

    assert tuple(file.status for file in second.acquisition.files) == ("reused",) * 3
    assert second_transport.calls == []
    assert second.publication.reused
    assert second.publication.artifact == first.publication.artifact


def test_cli_defaults_and_concise_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(arguments: Namespace) -> int:
        assert arguments.manifest == cli_module.DEFAULT_ESCI_MANIFEST
        assert arguments.config == cli_module.DEFAULT_CONFIG
        print("downloaded fixture (10 bytes verified)")
        print("validation passed")
        print("published validation artifact: raw-validation/example")
        return 0

    monkeypatch.setattr(cli_module, "_run_download_esci", fake_run)

    exit_code = cli_module.main(["data", "download-esci"])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "downloaded fixture" in output.out
    assert "validation passed" in output.out
    assert "validation artifact" in output.out
    assert output.err == ""


def test_cli_returns_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(arguments: Namespace) -> int:
        del arguments
        raise ExistingRawFileMismatchError("existing raw file is incompatible")

    monkeypatch.setattr(cli_module, "_run_download_esci", fail)

    exit_code = cli_module.main(["data", "download-esci"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err.strip() == "error: existing raw file is incompatible"
    assert "Traceback" not in output.err


def test_cli_summarizes_invalid_validation_but_exception_retains_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    release = _release(tmp_path)
    report = _valid_report(release, NOW, valid=False)

    def fail(arguments: Namespace) -> int:
        del arguments
        raise RawDataValidationError(report)

    monkeypatch.setattr(cli_module, "_run_download_esci", fail)

    exit_code = cli_module.main(["data", "download-esci"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err.strip() == "error: raw ESCI validation failed (1 checks failed)"
    assert "Traceback" not in output.err
    with pytest.raises(RawDataValidationError) as caught:
        fail(Namespace())
    assert caught.value.report == report


def test_importing_cli_and_downloader_opens_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_connection(*args: object, **kwargs: object) -> socket.socket:
        del args, kwargs
        raise AssertionError("imports must not open network connections")

    monkeypatch.setattr(socket, "create_connection", forbid_connection)

    importlib.reload(download_module)
    importlib.reload(cli_module)


def test_transport_protocol_is_satisfied() -> None:
    transport: DownloadTransport = HttpDownloadTransport()
    assert isinstance(transport, HttpDownloadTransport)
