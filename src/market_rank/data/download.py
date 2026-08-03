"""Explicit, streamed, retry-bounded ESCI acquisition and validation workflow."""

from __future__ import annotations

import http.client
import os
import ssl
import tempfile
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self
from urllib.parse import urljoin, urlsplit, urlunsplit

from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import (
    RawDataError,
    RawValidationPublication,
    RawValidationReport,
    ResolvedReleaseManifest,
    ensure_raw_validation_artifact,
    validate_raw_dataset,
)

DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_USER_AGENT = (
    "MarketRank/0.1 Goldfish-004A "
    "(+https://github.com/amazon-science/esci-data)"
)
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_REDIRECT_HTTP_STATUSES = frozenset({301, 302, 303, 307, 308})


class EsciDownloadError(RawDataError):
    """Base exception for explicit ESCI acquisition failures."""


class DownloadPathError(EsciDownloadError):
    """Raised when a download target is unsafe or unavailable."""


class ExistingRawFileMismatchError(EsciDownloadError):
    """Raised when an existing immutable raw file does not match its manifest."""


class DownloadedFileMismatchError(EsciDownloadError):
    """Raised when streamed bytes do not match the pinned size or checksum."""


class DownloadRequestError(EsciDownloadError):
    """Base transport error classified for retry policy."""

    retryable: bool = False


class TransientDownloadError(DownloadRequestError):
    """A transport failure that may succeed within the bounded retry policy."""

    retryable = True


class PermanentDownloadError(DownloadRequestError):
    """A transport failure that must not be retried."""


class DownloadRetryExhaustedError(EsciDownloadError):
    """Raised after all bounded attempts fail transiently."""


class ReadableStream(Protocol):
    """Minimal response body contract used by the downloader."""

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes."""


class DownloadTransport(Protocol):
    """Injectable network boundary; implementations open no connection at import time."""

    def open(
        self,
        url: str,
        *,
        connect_timeout_s: float,
        read_timeout_s: float,
        user_agent: str,
    ) -> AbstractContextManager[ReadableStream]:
        """Open one validated response body or raise a classified request error."""


class _HttpResponseStream(AbstractContextManager[ReadableStream]):
    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
    ) -> None:
        self._connection = connection
        self._response = response

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._response.close()
        self._connection.close()

    def read(self, size: int = -1) -> bytes:
        try:
            return self._response.read(size)
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise TransientDownloadError(f"response body read failed: {exc}") from exc


class HttpDownloadTransport:
    """Standard-library HTTP(S) transport with bounded redirect handling."""

    def __init__(self, *, max_redirects: int = 5) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects must be non-negative")
        self._max_redirects = max_redirects

    def open(
        self,
        url: str,
        *,
        connect_timeout_s: float,
        read_timeout_s: float,
        user_agent: str,
    ) -> AbstractContextManager[ReadableStream]:
        current_url = url
        for redirect_count in range(self._max_redirects + 1):
            connection = self._open_connection(current_url, connect_timeout_s)
            split_url = urlsplit(current_url)
            request_target = urlunsplit(("", "", split_url.path or "/", split_url.query, ""))
            try:
                connection.request(
                    "GET",
                    request_target,
                    headers={
                        "Accept": "application/octet-stream",
                        "User-Agent": user_agent,
                    },
                )
                if connection.sock is not None:
                    connection.sock.settimeout(read_timeout_s)
                response = connection.getresponse()
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                connection.close()
                raise TransientDownloadError(f"request failed: {exc}") from exc

            if response.status in _REDIRECT_HTTP_STATUSES:
                location = response.getheader("Location")
                response.close()
                connection.close()
                if location is None:
                    raise PermanentDownloadError(
                        f"HTTP {response.status} redirect omitted Location"
                    )
                if redirect_count == self._max_redirects:
                    raise PermanentDownloadError("redirect limit exceeded")
                redirected_url = urljoin(current_url, location)
                if (
                    urlsplit(current_url).scheme == "https"
                    and urlsplit(redirected_url).scheme != "https"
                ):
                    raise PermanentDownloadError("HTTPS download cannot redirect to HTTP")
                current_url = redirected_url
                continue

            if response.status == 200:
                return _HttpResponseStream(connection, response)

            status = response.status
            reason = response.reason
            response.close()
            connection.close()
            message = f"HTTP {status} {reason}"
            if status in _RETRYABLE_HTTP_STATUSES:
                raise TransientDownloadError(message)
            raise PermanentDownloadError(message)

        raise PermanentDownloadError("redirect limit exceeded")

    @staticmethod
    def _open_connection(url: str, connect_timeout_s: float) -> http.client.HTTPConnection:
        split_url = urlsplit(url)
        if split_url.scheme not in {"http", "https"}:
            raise PermanentDownloadError(f"unsupported URL scheme: {split_url.scheme!r}")
        if split_url.hostname is None or split_url.username is not None:
            raise PermanentDownloadError("download URL must have a host and no credentials")

        try:
            port = split_url.port
        except ValueError as exc:
            raise PermanentDownloadError(f"invalid URL port: {exc}") from exc
        if split_url.scheme == "https":
            return http.client.HTTPSConnection(
                split_url.hostname,
                port=port,
                timeout=connect_timeout_s,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(
            split_url.hostname,
            port=port,
            timeout=connect_timeout_s,
        )


@dataclass(frozen=True, slots=True)
class DownloadPolicy:
    """Bounded resource, timeout, redirect, and retry controls."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    connect_timeout_s: float = 15.0
    read_timeout_s: float = 60.0
    max_attempts: int = 3
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 4.0
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        if self.chunk_size < 4096 or self.chunk_size > 8 * 1024 * 1024:
            raise ValueError("chunk_size must be between 4 KiB and 8 MiB")
        if self.connect_timeout_s <= 0 or self.read_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_attempts < 1 or self.max_attempts > 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if self.initial_backoff_s < 0 or self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("backoff bounds are invalid")
        if not self.user_agent.strip():
            raise ValueError("user_agent must not be empty")

    def backoff_after(self, failed_attempt: int) -> float:
        """Return capped exponential backoff after a one-based failed attempt."""
        return min(self.max_backoff_s, self.initial_backoff_s * 2.0 ** (failed_attempt - 1))


FileAcquisitionStatus = Literal["downloaded", "reused"]


@dataclass(frozen=True, slots=True)
class FileAcquisition:
    """Verified local state for one pinned source file."""

    filename: str
    status: FileAcquisitionStatus
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Verified raw files and the actual completion timestamp of this acquisition pass."""

    raw_root: Path
    files: tuple[FileAcquisition, ...]
    completed_utc: datetime


@dataclass(frozen=True, slots=True)
class DownloadWorkflowResult:
    """Completed acquisition, validation, and artifact publication state."""

    acquisition: AcquisitionResult
    report: RawValidationReport
    publication: RawValidationPublication


ProgressCallback = Callable[[str], None]
SleepCallback = Callable[[float], None]
Clock = Callable[[], datetime]


class RawValidator(Protocol):
    """Injectable Goldfish 004 validation boundary."""

    def __call__(
        self,
        release: ResolvedReleaseManifest,
        raw_root: Path,
        *,
        retrieved_utc: datetime,
    ) -> RawValidationReport:
        """Validate all acquired raw files."""


class RawPublisher(Protocol):
    """Injectable Goldfish 003 publication boundary."""

    def __call__(
        self,
        release: ResolvedReleaseManifest,
        report: RawValidationReport,
        store: ArtifactStore,
        *,
        config_sha256: str,
        code_revision: str,
    ) -> RawValidationPublication:
        """Publish or reuse compatible validation evidence."""


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _path_is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _sha256_file(path: Path, chunk_size: int) -> tuple[int, str]:
    digest = sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()


def _verify_existing_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    chunk_size: int,
) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ExistingRawFileMismatchError(
            f"existing raw path is not a regular file and was left untouched: {path.name}"
        )
    try:
        size_bytes, file_sha256 = _sha256_file(path, chunk_size)
    except OSError as exc:
        raise ExistingRawFileMismatchError(
            f"cannot verify existing raw file {path.name}: {exc}"
        ) from exc
    if size_bytes != expected_size or file_sha256 != expected_sha256:
        raise ExistingRawFileMismatchError(
            f"existing raw file {path.name} does not match the pinned size/checksum "
            "and was left untouched"
        )
    return size_bytes, file_sha256


def _download_once(
    *,
    source_url: str,
    filename: str,
    target_path: Path,
    expected_size: int,
    expected_sha256: str,
    transport: DownloadTransport,
    policy: DownloadPolicy,
) -> FileAcquisition:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.partial-",
        suffix=".tmp",
        dir=target_path.parent,
    )
    temporary_path: Path | None = Path(temporary_name)
    digest = sha256()
    size_bytes = 0
    try:
        with (
            os.fdopen(descriptor, "wb") as output,
            transport.open(
                source_url,
                connect_timeout_s=policy.connect_timeout_s,
                read_timeout_s=policy.read_timeout_s,
                user_agent=policy.user_agent,
            ) as response,
        ):
            while chunk := response.read(policy.chunk_size):
                if not isinstance(chunk, bytes):
                    raise PermanentDownloadError("transport returned non-byte response data")
                output.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
                if size_bytes > expected_size:
                    raise DownloadedFileMismatchError(
                        f"downloaded {filename} exceeded pinned size {expected_size} bytes"
                    )
            output.flush()
            os.fsync(output.fileno())

        file_sha256 = digest.hexdigest()
        if size_bytes != expected_size or file_sha256 != expected_sha256:
            raise DownloadedFileMismatchError(
                f"downloaded {filename} failed pinned integrity: expected "
                f"{expected_size} bytes/{expected_sha256}, observed "
                f"{size_bytes} bytes/{file_sha256}"
            )

        if _path_is_present(target_path):
            reused_size, reused_sha256 = _verify_existing_file(
                target_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                chunk_size=policy.chunk_size,
            )
            return FileAcquisition(
                filename=filename,
                status="reused",
                size_bytes=reused_size,
                sha256=reused_sha256,
            )

        if temporary_path is None:
            raise EsciDownloadError("temporary download state was lost before promotion")
        os.rename(temporary_path, target_path)
        temporary_path = None
        _fsync_directory(target_path.parent)
        return FileAcquisition(
            filename=filename,
            status="downloaded",
            size_bytes=size_bytes,
            sha256=file_sha256,
        )
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _download_with_retries(
    *,
    source_url: str,
    filename: str,
    target_path: Path,
    expected_size: int,
    expected_sha256: str,
    transport: DownloadTransport,
    policy: DownloadPolicy,
    progress: ProgressCallback | None,
    sleeper: SleepCallback,
) -> FileAcquisition:
    for attempt in range(1, policy.max_attempts + 1):
        _emit(progress, f"downloading {filename} (attempt {attempt}/{policy.max_attempts})")
        try:
            return _download_once(
                source_url=source_url,
                filename=filename,
                target_path=target_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                transport=transport,
                policy=policy,
            )
        except TransientDownloadError as exc:
            if attempt == policy.max_attempts:
                raise DownloadRetryExhaustedError(
                    f"download failed transiently after {attempt} attempts for {filename}: {exc}"
                ) from exc
            sleeper(policy.backoff_after(attempt))

    raise DownloadRetryExhaustedError(f"download attempts exhausted for {filename}")


def acquire_esci_files(
    release: ResolvedReleaseManifest,
    raw_root: Path,
    *,
    transport: DownloadTransport | None = None,
    policy: DownloadPolicy | None = None,
    progress: ProgressCallback | None = None,
    sleeper: SleepCallback = time.sleep,
    clock: Clock | None = None,
) -> AcquisitionResult:
    """Reuse or stream all pinned files, promoting only exact verified bytes."""
    selected_policy = policy or DownloadPolicy()
    selected_transport = transport or HttpDownloadTransport()
    selected_clock = clock or (lambda: datetime.now(UTC))

    if raw_root.is_symlink():
        raise DownloadPathError(f"raw root cannot be a symbolic link: {raw_root}")
    try:
        raw_root.mkdir(parents=True, exist_ok=True)
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise DownloadPathError(f"cannot create raw root {raw_root}: {exc}") from exc
    if not root.is_dir():
        raise DownloadPathError(f"raw root is not a directory: {root}")

    acquired: list[FileAcquisition] = []
    for source in release.manifest.files:
        target_path = root / source.filename
        if _path_is_present(target_path):
            _emit(progress, f"verifying existing {source.filename}")
            size_bytes, file_sha256 = _verify_existing_file(
                target_path,
                expected_size=source.size_bytes,
                expected_sha256=source.sha256,
                chunk_size=selected_policy.chunk_size,
            )
            file_result = FileAcquisition(
                filename=source.filename,
                status="reused",
                size_bytes=size_bytes,
                sha256=file_sha256,
            )
        else:
            file_result = _download_with_retries(
                source_url=str(source.source_url),
                filename=source.filename,
                target_path=target_path,
                expected_size=source.size_bytes,
                expected_sha256=source.sha256,
                transport=selected_transport,
                policy=selected_policy,
                progress=progress,
                sleeper=sleeper,
            )
        acquired.append(file_result)
        _emit(
            progress,
            f"{file_result.status} {file_result.filename} "
            f"({file_result.size_bytes} bytes verified)",
        )

    completed_utc = selected_clock()
    if completed_utc.tzinfo is None or completed_utc.utcoffset() != timedelta(0):
        raise EsciDownloadError("acquisition clock must return timezone-aware UTC")
    return AcquisitionResult(
        raw_root=root,
        files=tuple(acquired),
        completed_utc=completed_utc,
    )


def _failed_check_count(report: RawValidationReport) -> int:
    return sum(not check.passed for file in report.files for check in file.checks) + sum(
        not check.passed for check in report.dataset_checks
    )


def download_validate_esci(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    raw_root: Path | None = None,
    artifact_store: ArtifactStore | None = None,
    transport: DownloadTransport | None = None,
    policy: DownloadPolicy | None = None,
    progress: ProgressCallback | None = None,
    sleeper: SleepCallback = time.sleep,
    clock: Clock | None = None,
    validator: RawValidator = validate_raw_dataset,
    publisher: RawPublisher = ensure_raw_validation_artifact,
) -> DownloadWorkflowResult:
    """Explicitly acquire, validate, and publish or reuse ESCI validation evidence."""
    selected_raw_root = raw_root or config.config.paths.data_dir / "raw" / "esci"
    selected_store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    acquisition = acquire_esci_files(
        release,
        selected_raw_root,
        transport=transport,
        policy=policy,
        progress=progress,
        sleeper=sleeper,
        clock=clock,
    )

    _emit(progress, "validating raw ESCI dataset")
    report = validator(
        release,
        acquisition.raw_root,
        retrieved_utc=acquisition.completed_utc,
    )
    if not report.valid:
        _emit(progress, f"validation failed ({_failed_check_count(report)} checks failed)")
        report.require_valid()
    _emit(progress, "validation passed")

    publication = publisher(
        release,
        report,
        selected_store,
        config_sha256=config.sha256,
        code_revision=code_revision,
    )
    publication_status = "reused" if publication.reused else "published"
    _emit(
        progress,
        f"{publication_status} validation artifact: {publication.artifact.manifest.artifact_id}",
    )
    return DownloadWorkflowResult(
        acquisition=acquisition,
        report=report,
        publication=publication,
    )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_USER_AGENT",
    "AcquisitionResult",
    "DownloadPathError",
    "DownloadPolicy",
    "DownloadRequestError",
    "DownloadRetryExhaustedError",
    "DownloadTransport",
    "DownloadWorkflowResult",
    "DownloadedFileMismatchError",
    "EsciDownloadError",
    "ExistingRawFileMismatchError",
    "FileAcquisition",
    "HttpDownloadTransport",
    "PermanentDownloadError",
    "TransientDownloadError",
    "acquire_esci_files",
    "download_validate_esci",
]
