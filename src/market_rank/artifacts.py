"""Strict artifact manifests and atomic stage-output promotion."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

MANIFEST_FILENAME = "manifest.json"
SUCCESS_FILENAME = "_SUCCESS"
_RESERVED_FILENAMES = frozenset({MANIFEST_FILENAME, SUCCESS_FILENAME})

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
ArtifactSegment = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]


class ArtifactError(RuntimeError):
    """Base exception for artifact protocol failures."""


class ArtifactPathError(ArtifactError):
    """Raised when an artifact or payload path escapes its allowed boundary."""


class ArtifactExistsError(ArtifactError):
    """Raised when immutable artifact coordinates have already been promoted."""


class ArtifactWriteError(ArtifactError):
    """Raised when a staged artifact cannot be safely promoted."""


class ArtifactValidationError(ArtifactError):
    """Raised when a persisted artifact fails integrity or compatibility checks."""


class _StrictModel(BaseModel):
    """Shared immutable, unknown-key-rejecting manifest behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_payload_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("payload paths must use POSIX '/' separators")

    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("payload path must be a normalized relative path")
    if path.as_posix() != value:
        raise ValueError("payload path must be normalized")
    if len(path.parts) == 1 and value in _RESERVED_FILENAMES:
        raise ValueError(f"payload path is reserved: {value}")
    return value


class ArtifactFile(_StrictModel):
    """Integrity metadata for one regular payload file."""

    relative_path: str
    size_bytes: int = Field(strict=True, ge=0)
    sha256: Sha256Digest

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Require a portable path that cannot escape the artifact directory."""
        return _validate_payload_path(value)


class ArtifactDependency(_StrictModel):
    """Exact parent-manifest identity required by a consumer artifact."""

    artifact_id: str = Field(strict=True, min_length=1)
    manifest_sha256: Sha256Digest

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        """Require the canonical five-segment artifact identifier."""
        _parse_artifact_id(value)
        return value


class ArtifactManifest(_StrictModel):
    """Versioned compatibility and integrity contract for one artifact."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    artifact_type: ArtifactSegment
    dataset_version: ArtifactSegment
    profile: ArtifactSegment
    component_version: ArtifactSegment
    config_sha256: Sha256Digest
    created_utc: datetime
    code_revision: str = Field(strict=True, min_length=1)
    dependencies: tuple[ArtifactDependency, ...] = ()
    files: tuple[ArtifactFile, ...] = Field(min_length=1)

    @field_validator("created_utc")
    @classmethod
    def validate_created_utc(cls, value: datetime) -> datetime:
        """Require an aware UTC timestamp so manifests are unambiguous."""
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("created_utc must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_identity_and_collections(self) -> Self:
        """Tie the ID to coordinates and require unique canonical ordering."""
        expected_id = _build_artifact_id(
            artifact_type=self.artifact_type,
            dataset_version=self.dataset_version,
            profile=self.profile,
            component_version=self.component_version,
            config_sha256=self.config_sha256,
        )
        if self.artifact_id != expected_id:
            raise ValueError(f"artifact_id must equal {expected_id!r}")

        dependency_ids = tuple(item.artifact_id for item in self.dependencies)
        if dependency_ids != tuple(sorted(dependency_ids)) or len(dependency_ids) != len(
            set(dependency_ids)
        ):
            raise ValueError("dependencies must be unique and sorted by artifact_id")

        file_paths = tuple(item.relative_path for item in self.files)
        if file_paths != tuple(sorted(file_paths)) or len(file_paths) != len(set(file_paths)):
            raise ValueError("files must be unique and sorted by relative_path")
        return self


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """A fully verified promoted artifact."""

    path: Path
    manifest: ArtifactManifest
    manifest_sha256: str


def _build_artifact_id(
    *,
    artifact_type: str,
    dataset_version: str,
    profile: str,
    component_version: str,
    config_sha256: str,
) -> str:
    return "/".join((artifact_type, dataset_version, profile, component_version, config_sha256))


def _parse_artifact_id(artifact_id: str) -> tuple[str, str, str, str, str]:
    if "\\" in artifact_id:
        raise ValueError("artifact_id must use POSIX '/' separators")
    parts = artifact_id.split("/")
    if len(parts) != 5:
        raise ValueError("artifact_id must contain exactly five path segments")

    artifact_type, dataset_version, profile, component_version, config_sha256 = parts
    for name, value in (
        ("artifact_type", artifact_type),
        ("dataset_version", dataset_version),
        ("profile", profile),
        ("component_version", component_version),
    ):
        if (
            not value
            or not value.isascii()
            or not value[0].isalnum()
            or any(not (character.isalnum() or character in "._-") for character in value)
        ):
            raise ValueError(f"invalid {name} segment: {value!r}")
    if len(config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in config_sha256
    ):
        raise ValueError("config_sha256 must be 64 lowercase hexadecimal characters")
    return artifact_type, dataset_version, profile, component_version, config_sha256


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
        os.fsync(stream.fileno())
    return size_bytes, digest.hexdigest()


def _canonical_manifest_bytes(manifest: ArtifactManifest) -> bytes:
    document = manifest.model_dump(mode="json")
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_durable_file(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _scan_payloads(
    directory: Path,
    *,
    ignore_protocol_files: bool = False,
) -> tuple[ArtifactFile, ...]:
    payloads: list[ArtifactFile] = []
    error_type = ArtifactValidationError if ignore_protocol_files else ArtifactWriteError
    for candidate in sorted(directory.rglob("*")):
        relative_path = candidate.relative_to(directory).as_posix()
        if ignore_protocol_files and relative_path in _RESERVED_FILENAMES:
            if candidate.is_symlink() or not candidate.is_file():
                raise ArtifactValidationError(
                    f"artifact protocol path must be a regular file: {candidate}"
                )
            continue
        if candidate.is_symlink():
            raise error_type(f"artifact payload cannot be a symbolic link: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise error_type(f"artifact payload must be a regular file: {candidate}")

        if relative_path in _RESERVED_FILENAMES:
            raise error_type(f"artifact payload uses reserved path: {relative_path}")
        size_bytes, file_sha256 = _sha256_file(candidate)
        payloads.append(
            ArtifactFile(
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=file_sha256,
            )
        )
    return tuple(payloads)


class _DuplicateJsonKeyError(ValueError):
    """Internal signal for duplicate manifest object keys."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        document[key] = value
    return document


def _read_manifest(path: Path) -> tuple[ArtifactManifest, bytes]:
    try:
        manifest_bytes = path.read_bytes()
        document = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise ArtifactValidationError(
            f"cannot read strict artifact manifest {path}: {exc}"
        ) from exc

    try:
        manifest = ArtifactManifest.model_validate(document)
    except ValidationError as exc:
        raise ArtifactValidationError(f"invalid artifact manifest {path}: {exc}") from exc
    return manifest, manifest_bytes


class ArtifactTransaction:
    """Same-filesystem temporary artifact directory with explicit atomic commit."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        artifact_type: str,
        dataset_version: str,
        profile: str,
        component_version: str,
        config_sha256: str,
        code_revision: str,
        dependencies: Sequence[ArtifactDependency],
    ) -> None:
        artifact_id = _build_artifact_id(
            artifact_type=artifact_type,
            dataset_version=dataset_version,
            profile=profile,
            component_version=component_version,
            config_sha256=config_sha256,
        )
        try:
            _parse_artifact_id(artifact_id)
        except ValueError as exc:
            raise ArtifactPathError(f"invalid artifact coordinates: {exc}") from exc
        if not code_revision:
            raise ArtifactWriteError("code_revision must not be empty")

        sorted_dependencies = tuple(sorted(dependencies, key=lambda item: item.artifact_id))
        if len({item.artifact_id for item in sorted_dependencies}) != len(sorted_dependencies):
            raise ArtifactWriteError("dependencies must have unique artifact IDs")

        self._store = store
        self.artifact_id = artifact_id
        self.target_path = store._path_for_id(artifact_id)
        self._artifact_type = artifact_type
        self._dataset_version = dataset_version
        self._profile = profile
        self._component_version = component_version
        self._config_sha256 = config_sha256
        self._code_revision = code_revision
        self._dependencies = sorted_dependencies
        self._temporary_path: Path | None = None
        self._state = "new"

    def __enter__(self) -> Self:
        if self._state != "new":
            raise ArtifactWriteError("artifact transaction cannot be entered more than once")
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = tempfile.mkdtemp(
            prefix=f".{self.target_path.name}.tmp-",
            dir=self.target_path.parent,
        )
        self._temporary_path = Path(temporary_path)
        self._state = "open"
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._state != "committed":
            self._discard_temporary_path()

    def path(self, relative_path: str) -> Path:
        """Return a safe writable payload path inside the open transaction."""
        if self._state != "open" or self._temporary_path is None:
            raise ArtifactWriteError("artifact transaction is not open")
        try:
            normalized_path = _validate_payload_path(relative_path)
        except ValueError as exc:
            raise ArtifactPathError(str(exc)) from exc

        destination = self._temporary_path.joinpath(*PurePosixPath(normalized_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def commit(self) -> LoadedArtifact:
        """Write integrity state and atomically promote the staged directory."""
        if self._state != "open" or self._temporary_path is None:
            raise ArtifactWriteError("only an open artifact transaction can be committed")
        if self.target_path.exists():
            raise ArtifactExistsError(f"immutable artifact already exists: {self.target_path}")

        files = _scan_payloads(self._temporary_path)
        if not files:
            raise ArtifactWriteError("artifact must contain at least one payload file")

        manifest = ArtifactManifest(
            artifact_id=self.artifact_id,
            artifact_type=self._artifact_type,
            dataset_version=self._dataset_version,
            profile=self._profile,
            component_version=self._component_version,
            config_sha256=self._config_sha256,
            created_utc=datetime.now(UTC),
            code_revision=self._code_revision,
            dependencies=self._dependencies,
            files=files,
        )
        manifest_bytes = _canonical_manifest_bytes(manifest)
        manifest_sha256 = sha256(manifest_bytes).hexdigest()
        _write_durable_file(self._temporary_path / MANIFEST_FILENAME, manifest_bytes)
        _write_durable_file(
            self._temporary_path / SUCCESS_FILENAME,
            f"{manifest_sha256}\n".encode("ascii"),
        )
        _fsync_directory(self._temporary_path)

        try:
            os.rename(self._temporary_path, self.target_path)
        except OSError as exc:
            raise ArtifactWriteError(
                f"cannot atomically promote {self.target_path}: {exc}"
            ) from exc

        self._temporary_path = None
        self._state = "committed"
        _fsync_directory(self.target_path.parent)
        return LoadedArtifact(
            path=self.target_path,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )

    def _discard_temporary_path(self) -> None:
        if self._temporary_path is not None and self._temporary_path.exists():
            shutil.rmtree(self._temporary_path)
        self._temporary_path = None
        self._state = "discarded"


class ArtifactStore:
    """Artifact protocol rooted at one explicit allowlisted directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)

    def stage(
        self,
        *,
        artifact_type: str,
        dataset_version: str,
        profile: str,
        component_version: str,
        config_sha256: str,
        code_revision: str,
        dependencies: Sequence[ArtifactDependency] = (),
    ) -> ArtifactTransaction:
        """Create a transaction for canonical immutable artifact coordinates."""
        return ArtifactTransaction(
            store=self,
            artifact_type=artifact_type,
            dataset_version=dataset_version,
            profile=profile,
            component_version=component_version,
            config_sha256=config_sha256,
            code_revision=code_revision,
            dependencies=dependencies,
        )

    def load(
        self,
        artifact_id: str,
    ) -> LoadedArtifact:
        """Recursively load a complete artifact and verify every declared parent."""
        return self._load(artifact_id, ancestry=(), cache={})

    def _load(
        self,
        artifact_id: str,
        *,
        ancestry: tuple[str, ...],
        cache: dict[str, LoadedArtifact],
    ) -> LoadedArtifact:
        cached = cache.get(artifact_id)
        if cached is not None:
            return cached
        if artifact_id in ancestry:
            dependency_chain = " -> ".join((*ancestry, artifact_id))
            raise ArtifactValidationError(f"artifact dependency cycle detected: {dependency_chain}")

        path = self._path_for_id(artifact_id)
        if path.is_symlink():
            raise ArtifactPathError(f"artifact directory cannot be a symbolic link: {path}")
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise ArtifactPathError(f"artifact does not resolve below {self.root}: {path}") from exc
        if not resolved_path.is_dir():
            raise ArtifactValidationError(f"artifact path is not a directory: {resolved_path}")

        manifest_path = resolved_path / MANIFEST_FILENAME
        success_path = resolved_path / SUCCESS_FILENAME
        if manifest_path.is_symlink() or success_path.is_symlink():
            raise ArtifactValidationError(
                f"artifact protocol files cannot be symbolic links: {resolved_path}"
            )

        manifest, manifest_bytes = _read_manifest(manifest_path)
        if manifest.artifact_id != artifact_id:
            raise ArtifactValidationError(
                f"manifest artifact ID {manifest.artifact_id!r} does not match {artifact_id!r}"
            )

        manifest_sha256 = sha256(manifest_bytes).hexdigest()
        try:
            success_hash = success_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ArtifactValidationError(
                f"artifact has no readable success marker: {resolved_path}"
            ) from exc
        if success_hash != manifest_sha256:
            raise ArtifactValidationError(
                f"success marker does not match manifest checksum for {artifact_id}"
            )

        actual_files = _scan_payloads(resolved_path, ignore_protocol_files=True)
        if actual_files != manifest.files:
            raise ArtifactValidationError(
                f"artifact payload integrity check failed for {artifact_id}"
            )

        next_ancestry = (*ancestry, artifact_id)
        for dependency in manifest.dependencies:
            parent = self._load(
                dependency.artifact_id,
                ancestry=next_ancestry,
                cache=cache,
            )
            if parent.manifest_sha256 != dependency.manifest_sha256:
                raise ArtifactValidationError(
                    f"dependency manifest checksum does not match for {artifact_id}: "
                    f"{dependency.artifact_id}"
                )

        loaded = LoadedArtifact(
            path=resolved_path,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )
        cache[artifact_id] = loaded
        return loaded

    def _path_for_id(self, artifact_id: str) -> Path:
        try:
            parts = _parse_artifact_id(artifact_id)
        except ValueError as exc:
            raise ArtifactPathError(f"invalid artifact ID {artifact_id!r}: {exc}") from exc
        return self.root.joinpath(*parts)


__all__ = [
    "MANIFEST_FILENAME",
    "SUCCESS_FILENAME",
    "ArtifactDependency",
    "ArtifactError",
    "ArtifactExistsError",
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactPathError",
    "ArtifactStore",
    "ArtifactTransaction",
    "ArtifactValidationError",
    "ArtifactWriteError",
    "LoadedArtifact",
]
