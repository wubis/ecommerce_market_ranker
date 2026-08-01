"""Tests for artifact manifests and atomic stage-output promotion."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from market_rank.artifacts import (
    MANIFEST_FILENAME,
    SUCCESS_FILENAME,
    ArtifactDependency,
    ArtifactExistsError,
    ArtifactPathError,
    ArtifactStore,
    ArtifactValidationError,
    ArtifactWriteError,
    LoadedArtifact,
)
from market_rank.config import load_config

REPOSITORY_ROOT = Path(__file__).parents[2]
BASE_CONFIG = REPOSITORY_ROOT / "configs" / "base.yaml"
CONFIG_SHA256 = "a" * 64


def _commit_text_artifact(
    store: ArtifactStore,
    *,
    artifact_type: str = "toy",
    component_version: str = "v1",
    config_sha256: str = CONFIG_SHA256,
    dependencies: tuple[ArtifactDependency, ...] = (),
) -> LoadedArtifact:
    with store.stage(
        artifact_type=artifact_type,
        dataset_version="dataset-v1",
        profile="development",
        component_version=component_version,
        config_sha256=config_sha256,
        code_revision="abc123",
        dependencies=dependencies,
    ) as transaction:
        transaction.path("nested/payload.txt").write_text("hello\n", encoding="utf-8")
        transaction.path("metadata.json").write_text('{"rows":1}\n', encoding="utf-8")
        return transaction.commit()


def test_stage_commit_and_verified_load(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    committed = _commit_text_artifact(store)

    expected_path = store.root / "toy" / "dataset-v1" / "development" / "v1" / CONFIG_SHA256
    assert committed.path == expected_path
    assert committed.manifest.artifact_id == (f"toy/dataset-v1/development/v1/{CONFIG_SHA256}")
    assert committed.manifest.created_utc.utcoffset() == timedelta(0)
    assert tuple(item.relative_path for item in committed.manifest.files) == (
        "metadata.json",
        "nested/payload.txt",
    )
    assert (committed.path / MANIFEST_FILENAME).is_file()
    assert (committed.path / SUCCESS_FILENAME).read_text(encoding="ascii").strip() == (
        committed.manifest_sha256
    )

    loaded = store.load(committed.manifest.artifact_id)
    assert loaded == committed


def test_resolved_configuration_hash_can_address_artifact(tmp_path: Path) -> None:
    config_sha256 = load_config([BASE_CONFIG]).sha256
    committed = _commit_text_artifact(store=ArtifactStore(tmp_path), config_sha256=config_sha256)

    assert committed.manifest.config_sha256 == config_sha256
    assert committed.path.name == config_sha256


def test_declared_parent_is_recursively_loaded_and_verified(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    parent = _commit_text_artifact(store, artifact_type="parent")
    dependency = ArtifactDependency(
        artifact_id=parent.manifest.artifact_id,
        manifest_sha256=parent.manifest_sha256,
    )
    child = _commit_text_artifact(
        store,
        artifact_type="child",
        dependencies=(dependency,),
    )

    loaded = store.load(child.manifest.artifact_id)
    assert loaded.manifest.dependencies == (dependency,)

    incompatible_child = _commit_text_artifact(
        store,
        artifact_type="incompatible-child",
        dependencies=(
            ArtifactDependency(
                artifact_id=parent.manifest.artifact_id,
                manifest_sha256="b" * 64,
            ),
        ),
    )
    with pytest.raises(ArtifactValidationError, match="dependency manifest checksum"):
        store.load(incompatible_child.manifest.artifact_id)


def test_payload_tampering_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = _commit_text_artifact(store)
    (artifact.path / "nested" / "payload.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="integrity"):
        store.load(artifact.manifest.artifact_id)


def test_unmanifested_payload_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = _commit_text_artifact(store)
    (artifact.path / "extra.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="integrity"):
        store.load(artifact.manifest.artifact_id)


def test_missing_success_marker_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = _commit_text_artifact(store)
    (artifact.path / SUCCESS_FILENAME).unlink()

    with pytest.raises(ArtifactValidationError, match="success marker"):
        store.load(artifact.manifest.artifact_id)


def test_manifest_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = _commit_text_artifact(store)
    manifest_path = artifact.path / MANIFEST_FILENAME
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ArtifactValidationError, match="success marker"):
        store.load(artifact.manifest.artifact_id)


def test_unknown_manifest_field_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = _commit_text_artifact(store)
    manifest_path = artifact.path / MANIFEST_FILENAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["unknown"] = True
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="invalid artifact manifest"):
        store.load(artifact.manifest.artifact_id)


def test_duplicate_manifest_key_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = _commit_text_artifact(store)
    manifest_path = artifact.path / MANIFEST_FILENAME
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text('{"schema_version":1,' + original[1:], encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="duplicate JSON key"):
        store.load(artifact.manifest.artifact_id)


def test_symbolic_link_payload_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    external = tmp_path / "external.txt"
    external.write_text("external", encoding="utf-8")

    with (
        pytest.raises(ArtifactWriteError, match="symbolic link"),
        store.stage(
            artifact_type="symlink",
            dataset_version="dataset-v1",
            profile="development",
            component_version="v1",
            config_sha256=CONFIG_SHA256,
            code_revision="abc123",
        ) as transaction,
    ):
        transaction.path("payload.txt").symlink_to(external)
        transaction.commit()


def test_exception_discards_temporary_output(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    transaction = store.stage(
        artifact_type="failed",
        dataset_version="dataset-v1",
        profile="development",
        component_version="v1",
        config_sha256=CONFIG_SHA256,
        code_revision="abc123",
    )

    with pytest.raises(RuntimeError, match="stage failed"), transaction:
        transaction.path("partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("stage failed")

    assert not transaction.target_path.exists()
    assert not tuple(transaction.target_path.parent.glob(f".{CONFIG_SHA256}.tmp-*"))


def test_exit_without_commit_discards_output(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    transaction = store.stage(
        artifact_type="abandoned",
        dataset_version="dataset-v1",
        profile="development",
        component_version="v1",
        config_sha256=CONFIG_SHA256,
        code_revision="abc123",
    )

    with transaction:
        transaction.path("partial.txt").write_text("partial", encoding="utf-8")

    assert not transaction.target_path.exists()


@pytest.mark.parametrize(
    "relative_path",
    ["../escape.txt", "/absolute.txt", "nested/../../escape.txt", "manifest.json"],
)
def test_payload_path_escape_and_reserved_names_are_rejected(
    tmp_path: Path,
    relative_path: str,
) -> None:
    store = ArtifactStore(tmp_path)
    with (
        store.stage(
            artifact_type="paths",
            dataset_version="dataset-v1",
            profile="development",
            component_version="v1",
            config_sha256=CONFIG_SHA256,
            code_revision="abc123",
        ) as transaction,
        pytest.raises(ArtifactPathError),
    ):
        transaction.path(relative_path)


def test_empty_artifact_cannot_be_committed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with (
        pytest.raises(ArtifactWriteError, match="at least one payload"),
        store.stage(
            artifact_type="empty",
            dataset_version="dataset-v1",
            profile="development",
            component_version="v1",
            config_sha256=CONFIG_SHA256,
            code_revision="abc123",
        ) as transaction,
    ):
        transaction.commit()


def test_promoted_artifact_is_immutable(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _commit_text_artifact(store)

    with pytest.raises(ArtifactExistsError, match="already exists"):
        _commit_text_artifact(store)


@pytest.mark.parametrize(
    "artifact_id",
    [f"../toy/dataset-v1/development/v1/{CONFIG_SHA256}", "too/few/segments"],
)
def test_invalid_artifact_id_is_rejected(tmp_path: Path, artifact_id: str) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(ArtifactPathError, match="invalid artifact ID"):
        store.load(artifact_id)
