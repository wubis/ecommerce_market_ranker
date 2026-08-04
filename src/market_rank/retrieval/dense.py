"""Pinned, checkpointed MiniLM embeddings and exact FAISS CPU retrieval."""

from __future__ import annotations

import importlib
import json
import math
import os
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, Protocol, Self, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from market_rank.artifacts import (
    ArtifactDependency,
    ArtifactExistsError,
    ArtifactStore,
    LoadedArtifact,
)
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.data.foundation import (
    CATALOG_MEMBERSHIP_FILENAME,
    FOUNDATION_MANIFEST_FILENAME,
    PRODUCT_DOCUMENTS_FILENAME,
    QUERIES_FILENAME,
    CatalogId,
    DataFoundationManifest,
    foundation_artifact_id,
    load_foundation_manifest,
)

DENSE_METADATA_FILENAME = "dense-index.json"
DENSE_DOCUMENT_MAP_FILENAME = "document-map.parquet"
EMBEDDINGS_FILENAME = "product-embeddings.npy"
FAISS_INDEX_FILENAME = "flatip.faiss"
CHECKPOINT_FILENAME = "checkpoint.json"
CHECKPOINT_EMBEDDINGS_FILENAME = "product-embeddings.partial.npy"

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
GitCommitDigest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$"),
]
FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


class DenseRetrievalError(RuntimeError):
    """Base exception for dense model, build, load, and query failures."""


class DenseModelError(DenseRetrievalError):
    """Raised when the exact pinned model is unavailable or incompatible."""


class DenseBuildError(DenseRetrievalError):
    """Raised when the dense artifact cannot be built safely."""


class DenseIndexValidationError(DenseRetrievalError):
    """Raised when persisted dense state is corrupt or incompatible."""


class DenseQueryError(DenseRetrievalError):
    """Raised when a dense search or pair request violates its contract."""


class DenseResourceError(DenseBuildError):
    """Raised when observed build RSS crosses the local promotion gate."""

    def __init__(self, measurement: DenseResourceMeasurement) -> None:
        super().__init__(
            f"dense build peak RSS {measurement.peak_rss_bytes} exceeds "
            f"the {measurement.rss_limit_bytes}-byte limit after "
            f"{measurement.completed_documents}/{measurement.total_documents} documents"
        )
        self.measurement = measurement


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DenseResourceMeasurement(_StrictModel):
    """Measured build resource facts, including a partial failure position."""

    build_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    embedding_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    faiss_build_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    completed_documents: int = Field(strict=True, ge=0)
    total_documents: int = Field(strict=True, ge=1)
    embedding_bytes: int = Field(strict=True, ge=0)
    faiss_index_bytes: int = Field(strict=True, ge=0)
    artifact_payload_bytes: int = Field(strict=True, ge=0)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if self.completed_documents > self.total_documents:
            raise ValueError("completed documents exceeds total documents")
        if self.passed != (
            self.peak_rss_bytes <= self.rss_limit_bytes
            and self.completed_documents == self.total_documents
        ):
            raise ValueError("resource passed status does not match RSS/completion gates")
        return self


class DenseLatencyMeasurement(_StrictModel):
    """Warm query-encoding plus exact-search smoke latency."""

    sample_queries: int = Field(strict=True, ge=1)
    p50_ms: float = Field(ge=0.0, allow_inf_nan=False)
    p95_ms: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_ms: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if not self.p50_ms <= self.p95_ms <= self.maximum_ms:
            raise ValueError("latency percentiles are not ordered")
        return self


class DenseCheck(_StrictModel):
    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class DenseIndexMetadata(_StrictModel):
    """Strict MiniLM/vector/FAISS identity and validation contract."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    foundation_artifact_id: str = Field(strict=True, min_length=1)
    foundation_manifest_sha256: Sha256Digest
    catalog_id: CatalogId
    catalog_membership_sha256: Sha256Digest
    product_document_version: Literal["product-document-v1"]
    component_version: Literal["minilm-l6-v2-flatip-v1"]
    model_id: Literal["sentence-transformers/all-MiniLM-L6-v2"]
    model_revision: Literal["c9745ed1d9f207416be6d2e6f8de32d1f16199bf"]
    encoder_backend: Literal["sentence-transformers-cpu-v1"] = "sentence-transformers-cpu-v1"
    embedding_dimension: Literal[384]
    embedding_dtype: Literal["float32"] = "float32"
    normalization: Literal["l2-unit-v1"] = "l2-unit-v1"
    index_type: Literal["IndexFlatIP"] = "IndexFlatIP"
    distance_metric: Literal["inner_product"] = "inner_product"
    document_count: int = Field(strict=True, ge=1)
    embedding_batch_size: int = Field(strict=True, ge=1)
    default_top_k: int = Field(strict=True, ge=1)
    max_top_k: int = Field(strict=True, ge=1)
    resumed_documents: int = Field(strict=True, ge=0)
    minimum_norm: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_norm: float = Field(gt=0.0, allow_inf_nan=False)
    resource: DenseResourceMeasurement
    latency: DenseLatencyMeasurement
    checks: tuple[DenseCheck, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k exceeds max_top_k")
        if self.resumed_documents > self.document_count:
            raise ValueError("resumed document count exceeds catalog")
        if not 0.999 <= self.minimum_norm <= self.maximum_norm <= 1.001:
            raise ValueError("persisted embeddings are not unit normalized")
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(check_ids) != len(set(check_ids)):
            raise ValueError("checks must be unique and sorted")
        return self


class DenseCheckpoint(_StrictModel):
    """Durable contiguous-row checkpoint retained after an interrupted build."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    foundation_manifest_sha256: Sha256Digest
    model_id: str = Field(strict=True, min_length=1)
    model_revision: GitCommitDigest
    document_count: int = Field(strict=True, ge=1)
    embedding_dimension: int = Field(strict=True, ge=1)
    completed_documents: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if self.completed_documents > self.document_count:
            raise ValueError("checkpoint completion exceeds document count")
        return self


@dataclass(frozen=True, slots=True)
class DenseBuildResult:
    artifact: LoadedArtifact
    metadata: DenseIndexMetadata
    reused: bool


@dataclass(frozen=True, slots=True)
class DenseCandidate:
    product_id: str
    locale: str
    raw_score: float
    one_based_rank: int
    retriever_id: str
    index_id: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class DensePairScore:
    product_id: str
    locale: str
    raw_score: float
    retriever_id: str
    index_id: str


class DenseEncoder(Protocol):
    """Injectable encoder boundary used by the real runtime and offline fixtures."""

    model_id: str
    model_revision: str
    dimension: int

    def encode_documents(self, documents: tuple[str, ...]) -> FloatMatrix: ...

    def encode_query(self, query: str) -> FloatVector: ...


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _catalog_membership_hash(manifest: DataFoundationManifest) -> str:
    for table in manifest.tables:
        if table.filename == CATALOG_MEMBERSHIP_FILENAME:
            return table.sha256
    raise DenseBuildError("foundation manifest omits catalog membership integrity")


def _load_foundation_dependency(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    store: ArtifactStore,
) -> tuple[LoadedArtifact, DataFoundationManifest, pl.DataFrame, pl.DataFrame, tuple[str, ...]]:
    expected_id = foundation_artifact_id(release, config.sha256)
    try:
        artifact = store.load(expected_id)
        manifest = load_foundation_manifest(artifact.path / FOUNDATION_MANIFEST_FILENAME)
        membership = pl.read_parquet(artifact.path / CATALOG_MEMBERSHIP_FILENAME)
        documents = pl.read_parquet(
            artifact.path / PRODUCT_DOCUMENTS_FILENAME,
            columns=["locale", "product_id", "document", "document_sha256", "document_version"],
        )
        queries = tuple(
            str(value)
            for value in pl.read_parquet(
                artifact.path / QUERIES_FILENAME, columns=["normalized_query"]
            )["normalized_query"].to_list()
        )
    except (OSError, RuntimeError, pl.exceptions.PolarsError) as exc:
        raise DenseBuildError(
            "a compatible Goldfish 006 data foundation is required; run "
            "`market-rank data build-esci-foundation` first"
        ) from exc
    if (
        manifest.release_manifest_sha256 != release.sha256
        or manifest.config_sha256 != config.sha256
        or manifest.catalog_products != membership.height
        or documents.height != membership.height
        or not queries
    ):
        raise DenseBuildError("data foundation lineage or catalog/query counts are incompatible")
    return artifact, manifest, membership, documents, queries


def _aligned_documents(membership: pl.DataFrame, documents: pl.DataFrame) -> pl.DataFrame:
    ordered = membership.sort("catalog_ordinal")
    if ordered["catalog_ordinal"].to_list() != list(range(membership.height)):
        raise DenseBuildError("catalog ordinals are not contiguous and zero-based")
    aligned = ordered.join(
        documents,
        on=("locale", "product_id", "document_sha256"),
        how="inner",
        validate="1:1",
    ).sort("catalog_ordinal")
    if aligned.height != membership.height:
        raise DenseBuildError("document/catalog keys or hashes are misaligned")
    if aligned["document_version"].n_unique() != 1:
        raise DenseBuildError("catalog contains multiple product document versions")
    return aligned


def _snapshot_download(
    *, repo_id: str, revision: str, cache_dir: str, local_files_only: bool
) -> str:
    module = importlib.import_module("huggingface_hub")
    snapshot_download = cast(Any, module).snapshot_download
    return str(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def cache_dense_model(config: ResolvedConfig, *, allow_network: bool) -> Path:
    """Resolve the exact model snapshot, downloading only after explicit opt-in."""
    dense = config.config.retrieval.dense
    try:
        resolved = _snapshot_download(
            repo_id=dense.model_id,
            revision=dense.model_revision,
            cache_dir=str(dense.model_cache_dir),
            local_files_only=not allow_network,
        )
    except Exception as exc:
        action = (
            "explicit download failed"
            if allow_network
            else "model is not cached; run `market-rank retrieval cache-minilm --allow-network`"
        )
        raise DenseModelError(f"{action}: {exc}") from exc
    path = Path(resolved)
    if not path.is_dir():
        raise DenseModelError("resolved MiniLM snapshot is not a directory")
    return path


class SentenceTransformerEncoder:
    """CPU-only adapter that can load only the exact locally cached snapshot."""

    def __init__(self, config: ResolvedConfig) -> None:
        dense = config.config.retrieval.dense
        snapshot = cache_dense_model(config, allow_network=False)
        try:
            torch = cast(Any, importlib.import_module("torch"))
            torch.set_num_threads(config.config.runtime.max_threads)
            sentence_transformers = cast(Any, importlib.import_module("sentence_transformers"))
            self._model = sentence_transformers.SentenceTransformer(
                str(snapshot),
                device="cpu",
                local_files_only=True,
                trust_remote_code=False,
            )
            observed_dimension = self._model.get_sentence_embedding_dimension()
        except Exception as exc:
            raise DenseModelError(f"cannot load cached MiniLM snapshot on CPU: {exc}") from exc
        if observed_dimension != dense.embedding_dimension:
            raise DenseModelError(
                f"MiniLM dimension {observed_dimension} != {dense.embedding_dimension}"
            )
        self.model_id = dense.model_id
        self.model_revision = dense.model_revision
        self.dimension = dense.embedding_dimension
        self._batch_size = dense.embedding_batch_size

    def _encode(self, values: tuple[str, ...]) -> FloatMatrix:
        try:
            encoded = self._model.encode(
                list(values),
                batch_size=min(self._batch_size, len(values)),
                show_progress_bar=False,
                output_value="sentence_embedding",
                precision="float32",
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise DenseModelError(f"MiniLM CPU encoding failed: {exc}") from exc
        return np.asarray(encoded, dtype=np.float32, order="C")

    def encode_documents(self, documents: tuple[str, ...]) -> FloatMatrix:
        return self._encode(documents)

    def encode_query(self, query: str) -> FloatVector:
        return cast(FloatVector, self._encode((query,))[0])


def dense_artifact_id(release: ResolvedReleaseManifest, config_sha256: str) -> str:
    """Return deterministic Goldfish 008 dense artifact coordinates."""
    return "/".join(
        (
            "dense-index",
            release.manifest.dataset_version,
            "portfolio",
            "minilm-l6-v2-flatip-v1",
            config_sha256,
        )
    )


def _workspace_path(store: ArtifactStore, artifact_id: str) -> Path:
    key = sha256(artifact_id.encode("utf-8")).hexdigest()
    return store.root / ".dense-build" / key


def _checkpoint_for(
    artifact_id: str,
    config: ResolvedConfig,
    foundation: LoadedArtifact,
    document_count: int,
    completed: int,
) -> DenseCheckpoint:
    dense = config.config.retrieval.dense
    return DenseCheckpoint(
        artifact_id=artifact_id,
        config_sha256=config.sha256,
        foundation_manifest_sha256=foundation.manifest_sha256,
        model_id=dense.model_id,
        model_revision=dense.model_revision,
        document_count=document_count,
        embedding_dimension=dense.embedding_dimension,
        completed_documents=completed,
    )


def _write_checkpoint(path: Path, checkpoint: DenseCheckpoint) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_canonical_json(checkpoint), encoding="utf-8")
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> DenseCheckpoint:
    try:
        return DenseCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise DenseBuildError(f"cannot resume dense checkpoint {path}: {exc}") from exc


def _validate_vectors(
    vectors: NDArray[Any], expected_rows: int, dimension: int
) -> tuple[float, float]:
    if vectors.shape != (expected_rows, dimension):
        raise DenseBuildError(
            f"encoder returned shape {vectors.shape}, expected {(expected_rows, dimension)}"
        )
    if vectors.dtype != np.float32:
        raise DenseBuildError(f"encoder returned {vectors.dtype}, expected float32")
    if not np.isfinite(vectors).all():
        raise DenseBuildError("encoder returned non-finite product vectors")
    norms = np.linalg.norm(vectors, axis=1)
    minimum = float(norms.min())
    maximum = float(norms.max())
    if minimum < 0.999 or maximum > 1.001:
        raise DenseBuildError(
            f"encoder returned non-unit vectors with norm range [{minimum}, {maximum}]"
        )
    return minimum, maximum


def _resource_measurement(
    *,
    started: float,
    embedding_seconds: float,
    faiss_seconds: float,
    config: ResolvedConfig,
    completed: int,
    total: int,
    embedding_path: Path,
    faiss_path: Path,
    payload_bytes: int = 0,
) -> DenseResourceMeasurement:
    peak = _peak_rss_bytes()
    limit = config.config.runtime.rss_limit_mb * 1024 * 1024
    return DenseResourceMeasurement(
        build_seconds=max(0.0, time.perf_counter() - started),
        embedding_seconds=max(0.0, embedding_seconds),
        faiss_build_seconds=max(0.0, faiss_seconds),
        peak_rss_bytes=peak,
        rss_limit_bytes=limit,
        completed_documents=completed,
        total_documents=total,
        embedding_bytes=embedding_path.stat().st_size if embedding_path.exists() else 0,
        faiss_index_bytes=faiss_path.stat().st_size if faiss_path.exists() else 0,
        artifact_payload_bytes=payload_bytes,
        passed=peak <= limit and completed == total,
    )


def _validate_encoder(encoder: DenseEncoder, config: ResolvedConfig) -> None:
    dense = config.config.retrieval.dense
    if (
        encoder.model_id != dense.model_id
        or encoder.model_revision != dense.model_revision
        or encoder.dimension != dense.embedding_dimension
    ):
        raise DenseModelError("encoder identity, revision, or dimension is incompatible")


def _embed_with_checkpoint(
    workspace: Path,
    artifact_id: str,
    aligned: pl.DataFrame,
    config: ResolvedConfig,
    foundation: LoadedArtifact,
    encoder: DenseEncoder,
    *,
    started: float,
) -> tuple[Path, int, float, float, float]:
    dense = config.config.retrieval.dense
    workspace.mkdir(parents=True, exist_ok=True)
    embedding_path = workspace / CHECKPOINT_EMBEDDINGS_FILENAME
    checkpoint_path = workspace / CHECKPOINT_FILENAME
    expected = _checkpoint_for(artifact_id, config, foundation, aligned.height, 0)
    resumed = 0
    if checkpoint_path.exists() or embedding_path.exists():
        if not checkpoint_path.is_file() or not embedding_path.is_file():
            raise DenseBuildError("dense checkpoint is incomplete; remove its scoped workspace")
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.model_copy(update={"completed_documents": 0}) != expected:
            raise DenseBuildError("dense checkpoint lineage or shape is incompatible")
        resumed = checkpoint.completed_documents
        embeddings = np.lib.format.open_memmap(embedding_path, mode="r+")
        if embeddings.shape != (aligned.height, dense.embedding_dimension):
            raise DenseBuildError("checkpoint embedding shape is incompatible")
        if resumed:
            _validate_vectors(embeddings[:resumed], resumed, dense.embedding_dimension)
    else:
        embeddings = np.lib.format.open_memmap(
            embedding_path,
            mode="w+",
            dtype=np.float32,
            shape=(aligned.height, dense.embedding_dimension),
        )
        _write_checkpoint(checkpoint_path, expected)

    embedding_started = time.perf_counter()
    completed = resumed
    try:
        documents = aligned["document"]
        while completed < aligned.height:
            stop = min(completed + dense.embedding_batch_size, aligned.height)
            batch = tuple(str(value) for value in documents[completed:stop].to_list())
            try:
                vectors = encoder.encode_documents(batch)
            except DenseRetrievalError:
                raise
            except Exception as exc:
                raise DenseBuildError(
                    f"product encoding failed after {completed} completed documents: {exc}"
                ) from exc
            _validate_vectors(vectors, stop - completed, dense.embedding_dimension)
            embeddings[completed:stop] = vectors
            embeddings.flush()
            completed = stop
            _write_checkpoint(
                checkpoint_path,
                _checkpoint_for(artifact_id, config, foundation, aligned.height, completed),
            )
            measurement = _resource_measurement(
                started=started,
                embedding_seconds=time.perf_counter() - embedding_started,
                faiss_seconds=0.0,
                config=config,
                completed=completed,
                total=aligned.height,
                embedding_path=embedding_path,
                faiss_path=workspace / FAISS_INDEX_FILENAME,
            )
            if measurement.peak_rss_bytes > measurement.rss_limit_bytes:
                raise DenseResourceError(measurement)
        minimum, maximum = _validate_vectors(embeddings, aligned.height, dense.embedding_dimension)
    finally:
        embeddings.flush()
        del embeddings
    return embedding_path, resumed, time.perf_counter() - embedding_started, minimum, maximum


def _faiss_module(max_threads: int) -> Any:
    try:
        faiss = cast(Any, importlib.import_module("faiss"))
        faiss.omp_set_num_threads(max_threads)
        return faiss
    except Exception as exc:
        raise DenseBuildError(f"FAISS CPU is unavailable: {exc}") from exc


def _stable_search(
    index: Any,
    vector: FloatVector,
    product_ids: tuple[str, ...],
    top_k: int,
) -> tuple[tuple[int, float], ...]:
    catalog_size = len(product_ids)
    target = min(top_k, catalog_size)
    requested = min(catalog_size, max(target * 2, 64))
    while True:
        scores, ordinals = index.search(vector.reshape(1, -1), requested)
        observed = [
            (int(ordinal), float(score))
            for ordinal, score in zip(ordinals[0], scores[0], strict=True)
            if int(ordinal) >= 0
        ]
        if len(observed) != requested or any(not math.isfinite(score) for _, score in observed):
            raise DenseIndexValidationError(
                "FAISS returned incomplete or non-finite requested scores"
            )
        observed.sort(key=lambda item: (-item[1], product_ids[item[0]]))
        if requested == catalog_size or observed[target - 1][1] > observed[-1][1]:
            return tuple(observed[:target])
        requested = min(catalog_size, requested * 2)


def _validate_faiss_vector_parity(index: Any, embeddings: NDArray[Any]) -> None:
    """Compare persisted FAISS rows to the neutral memmap without a full duplicate."""
    batch_size = 4096
    try:
        for start in range(0, embeddings.shape[0], batch_size):
            count = min(batch_size, embeddings.shape[0] - start)
            reconstructed = np.asarray(index.reconstruct_n(start, count), dtype=np.float32)
            if not np.array_equal(reconstructed, embeddings[start : start + count]):
                raise DenseIndexValidationError(
                    f"FAISS/vector row parity failed at catalog ordinal {start}"
                )
    except DenseIndexValidationError:
        raise
    except Exception as exc:
        raise DenseIndexValidationError(f"cannot reconstruct FAISS vectors: {exc}") from exc


def _build_faiss(
    workspace: Path,
    embeddings_path: Path,
    config: ResolvedConfig,
) -> tuple[Any, float]:
    dense = config.config.retrieval.dense
    faiss = _faiss_module(config.config.runtime.max_threads)
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    started = time.perf_counter()
    try:
        index = faiss.IndexFlatIP(dense.embedding_dimension)
        index.add(np.asarray(embeddings, dtype=np.float32, order="C"))
        faiss.write_index(index, str(workspace / FAISS_INDEX_FILENAME))
        index = faiss.read_index(str(workspace / FAISS_INDEX_FILENAME))
        _validate_faiss_vector_parity(index, embeddings)
    except Exception as exc:
        raise DenseBuildError(f"cannot build/persist exact FAISS IndexFlatIP: {exc}") from exc
    finally:
        del embeddings
    return index, time.perf_counter() - started


def _latency_measurement(
    index: Any,
    encoder: DenseEncoder,
    queries: tuple[str, ...],
    product_ids: tuple[str, ...],
    config: ResolvedConfig,
) -> DenseLatencyMeasurement:
    sample = queries[: config.config.retrieval.dense.latency_sample_queries]
    observations: list[float] = []
    for query in sample:
        started = time.perf_counter()
        vector = encoder.encode_query(query)
        _validate_vectors(vector.reshape(1, -1), 1, encoder.dimension)
        _stable_search(index, vector, product_ids, 1)
        observations.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(observations, dtype=np.float64)
    return DenseLatencyMeasurement(
        sample_queries=len(observations),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        maximum_ms=float(values.max()),
    )


def load_dense_metadata(path: Path) -> DenseIndexMetadata:
    """Load strict dense metadata without loading vectors or the model."""
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        return DenseIndexMetadata.model_validate(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise DenseIndexValidationError(f"cannot load dense metadata {path}: {exc}") from exc


def _reuse_dense(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    foundation: LoadedArtifact,
    store: ArtifactStore,
) -> DenseBuildResult:
    artifact = store.load(dense_artifact_id(release, config.sha256))
    dependency = ArtifactDependency(
        artifact_id=foundation.manifest.artifact_id,
        manifest_sha256=foundation.manifest_sha256,
    )
    if artifact.manifest.dependencies != (dependency,):
        raise DenseIndexValidationError("existing dense index has an incompatible foundation")
    metadata = load_dense_metadata(artifact.path / DENSE_METADATA_FILENAME)
    if (
        metadata.artifact_id != artifact.manifest.artifact_id
        or metadata.config_sha256 != config.sha256
        or metadata.foundation_manifest_sha256 != foundation.manifest_sha256
    ):
        raise DenseIndexValidationError("existing dense metadata has incompatible lineage")
    return DenseBuildResult(artifact=artifact, metadata=metadata, reused=True)


def build_dense_index(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    artifact_store: ArtifactStore | None = None,
    encoder: DenseEncoder | None = None,
) -> DenseBuildResult:
    """Build or resume the normalized MiniLM vectors and exact FAISS index."""
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    foundation, foundation_manifest, membership, documents, queries = _load_foundation_dependency(
        release, config, store
    )
    artifact_id = dense_artifact_id(release, config.sha256)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse_dense(release, config, foundation, store)
    aligned = _aligned_documents(membership, documents)
    del membership, documents
    active_encoder = (
        encoder if encoder is not None else cast(DenseEncoder, SentenceTransformerEncoder(config))
    )
    _validate_encoder(active_encoder, config)
    product_ids = tuple(str(value) for value in aligned["product_id"].to_list())
    workspace = _workspace_path(store, artifact_id)
    started = time.perf_counter()
    embeddings_path, resumed, embedding_seconds, minimum_norm, maximum_norm = (
        _embed_with_checkpoint(
            workspace,
            artifact_id,
            aligned,
            config,
            foundation,
            active_encoder,
            started=started,
        )
    )
    index, faiss_seconds = _build_faiss(workspace, embeddings_path, config)
    latency = _latency_measurement(index, active_encoder, queries, product_ids, config)
    faiss_path = workspace / FAISS_INDEX_FILENAME
    dense = config.config.retrieval.dense
    dependency = ArtifactDependency(
        artifact_id=foundation.manifest.artifact_id,
        manifest_sha256=foundation.manifest_sha256,
    )
    try:
        with store.stage(
            artifact_type="dense-index",
            dataset_version=release.manifest.dataset_version,
            profile="portfolio",
            component_version=dense.component_version,
            config_sha256=config.sha256,
            code_revision=code_revision,
            dependencies=(dependency,),
        ) as transaction:
            root = transaction.path(DENSE_METADATA_FILENAME).parent
            shutil.copyfile(embeddings_path, root / EMBEDDINGS_FILENAME)
            shutil.copyfile(faiss_path, root / FAISS_INDEX_FILENAME)
            aligned.select(
                "catalog_ordinal", "locale", "product_id", "document_sha256", "document_version"
            ).write_parquet(root / DENSE_DOCUMENT_MAP_FILENAME, compression="zstd", statistics=True)
            payload_bytes = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
            resource_measurement = _resource_measurement(
                started=started,
                embedding_seconds=embedding_seconds,
                faiss_seconds=faiss_seconds,
                config=config,
                completed=aligned.height,
                total=aligned.height,
                embedding_path=root / EMBEDDINGS_FILENAME,
                faiss_path=root / FAISS_INDEX_FILENAME,
                payload_bytes=payload_bytes,
            )
            if not resource_measurement.passed:
                raise DenseResourceError(resource_measurement)
            checks = tuple(
                sorted(
                    (
                        DenseCheck(
                            check_id="catalog_alignment",
                            detail=f"all {aligned.height} catalog rows align to one vector",
                        ),
                        DenseCheck(
                            check_id="checkpoint_contiguity",
                            detail="only verified contiguous embedding prefixes are resumable",
                        ),
                        DenseCheck(
                            check_id="exact_flatip",
                            detail="normalized float32 vectors use CPU IndexFlatIP",
                        ),
                        DenseCheck(
                            check_id="reload_parity",
                            detail="persisted FAISS and memmap identities are validated on load",
                        ),
                        DenseCheck(
                            check_id="resource_gate",
                            detail=(
                                f"peak RSS {resource_measurement.peak_rss_bytes} <= "
                                f"{resource_measurement.rss_limit_bytes} bytes"
                            ),
                        ),
                    ),
                    key=lambda item: item.check_id,
                )
            )
            metadata = DenseIndexMetadata(
                artifact_id=artifact_id,
                dataset_version=release.manifest.dataset_version,
                config_sha256=config.sha256,
                foundation_artifact_id=foundation.manifest.artifact_id,
                foundation_manifest_sha256=foundation.manifest_sha256,
                catalog_id=foundation_manifest.catalog_id,
                catalog_membership_sha256=_catalog_membership_hash(foundation_manifest),
                product_document_version=foundation_manifest.product_document_version,
                component_version=dense.component_version,
                model_id=dense.model_id,
                model_revision=dense.model_revision,
                embedding_dimension=dense.embedding_dimension,
                document_count=aligned.height,
                embedding_batch_size=dense.embedding_batch_size,
                default_top_k=dense.default_top_k,
                max_top_k=dense.max_top_k,
                resumed_documents=resumed,
                minimum_norm=minimum_norm,
                maximum_norm=maximum_norm,
                resource=resource_measurement,
                latency=latency,
                checks=checks,
            )
            transaction.path(DENSE_METADATA_FILENAME).write_text(
                _canonical_json(metadata), encoding="utf-8"
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse_dense(release, config, foundation, store)
    shutil.rmtree(workspace)
    return DenseBuildResult(artifact=artifact, metadata=metadata, reused=False)


class DenseIndex:
    """Offline-loaded exact dense catalog index with injected/local query encoder."""

    def __init__(
        self,
        artifact: LoadedArtifact,
        metadata: DenseIndexMetadata,
        encoder: DenseEncoder,
        *,
        max_threads: int = 4,
    ) -> None:
        self.artifact = artifact
        self.metadata = metadata
        self._encoder = encoder
        self._closed = False
        if (
            encoder.model_id != metadata.model_id
            or encoder.model_revision != metadata.model_revision
            or encoder.dimension != metadata.embedding_dimension
        ):
            raise DenseIndexValidationError("query encoder is incompatible with dense metadata")
        try:
            document_map = pl.read_parquet(artifact.path / DENSE_DOCUMENT_MAP_FILENAME).sort(
                "catalog_ordinal"
            )
            self._vectors: Any = np.load(
                artifact.path / EMBEDDINGS_FILENAME, mmap_mode="r", allow_pickle=False
            )
            faiss = _faiss_module(max_threads)
            self._index = faiss.read_index(str(artifact.path / FAISS_INDEX_FILENAME))
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            raise DenseIndexValidationError(f"cannot load dense index payloads: {exc}") from exc
        if document_map.height != metadata.document_count:
            raise DenseIndexValidationError("dense document map count does not match metadata")
        if document_map["catalog_ordinal"].to_list() != list(range(metadata.document_count)):
            raise DenseIndexValidationError("dense document map ordinals are not contiguous")
        if self._vectors.shape != (metadata.document_count, metadata.embedding_dimension):
            raise DenseIndexValidationError("persisted embedding matrix has an incompatible shape")
        if self._vectors.dtype != np.float32:
            raise DenseIndexValidationError("persisted embeddings are not float32")
        try:
            minimum, maximum = _validate_vectors(
                self._vectors, metadata.document_count, metadata.embedding_dimension
            )
        except DenseBuildError as exc:
            raise DenseIndexValidationError(str(exc)) from exc
        if minimum != metadata.minimum_norm or maximum != metadata.maximum_norm:
            raise DenseIndexValidationError("persisted vector norm facts changed")
        if (
            type(self._index).__name__ != metadata.index_type
            or int(self._index.d) != metadata.embedding_dimension
            or int(self._index.ntotal) != metadata.document_count
        ):
            raise DenseIndexValidationError("persisted FAISS type or dimensions are incompatible")
        _validate_faiss_vector_parity(self._index, self._vectors)
        self._product_ids = tuple(str(value) for value in document_map["product_id"].to_list())
        self._locales = tuple(str(value) for value in document_map["locale"].to_list())
        if len(set(self._product_ids)) != len(self._product_ids):
            raise DenseIndexValidationError("product IDs are not unique in the dense catalog")
        self._ordinals = {product_id: index for index, product_id in enumerate(self._product_ids)}

    @property
    def retriever_id(self) -> str:
        return (
            f"dense:{self.metadata.component_version}:{self.metadata.model_id}@"
            f"{self.metadata.model_revision}"
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        mapping = getattr(self._vectors, "_mmap", None)
        if mapping is not None:
            mapping.close()
        self._closed = True

    def _validate_query(self, query: str) -> FloatVector | None:
        if self._closed:
            raise DenseQueryError("dense index is closed")
        if not isinstance(query, str) or len(query) > 4096:
            raise DenseQueryError("query must be a string of at most 4096 characters")
        if not query.strip():
            return None
        vector = self._encoder.encode_query(query)
        try:
            _validate_vectors(vector.reshape(1, -1), 1, self.metadata.embedding_dimension)
        except DenseBuildError as exc:
            raise DenseQueryError(f"query encoder violated the dense contract: {exc}") from exc
        return vector

    def search(self, query: str, top_k: int | None = None) -> tuple[DenseCandidate, ...]:
        """Return exact cosine candidates with stable product-ID tie breaking."""
        selected_k = self.metadata.default_top_k if top_k is None else top_k
        if not isinstance(selected_k, int) or isinstance(selected_k, bool):
            raise DenseQueryError("top_k must be an integer")
        if selected_k < 1 or selected_k > self.metadata.max_top_k:
            raise DenseQueryError(f"top_k must be between 1 and {self.metadata.max_top_k}")
        started = time.perf_counter()
        vector = self._validate_query(query)
        if vector is None:
            return ()
        ranked = _stable_search(
            self._index,
            vector,
            self._product_ids,
            min(selected_k, self.metadata.document_count),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return tuple(
            DenseCandidate(
                product_id=self._product_ids[ordinal],
                locale=self._locales[ordinal],
                raw_score=score,
                one_based_rank=rank,
                retriever_id=self.retriever_id,
                index_id=self.artifact.manifest.artifact_id,
                latency_ms=latency_ms,
            )
            for rank, (ordinal, score) in enumerate(ranked, start=1)
        )

    def score_pairs(self, query: str, product_ids: tuple[str, ...]) -> tuple[DensePairScore, ...]:
        """Score every requested known product directly from the normalized memmap."""
        if len(product_ids) != len(set(product_ids)):
            raise DenseQueryError("explicit pair product IDs must be unique")
        unknown = tuple(
            product_id for product_id in product_ids if product_id not in self._ordinals
        )
        if unknown:
            raise DenseQueryError(f"explicit pair products are outside the catalog: {unknown}")
        vector = self._validate_query(query)
        if vector is None:
            scores = np.zeros(len(product_ids), dtype=np.float32)
        else:
            ordinals = [self._ordinals[product_id] for product_id in product_ids]
            scores = np.asarray(self._vectors[ordinals] @ vector, dtype=np.float32)
        if not np.isfinite(scores).all():
            raise DenseQueryError("dense pair scoring produced non-finite values")
        return tuple(
            DensePairScore(
                product_id=product_id,
                locale=self._locales[self._ordinals[product_id]],
                raw_score=float(score),
                retriever_id=self.retriever_id,
                index_id=self.artifact.manifest.artifact_id,
            )
            for product_id, score in zip(product_ids, scores, strict=True)
        )


def load_dense_index(
    artifact_store: ArtifactStore,
    artifact_id: str,
    *,
    encoder: DenseEncoder,
    max_threads: int = 4,
) -> DenseIndex:
    """Recursively verify and load a promoted dense index without downloads/builds."""
    artifact = artifact_store.load(artifact_id)
    if artifact.manifest.artifact_type != "dense-index":
        raise DenseIndexValidationError("artifact is not a dense index")
    metadata = load_dense_metadata(artifact.path / DENSE_METADATA_FILENAME)
    if metadata.artifact_id != artifact.manifest.artifact_id:
        raise DenseIndexValidationError("dense metadata artifact ID does not match its manifest")
    if len(artifact.manifest.dependencies) != 1 or (
        metadata.foundation_artifact_id != artifact.manifest.dependencies[0].artifact_id
        or metadata.foundation_manifest_sha256 != artifact.manifest.dependencies[0].manifest_sha256
    ):
        raise DenseIndexValidationError("dense metadata foundation dependency is incompatible")
    return DenseIndex(artifact, metadata, encoder, max_threads=max_threads)


__all__ = [
    "DENSE_DOCUMENT_MAP_FILENAME",
    "DENSE_METADATA_FILENAME",
    "EMBEDDINGS_FILENAME",
    "FAISS_INDEX_FILENAME",
    "DenseBuildError",
    "DenseBuildResult",
    "DenseCandidate",
    "DenseCheckpoint",
    "DenseEncoder",
    "DenseIndex",
    "DenseIndexMetadata",
    "DenseIndexValidationError",
    "DenseLatencyMeasurement",
    "DenseModelError",
    "DensePairScore",
    "DenseQueryError",
    "DenseResourceError",
    "DenseResourceMeasurement",
    "DenseRetrievalError",
    "SentenceTransformerEncoder",
    "build_dense_index",
    "cache_dense_model",
    "dense_artifact_id",
    "load_dense_index",
    "load_dense_metadata",
]
