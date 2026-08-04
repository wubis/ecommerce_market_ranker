"""Deterministic, persisted, CPU-first BM25 retrieval for the fixed ESCI catalog."""

from __future__ import annotations

import heapq
import json
import math
import mmap
import os
import re
import resource
import sqlite3
import struct
import sys
import time
import unicodedata
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, BinaryIO, Literal, Self

import polars as pl
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
    CatalogId,
    DataFoundationManifest,
    foundation_artifact_id,
    load_foundation_manifest,
)

SPARSE_METADATA_FILENAME = "sparse-index.json"
DOCUMENT_MAP_FILENAME = "document-map.parquet"
VOCABULARY_FILENAME = "vocabulary.txt"
POSTINGS_OFFSETS_FILENAME = "postings-offsets.u64"
POSTING_DOC_IDS_FILENAME = "posting-doc-ids.u32"
POSTING_TFS_FILENAME = "posting-term-frequencies.u32"
DOCUMENT_LENGTHS_FILENAME = "document-lengths.u32"
DOCUMENT_FREQUENCIES_FILENAME = "document-frequencies.u32"
INVERSE_DOCUMENT_FREQUENCIES_FILENAME = "inverse-document-frequencies.f32"
TOKEN_PATTERN: Literal[r"[^\W_]+(?:[-'][^\W_]+)*"] = r"[^\W_]+(?:[-'][^\W_]+)*"

_TOKEN_RE = re.compile(TOKEN_PATTERN, flags=re.UNICODE)

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class SparseRetrievalError(RuntimeError):
    """Base exception for sparse index construction, loading, and querying."""


class SparseBuildError(SparseRetrievalError):
    """Raised when a fixed-catalog BM25 index cannot be constructed safely."""


class SparseIndexValidationError(SparseRetrievalError):
    """Raised when persisted sparse state is incomplete or incompatible."""


class SparseQueryError(SparseRetrievalError):
    """Raised when a search or explicit pair-score request is invalid."""


class SparseResourceError(SparseBuildError):
    """Raised when observed sparse-build RSS exceeds the configured gate."""

    def __init__(self, measurement: SparseResourceMeasurement) -> None:
        super().__init__(
            f"sparse build peak RSS {measurement.peak_rss_bytes} exceeds "
            f"the {measurement.rss_limit_bytes}-byte limit"
        )
        self.measurement = measurement


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SparseResourceMeasurement(_StrictModel):
    """Observed local construction resource facts."""

    build_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    peak_rss_bytes: int = Field(strict=True, ge=0)
    rss_limit_bytes: int = Field(strict=True, ge=1)
    index_payload_bytes: int = Field(strict=True, ge=0)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        if self.passed != (self.peak_rss_bytes <= self.rss_limit_bytes):
            raise ValueError("resource passed status does not match the RSS gate")
        return self


class SparseCheck(_StrictModel):
    """One successful persisted-index invariant."""

    check_id: str = Field(strict=True, min_length=1)
    passed: Literal[True] = True
    detail: str = Field(strict=True, min_length=1)


class SparseIndexMetadata(_StrictModel):
    """Strict identity, algorithm, alignment, and resource contract for BM25."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(strict=True, min_length=1)
    dataset_version: str = Field(strict=True, min_length=1)
    config_sha256: Sha256Digest
    foundation_artifact_id: str = Field(strict=True, min_length=1)
    foundation_manifest_sha256: Sha256Digest
    catalog_id: CatalogId
    catalog_membership_sha256: Sha256Digest
    product_document_version: Literal["product-document-v1"]
    component_version: Literal["bm25-v1"]
    tokenizer_version: Literal["unicode-word-v1"]
    token_pattern: Literal[r"[^\W_]+(?:[-'][^\W_]+)*"] = TOKEN_PATTERN
    query_term_policy: Literal["unique"] = "unique"
    k1: float = Field(gt=0.0, le=5.0, allow_inf_nan=False)
    b: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    document_count: int = Field(strict=True, ge=1)
    vocabulary_size: int = Field(strict=True, ge=1)
    posting_count: int = Field(strict=True, ge=1)
    total_tokens: int = Field(strict=True, ge=1)
    zero_token_documents: int = Field(strict=True, ge=0)
    average_document_length: float = Field(gt=0.0, allow_inf_nan=False)
    default_top_k: int = Field(strict=True, ge=1)
    max_top_k: int = Field(strict=True, ge=1)
    resource: SparseResourceMeasurement
    checks: tuple[SparseCheck, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k exceeds max_top_k")
        if self.zero_token_documents > self.document_count:
            raise ValueError("zero-token count exceeds document count")
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(check_ids) != len(set(check_ids)):
            raise ValueError("checks must be unique and sorted")
        return self


@dataclass(frozen=True, slots=True)
class SparseBuildResult:
    artifact: LoadedArtifact
    metadata: SparseIndexMetadata
    reused: bool


@dataclass(frozen=True, slots=True)
class SparseCandidate:
    product_id: str
    locale: str
    raw_score: float
    one_based_rank: int
    retriever_id: str
    index_id: str


@dataclass(frozen=True, slots=True)
class SparsePairScore:
    product_id: str
    locale: str
    raw_score: float
    retriever_id: str
    index_id: str


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize with the versioned Unicode word rule used at build and query time."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(match.group(0) for match in _TOKEN_RE.finditer(normalized))


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


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def _write_array(path: Path, typecode: str, values: list[int] | list[float]) -> None:
    output = array(typecode, values)
    if sys.byteorder != "little":
        output.byteswap()
    try:
        with path.open("xb") as stream:
            output.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise SparseBuildError(f"cannot persist {path.name}: {exc}") from exc


def _flush_array(stream: BinaryIO, typecode: str, values: list[int]) -> None:
    output = array(typecode, values)
    if sys.byteorder != "little":
        output.byteswap()
    output.tofile(stream)


def _load_foundation_dependency(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    store: ArtifactStore,
) -> tuple[LoadedArtifact, DataFoundationManifest, pl.DataFrame, pl.DataFrame]:
    expected_id = foundation_artifact_id(release, config.sha256)
    try:
        artifact = store.load(expected_id)
        manifest = load_foundation_manifest(artifact.path / FOUNDATION_MANIFEST_FILENAME)
        membership = pl.read_parquet(artifact.path / CATALOG_MEMBERSHIP_FILENAME)
        documents = pl.read_parquet(
            artifact.path / PRODUCT_DOCUMENTS_FILENAME,
            columns=[
                "locale",
                "product_id",
                "document",
                "document_sha256",
                "document_version",
            ],
        )
    except (OSError, RuntimeError, pl.exceptions.PolarsError) as exc:
        raise SparseBuildError(
            "a compatible Goldfish 006 data foundation is required; run "
            "`market-rank data build-esci-foundation` first"
        ) from exc
    if (
        manifest.release_manifest_sha256 != release.sha256
        or manifest.config_sha256 != config.sha256
        or manifest.catalog_products != membership.height
        or documents.height != membership.height
    ):
        raise SparseBuildError("data foundation lineage or catalog counts are incompatible")
    return artifact, manifest, membership, documents


def _aligned_documents(membership: pl.DataFrame, documents: pl.DataFrame) -> pl.DataFrame:
    expected_ordinals = list(range(membership.height))
    ordered_membership = membership.sort("catalog_ordinal")
    if ordered_membership["catalog_ordinal"].to_list() != expected_ordinals:
        raise SparseBuildError("catalog ordinals are not contiguous and zero-based")
    aligned = ordered_membership.join(
        documents,
        on=("locale", "product_id", "document_sha256"),
        how="inner",
        validate="1:1",
    ).sort("catalog_ordinal")
    if aligned.height != membership.height:
        raise SparseBuildError("document/catalog keys or document hashes are misaligned")
    if aligned["document_version"].n_unique() != 1:
        raise SparseBuildError("catalog contains multiple product document versions")
    return aligned


def _create_posting_database(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE postings ("
            "token TEXT NOT NULL, document_ordinal INTEGER NOT NULL, "
            "term_frequency INTEGER NOT NULL, "
            "PRIMARY KEY (token, document_ordinal)) WITHOUT ROWID"
        )
        return connection
    except sqlite3.Error as exc:
        raise SparseBuildError(f"cannot initialize disk-backed posting build: {exc}") from exc


def _populate_postings(
    connection: sqlite3.Connection,
    aligned: pl.DataFrame,
    *,
    batch_size: int,
) -> tuple[list[int], int, int]:
    document_lengths: list[int] = []
    pending: list[tuple[str, int, int]] = []
    total_tokens = 0
    zero_token_documents = 0
    try:
        for row in aligned.select("catalog_ordinal", "document").iter_rows(named=True):
            ordinal = int(row["catalog_ordinal"])
            document = row["document"]
            if not isinstance(document, str):
                raise SparseBuildError(f"document ordinal {ordinal} has invalid text")
            counts = Counter(tokenize(document))
            document_length = sum(counts.values())
            document_lengths.append(document_length)
            total_tokens += document_length
            zero_token_documents += document_length == 0
            pending.extend((token, ordinal, frequency) for token, frequency in counts.items())
            if len(pending) >= batch_size:
                connection.executemany("INSERT INTO postings VALUES (?, ?, ?)", pending)
                connection.commit()
                pending.clear()
        if pending:
            connection.executemany("INSERT INTO postings VALUES (?, ?, ?)", pending)
            connection.commit()
    except (sqlite3.Error, OverflowError) as exc:
        raise SparseBuildError(f"cannot build disk-backed postings: {exc}") from exc
    if not total_tokens:
        raise SparseBuildError("catalog product documents produced zero indexable tokens")
    return document_lengths, total_tokens, zero_token_documents


def _persist_vocabulary_and_statistics(
    connection: sqlite3.Connection,
    root: Path,
    document_count: int,
) -> tuple[int, int]:
    offsets = [0]
    document_frequencies: list[int] = []
    inverse_document_frequencies: list[float] = []
    posting_count = 0
    vocabulary_size = 0
    try:
        with (root / VOCABULARY_FILENAME).open("x", encoding="utf-8", newline="\n") as vocab:
            for token, observed_df in connection.execute(
                "SELECT token, COUNT(*) FROM postings GROUP BY token ORDER BY token"
            ):
                document_frequency = int(observed_df)
                posting_count += document_frequency
                vocabulary_size += 1
                vocab.write(f"{token}\n")
                offsets.append(posting_count)
                document_frequencies.append(document_frequency)
                inverse_document_frequencies.append(
                    math.log(
                        1.0
                        + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                    )
                )
            vocab.flush()
            os.fsync(vocab.fileno())
    except (OSError, sqlite3.Error) as exc:
        raise SparseBuildError(f"cannot persist vocabulary/statistics: {exc}") from exc
    if not vocabulary_size or not posting_count:
        raise SparseBuildError("BM25 vocabulary or postings are empty")
    _write_array(root / POSTINGS_OFFSETS_FILENAME, "Q", offsets)
    _write_array(root / DOCUMENT_FREQUENCIES_FILENAME, "I", document_frequencies)
    _write_array(root / INVERSE_DOCUMENT_FREQUENCIES_FILENAME, "f", inverse_document_frequencies)
    return vocabulary_size, posting_count


def _persist_postings(connection: sqlite3.Connection, root: Path, batch_size: int) -> None:
    doc_ids: list[int] = []
    frequencies: list[int] = []
    try:
        with (
            (root / POSTING_DOC_IDS_FILENAME).open("xb") as doc_stream,
            (root / POSTING_TFS_FILENAME).open("xb") as tf_stream,
        ):
            for document_ordinal, term_frequency in connection.execute(
                "SELECT document_ordinal, term_frequency "
                "FROM postings ORDER BY token, document_ordinal"
            ):
                doc_ids.append(int(document_ordinal))
                frequencies.append(int(term_frequency))
                if len(doc_ids) >= batch_size:
                    _flush_array(doc_stream, "I", doc_ids)
                    _flush_array(tf_stream, "I", frequencies)
                    doc_ids.clear()
                    frequencies.clear()
            if doc_ids:
                _flush_array(doc_stream, "I", doc_ids)
                _flush_array(tf_stream, "I", frequencies)
            for stream in (doc_stream, tf_stream):
                stream.flush()
                os.fsync(stream.fileno())
    except (OSError, sqlite3.Error) as exc:
        raise SparseBuildError(f"cannot persist posting arrays: {exc}") from exc


def _catalog_membership_hash(manifest: DataFoundationManifest) -> str:
    for table in manifest.tables:
        if table.filename == CATALOG_MEMBERSHIP_FILENAME:
            return table.sha256
    raise SparseBuildError("foundation manifest omits catalog membership integrity")


def _build_index_payload(
    root: Path,
    aligned: pl.DataFrame,
    config: ResolvedConfig,
) -> tuple[int, int, int, int, int]:
    database_path = root / ".postings-build.sqlite3"
    connection = _create_posting_database(database_path)
    sparse = config.config.retrieval.sparse
    try:
        document_lengths, total_tokens, zero_token_documents = _populate_postings(
            connection,
            aligned,
            batch_size=sparse.sqlite_batch_rows,
        )
        vocabulary_size, posting_count = _persist_vocabulary_and_statistics(
            connection, root, aligned.height
        )
        _persist_postings(connection, root, sparse.sqlite_batch_rows)
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()
    _write_array(root / DOCUMENT_LENGTHS_FILENAME, "I", document_lengths)
    aligned.select(
        "catalog_ordinal", "locale", "product_id", "document_sha256", "document_version"
    ).write_parquet(root / DOCUMENT_MAP_FILENAME, compression="zstd", statistics=True)
    return (
        vocabulary_size,
        posting_count,
        total_tokens,
        zero_token_documents,
        sum(document_lengths),
    )


def _metadata_for(
    root: Path,
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    foundation: LoadedArtifact,
    foundation_manifest: DataFoundationManifest,
    *,
    started: float,
    vocabulary_size: int,
    posting_count: int,
    total_tokens: int,
    zero_token_documents: int,
    document_length_sum: int,
) -> SparseIndexMetadata:
    sparse = config.config.retrieval.sparse
    elapsed = max(0.0, time.perf_counter() - started)
    peak_rss_bytes = _peak_rss_bytes()
    rss_limit_bytes = config.config.runtime.rss_limit_mb * 1024 * 1024
    measurement = SparseResourceMeasurement(
        build_seconds=elapsed,
        peak_rss_bytes=peak_rss_bytes,
        rss_limit_bytes=rss_limit_bytes,
        index_payload_bytes=_directory_bytes(root),
        passed=peak_rss_bytes <= rss_limit_bytes,
    )
    if not measurement.passed:
        raise SparseResourceError(measurement)
    document_count = foundation_manifest.catalog_products
    checks = tuple(
        sorted(
            (
                SparseCheck(
                    check_id="catalog_alignment",
                    detail=f"all {document_count} catalog ordinals align to one document",
                ),
                SparseCheck(
                    check_id="pair_scoring_complete",
                    detail="explicit known product IDs always receive a finite BM25 score",
                ),
                SparseCheck(
                    check_id="resource_gate",
                    detail=(
                        f"peak RSS {measurement.peak_rss_bytes} <= "
                        f"{measurement.rss_limit_bytes} bytes"
                    ),
                ),
                SparseCheck(
                    check_id="typed_persistence",
                    detail="postings, offsets, lengths, frequencies, and IDF use typed arrays",
                ),
            ),
            key=lambda item: item.check_id,
        )
    )
    return SparseIndexMetadata(
        artifact_id=sparse_artifact_id(release, config.sha256),
        dataset_version=release.manifest.dataset_version,
        config_sha256=config.sha256,
        foundation_artifact_id=foundation.manifest.artifact_id,
        foundation_manifest_sha256=foundation.manifest_sha256,
        catalog_id=foundation_manifest.catalog_id,
        catalog_membership_sha256=_catalog_membership_hash(foundation_manifest),
        product_document_version=foundation_manifest.product_document_version,
        component_version=sparse.component_version,
        tokenizer_version=sparse.tokenizer_version,
        k1=sparse.k1,
        b=sparse.b,
        document_count=document_count,
        vocabulary_size=vocabulary_size,
        posting_count=posting_count,
        total_tokens=total_tokens,
        zero_token_documents=zero_token_documents,
        average_document_length=document_length_sum / document_count,
        default_top_k=sparse.default_top_k,
        max_top_k=sparse.max_top_k,
        resource=measurement,
        checks=checks,
    )


def load_sparse_metadata(path: Path) -> SparseIndexMetadata:
    """Load strict sparse index metadata."""
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        return SparseIndexMetadata.model_validate(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise SparseIndexValidationError(f"cannot load sparse metadata {path}: {exc}") from exc


def sparse_artifact_id(release: ResolvedReleaseManifest, config_sha256: str) -> str:
    """Return deterministic Goldfish 007 sparse artifact coordinates."""
    return "/".join(
        ("sparse-index", release.manifest.dataset_version, "portfolio", "bm25-v1", config_sha256)
    )


def _reuse_sparse(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    foundation: LoadedArtifact,
    store: ArtifactStore,
) -> SparseBuildResult:
    artifact = store.load(sparse_artifact_id(release, config.sha256))
    dependency = ArtifactDependency(
        artifact_id=foundation.manifest.artifact_id,
        manifest_sha256=foundation.manifest_sha256,
    )
    if artifact.manifest.dependencies != (dependency,):
        raise SparseIndexValidationError("existing sparse index has an incompatible foundation")
    metadata = load_sparse_metadata(artifact.path / SPARSE_METADATA_FILENAME)
    if (
        metadata.artifact_id != artifact.manifest.artifact_id
        or metadata.config_sha256 != config.sha256
        or metadata.foundation_manifest_sha256 != foundation.manifest_sha256
    ):
        raise SparseIndexValidationError("existing sparse metadata has incompatible lineage")
    return SparseBuildResult(artifact=artifact, metadata=metadata, reused=True)


def build_sparse_index(
    release: ResolvedReleaseManifest,
    config: ResolvedConfig,
    *,
    code_revision: str,
    artifact_store: ArtifactStore | None = None,
) -> SparseBuildResult:
    """Build or reuse the persisted BM25 index over the fixed catalog."""
    store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
    foundation, foundation_manifest, membership, documents = _load_foundation_dependency(
        release, config, store
    )
    artifact_id = sparse_artifact_id(release, config.sha256)
    artifact_path = store.root.joinpath(*artifact_id.split("/"))
    if artifact_path.exists() or artifact_path.is_symlink():
        return _reuse_sparse(release, config, foundation, store)

    aligned = _aligned_documents(membership, documents)
    dependency = ArtifactDependency(
        artifact_id=foundation.manifest.artifact_id,
        manifest_sha256=foundation.manifest_sha256,
    )
    started = time.perf_counter()
    try:
        with store.stage(
            artifact_type="sparse-index",
            dataset_version=release.manifest.dataset_version,
            profile="portfolio",
            component_version=config.config.retrieval.sparse.component_version,
            config_sha256=config.sha256,
            code_revision=code_revision,
            dependencies=(dependency,),
        ) as transaction:
            root = transaction.path(SPARSE_METADATA_FILENAME).parent
            vocabulary_size, posting_count, total_tokens, zero_tokens, length_sum = (
                _build_index_payload(root, aligned, config)
            )
            metadata = _metadata_for(
                root,
                release,
                config,
                foundation,
                foundation_manifest,
                started=started,
                vocabulary_size=vocabulary_size,
                posting_count=posting_count,
                total_tokens=total_tokens,
                zero_token_documents=zero_tokens,
                document_length_sum=length_sum,
            )
            transaction.path(SPARSE_METADATA_FILENAME).write_text(
                _canonical_json(metadata), encoding="utf-8"
            )
            artifact = transaction.commit()
    except ArtifactExistsError:
        return _reuse_sparse(release, config, foundation, store)
    return SparseBuildResult(artifact=artifact, metadata=metadata, reused=False)


class _MappedArray:
    def __init__(self, path: Path, format_code: Literal["I", "Q", "f"], expected: int) -> None:
        if sys.byteorder != "little":
            raise SparseIndexValidationError("typed sparse arrays require a little-endian host")
        item_size = struct.calcsize(format_code)
        if path.stat().st_size != expected * item_size:
            raise SparseIndexValidationError(
                f"typed array {path.name} has an incompatible byte length"
            )
        self._stream = path.open("rb")
        self._mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.values = memoryview(self._mapping).cast(format_code)

    def close(self) -> None:
        self.values.release()
        self._mapping.close()
        self._stream.close()


class SparseIndex:
    """Memory-mapped persisted BM25 index with deterministic search and pair scoring."""

    def __init__(self, artifact: LoadedArtifact, metadata: SparseIndexMetadata) -> None:
        self.artifact = artifact
        self.metadata = metadata
        self._closed = False
        try:
            vocabulary = (
                (artifact.path / VOCABULARY_FILENAME).read_text(encoding="utf-8").splitlines()
            )
            document_map = pl.read_parquet(artifact.path / DOCUMENT_MAP_FILENAME).sort(
                "catalog_ordinal"
            )
        except (OSError, UnicodeDecodeError, pl.exceptions.PolarsError) as exc:
            raise SparseIndexValidationError(f"cannot load sparse identity maps: {exc}") from exc
        if len(vocabulary) != metadata.vocabulary_size:
            raise SparseIndexValidationError("vocabulary size does not match metadata")
        if document_map.height != metadata.document_count:
            raise SparseIndexValidationError("document map count does not match metadata")
        if document_map["catalog_ordinal"].to_list() != list(range(metadata.document_count)):
            raise SparseIndexValidationError("document map ordinals are not contiguous")
        self._vocabulary = {token: index for index, token in enumerate(vocabulary)}
        self._product_ids = tuple(str(value) for value in document_map["product_id"].to_list())
        self._locales = tuple(str(value) for value in document_map["locale"].to_list())
        if len(set(self._product_ids)) != len(self._product_ids):
            raise SparseIndexValidationError("product IDs are not unique in the US catalog")
        self._ordinals = {product_id: index for index, product_id in enumerate(self._product_ids)}
        root = artifact.path
        self._offsets = _MappedArray(
            root / POSTINGS_OFFSETS_FILENAME, "Q", metadata.vocabulary_size + 1
        )
        self._doc_ids = _MappedArray(root / POSTING_DOC_IDS_FILENAME, "I", metadata.posting_count)
        self._term_frequencies = _MappedArray(
            root / POSTING_TFS_FILENAME, "I", metadata.posting_count
        )
        self._document_lengths = _MappedArray(
            root / DOCUMENT_LENGTHS_FILENAME, "I", metadata.document_count
        )
        self._document_frequencies = _MappedArray(
            root / DOCUMENT_FREQUENCIES_FILENAME, "I", metadata.vocabulary_size
        )
        self._idf = _MappedArray(
            root / INVERSE_DOCUMENT_FREQUENCIES_FILENAME, "f", metadata.vocabulary_size
        )
        if self._offsets.values[-1] != metadata.posting_count:
            self.close()
            raise SparseIndexValidationError("final posting offset does not match posting count")

    @property
    def retriever_id(self) -> str:
        return (
            f"bm25:{self.metadata.component_version}:{self.metadata.tokenizer_version}:"
            f"k1={self.metadata.k1}:b={self.metadata.b}"
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
        for mapped in (
            self._idf,
            self._document_frequencies,
            self._document_lengths,
            self._term_frequencies,
            self._doc_ids,
            self._offsets,
        ):
            mapped.close()
        self._closed = True

    def _validate_query(self, query: str) -> tuple[int, ...]:
        if self._closed:
            raise SparseQueryError("sparse index is closed")
        if not isinstance(query, str) or len(query) > 4096:
            raise SparseQueryError("query must be a string of at most 4096 characters")
        return tuple(
            sorted(
                {
                    term_id
                    for token in tokenize(query)
                    if (term_id := self._vocabulary.get(token)) is not None
                }
            )
        )

    def _term_score(self, term_id: int, posting_index: int) -> float:
        document_id = int(self._doc_ids.values[posting_index])
        frequency = float(self._term_frequencies.values[posting_index])
        document_length = float(self._document_lengths.values[document_id])
        denominator = frequency + self.metadata.k1 * (
            1.0
            - self.metadata.b
            + self.metadata.b * document_length / self.metadata.average_document_length
        )
        return float(self._idf.values[term_id]) * frequency * (self.metadata.k1 + 1.0) / denominator

    def search(self, query: str, top_k: int | None = None) -> tuple[SparseCandidate, ...]:
        """Return fixed-catalog BM25 candidates with stable product-ID tie breaks."""
        selected_k = self.metadata.default_top_k if top_k is None else top_k
        if not isinstance(selected_k, int) or isinstance(selected_k, bool):
            raise SparseQueryError("top_k must be an integer")
        if selected_k < 1 or selected_k > self.metadata.max_top_k:
            raise SparseQueryError(f"top_k must be between 1 and {self.metadata.max_top_k}")
        term_ids = self._validate_query(query)
        if not term_ids:
            return ()
        scores = array("d", [0.0]) * self.metadata.document_count
        touched_flags = bytearray(self.metadata.document_count)
        touched: list[int] = []
        for term_id in term_ids:
            start = int(self._offsets.values[term_id])
            stop = int(self._offsets.values[term_id + 1])
            for posting_index in range(start, stop):
                document_id = int(self._doc_ids.values[posting_index])
                scores[document_id] += self._term_score(term_id, posting_index)
                if not touched_flags[document_id]:
                    touched_flags[document_id] = 1
                    touched.append(document_id)
        winners = heapq.nsmallest(
            min(selected_k, len(touched)),
            touched,
            key=lambda document_id: (-scores[document_id], self._product_ids[document_id]),
        )
        return tuple(
            SparseCandidate(
                product_id=self._product_ids[document_id],
                locale=self._locales[document_id],
                raw_score=float(scores[document_id]),
                one_based_rank=rank,
                retriever_id=self.retriever_id,
                index_id=self.artifact.manifest.artifact_id,
            )
            for rank, document_id in enumerate(winners, start=1)
        )

    def score_pairs(self, query: str, product_ids: tuple[str, ...]) -> tuple[SparsePairScore, ...]:
        """Score every explicit known product, including zero-match products."""
        if len(product_ids) != len(set(product_ids)):
            raise SparseQueryError("explicit product IDs must be unique")
        unknown = tuple(
            product_id for product_id in product_ids if product_id not in self._ordinals
        )
        if unknown:
            raise SparseQueryError(f"explicit product IDs are outside the catalog: {unknown[:3]}")
        term_ids = self._validate_query(query)
        requested = {self._ordinals[product_id]: product_id for product_id in product_ids}
        scores = {ordinal: 0.0 for ordinal in requested}
        for term_id in term_ids:
            start = int(self._offsets.values[term_id])
            stop = int(self._offsets.values[term_id + 1])
            for posting_index in range(start, stop):
                document_id = int(self._doc_ids.values[posting_index])
                if document_id in requested:
                    scores[document_id] += self._term_score(term_id, posting_index)
        return tuple(
            SparsePairScore(
                product_id=product_id,
                locale=self._locales[self._ordinals[product_id]],
                raw_score=scores[self._ordinals[product_id]],
                retriever_id=self.retriever_id,
                index_id=self.artifact.manifest.artifact_id,
            )
            for product_id in product_ids
        )

    def query_idf_values(self, query: str) -> tuple[float, ...]:
        """Return unique in-vocabulary query-token IDFs for shared feature formulas."""
        term_ids = self._validate_query(query)
        return tuple(float(self._idf.values[term_id]) for term_id in term_ids)


def load_sparse_index(store: ArtifactStore, artifact_id: str) -> SparseIndex:
    """Recursively verify and memory-map one immutable sparse index artifact."""
    artifact = store.load(artifact_id)
    metadata = load_sparse_metadata(artifact.path / SPARSE_METADATA_FILENAME)
    if metadata.artifact_id != artifact.manifest.artifact_id:
        raise SparseIndexValidationError("sparse metadata artifact ID is incompatible")
    return SparseIndex(artifact, metadata)


__all__ = [
    "DOCUMENT_MAP_FILENAME",
    "SPARSE_METADATA_FILENAME",
    "SparseBuildError",
    "SparseBuildResult",
    "SparseCandidate",
    "SparseIndex",
    "SparseIndexMetadata",
    "SparseIndexValidationError",
    "SparsePairScore",
    "SparseQueryError",
    "SparseResourceError",
    "SparseResourceMeasurement",
    "SparseRetrievalError",
    "build_sparse_index",
    "load_sparse_index",
    "load_sparse_metadata",
    "sparse_artifact_id",
    "tokenize",
]
