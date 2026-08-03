"""Persisted candidate-retrieval contracts."""

from market_rank.retrieval.sparse import (
    SparseBuildError,
    SparseBuildResult,
    SparseCandidate,
    SparseIndex,
    SparseIndexMetadata,
    SparseIndexValidationError,
    SparsePairScore,
    SparseQueryError,
    SparseResourceError,
    SparseResourceMeasurement,
    SparseRetrievalError,
    build_sparse_index,
    load_sparse_index,
    sparse_artifact_id,
    tokenize,
)

__all__ = [
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
    "sparse_artifact_id",
    "tokenize",
]
