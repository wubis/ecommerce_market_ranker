"""Deterministic reciprocal-rank fusion with complete source provenance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class HybridRetrievalError(ValueError):
    """Raised when source candidates or fusion bounds violate the RRF contract."""


class RankedCandidate(Protocol):
    @property
    def product_id(self) -> str: ...

    @property
    def locale(self) -> str: ...

    @property
    def raw_score(self) -> float: ...

    @property
    def one_based_rank(self) -> int: ...

    @property
    def retriever_id(self) -> str: ...

    @property
    def index_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class HybridCandidate:
    """One deduplicated union candidate with nullable per-source evidence."""

    product_id: str
    locale: str
    rrf_score: float
    one_based_rank: int
    best_source_rank: int
    sparse_score: float | None
    sparse_rank: int | None
    sparse_retriever_id: str | None
    sparse_index_id: str | None
    dense_score: float | None
    dense_rank: int | None
    dense_retriever_id: str | None
    dense_index_id: str | None
    source_count: int
    retriever_id: str


@dataclass(frozen=True, slots=True)
class HybridResult:
    """Bounded fused candidates plus explicit source availability."""

    candidates: tuple[HybridCandidate, ...]
    sparse_count: int
    dense_count: int
    union_count_before_truncation: int
    degraded_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Evidence:
    product_id: str
    locale: str
    sparse_score: float | None = None
    sparse_rank: int | None = None
    sparse_retriever_id: str | None = None
    sparse_index_id: str | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    dense_retriever_id: str | None = None
    dense_index_id: str | None = None


def _validate_source(name: str, candidates: Sequence[RankedCandidate]) -> None:
    observed_products: set[str] = set()
    for expected_rank, candidate in enumerate(candidates, start=1):
        if not candidate.product_id or not candidate.locale:
            raise HybridRetrievalError(f"{name} candidates require product and locale keys")
        if candidate.product_id in observed_products:
            raise HybridRetrievalError(f"{name} candidates contain duplicate product IDs")
        if candidate.one_based_rank != expected_rank:
            raise HybridRetrievalError(f"{name} candidate ranks must be contiguous and one-based")
        if not math.isfinite(candidate.raw_score):
            raise HybridRetrievalError(f"{name} candidate scores must be finite")
        if not candidate.retriever_id or not candidate.index_id:
            raise HybridRetrievalError(f"{name} candidates require retriever/index provenance")
        observed_products.add(candidate.product_id)


def fuse_rrf(
    sparse_candidates: Sequence[RankedCandidate],
    dense_candidates: Sequence[RankedCandidate],
    *,
    rrf_constant: int = 60,
    top_k: int = 200,
    max_top_k: int = 1000,
) -> HybridResult:
    """Union, deduplicate, retain source evidence, and rank by deterministic RRF."""
    for name, value in (
        ("rrf_constant", rrf_constant),
        ("top_k", top_k),
        ("max_top_k", max_top_k),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise HybridRetrievalError(f"{name} must be a positive integer")
    if top_k > max_top_k:
        raise HybridRetrievalError("top_k exceeds max_top_k")
    _validate_source("sparse", sparse_candidates)
    _validate_source("dense", dense_candidates)

    evidence: dict[str, _Evidence] = {}
    for candidate in sparse_candidates:
        evidence[candidate.product_id] = _Evidence(
            product_id=candidate.product_id,
            locale=candidate.locale,
            sparse_score=candidate.raw_score,
            sparse_rank=candidate.one_based_rank,
            sparse_retriever_id=candidate.retriever_id,
            sparse_index_id=candidate.index_id,
        )
    for candidate in dense_candidates:
        existing = evidence.get(candidate.product_id)
        if existing is not None and existing.locale != candidate.locale:
            raise HybridRetrievalError("source candidates disagree on product locale")
        evidence[candidate.product_id] = _Evidence(
            product_id=candidate.product_id,
            locale=candidate.locale,
            sparse_score=existing.sparse_score if existing else None,
            sparse_rank=existing.sparse_rank if existing else None,
            sparse_retriever_id=existing.sparse_retriever_id if existing else None,
            sparse_index_id=existing.sparse_index_id if existing else None,
            dense_score=candidate.raw_score,
            dense_rank=candidate.one_based_rank,
            dense_retriever_id=candidate.retriever_id,
            dense_index_id=candidate.index_id,
        )

    scored: list[tuple[float, int, str, _Evidence]] = []
    for item in evidence.values():
        ranks = tuple(rank for rank in (item.sparse_rank, item.dense_rank) if rank is not None)
        rrf_score = sum(1.0 / (rrf_constant + rank) for rank in ranks)
        scored.append((rrf_score, min(ranks), item.product_id, item))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))

    retriever_id = f"rrf:rrf-v1:k={rrf_constant}"
    candidates = tuple(
        HybridCandidate(
            product_id=item.product_id,
            locale=item.locale,
            rrf_score=rrf_score,
            one_based_rank=rank,
            best_source_rank=best_source_rank,
            sparse_score=item.sparse_score,
            sparse_rank=item.sparse_rank,
            sparse_retriever_id=item.sparse_retriever_id,
            sparse_index_id=item.sparse_index_id,
            dense_score=item.dense_score,
            dense_rank=item.dense_rank,
            dense_retriever_id=item.dense_retriever_id,
            dense_index_id=item.dense_index_id,
            source_count=int(item.sparse_rank is not None) + int(item.dense_rank is not None),
            retriever_id=retriever_id,
        )
        for rank, (rrf_score, best_source_rank, _, item) in enumerate(scored[:top_k], start=1)
    )
    degraded = tuple(
        source
        for source, source_candidates in (
            ("sparse", sparse_candidates),
            ("dense", dense_candidates),
        )
        if not source_candidates
    )
    return HybridResult(
        candidates=candidates,
        sparse_count=len(sparse_candidates),
        dense_count=len(dense_candidates),
        union_count_before_truncation=len(evidence),
        degraded_sources=degraded,
    )


__all__ = [
    "HybridCandidate",
    "HybridResult",
    "HybridRetrievalError",
    "RankedCandidate",
    "fuse_rrf",
]
