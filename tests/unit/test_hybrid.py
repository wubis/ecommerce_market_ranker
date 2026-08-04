"""Toy reciprocal-rank fusion tests for Goldfish 009."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from market_rank.retrieval.hybrid import HybridRetrievalError, fuse_rrf


@dataclass(frozen=True)
class Candidate:
    product_id: str
    locale: str
    raw_score: float
    one_based_rank: int
    retriever_id: str
    index_id: str


def _candidate(product_id: str, rank: int, source: str, score: float = 1.0) -> Candidate:
    return Candidate(
        product_id=product_id,
        locale="us",
        raw_score=score,
        one_based_rank=rank,
        retriever_id=source,
        index_id=f"{source}-index",
    )


def test_rrf_unions_deduplicates_and_retains_nullable_source_evidence() -> None:
    sparse = (_candidate("a", 1, "bm25", 4.0), _candidate("b", 2, "bm25", 3.0))
    dense = (_candidate("b", 1, "dense", 0.9), _candidate("c", 2, "dense", 0.8))

    result = fuse_rrf(sparse, dense, rrf_constant=60, top_k=3)

    assert tuple(item.product_id for item in result.candidates) == ("b", "a", "c")
    shared, sparse_only, dense_only = result.candidates
    assert shared.rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert shared.sparse_rank == 2
    assert shared.dense_rank == 1
    assert shared.source_count == 2
    assert sparse_only.dense_rank is None
    assert dense_only.sparse_score is None
    assert result.union_count_before_truncation == 3
    assert result.degraded_sources == ()


def test_rrf_ties_use_best_source_rank_then_product_id() -> None:
    sparse = (_candidate("z", 1, "bm25"), _candidate("a", 2, "bm25"))
    dense = (_candidate("a", 1, "dense"), _candidate("z", 2, "dense"))

    result = fuse_rrf(sparse, dense, top_k=2)

    assert result.candidates[0].rrf_score == result.candidates[1].rrf_score
    assert tuple(item.product_id for item in result.candidates) == ("a", "z")
    assert tuple(item.one_based_rank for item in result.candidates) == (1, 2)


def test_one_empty_source_is_valid_and_explicitly_degraded() -> None:
    dense = (_candidate("a", 1, "dense", 0.5),)

    result = fuse_rrf((), dense)

    assert tuple(item.product_id for item in result.candidates) == ("a",)
    assert result.degraded_sources == ("sparse",)
    assert result.candidates[0].rrf_score == pytest.approx(1 / 61)


def test_two_empty_sources_produce_a_structured_empty_union() -> None:
    result = fuse_rrf((), ())

    assert result.candidates == ()
    assert result.union_count_before_truncation == 0
    assert result.degraded_sources == ("sparse", "dense")


@pytest.mark.parametrize(
    ("sparse", "dense", "match"),
    [
        (
            (_candidate("a", 1, "bm25"), _candidate("a", 2, "bm25")),
            (),
            "duplicate",
        ),
        ((_candidate("a", 2, "bm25"),), (), "contiguous"),
        ((replace(_candidate("a", 1, "bm25"), raw_score=float("nan")),), (), "finite"),
        (
            (_candidate("a", 1, "bm25"),),
            (replace(_candidate("a", 1, "dense"), locale="ca"),),
            "locale",
        ),
    ],
)
def test_invalid_source_candidates_fail(
    sparse: tuple[Candidate, ...],
    dense: tuple[Candidate, ...],
    match: str,
) -> None:
    with pytest.raises(HybridRetrievalError, match=match):
        fuse_rrf(sparse, dense)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rrf_constant": 0},
        {"top_k": 0},
        {"top_k": 11, "max_top_k": 10},
        {"top_k": True},
    ],
)
def test_invalid_fusion_bounds_fail(kwargs: dict[str, int]) -> None:
    with pytest.raises(HybridRetrievalError):
        fuse_rrf((), (), **kwargs)
