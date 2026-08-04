"""Toy proofs for protocol-safe Goldfish 007 ranking and retrieval metrics."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from market_rank.evaluation.metrics import (
    CLOSED_POOL_PROTOCOL,
    END_TO_END_PROTOCOL,
    RETRIEVAL_PROTOCOL,
    Judgment,
    MetricProtocolError,
    MetricRecord,
    dcg_at_k,
    evaluate_ranked_products,
    ndcg_at_k,
)


def _judgments() -> tuple[Judgment, ...]:
    return (
        Judgment(product_id="exact", label="E", gain=1.0),
        Judgment(product_id="substitute", label="S", gain=0.1),
        Judgment(product_id="complement", label="C", gain=0.01),
        Judgment(product_id="irrelevant", label="I", gain=0.0),
    )


def _values(records: tuple[MetricRecord, ...]) -> dict[str, float]:
    return {record.metric: record.value for record in records}


def test_dcg_uses_official_gains_directly_without_exponential_transform() -> None:
    observed = dcg_at_k((1.0, 0.1, 0.01), 3)
    expected = 1.0 + 0.1 / math.log2(3) + 0.01 / math.log2(4)

    assert observed == pytest.approx(expected)


def test_ndcg_matches_hand_calculated_ideal_order() -> None:
    assert ndcg_at_k((1.0, 0.1, 0.01, 0.0), (0.0, 0.01, 0.1, 1.0), 4) == (pytest.approx(1.0))
    assert ndcg_at_k((0.0, 0.01), (0.0, 0.0), 2) == 0.0


def test_closed_pool_metrics_require_the_complete_judged_set() -> None:
    with pytest.raises(MetricProtocolError, match="complete"):
        evaluate_ranked_products(
            CLOSED_POOL_PROTOCOL,
            ("exact", "substitute"),
            _judgments(),
            k=2,
        )


def test_closed_pool_metrics_report_ndcg_precision_map_mrr_and_exact_hit() -> None:
    records = evaluate_ranked_products(
        CLOSED_POOL_PROTOCOL,
        ("irrelevant", "exact", "substitute", "complement"),
        _judgments(),
        k=3,
    )
    values = _values(records)

    assert set(values) == {"ndcg_official_gain", "precision", "map", "mrr", "exact_hit"}
    assert values["precision"] == pytest.approx(2 / 3)
    assert values["mrr"] == pytest.approx(0.5)
    assert values["map"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert values["exact_hit"] == 1.0
    assert all(record.protocol == CLOSED_POOL_PROTOCOL for record in records)


def test_catalog_retrieval_metrics_do_not_emit_naive_precision_map_or_ndcg() -> None:
    records = evaluate_ranked_products(
        RETRIEVAL_PROTOCOL,
        ("unknown-a", "substitute", "unknown-b", "exact"),
        _judgments(),
        k=4,
    )
    values = _values(records)

    assert set(values) == {
        "judged_recall",
        "exact_hit",
        "judged_mrr",
        "known_judgment_coverage",
        "unjudged_rate",
    }
    assert values["judged_recall"] == 1.0
    assert values["judged_mrr"] == pytest.approx(0.5)
    assert values["known_judgment_coverage"] == 0.5
    assert values["unjudged_rate"] == 0.5
    assert all(record.unjudged_count == 2 for record in records)


def test_end_to_end_diagnostic_reuses_only_retrieval_safe_metrics() -> None:
    records = evaluate_ranked_products(
        END_TO_END_PROTOCOL,
        ("unknown", "exact", "substitute"),
        _judgments(),
        k=3,
    )

    assert {record.metric for record in records} == {
        "judged_recall",
        "exact_hit",
        "judged_mrr",
        "known_judgment_coverage",
        "unjudged_rate",
    }
    assert all(record.protocol == END_TO_END_PROTOCOL for record in records)


def test_empty_catalog_result_remains_in_evaluation_with_zero_metrics() -> None:
    records = evaluate_ranked_products(RETRIEVAL_PROTOCOL, (), _judgments(), k=10)

    assert all(record.value == 0.0 for record in records)
    assert all(record.returned_count == 0 for record in records)


def test_relevance_threshold_is_explicit_and_supports_exact_only() -> None:
    records = evaluate_ranked_products(
        RETRIEVAL_PROTOCOL,
        ("substitute",),
        _judgments(),
        k=1,
        relevant_labels=frozenset({"E"}),
    )

    assert _values(records)["judged_recall"] == 0.0


def test_duplicate_ranked_products_are_rejected() -> None:
    with pytest.raises(MetricProtocolError, match="duplicates"):
        evaluate_ranked_products(
            RETRIEVAL_PROTOCOL,
            ("exact", "exact"),
            _judgments(),
            k=2,
        )


def test_invalid_cutoffs_and_relevance_labels_are_rejected() -> None:
    with pytest.raises(MetricProtocolError, match="positive"):
        evaluate_ranked_products(RETRIEVAL_PROTOCOL, (), _judgments(), k=0)
    with pytest.raises(MetricProtocolError, match="non-empty"):
        evaluate_ranked_products(
            RETRIEVAL_PROTOCOL,
            (),
            _judgments(),
            k=1,
            relevant_labels=frozenset(),
        )


def test_official_label_gain_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="official ESCI"):
        Judgment(product_id="wrong", label="E", gain=0.1)
