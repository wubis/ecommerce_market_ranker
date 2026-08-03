"""Metric primitives that prevent closed-pool and catalog-retrieval protocol mixing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CLOSED_POOL_PROTOCOL: Literal["closed_pool_task1_v1"] = "closed_pool_task1_v1"
RETRIEVAL_PROTOCOL: Literal["retrieval_catalog_task1_us_v1"] = "retrieval_catalog_task1_us_v1"
MetricProtocol = Literal["closed_pool_task1_v1", "retrieval_catalog_task1_us_v1"]
EsciLabel = Literal["I", "C", "S", "E"]

_OFFICIAL_GAINS: dict[EsciLabel, float] = {"I": 0.0, "C": 0.01, "S": 0.1, "E": 1.0}


class MetricProtocolError(ValueError):
    """Raised when a metric request violates its candidate-population protocol."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Judgment(_StrictModel):
    """One official query-product judgment with the published non-exponential gain."""

    product_id: str = Field(strict=True, min_length=1)
    label: EsciLabel
    gain: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_gain(self) -> Judgment:
        if self.gain != _OFFICIAL_GAINS[self.label]:
            raise ValueError("gain does not match the official ESCI Task-1 label mapping")
        return self


class MetricRecord(_StrictModel):
    """One protocol-labelled query-level metric result."""

    protocol: MetricProtocol
    metric: str = Field(strict=True, min_length=1)
    cutoff: int = Field(strict=True, ge=1)
    value: float = Field(ge=0.0, allow_inf_nan=False)
    returned_count: int = Field(strict=True, ge=0)
    judged_count: int = Field(strict=True, ge=0)
    unjudged_count: int = Field(strict=True, ge=0)
    relevant_judgment_count: int = Field(strict=True, ge=0)


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise MetricProtocolError("metric cutoff must be a positive integer")


def _validate_ranked(ranked_product_ids: Sequence[str]) -> tuple[str, ...]:
    ranked = tuple(ranked_product_ids)
    if any(not isinstance(product_id, str) or not product_id for product_id in ranked):
        raise MetricProtocolError("ranked product IDs must be non-empty strings")
    if len(ranked) != len(set(ranked)):
        raise MetricProtocolError("ranked product IDs must not contain duplicates")
    return ranked


def _judgment_map(judgments: Sequence[Judgment]) -> dict[str, Judgment]:
    result = {judgment.product_id: judgment for judgment in judgments}
    if len(result) != len(judgments):
        raise MetricProtocolError("official judgments must have unique product IDs")
    return result


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Compute DCG with gains used directly, never an additional exponential transform."""
    _validate_k(k)
    if any(not math.isfinite(gain) or gain < 0.0 for gain in gains):
        raise MetricProtocolError("DCG gains must be finite and non-negative")
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains[:k], start=1))


def ndcg_at_k(ranked_gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> float:
    """Compute official-gain NDCG for a complete judged candidate pool."""
    _validate_k(k)
    ideal = dcg_at_k(sorted(ideal_gains, reverse=True), k)
    return 0.0 if ideal == 0.0 else dcg_at_k(ranked_gains, k) / ideal


def _record(
    protocol: MetricProtocol,
    metric: str,
    k: int,
    value: float,
    *,
    returned_count: int,
    judged_count: int,
    unjudged_count: int,
    relevant_judgment_count: int,
) -> MetricRecord:
    return MetricRecord(
        protocol=protocol,
        metric=metric,
        cutoff=k,
        value=value,
        returned_count=returned_count,
        judged_count=judged_count,
        unjudged_count=unjudged_count,
        relevant_judgment_count=relevant_judgment_count,
    )


def _closed_pool_metrics(
    ranked: tuple[str, ...],
    judgments: dict[str, Judgment],
    k: int,
    relevant_labels: frozenset[EsciLabel],
) -> tuple[MetricRecord, ...]:
    if set(ranked) != set(judgments):
        raise MetricProtocolError(
            "closed-pool metrics require exactly the complete official judged product set"
        )
    head = ranked[:k]
    relevant = {
        product_id for product_id, item in judgments.items() if item.label in relevant_labels
    }
    ranked_relevance = [product_id in relevant for product_id in head]
    relevant_count = len(relevant)
    precision_denominator = min(k, len(ranked))
    precision = sum(ranked_relevance) / precision_denominator if precision_denominator else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, is_relevant in enumerate(ranked_relevance, start=1) if is_relevant),
        0.0,
    )
    precision_sum = 0.0
    observed_relevant = 0
    for rank, is_relevant in enumerate(ranked_relevance, start=1):
        if is_relevant:
            observed_relevant += 1
            precision_sum += observed_relevant / rank
    average_precision = precision_sum / relevant_count if relevant_count else 0.0
    ranked_gains = [judgments[product_id].gain for product_id in ranked]
    ideal_gains = [item.gain for item in judgments.values()]
    common = {
        "returned_count": len(head),
        "judged_count": len(head),
        "unjudged_count": 0,
        "relevant_judgment_count": relevant_count,
    }
    return (
        _record(
            CLOSED_POOL_PROTOCOL,
            "ndcg_official_gain",
            k,
            ndcg_at_k(ranked_gains, ideal_gains, k),
            **common,
        ),
        _record(CLOSED_POOL_PROTOCOL, "precision", k, precision, **common),
        _record(CLOSED_POOL_PROTOCOL, "map", k, average_precision, **common),
        _record(CLOSED_POOL_PROTOCOL, "mrr", k, reciprocal_rank, **common),
        _record(
            CLOSED_POOL_PROTOCOL,
            "exact_hit",
            k,
            float(any(judgments[product_id].label == "E" for product_id in head)),
            **common,
        ),
    )


def _retrieval_metrics(
    ranked: tuple[str, ...],
    judgments: dict[str, Judgment],
    k: int,
    relevant_labels: frozenset[EsciLabel],
) -> tuple[MetricRecord, ...]:
    head = ranked[:k]
    known = [product_id for product_id in head if product_id in judgments]
    relevant = {
        product_id for product_id, item in judgments.items() if item.label in relevant_labels
    }
    retrieved_relevant = [product_id for product_id in head if product_id in relevant]
    relevant_count = len(relevant)
    recall = len(retrieved_relevant) / relevant_count if relevant_count else 0.0
    judged_mrr = next(
        (1.0 / rank for rank, product_id in enumerate(head, start=1) if product_id in relevant),
        0.0,
    )
    known_coverage = len(known) / len(head) if head else 0.0
    unjudged_count = len(head) - len(known)
    common = {
        "returned_count": len(head),
        "judged_count": len(known),
        "unjudged_count": unjudged_count,
        "relevant_judgment_count": relevant_count,
    }
    return (
        _record(RETRIEVAL_PROTOCOL, "judged_recall", k, recall, **common),
        _record(
            RETRIEVAL_PROTOCOL,
            "exact_hit",
            k,
            float(
                any(
                    product_id in judgments and judgments[product_id].label == "E"
                    for product_id in head
                )
            ),
            **common,
        ),
        _record(RETRIEVAL_PROTOCOL, "judged_mrr", k, judged_mrr, **common),
        _record(RETRIEVAL_PROTOCOL, "known_judgment_coverage", k, known_coverage, **common),
        _record(
            RETRIEVAL_PROTOCOL,
            "unjudged_rate",
            k,
            unjudged_count / len(head) if head else 0.0,
            **common,
        ),
    )


def evaluate_ranked_products(
    protocol: MetricProtocol,
    ranked_product_ids: Sequence[str],
    judgments: Sequence[Judgment],
    *,
    k: int,
    relevant_labels: frozenset[EsciLabel] = frozenset({"E", "S"}),
) -> tuple[MetricRecord, ...]:
    """Evaluate one query while enforcing its named candidate-population protocol."""
    _validate_k(k)
    if not relevant_labels or not relevant_labels <= frozenset(_OFFICIAL_GAINS):
        raise MetricProtocolError("relevant_labels must be a non-empty ESCI label subset")
    ranked = _validate_ranked(ranked_product_ids)
    judgment_by_product = _judgment_map(judgments)
    if protocol == CLOSED_POOL_PROTOCOL:
        return _closed_pool_metrics(ranked, judgment_by_product, k, relevant_labels)
    if protocol == RETRIEVAL_PROTOCOL:
        return _retrieval_metrics(ranked, judgment_by_product, k, relevant_labels)
    raise MetricProtocolError(f"unsupported metric protocol: {protocol}")


__all__ = [
    "CLOSED_POOL_PROTOCOL",
    "RETRIEVAL_PROTOCOL",
    "Judgment",
    "MetricProtocolError",
    "MetricRecord",
    "dcg_at_k",
    "evaluate_ranked_products",
    "ndcg_at_k",
]
