"""Protocol-safe offline evaluation primitives."""

from market_rank.evaluation.metrics import (
    CLOSED_POOL_PROTOCOL,
    RETRIEVAL_PROTOCOL,
    Judgment,
    MetricProtocolError,
    MetricRecord,
    dcg_at_k,
    evaluate_ranked_products,
    ndcg_at_k,
)

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
