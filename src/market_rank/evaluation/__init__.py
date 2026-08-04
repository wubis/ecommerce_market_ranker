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
from market_rank.evaluation.retrieval import (
    CombinedResourceMeasurement,
    HybridResourceError,
    RetrievalEvaluationBuildError,
    RetrievalEvaluationBuildResult,
    RetrievalEvaluationError,
    RetrievalEvaluationManifest,
    RetrievalEvaluationValidationError,
    build_retrieval_evaluation,
    load_retrieval_evaluation_manifest,
    retrieval_evaluation_artifact_id,
)

__all__ = [
    "CLOSED_POOL_PROTOCOL",
    "RETRIEVAL_PROTOCOL",
    "CombinedResourceMeasurement",
    "HybridResourceError",
    "Judgment",
    "MetricProtocolError",
    "MetricRecord",
    "RetrievalEvaluationBuildError",
    "RetrievalEvaluationBuildResult",
    "RetrievalEvaluationError",
    "RetrievalEvaluationManifest",
    "RetrievalEvaluationValidationError",
    "build_retrieval_evaluation",
    "dcg_at_k",
    "evaluate_ranked_products",
    "load_retrieval_evaluation_manifest",
    "ndcg_at_k",
    "retrieval_evaluation_artifact_id",
]
