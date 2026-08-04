"""Lightweight HTTP and orchestration contracts shared by API clients."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SearchMode = Literal["active", "bm25", "dense", "hybrid", "pointwise", "lambdamart"]
ResolvedStage = Literal["bm25", "dense", "rrf", "pointwise", "lambdamart"]
ComponentState = Literal["ready", "degraded", "unavailable"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchRequest(_StrictModel):
    query: str = Field(strict=True, min_length=1, max_length=4096)
    top_k: int | None = Field(default=None, strict=True, ge=1, le=100)
    mode: SearchMode = "active"
    neural_rerank: bool = Field(default=False, strict=True)
    diversify: bool = Field(default=False, strict=True)
    deadline_ms: int | None = Field(default=None, strict=True, ge=100, le=30000)
    debug: bool = Field(default=False, strict=True)


class ComponentStatus(_StrictModel):
    component: Literal["bundle", "product_store", "sparse", "dense", "rankers"]
    state: ComponentState
    detail: str = Field(strict=True, min_length=1)


class FallbackEvent(_StrictModel):
    component: str = Field(strict=True, min_length=1)
    requested_stage: str = Field(strict=True, min_length=1)
    resolved_stage: str = Field(strict=True, min_length=1)
    reason_code: str = Field(strict=True, min_length=1)


class SearchTimings(_StrictModel):
    parse_ms: float = Field(ge=0.0, allow_inf_nan=False)
    sparse_ms: float = Field(ge=0.0, allow_inf_nan=False)
    dense_ms: float = Field(ge=0.0, allow_inf_nan=False)
    fusion_ms: float = Field(ge=0.0, allow_inf_nan=False)
    features_ms: float = Field(ge=0.0, allow_inf_nan=False)
    ranker_ms: float = Field(ge=0.0, allow_inf_nan=False)
    product_lookup_ms: float = Field(ge=0.0, allow_inf_nan=False)
    total_ms: float = Field(ge=0.0, allow_inf_nan=False)


class RetrievalProvenance(_StrictModel):
    bm25_score: float | None = Field(default=None, allow_inf_nan=False)
    bm25_rank: int | None = Field(default=None, strict=True, ge=1)
    sparse_retriever_id: str | None = None
    sparse_index_id: str | None = None
    dense_score: float | None = Field(default=None, allow_inf_nan=False)
    dense_rank: int | None = Field(default=None, strict=True, ge=1)
    dense_retriever_id: str | None = None
    dense_index_id: str | None = None
    rrf_score: float = Field(ge=0.0, allow_inf_nan=False)
    rrf_rank: int = Field(strict=True, ge=1)
    source_count: int = Field(strict=True, ge=1, le=2)


class ResultDebug(_StrictModel):
    feature_values: tuple[tuple[str, float], ...]


class SearchResult(_StrictModel):
    product_id: str = Field(strict=True, min_length=1)
    locale: Literal["us"]
    rank: int = Field(strict=True, ge=1)
    score: float = Field(allow_inf_nan=False)
    score_field: str = Field(strict=True, min_length=1)
    title: str
    brand: str
    color: str
    bullets: str
    description_snippet: str
    provenance: RetrievalProvenance
    debug: ResultDebug | None = None


class SearchResponse(_StrictModel):
    query_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    bundle_id: str = Field(strict=True, min_length=1)
    catalog_id: str = Field(strict=True, min_length=1)
    config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    requested_mode: SearchMode
    promoted_stage: Literal["rrf", "pointwise", "lambdamart"]
    resolved_stage: ResolvedStage
    score_field: str = Field(strict=True, min_length=1)
    score_comparable_with_promoted_stage: bool = Field(strict=True)
    degraded: bool = Field(strict=True)
    fallbacks: tuple[FallbackEvent, ...]
    candidate_count: int = Field(strict=True, ge=0)
    results: tuple[SearchResult, ...]
    timings: SearchTimings

    @model_validator(mode="after")
    def validate_results(self) -> Self:
        if self.results and tuple(result.rank for result in self.results) != tuple(
            range(1, len(self.results) + 1)
        ):
            raise ValueError("result ranks must be contiguous and one-based")
        if len({result.product_id for result in self.results}) != len(self.results):
            raise ValueError("search results must have unique product IDs")
        return self


class RuntimeInfo(_StrictModel):
    ready: bool = Field(strict=True)
    degraded: bool = Field(strict=True)
    bundle_id: str = Field(strict=True, min_length=1)
    catalog_id: str = Field(strict=True, min_length=1)
    config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    active_stage: Literal["rrf", "pointwise", "lambdamart"]
    components: tuple[ComponentStatus, ...]


class LivenessResponse(_StrictModel):
    status: str = Field(default="live", pattern="^live$")


class ReadinessResponse(_StrictModel):
    status: str = Field(pattern="^(ready|not_ready)$")
    ready: bool = Field(strict=True)
    degraded: bool = Field(strict=True)
    bundle_id: str | None
    active_stage: str | None
    components: tuple[ComponentStatus, ...]


class ModelInfoResponse(_StrictModel):
    bundle_id: str
    active_stage: str
    active_score_field: str
    active_score_comparable: bool
    fallback_contract: str
    feature_set_id: str
    feature_names: tuple[str, ...]
    parser_state_sha256: str
    ranking_models_artifact_id: str


class ArtifactComponentResponse(_StrictModel):
    component: str
    artifact_id: str
    manifest_sha256: str


class ArtifactInfoResponse(_StrictModel):
    bundle_id: str
    dataset_version: str
    profile: str
    catalog_id: str
    catalog_membership_sha256: str
    config_sha256: str
    offline_startup_required: bool
    components: tuple[ArtifactComponentResponse, ...]


__all__ = [
    "ArtifactComponentResponse",
    "ArtifactInfoResponse",
    "ComponentState",
    "ComponentStatus",
    "FallbackEvent",
    "LivenessResponse",
    "ModelInfoResponse",
    "ReadinessResponse",
    "ResolvedStage",
    "ResultDebug",
    "RetrievalProvenance",
    "RuntimeInfo",
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchTimings",
]
