"""Pure comparison and presentation metrics for the Streamlit demo."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_rank.serving.contracts import SearchMode, SearchResponse

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


class DemoPresentationError(ValueError):
    """Raised when API responses cannot be compared honestly."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListMetrics(_StrictModel):
    result_count: int = Field(strict=True, ge=0)
    unique_brand_count: int = Field(strict=True, ge=0)
    missing_brand_count: int = Field(strict=True, ge=0)
    maximum_brand_concentration: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    brand_entropy_bits: float = Field(ge=0.0, allow_inf_nan=False)
    title_token_ild: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ModeSummary(_StrictModel):
    requested_mode: SearchMode
    resolved_stage: str = Field(strict=True, min_length=1)
    promoted_stage: str = Field(strict=True, min_length=1)
    score_field: str = Field(strict=True, min_length=1)
    comparable_with_promoted_stage: bool = Field(strict=True)
    degraded: bool = Field(strict=True)
    fallback_reason_codes: tuple[str, ...]
    candidate_count: int = Field(strict=True, ge=0)
    total_ms: float = Field(ge=0.0, allow_inf_nan=False)
    metrics: ListMetrics


class RankPosition(_StrictModel):
    mode: SearchMode
    rank: int | None = Field(default=None, strict=True, ge=1)
    change_from_baseline: int | None = Field(default=None, strict=True)


class RankChange(_StrictModel):
    product_id: str = Field(strict=True, min_length=1)
    title: str
    brand: str
    baseline_mode: SearchMode
    positions: tuple[RankPosition, ...]


class ComparisonReport(_StrictModel):
    query_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    bundle_id: str = Field(strict=True, min_length=1)
    catalog_id: str = Field(strict=True, min_length=1)
    config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    baseline_mode: SearchMode
    summaries: tuple[ModeSummary, ...]
    rank_changes: tuple[RankChange, ...]

    @model_validator(mode="after")
    def validate_modes(self) -> Self:
        modes = tuple(summary.requested_mode for summary in self.summaries)
        if not modes or modes[0] != self.baseline_mode or len(set(modes)) != len(modes):
            raise ValueError("comparison modes must be unique and begin with the baseline")
        return self


def _title_dissimilarity(left: str, right: str) -> float:
    left_tokens = set(_display_tokens(left))
    right_tokens = set(_display_tokens(right))
    union = left_tokens | right_tokens
    return 0.0 if not union else 1.0 - len(left_tokens & right_tokens) / len(union)


def _display_tokens(value: str) -> tuple[str, ...]:
    """Tokenize titles locally so the UI never imports retrieval/model code."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def compute_list_metrics(response: SearchResponse) -> ListMetrics:
    """Compute transparent display diagnostics; none are relevance judgments."""
    results = response.results
    if not results:
        return ListMetrics(
            result_count=0,
            unique_brand_count=0,
            missing_brand_count=0,
            maximum_brand_concentration=0.0,
            brand_entropy_bits=0.0,
            title_token_ild=0.0,
        )
    brands = tuple(result.brand.strip().casefold() or "(missing)" for result in results)
    counts = Counter(brands)
    total = len(brands)
    probabilities = tuple(count / total for count in counts.values())
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    pair_dissimilarities = tuple(
        _title_dissimilarity(results[left].title, results[right].title)
        for left in range(len(results))
        for right in range(left + 1, len(results))
    )
    return ListMetrics(
        result_count=total,
        unique_brand_count=len({brand for brand in brands if brand != "(missing)"}),
        missing_brand_count=counts.get("(missing)", 0),
        maximum_brand_concentration=max(counts.values()) / total,
        brand_entropy_bits=entropy,
        title_token_ild=(
            sum(pair_dissimilarities) / len(pair_dissimilarities) if pair_dissimilarities else 0.0
        ),
    )


def compare_responses(responses: tuple[SearchResponse, ...]) -> ComparisonReport:
    """Align mode outputs and compute signed movement from the first response."""
    if not responses:
        raise DemoPresentationError("at least one search response is required")
    identity = (
        responses[0].query_sha256,
        responses[0].bundle_id,
        responses[0].catalog_id,
        responses[0].config_sha256,
    )
    if any(
        (
            response.query_sha256,
            response.bundle_id,
            response.catalog_id,
            response.config_sha256,
        )
        != identity
        for response in responses[1:]
    ):
        raise DemoPresentationError("comparison responses have incompatible query or lineage")
    modes = tuple(response.requested_mode for response in responses)
    if len(set(modes)) != len(modes):
        raise DemoPresentationError("comparison responses contain duplicate modes")

    summaries = tuple(
        ModeSummary(
            requested_mode=response.requested_mode,
            resolved_stage=response.resolved_stage,
            promoted_stage=response.promoted_stage,
            score_field=response.score_field,
            comparable_with_promoted_stage=response.score_comparable_with_promoted_stage,
            degraded=response.degraded,
            fallback_reason_codes=tuple(event.reason_code for event in response.fallbacks),
            candidate_count=response.candidate_count,
            total_ms=response.timings.total_ms,
            metrics=compute_list_metrics(response),
        )
        for response in responses
    )
    result_by_mode = {
        response.requested_mode: {result.product_id: result for result in response.results}
        for response in responses
    }
    product_order: list[str] = []
    for response in responses:
        for result in response.results:
            if result.product_id not in product_order:
                product_order.append(result.product_id)
    baseline_mode = modes[0]
    baseline = result_by_mode[baseline_mode]
    changes: list[RankChange] = []
    for product_id in product_order:
        display = next(
            result_by_mode[mode][product_id] for mode in modes if product_id in result_by_mode[mode]
        )
        baseline_rank = baseline[product_id].rank if product_id in baseline else None
        positions = tuple(
            RankPosition(
                mode=mode,
                rank=(
                    result_by_mode[mode][product_id].rank
                    if product_id in result_by_mode[mode]
                    else None
                ),
                change_from_baseline=(
                    baseline_rank - result_by_mode[mode][product_id].rank
                    if baseline_rank is not None and product_id in result_by_mode[mode]
                    else None
                ),
            )
            for mode in modes
        )
        changes.append(
            RankChange(
                product_id=product_id,
                title=display.title,
                brand=display.brand,
                baseline_mode=baseline_mode,
                positions=positions,
            )
        )
    return ComparisonReport(
        query_sha256=identity[0],
        bundle_id=identity[1],
        catalog_id=identity[2],
        config_sha256=identity[3],
        baseline_mode=baseline_mode,
        summaries=summaries,
        rank_changes=tuple(changes),
    )


__all__ = [
    "ComparisonReport",
    "DemoPresentationError",
    "ListMetrics",
    "ModeSummary",
    "RankChange",
    "RankPosition",
    "compare_responses",
    "compute_list_metrics",
]
