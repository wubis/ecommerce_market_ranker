"""Pure `ltr_core_v1` formulas shared by offline materialization and serving."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from market_rank.features.registry import FEATURE_NAMES
from market_rank.query.parser import ParsedQuery
from market_rank.retrieval.sparse import tokenize

_MODEL_RE = re.compile(r"(?i)[a-z0-9]+(?:[-_.][a-z0-9]+)*")


class FeatureFormulaError(ValueError):
    """Raised when pair inputs cannot produce a valid primary feature row."""


@dataclass(frozen=True, slots=True)
class ProductFeatureView:
    """Only official product fields required by shared feature formulas."""

    locale: str
    title: str
    brand: str
    color: str
    bullets: str
    description: str
    normalized_brand: str
    normalized_color: str
    title_missing: bool
    brand_missing: bool
    color_missing: bool
    bullets_missing: bool
    description_missing: bool


def rank_fractions(scores: dict[str, float]) -> tuple[dict[str, int], dict[str, float]]:
    """Rank finite scores descending with product-ID ties and bound rank to `[0,1]`."""
    if not scores:
        return {}, {}
    if any(not product_id or not math.isfinite(score) for product_id, score in scores.items()):
        raise FeatureFormulaError("rank inputs require nonempty product IDs and finite scores")
    ordered = sorted(scores, key=lambda product_id: (-scores[product_id], product_id))
    denominator = max(len(ordered) - 1, 1)
    ranks = {product_id: rank for rank, product_id in enumerate(ordered, start=1)}
    fractions = {
        product_id: np.float32((rank - 1) / denominator).item()
        for product_id, rank in ranks.items()
    }
    return ranks, fractions


def _coverage(query: set[str], product: set[str]) -> float:
    return 0.0 if not query else len(query & product) / len(query)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def _phrase_match(query_tokens: tuple[str, ...], product_tokens: tuple[str, ...]) -> int:
    width = len(query_tokens)
    if not width or width > len(product_tokens):
        return 0
    return int(
        any(
            product_tokens[start : start + width] == query_tokens
            for start in range(len(product_tokens) - width + 1)
        )
    )


def _category_code(value: str, mapping: dict[str, int]) -> int:
    if not value:
        return 0
    return mapping.get(value, 1)


def compute_core_features(
    parsed: ParsedQuery,
    product: ProductFeatureView,
    *,
    lexical_specificity: float,
    brand_codes: dict[str, int],
    color_codes: dict[str, int],
    bm25_score: float,
    bm25_rank_fraction: float,
    dense_score: float,
    dense_rank_fraction: float,
    rrf_score: float,
    rrf_rank_fraction: float,
) -> dict[str, float | int]:
    """Compute one label-free model row using the authoritative registry formulas."""
    numeric = (
        lexical_specificity,
        bm25_score,
        bm25_rank_fraction,
        dense_score,
        dense_rank_fraction,
        rrf_score,
        rrf_rank_fraction,
    )
    if product.locale != "us":
        raise FeatureFormulaError("ltr_core_v1 currently supports only the US locale")
    if any(not math.isfinite(value) for value in numeric):
        raise FeatureFormulaError("feature inputs must be finite")
    if any(
        not 0.0 <= value <= 1.0
        for value in (bm25_rank_fraction, dense_rank_fraction, rrf_rank_fraction)
    ):
        raise FeatureFormulaError("bounded rank fractions must be in [0,1]")

    query_tokens = set(parsed.tokens)
    title_tokens_tuple = tokenize(product.title)
    title_tokens = set(title_tokens_tuple)
    description_tokens_tuple = tokenize(product.description)
    description_tokens = set(description_tokens_tuple)
    bullet_tokens_tuple = tokenize(product.bullets)
    bullet_tokens = set(bullet_tokens_tuple)
    product_tokens_tuple = tokenize(
        " ".join(
            (product.title, product.brand, product.color, product.bullets, product.description)
        )
    )
    product_tokens = set(product_tokens_tuple)
    product_models = {
        token
        for token in product_tokens_tuple
        if _MODEL_RE.fullmatch(token)
        and any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    }
    query_models = set(parsed.model_tokens)
    source_values = (
        product.title,
        product.brand,
        product.color,
        product.bullets,
        product.description,
    )
    completeness = sum(bool(value) for value in source_values) / len(source_values)
    title_log = math.log1p(len(title_tokens_tuple))
    length_ratio = (
        0.0 if title_log == 0.0 else min(math.log1p(len(parsed.tokens)) / title_log, 4.0) / 4.0
    )
    brand_value = parsed.brand.value if parsed.brand is not None else ""
    color_value = parsed.color.value if parsed.color is not None else ""

    row: dict[str, float | int] = {
        "query_char_count": float(len(parsed.normalized_text)),
        "query_token_count": float(len(parsed.tokens)),
        "query_unique_token_ratio": len(query_tokens) / len(parsed.tokens),
        "query_digit_token_count": float(
            sum(any(character.isdigit() for character in token) for token in parsed.tokens)
        ),
        "query_model_token_count": float(len(parsed.model_tokens)),
        "query_brand_detected": int(parsed.brand is not None),
        "query_brand_confidence": parsed.brand.confidence if parsed.brand is not None else 0.0,
        "query_color_detected": int(parsed.color is not None),
        "query_color_confidence": parsed.color.confidence if parsed.color is not None else 0.0,
        "query_lexical_specificity": lexical_specificity,
        "locale_code": 1,
        "title_char_count": float(len(product.title)),
        "title_token_count": float(len(title_tokens_tuple)),
        "description_char_count": float(len(product.description)),
        "description_token_count": float(len(description_tokens_tuple)),
        "bullet_char_count": float(len(product.bullets)),
        "bullet_token_count": float(len(bullet_tokens_tuple)),
        "title_missing": int(product.title_missing),
        "brand_missing": int(product.brand_missing),
        "color_missing": int(product.color_missing),
        "bullets_missing": int(product.bullets_missing),
        "description_missing": int(product.description_missing),
        "product_text_completeness": completeness,
        "brand_code": _category_code(product.normalized_brand, brand_codes),
        "color_code": _category_code(product.normalized_color, color_codes),
        "direct_bm25_score": bm25_score,
        "bm25_rank_fraction": bm25_rank_fraction,
        "direct_dense_cosine": dense_score,
        "dense_rank_fraction": dense_rank_fraction,
        "closed_rrf_score": rrf_score,
        "closed_rrf_rank_fraction": rrf_rank_fraction,
        "title_token_jaccard": _jaccard(query_tokens, title_tokens),
        "title_query_coverage": _coverage(query_tokens, title_tokens),
        "description_query_coverage": _coverage(query_tokens, description_tokens),
        "bullet_query_coverage": _coverage(query_tokens, bullet_tokens),
        "exact_phrase_match": _phrase_match(parsed.tokens, product_tokens_tuple),
        "brand_match": int(bool(brand_value) and brand_value == product.normalized_brand),
        "brand_conflict": int(
            bool(brand_value and product.normalized_brand)
            and brand_value != product.normalized_brand
        ),
        "color_match": int(bool(color_value) and color_value == product.normalized_color),
        "color_conflict": int(
            bool(color_value and product.normalized_color)
            and color_value != product.normalized_color
        ),
        "model_token_match": int(bool(query_models & product_models)),
        "model_token_conflict": int(
            bool(query_models and product_models) and not bool(query_models & product_models)
        ),
        "compatibility_term_match": int(
            bool(parsed.compatibility_tokens)
            and bool(set(parsed.compatibility_tokens) & product_tokens)
        ),
        "query_title_log_length_ratio": length_ratio,
    }
    if tuple(row) != FEATURE_NAMES:
        raise FeatureFormulaError("feature implementation order differs from the registry")
    return row


__all__ = [
    "FeatureFormulaError",
    "ProductFeatureView",
    "compute_core_features",
    "rank_fractions",
]
