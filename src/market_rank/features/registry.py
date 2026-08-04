"""Immutable leakage-reviewed registry for the primary LTR feature contract."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

FeatureDtype = Literal["float32", "uint8", "uint32"]
FeatureClass = Literal["query", "product", "retrieval", "interaction"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureSpec(_StrictModel):
    """One stable model-input formula and its leakage/storage policy."""

    name: str = Field(strict=True, pattern=r"^[a-z][a-z0-9_]*$")
    dtype: FeatureDtype
    feature_class: FeatureClass
    provenance: str = Field(strict=True, min_length=1)
    default: float | int
    missing_behavior: str = Field(strict=True, min_length=1)
    formula: str = Field(strict=True, min_length=1)
    online_available: Literal[True] = True
    uses_label: Literal[False] = False
    uses_product_id: Literal[False] = False
    uses_source_topk_provenance: Literal[False] = False


class FeatureRegistry(_StrictModel):
    """Ordered primary feature schema consumed by training and serving."""

    schema_version: Literal[1] = 1
    registry_version: Literal["feature-registry-v1"] = "feature-registry-v1"
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    features: tuple[FeatureSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        names = tuple(feature.name for feature in self.features)
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        return self


def _spec(
    name: str,
    dtype: FeatureDtype,
    feature_class: FeatureClass,
    formula: str,
    *,
    provenance: str = "derived:ranking-features-v1",
    default: float | int = 0.0,
    missing: str = "zero means absent or not applicable",
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        dtype=dtype,
        feature_class=feature_class,
        provenance=provenance,
        default=default,
        missing_behavior=missing,
        formula=formula,
    )


def ltr_core_v1_registry() -> FeatureRegistry:
    """Return the authoritative ordered, label-free `ltr_core_v1` registry."""
    features = (
        _spec("query_char_count", "float32", "query", "Unicode character count"),
        _spec("query_token_count", "float32", "query", "query-parser token count"),
        _spec("query_unique_token_ratio", "float32", "query", "unique tokens / token count"),
        _spec("query_digit_token_count", "float32", "query", "tokens containing a digit"),
        _spec("query_model_token_count", "float32", "query", "conservative model-token count"),
        _spec("query_brand_detected", "uint8", "query", "1 when catalog brand detected", default=0),
        _spec("query_brand_confidence", "float32", "query", "parser brand confidence"),
        _spec(
            "query_color_detected",
            "uint8",
            "query",
            "1 when catalog/alias color detected",
            default=0,
        ),
        _spec("query_color_confidence", "float32", "query", "parser color confidence"),
        _spec(
            "query_lexical_specificity",
            "float32",
            "query",
            "mean catalog BM25 IDF of unique query tokens",
        ),
        _spec("locale_code", "uint8", "query", "1 for the supported US locale", default=1),
        _spec(
            "title_char_count",
            "float32",
            "product",
            "clean title character count",
            provenance="esci",
        ),
        _spec(
            "title_token_count", "float32", "product", "clean title token count", provenance="esci"
        ),
        _spec(
            "description_char_count",
            "float32",
            "product",
            "clean description character count",
            provenance="esci",
        ),
        _spec(
            "description_token_count",
            "float32",
            "product",
            "clean description token count",
            provenance="esci",
        ),
        _spec(
            "bullet_char_count",
            "float32",
            "product",
            "clean bullet character count",
            provenance="esci",
        ),
        _spec(
            "bullet_token_count",
            "float32",
            "product",
            "clean bullet token count",
            provenance="esci",
        ),
        _spec(
            "title_missing",
            "uint8",
            "product",
            "official title missing flag",
            provenance="esci",
            default=0,
        ),
        _spec(
            "brand_missing",
            "uint8",
            "product",
            "official brand missing flag",
            provenance="esci",
            default=0,
        ),
        _spec(
            "color_missing",
            "uint8",
            "product",
            "official color missing flag",
            provenance="esci",
            default=0,
        ),
        _spec(
            "bullets_missing",
            "uint8",
            "product",
            "official bullets missing flag",
            provenance="esci",
            default=0,
        ),
        _spec(
            "description_missing",
            "uint8",
            "product",
            "official description missing flag",
            provenance="esci",
            default=0,
        ),
        _spec(
            "product_text_completeness",
            "float32",
            "product",
            "nonempty source fields / 5",
            provenance="esci",
        ),
        _spec(
            "brand_code",
            "uint32",
            "product",
            "train-fitted brand code; 0 missing, 1 unknown",
            default=1,
            missing="0 missing; 1 unknown",
        ),
        _spec(
            "color_code",
            "uint32",
            "product",
            "train-fitted color code; 0 missing, 1 unknown",
            default=1,
            missing="0 missing; 1 unknown",
        ),
        _spec(
            "direct_bm25_score", "float32", "retrieval", "explicit-pair BM25 score for every row"
        ),
        _spec(
            "bm25_rank_fraction", "float32", "retrieval", "(rank-1)/max(n-1,1) within current set"
        ),
        _spec(
            "direct_dense_cosine", "float32", "retrieval", "explicit-pair normalized dot product"
        ),
        _spec(
            "dense_rank_fraction", "float32", "retrieval", "(rank-1)/max(n-1,1) within current set"
        ),
        _spec("closed_rrf_score", "float32", "retrieval", "RRF over direct BM25/dense ranks"),
        _spec(
            "closed_rrf_rank_fraction",
            "float32",
            "retrieval",
            "bounded rank fraction of direct-score RRF",
        ),
        _spec(
            "title_token_jaccard",
            "float32",
            "interaction",
            "query/title token intersection / union",
        ),
        _spec(
            "title_query_coverage",
            "float32",
            "interaction",
            "query tokens present in title / query tokens",
        ),
        _spec(
            "description_query_coverage",
            "float32",
            "interaction",
            "query tokens present in description / query tokens",
        ),
        _spec(
            "bullet_query_coverage",
            "float32",
            "interaction",
            "query tokens present in bullets / query tokens",
        ),
        _spec(
            "exact_phrase_match",
            "uint8",
            "interaction",
            "boundary-aware normalized query token sequence in product text",
            default=0,
        ),
        _spec(
            "brand_match",
            "uint8",
            "interaction",
            "parsed brand equals official normalized brand",
            default=0,
        ),
        _spec(
            "brand_conflict",
            "uint8",
            "interaction",
            "parsed and official brands are both present and differ",
            default=0,
        ),
        _spec(
            "color_match",
            "uint8",
            "interaction",
            "parsed color equals official normalized color",
            default=0,
        ),
        _spec(
            "color_conflict",
            "uint8",
            "interaction",
            "parsed and official colors are both present and differ",
            default=0,
        ),
        _spec(
            "model_token_match",
            "uint8",
            "interaction",
            "query/product model token sets intersect",
            default=0,
        ),
        _spec(
            "model_token_conflict",
            "uint8",
            "interaction",
            "nonempty query/product model token sets do not intersect",
            default=0,
        ),
        _spec(
            "compatibility_term_match",
            "uint8",
            "interaction",
            "query compatibility signal appears in product text",
            default=0,
        ),
        _spec(
            "query_title_log_length_ratio",
            "float32",
            "interaction",
            "min(log1p(query tokens)/log1p(title tokens),4)/4",
        ),
    )
    return FeatureRegistry(features=features)


FEATURE_NAMES = tuple(feature.name for feature in ltr_core_v1_registry().features)


__all__ = [
    "FEATURE_NAMES",
    "FeatureRegistry",
    "FeatureSpec",
    "ltr_core_v1_registry",
]
