"""Registry, ranking, and shared feature formula tests for Goldfish 010."""

from __future__ import annotations

import math

import pytest

from market_rank.config import QueryUnderstandingConfig
from market_rank.features.core import ProductFeatureView, compute_core_features, rank_fractions
from market_rank.features.registry import FEATURE_NAMES, ltr_core_v1_registry
from market_rank.query.parser import ParsedQuery, QueryParser, build_parser_state


def _parsed() -> ParsedQuery:
    state = build_parser_state(("Acme",), ("Red",), QueryUnderstandingConfig())
    return QueryParser(state, QueryUnderstandingConfig()).parse("acme red x100 case")


def _product() -> ProductFeatureView:
    return ProductFeatureView(
        locale="us",
        title="Acme red X100 protective case",
        brand="Acme",
        color="Red",
        bullets="Compatible protective shell",
        description="Case for model X100",
        normalized_brand="acme",
        normalized_color="red",
        title_missing=False,
        brand_missing=False,
        color_missing=False,
        bullets_missing=False,
        description_missing=False,
    )


def test_ltr_core_registry_is_ordered_online_available_and_leakage_reviewed() -> None:
    registry = ltr_core_v1_registry()
    prohibited = {"product_id", "label_id", "gain", "source_count", "absolute_candidate_rank"}

    assert tuple(feature.name for feature in registry.features) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert not set(FEATURE_NAMES) & prohibited
    assert all(feature.online_available and not feature.uses_label for feature in registry.features)
    assert all(not feature.uses_product_id for feature in registry.features)
    assert all(not feature.uses_source_topk_provenance for feature in registry.features)


def test_rank_fractions_are_stable_bounded_and_handle_singletons() -> None:
    ranks, fractions = rank_fractions({"p2": 1.0, "p1": 1.0, "p3": -1.0})

    assert ranks == {"p1": 1, "p2": 2, "p3": 3}
    assert fractions == {"p1": 0.0, "p2": 0.5, "p3": 1.0}
    assert rank_fractions({"only": 4.0}) == ({"only": 1}, {"only": 0.0})


def test_shared_formula_emits_exact_registry_order_and_expected_interactions() -> None:
    parsed = _parsed()
    row = compute_core_features(
        parsed,
        _product(),
        lexical_specificity=2.5,
        brand_codes={"acme": 2},
        color_codes={"red": 2},
        bm25_score=3.0,
        bm25_rank_fraction=0.25,
        dense_score=0.75,
        dense_rank_fraction=0.0,
        rrf_score=0.03,
        rrf_rank_fraction=0.5,
    )

    assert tuple(row) == FEATURE_NAMES
    assert row["brand_code"] == 2
    assert row["color_code"] == 2
    assert row["brand_match"] == 1
    assert row["color_match"] == 1
    assert row["model_token_match"] == 1
    assert row["exact_phrase_match"] == 0
    assert row["product_text_completeness"] == 1.0
    assert all(math.isfinite(float(value)) for value in row.values())


def test_unknown_and_missing_categories_have_reserved_distinct_codes() -> None:
    parsed = _parsed()
    unknown = _product()
    row = compute_core_features(
        parsed,
        unknown,
        lexical_specificity=0.0,
        brand_codes={},
        color_codes={},
        bm25_score=0.0,
        bm25_rank_fraction=0.0,
        dense_score=0.0,
        dense_rank_fraction=0.0,
        rrf_score=0.0,
        rrf_rank_fraction=0.0,
    )

    assert row["brand_code"] == 1
    assert row["color_code"] == 1


def test_nonfinite_or_out_of_range_retrieval_inputs_are_rejected() -> None:
    parsed = _parsed()
    with pytest.raises(ValueError):
        compute_core_features(
            parsed,
            _product(),
            lexical_specificity=0.0,
            brand_codes={},
            color_codes={},
            bm25_score=float("nan"),
            bm25_rank_fraction=2.0,
            dense_score=0.0,
            dense_rank_fraction=0.0,
            rrf_score=0.0,
            rrf_rank_fraction=0.0,
        )
