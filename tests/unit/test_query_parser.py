"""Deterministic, bounded query-understanding tests for Goldfish 010."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_rank.config import QueryUnderstandingConfig
from market_rank.query.parser import (
    QueryParser,
    QueryParserError,
    QueryParserState,
    build_parser_state,
)


def _state() -> QueryParserState:
    return build_parser_state(
        ("Acme", "Acme Pro", "Other"),
        ("Blue", "Gray", "Red"),
        QueryUnderstandingConfig(),
    )


def test_parser_normalizes_extracts_and_hashes_deterministically() -> None:
    parser = QueryParser(_state(), QueryUnderstandingConfig())
    first = parser.parse("  ACME\u00a0PRO wireles XPS-13 16 GB compatible with RED ")
    second = parser.parse("  ACME\u00a0PRO wireles XPS-13 16 GB compatible with RED ")

    assert first == second
    assert first.raw_text.startswith("  ACME")
    assert first.normalized_text == "acme pro wireles xps-13 16 gb compatible with red"
    assert "wireless" in first.tokens
    assert first.brand is not None and first.brand.value == "acme pro"
    assert first.color is not None and first.color.value == "red"
    assert first.model_tokens == ("xps-13",)
    assert first.numbers == ("13", "16")
    assert first.units == ("gb",)
    assert first.measurements == ("16 gb",)
    assert "compatible" in first.compatibility_tokens
    assert first.compatibility_phrases == ("compatible with",)
    assert first.warnings == ("conservative_spelling_alias_applied",)
    assert len(first.query_sha256) == 64


def test_longest_brand_boundary_and_color_alias_are_non_filtering_signals() -> None:
    parsed = QueryParser(_state(), QueryUnderstandingConfig()).parse("acme pro charcoal case")

    assert parsed.brand is not None and parsed.brand.value == "acme pro"
    assert parsed.brand.confidence == 1.0
    assert parsed.color is not None and parsed.color.value == "gray"
    assert parsed.color.source == "alias"
    assert parsed.color.confidence == 0.9


@pytest.mark.parametrize("text", ["", " \t\n", "x" * 513])
def test_empty_or_over_limit_queries_are_rejected(text: str) -> None:
    with pytest.raises(QueryParserError):
        QueryParser(_state(), QueryUnderstandingConfig()).parse(text)


def test_token_limit_is_enforced_without_silent_truncation() -> None:
    config = QueryUnderstandingConfig(max_query_tokens=2)
    with pytest.raises(QueryParserError, match="token limit"):
        QueryParser(_state(), config).parse("one two three")


def test_parser_state_hash_detects_dictionary_tampering() -> None:
    document = _state().model_dump(mode="json")
    document["brands"] = ["changed"]

    with pytest.raises(ValidationError, match="state hash"):
        QueryParserState.model_validate(document)
