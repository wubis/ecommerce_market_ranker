"""Bounded deterministic query parsing backed only by persisted catalog dictionaries."""

from __future__ import annotations

import json
import re
import unicodedata
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from market_rank.config import QueryUnderstandingConfig
from market_rank.retrieval.sparse import tokenize

Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]

_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?)(?!\w)")
_MODEL_RE = re.compile(r"(?i)[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_UNIT_ALIASES = {
    "centimeter": "cm",
    "centimeters": "cm",
    "cm": "cm",
    "foot": "ft",
    "feet": "ft",
    "ft": "ft",
    "gigabyte": "gb",
    "gigabytes": "gb",
    "gb": "gb",
    "inch": "in",
    "inches": "in",
    "in": "in",
    "kilogram": "kg",
    "kilograms": "kg",
    "kg": "kg",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "l": "l",
    "megabyte": "mb",
    "megabytes": "mb",
    "mb": "mb",
    "millimeter": "mm",
    "millimeters": "mm",
    "mm": "mm",
    "ounce": "oz",
    "ounces": "oz",
    "oz": "oz",
    "pound": "lb",
    "pounds": "lb",
    "lb": "lb",
    "terabyte": "tb",
    "terabytes": "tb",
    "tb": "tb",
}
_COMPATIBILITY_TOKENS = frozenset(
    {"compatible", "compatibility", "fits", "fit", "replacement", "works", "with", "for"}
)
_NEGATIONS = frozenset({"no", "not", "without", "never"})
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "from",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
    }
)
_SPELLING_ALIASES = {
    "bluetooths": "bluetooth",
    "headfone": "headphone",
    "headfones": "headphones",
    "wireles": "wireless",
}
_COLOR_ALIASES = {
    "aqua": "blue",
    "charcoal": "gray",
    "grey": "gray",
    "navy blue": "navy",
}
_COMPATIBILITY_PHRASES = (
    "compatible with",
    "fits",
    "replacement for",
    "works with",
)


class QueryParserError(ValueError):
    """Raised when query text or persisted parser state violates the contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DetectedEntity(_StrictModel):
    """One non-filtering parser signal with explicit confidence and provenance."""

    value: str = Field(strict=True, min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source: Literal["catalog", "alias", "rule"]


class QueryParserState(_StrictModel):
    """Persisted label-free parser dictionaries and their deterministic identity."""

    schema_version: Literal[1] = 1
    parser_version: Literal["query-parser-v1"] = "query-parser-v1"
    brands: tuple[str, ...]
    colors: tuple[str, ...]
    color_aliases: tuple[tuple[str, str], ...]
    spelling_aliases: tuple[tuple[str, str], ...]
    state_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.brands != tuple(sorted(set(self.brands))):
            raise ValueError("brands must be unique and sorted")
        if self.colors != tuple(sorted(set(self.colors))):
            raise ValueError("colors must be unique and sorted")
        for name, entries in (
            ("color aliases", self.color_aliases),
            ("spelling aliases", self.spelling_aliases),
        ):
            if entries != tuple(sorted(entries)) or len({key for key, _ in entries}) != len(
                entries
            ):
                raise ValueError(f"{name} must be key-unique and sorted")
        expected = _state_sha256(
            self.parser_version,
            self.brands,
            self.colors,
            self.color_aliases,
            self.spelling_aliases,
        )
        if self.state_sha256 != expected:
            raise ValueError("parser state hash does not match its dictionaries")
        return self


class ParsedQuery(_StrictModel):
    """Complete deterministic query-understanding payload shared offline and online."""

    raw_text: str
    normalized_text: str = Field(strict=True, min_length=1)
    tokens: tuple[str, ...] = Field(min_length=1)
    reduced_tokens: tuple[str, ...] = Field(min_length=1)
    numbers: tuple[str, ...]
    units: tuple[str, ...]
    measurements: tuple[str, ...]
    model_tokens: tuple[str, ...]
    compatibility_tokens: tuple[str, ...]
    compatibility_phrases: tuple[str, ...]
    brand: DetectedEntity | None
    color: DetectedEntity | None
    parser_version: Literal["query-parser-v1"] = "query-parser-v1"
    parser_state_sha256: Sha256Digest
    query_sha256: Sha256Digest
    warnings: tuple[str, ...]


def normalize_query_text(text: str) -> str:
    """Apply the versioned NFKC/casefold/whitespace query view."""
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _canonical_hash(document: object) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _state_sha256(
    parser_version: str,
    brands: tuple[str, ...],
    colors: tuple[str, ...],
    color_aliases: tuple[tuple[str, str], ...],
    spelling_aliases: tuple[tuple[str, str], ...],
) -> str:
    return _canonical_hash(
        {
            "parser_version": parser_version,
            "brands": brands,
            "colors": colors,
            "color_aliases": color_aliases,
            "spelling_aliases": spelling_aliases,
        }
    )


def _normalized_values(values: tuple[str, ...], *, min_chars: int = 1) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                normalized
                for value in values
                if (normalized := normalize_query_text(value)) and len(normalized) >= min_chars
            }
        )
    )


def build_parser_state(
    brands: tuple[str, ...],
    colors: tuple[str, ...],
    config: QueryUnderstandingConfig,
) -> QueryParserState:
    """Fit label-free brand/color dictionaries from official catalog attributes."""
    normalized_brands = _normalized_values(brands, min_chars=config.brand_min_chars)
    normalized_colors = _normalized_values(colors)
    color_aliases = tuple(
        sorted(
            (alias, canonical)
            for alias, canonical in _COLOR_ALIASES.items()
            if canonical in normalized_colors
        )
    )
    spelling_aliases = tuple(sorted(_SPELLING_ALIASES.items()))
    return QueryParserState(
        parser_version=config.parser_version,
        brands=normalized_brands,
        colors=normalized_colors,
        color_aliases=color_aliases,
        spelling_aliases=spelling_aliases,
        state_sha256=_state_sha256(
            config.parser_version,
            normalized_brands,
            normalized_colors,
            color_aliases,
            spelling_aliases,
        ),
    )


def _longest_boundary_match(text_tokens: tuple[str, ...], values: tuple[str, ...]) -> str | None:
    token_count = len(text_tokens)
    candidates = sorted(values, key=lambda value: (-len(tokenize(value)), -len(value), value))
    for value in candidates:
        value_tokens = tokenize(value)
        width = len(value_tokens)
        if width and any(
            text_tokens[start : start + width] == value_tokens
            for start in range(token_count - width + 1)
        ):
            return value
    return None


class QueryParser:
    """Pure parser whose only fitted input is an immutable parser state."""

    def __init__(self, state: QueryParserState, config: QueryUnderstandingConfig) -> None:
        if state.parser_version != config.parser_version:
            raise QueryParserError("parser state and configured parser versions differ")
        self.state = state
        self.config = config

    def parse(self, raw_text: str) -> ParsedQuery:
        if not isinstance(raw_text, str):
            raise QueryParserError("query must be a string")
        try:
            byte_count = len(raw_text.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise QueryParserError("query is not valid UTF-8 text") from exc
        if len(raw_text) > self.config.max_query_chars or byte_count > self.config.max_query_bytes:
            raise QueryParserError("query exceeds configured character or UTF-8 byte limit")
        normalized = normalize_query_text(raw_text)
        if not normalized:
            raise QueryParserError("query is empty after normalization")

        initial_tokens = tokenize(normalized)
        if not initial_tokens:
            raise QueryParserError("query has no tokens after normalization")
        if len(initial_tokens) > self.config.max_query_tokens:
            raise QueryParserError("query exceeds configured token limit")
        spelling_aliases = dict(self.state.spelling_aliases)
        corrected = tuple(spelling_aliases.get(token, token) for token in initial_tokens)
        warnings = ("conservative_spelling_alias_applied",) if corrected != initial_tokens else ()
        corrected_text = " ".join(corrected)

        brand_value = _longest_boundary_match(corrected, self.state.brands)
        color_value = _longest_boundary_match(corrected, self.state.colors)
        color_source: Literal["catalog", "alias"] = "catalog"
        if color_value is None:
            alias = _longest_boundary_match(
                corrected, tuple(key for key, _ in self.state.color_aliases)
            )
            if alias is not None:
                color_value = dict(self.state.color_aliases)[alias]
                color_source = "alias"

        models = tuple(
            sorted(
                {
                    token
                    for token in corrected
                    if _MODEL_RE.fullmatch(token)
                    and any(character.isalpha() for character in token)
                    and any(character.isdigit() for character in token)
                }
            )
        )
        units = tuple(
            sorted({_UNIT_ALIASES[token] for token in corrected if token in _UNIT_ALIASES})
        )
        measurements = tuple(
            sorted(
                {
                    f"{token} {_UNIT_ALIASES[corrected[index + 1]]}"
                    for index, token in enumerate(corrected[:-1])
                    if _NUMBER_RE.fullmatch(token) and corrected[index + 1] in _UNIT_ALIASES
                }
            )
        )
        compatibility = tuple(sorted(set(corrected) & _COMPATIBILITY_TOKENS))
        compatibility_phrases = tuple(
            phrase for phrase in _COMPATIBILITY_PHRASES if phrase in corrected_text
        )
        preserved = _NEGATIONS | _COMPATIBILITY_TOKENS | frozenset(_UNIT_ALIASES)
        reduced = tuple(
            token
            for token in corrected
            if token not in _STOPWORDS or token in preserved or token in models
        )
        if not reduced:
            reduced = corrected
        query_hash = _canonical_hash(
            {
                "normalized_text": normalized,
                "parser_state_sha256": self.state.state_sha256,
                "parser_version": self.state.parser_version,
            }
        )
        return ParsedQuery(
            raw_text=raw_text,
            normalized_text=normalized,
            tokens=corrected,
            reduced_tokens=reduced,
            numbers=tuple(_NUMBER_RE.findall(corrected_text)),
            units=units,
            measurements=measurements,
            model_tokens=models,
            compatibility_tokens=compatibility,
            compatibility_phrases=compatibility_phrases,
            brand=(
                DetectedEntity(value=brand_value, confidence=1.0, source="catalog")
                if brand_value is not None
                else None
            ),
            color=(
                DetectedEntity(
                    value=color_value,
                    confidence=1.0 if color_source == "catalog" else 0.9,
                    source=color_source,
                )
                if color_value is not None
                else None
            ),
            parser_state_sha256=self.state.state_sha256,
            query_sha256=query_hash,
            warnings=warnings,
        )


__all__ = [
    "DetectedEntity",
    "ParsedQuery",
    "QueryParser",
    "QueryParserError",
    "QueryParserState",
    "build_parser_state",
    "normalize_query_text",
]
