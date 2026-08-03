"""Strict, deterministic configuration loading for MarketRank."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver


class ConfigError(ValueError):
    """Base exception for configuration loading and validation failures."""


class ConfigFileError(ConfigError):
    """Raised when a configuration file cannot be read as a strict YAML mapping."""


class ConfigOverrideError(ConfigError):
    """Raised when a dotted override cannot be applied to the configuration tree."""


class ConfigValidationError(ConfigError):
    """Raised when the resolved configuration violates the typed schema."""


class _StrictModel(BaseModel):
    """Shared immutable, unknown-key-rejecting model behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_StrictModel):
    """Stable project identity fields."""

    name: Literal["market-rank"] = "market-rank"
    locale: Literal["us"] = "us"


class PathsConfig(_StrictModel):
    """Repository-relative lifecycle paths."""

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    experiments_dir: Path = Path("experiments")
    reports_dir: Path = Path("reports")


class RuntimeConfig(_StrictModel):
    """Local deterministic and resource controls."""

    seed: int = Field(default=20260801, strict=True, ge=0, le=2**32 - 1)
    max_threads: int = Field(default=4, strict=True, ge=1, le=16)
    rss_limit_mb: int = Field(default=5632, strict=True, ge=512, le=8192)
    offline: bool = Field(default=True, strict=True)


class DatasetConfig(_StrictModel):
    """Deterministic Task-1 cohort, split, and nested-profile controls."""

    query_normalization_version: Literal["nfkc-casefold-ws-v1"] = "nfkc-casefold-ws-v1"
    split_version: Literal["normalized-query-sha256-v1"] = "normalized-query-sha256-v1"
    profile_version: Literal["nested-query-sha256-v1"] = "nested-query-sha256-v1"
    train_basis_points: int = Field(default=8500, strict=True, ge=1, le=9999)
    development_query_groups: int = Field(default=5000, strict=True, ge=1)
    portfolio_query_groups: int = Field(default=20000, strict=True, ge=1)
    product_document_version: Literal["product-document-v1"] = "product-document-v1"
    title_max_chars: int = Field(default=512, strict=True, ge=32, le=4096)
    brand_max_chars: int = Field(default=128, strict=True, ge=16, le=1024)
    color_max_chars: int = Field(default=128, strict=True, ge=16, le=1024)
    bullets_max_chars: int = Field(default=2048, strict=True, ge=128, le=16384)
    description_max_chars: int = Field(default=4096, strict=True, ge=256, le=32768)
    m2_runtime_reserve_mb: int = Field(default=512, strict=True, ge=64, le=4096)

    @model_validator(mode="after")
    def validate_nested_targets(self) -> Self:
        if self.development_query_groups > self.portfolio_query_groups:
            raise ValueError("development_query_groups must not exceed portfolio_query_groups")
        return self


class LoggingConfig(_StrictModel):
    """Safe local logging defaults."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    log_raw_queries: bool = Field(default=False, strict=True)


class AppConfig(_StrictModel):
    """Versioned root configuration model."""

    schema_version: Literal[1] = 1
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    dataset: DatasetConfig = DatasetConfig()
    logging: LoggingConfig = LoggingConfig()


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Validated configuration plus deterministic identity and source lineage."""

    config: AppConfig
    canonical_json: str
    sha256: str
    source_paths: tuple[Path, ...]

    @property
    def short_hash(self) -> str:
        """Return a human-readable, non-authoritative hash prefix."""
        return self.sha256[:12]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and non-string mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    mapping: dict[str, Any] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConstructorError(
                "while constructing a configuration mapping",
                node.start_mark,
                "configuration keys must be strings",
                key_node.start_mark,
            )
        if key in mapping:
            raise ConstructorError(
                "while constructing a configuration mapping",
                node.start_mark,
                f"duplicate key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(f"cannot read configuration file {path}: {exc}") from exc

    try:
        document = yaml.load(raw_text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"invalid configuration YAML in {path}: {exc}") from exc

    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ConfigFileError(f"configuration root must be a mapping: {path}")
    return document


def _deep_merge(base: Mapping[str, Any], later: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, later_value in later.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(later_value, Mapping):
            merged[key] = _deep_merge(base_value, later_value)
        else:
            merged[key] = deepcopy(later_value)
    return merged


def _apply_dotted_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, object],
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if not parts or any(not part for part in parts):
            raise ConfigOverrideError(f"invalid dotted override key: {dotted_key!r}")

        cursor: dict[str, Any] = resolved
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                child: dict[str, Any] = {}
                cursor[part] = child
                cursor = child
            elif isinstance(existing, dict):
                cursor = existing
            else:
                raise ConfigOverrideError(f"cannot apply {dotted_key!r}: {part!r} is not a mapping")
        cursor[parts[-1]] = deepcopy(value)

    return resolved


def _canonicalize(config: AppConfig) -> str:
    semantic_document = config.model_dump(mode="json")
    return json.dumps(
        semantic_document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def load_config(
    paths: Sequence[Path],
    overrides: Mapping[str, object] | None = None,
) -> ResolvedConfig:
    """Load ordered YAML layers, apply overrides, validate, and hash semantics."""
    if not paths:
        raise ConfigFileError("at least one configuration path is required")

    merged: dict[str, Any] = {}
    source_paths: list[Path] = []
    for input_path in paths:
        path = Path(input_path)
        merged = _deep_merge(merged, _load_yaml_mapping(path))
        source_paths.append(path.resolve(strict=False))

    if overrides:
        merged = _apply_dotted_overrides(merged, overrides)

    try:
        config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigValidationError(f"resolved configuration is invalid: {exc}") from exc

    canonical_json = _canonicalize(config)
    config_sha256 = sha256(canonical_json.encode("utf-8")).hexdigest()
    return ResolvedConfig(
        config=config,
        canonical_json=canonical_json,
        sha256=config_sha256,
        source_paths=tuple(source_paths),
    )


__all__ = [
    "AppConfig",
    "ConfigError",
    "ConfigFileError",
    "ConfigOverrideError",
    "ConfigValidationError",
    "DatasetConfig",
    "LoggingConfig",
    "PathsConfig",
    "ProjectConfig",
    "ResolvedConfig",
    "RuntimeConfig",
    "load_config",
]
