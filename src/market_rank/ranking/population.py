"""Leakage-safe exact grouped population construction for Goldfish 011."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Self

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_rank.config import RankerTrainingConfig
from market_rank.features.registry import FEATURE_NAMES, FeatureRegistry

OFFICIAL_GAIN_MAPPING: tuple[tuple[int, float], ...] = (
    (0, 0.0),
    (1, 0.01),
    (2, 0.1),
    (3, 1.0),
)
ProjectSplit = Literal["train", "validation"]
ExclusionReason = Literal["eligible", "too_few_rows", "single_label"]


class TrainingPopulationError(ValueError):
    """Raised when feature rows cannot form an exact leakage-safe training population."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PopulationSplitSummary(_StrictModel):
    project_split: ProjectSplit
    observed_query_groups: int = Field(strict=True, ge=1)
    observed_rows: int = Field(strict=True, ge=1)
    eligible_query_groups: int = Field(strict=True, ge=1)
    eligible_rows: int = Field(strict=True, ge=2)
    excluded_too_few_rows: int = Field(strict=True, ge=0)
    excluded_single_label: int = Field(strict=True, ge=0)
    group_sizes: tuple[int, ...] = Field(min_length=1)
    eligible_query_ids: tuple[int, ...] = Field(min_length=1)
    matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        if any(size < 2 for size in self.group_sizes):
            raise ValueError("eligible group sizes must be at least two")
        if sum(self.group_sizes) != self.eligible_rows:
            raise ValueError("group sizes do not sum to eligible rows")
        if len(self.group_sizes) != self.eligible_query_groups:
            raise ValueError("group count does not match group sizes")
        if len(self.eligible_query_ids) != self.eligible_query_groups:
            raise ValueError("eligible query IDs do not match group count")
        if self.eligible_query_ids != tuple(sorted(set(self.eligible_query_ids))):
            raise ValueError("eligible query IDs must be unique and sorted")
        if (
            self.eligible_query_groups + self.excluded_too_few_rows + self.excluded_single_label
            != self.observed_query_groups
        ):
            raise ValueError("eligible and excluded queries do not partition observed groups")
        return self


class TrainingPopulationManifest(_StrictModel):
    """Exact predicates, group arrays, exclusions, gains, and matrix identities."""

    schema_version: Literal[1] = 1
    population_version: Literal["training-population-v1"] = "training-population-v1"
    population_id: Literal["esci_task1_us_catalog_eligible_closed_train_validation_v1"] = (
        "esci_task1_us_catalog_eligible_closed_train_validation_v1"
    )
    locale: Literal["us"] = "us"
    small_version: Literal[1] = 1
    source_population: Literal["closed_judged"] = "closed_judged"
    included_project_splits: tuple[ProjectSplit, ProjectSplit] = ("train", "validation")
    excluded_project_split: Literal["test"] = "test"
    catalog_eligible_only: Literal[True] = True
    judged_rows_only: Literal[True] = True
    feature_set_id: Literal["ltr_core_v1"] = "ltr_core_v1"
    feature_names: tuple[str, ...] = Field(min_length=1)
    categorical_features: tuple[str, ...]
    label_column: Literal["label_id"] = "label_id"
    gain_column: Literal["gain"] = "gain"
    official_gain_mapping: tuple[tuple[int, float], ...] = OFFICIAL_GAIN_MAPPING
    minimum_group_rows: int = Field(strict=True, ge=2)
    minimum_distinct_labels: int = Field(strict=True, ge=2, le=4)
    train: PopulationSplitSummary
    validation: PopulationSplitSummary
    normalized_group_split_overlap: Literal[0] = 0

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("population feature order differs from ltr_core_v1")
        if not set(self.categorical_features) <= set(self.feature_names):
            raise ValueError("categorical features must belong to ltr_core_v1")
        if self.train.project_split != "train" or self.validation.project_split != "validation":
            raise ValueError("population summaries are assigned to the wrong split")
        return self


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    project_split: ProjectSplit
    frame: pl.DataFrame
    features: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    labels: np.ndarray[tuple[int], np.dtype[np.int32]]
    gains: np.ndarray[tuple[int], np.dtype[np.float32]]
    group_sizes: tuple[int, ...]
    query_ids: tuple[int, ...]
    summary: PopulationSplitSummary


@dataclass(frozen=True, slots=True)
class PreparedPopulation:
    train: PreparedSplit
    validation: PreparedSplit
    audit: pl.DataFrame
    manifest: TrainingPopulationManifest


def _matrix_sha256(
    frame: pl.DataFrame,
    features: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    labels: np.ndarray[tuple[int], np.dtype[np.int32]],
    group_sizes: tuple[int, ...],
) -> str:
    digest = sha256()
    keys = frame.select("query_id", "product_id").to_dicts()
    digest.update(json.dumps(keys, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(features.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))
    digest.update(np.asarray(group_sizes, dtype=np.uint32).tobytes(order="C"))
    return digest.hexdigest()


def _validate_frame(frame: pl.DataFrame, registry: FeatureRegistry) -> None:
    required = {
        "population",
        "profile",
        "query_id",
        "normalized_query_sha256",
        "project_split",
        "locale",
        "product_id",
        "feature_set_id",
        "label_id",
        "gain",
        *FEATURE_NAMES,
    }
    missing = required - set(frame.columns)
    if missing:
        raise TrainingPopulationError(f"closed feature matrix lacks columns: {sorted(missing)}")
    if frame.is_empty():
        raise TrainingPopulationError("closed feature matrix is empty")
    if set(frame["project_split"].unique()) - {"train", "validation"}:
        raise TrainingPopulationError("training input must exclude project test rows")
    feature_type_map: dict[str, type[pl.DataType]] = {
        "float32": pl.Float32,
        "uint8": pl.UInt8,
        "uint32": pl.UInt32,
    }
    expected_dtypes: dict[str, type[pl.DataType]] = {
        "population": pl.String,
        "profile": pl.String,
        "query_id": pl.Int64,
        "normalized_query_sha256": pl.String,
        "project_split": pl.String,
        "locale": pl.String,
        "product_id": pl.String,
        "feature_set_id": pl.String,
        "label_id": pl.UInt8,
        "gain": pl.Float32,
    }
    expected_dtypes.update(
        {feature.name: feature_type_map[feature.dtype] for feature in registry.features}
    )
    incompatible = {
        name: (str(frame.schema[name]), str(expected))
        for name, expected in expected_dtypes.items()
        if frame.schema[name] != expected
    }
    if incompatible:
        raise TrainingPopulationError(
            f"closed feature matrix has incompatible dtypes: {incompatible}"
        )
    if frame["population"].unique().to_list() != ["closed_judged"]:
        raise TrainingPopulationError("training requires the closed judged population")
    if frame["feature_set_id"].unique().to_list() != [registry.feature_set_id]:
        raise TrainingPopulationError("training feature-set identity is incompatible")
    if frame["locale"].unique().to_list() != ["us"]:
        raise TrainingPopulationError("training currently supports only the US locale")
    if frame.select(pl.struct("query_id", "product_id").n_unique()).item() != frame.height:
        raise TrainingPopulationError("training query-product keys are not unique")
    if frame.select((*FEATURE_NAMES, "label_id", "gain")).null_count().sum_horizontal().item() != 0:
        raise TrainingPopulationError("training features, labels, and gains cannot be null")
    numeric = frame.select(FEATURE_NAMES).to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise TrainingPopulationError("training feature values must be finite")
    expected_gains = dict(OFFICIAL_GAIN_MAPPING)
    for label, gain in frame.select("label_id", "gain").iter_rows():
        label_id = int(label)
        if label_id not in expected_gains or not math.isclose(
            float(gain), expected_gains[label_id], abs_tol=1e-6
        ):
            raise TrainingPopulationError("label IDs and persisted gains violate official mapping")

    split_counts = frame.group_by("query_id").agg(
        pl.col("project_split").n_unique().alias("splits"),
        pl.col("normalized_query_sha256").n_unique().alias("groups"),
    )
    if split_counts.filter((pl.col("splits") != 1) | (pl.col("groups") != 1)).height:
        raise TrainingPopulationError("query rows disagree on split or normalized-query group")
    overlap = (
        frame.select("normalized_query_sha256", "project_split")
        .unique()
        .group_by("normalized_query_sha256")
        .agg(pl.col("project_split").n_unique().alias("splits"))
        .filter(pl.col("splits") > 1)
    )
    if overlap.height:
        raise TrainingPopulationError("normalized-query groups cross train and validation")


def _prepare_split(
    frame: pl.DataFrame,
    project_split: ProjectSplit,
    config: RankerTrainingConfig,
) -> tuple[PreparedSplit, list[dict[str, object]]]:
    selected = frame.filter(pl.col("project_split") == project_split).sort("query_id", "product_id")
    if selected.is_empty():
        raise TrainingPopulationError(f"{project_split} split has no observed rows")
    audit_rows: list[dict[str, object]] = []
    eligible_query_ids: list[int] = []
    group_sizes: list[int] = []
    excluded_too_few = 0
    excluded_single_label = 0
    for group in selected.partition_by("query_id", maintain_order=True):
        query_id = int(group["query_id"][0])
        distinct_labels = group["label_id"].n_unique()
        reason: ExclusionReason = "eligible"
        if group.height < config.min_group_rows:
            reason = "too_few_rows"
            excluded_too_few += 1
        elif distinct_labels < config.min_distinct_labels:
            reason = "single_label"
            excluded_single_label += 1
        else:
            eligible_query_ids.append(query_id)
            group_sizes.append(group.height)
        audit_rows.append(
            {
                "project_split": project_split,
                "query_id": query_id,
                "normalized_query_sha256": group["normalized_query_sha256"][0],
                "observed_rows": group.height,
                "distinct_labels": distinct_labels,
                "eligible_for_fit": reason == "eligible",
                "exclusion_reason": reason,
            }
        )
    eligible = selected.filter(pl.col("query_id").is_in(eligible_query_ids))
    maximum = config.max_train_rows if project_split == "train" else config.max_validation_rows
    if eligible.is_empty():
        raise TrainingPopulationError(f"{project_split} has no eligible multi-label groups")
    if eligible.height > maximum:
        raise TrainingPopulationError(
            f"{project_split} eligible rows {eligible.height} exceed configured maximum {maximum}"
        )
    features = np.ascontiguousarray(eligible.select(FEATURE_NAMES).to_numpy(), dtype=np.float32)
    labels = np.ascontiguousarray(eligible["label_id"].to_numpy(), dtype=np.int32)
    gains = np.ascontiguousarray(eligible["gain"].to_numpy(), dtype=np.float32)
    sizes = tuple(group_sizes)
    query_ids = tuple(eligible_query_ids)
    summary = PopulationSplitSummary(
        project_split=project_split,
        observed_query_groups=selected["query_id"].n_unique(),
        observed_rows=selected.height,
        eligible_query_groups=len(query_ids),
        eligible_rows=eligible.height,
        excluded_too_few_rows=excluded_too_few,
        excluded_single_label=excluded_single_label,
        group_sizes=sizes,
        eligible_query_ids=query_ids,
        matrix_sha256=_matrix_sha256(eligible, features, labels, sizes),
    )
    return (
        PreparedSplit(
            project_split=project_split,
            frame=eligible,
            features=features,
            labels=labels,
            gains=gains,
            group_sizes=sizes,
            query_ids=query_ids,
            summary=summary,
        ),
        audit_rows,
    )


def prepare_training_population(
    frame: pl.DataFrame,
    registry: FeatureRegistry,
    config: RankerTrainingConfig,
) -> PreparedPopulation:
    """Validate rows and build identical pointwise/LambdaMART matrices and group arrays."""
    if tuple(feature.name for feature in registry.features) != FEATURE_NAMES:
        raise TrainingPopulationError("loaded feature registry order is incompatible")
    _validate_frame(frame, registry)
    train, train_audit = _prepare_split(frame, "train", config)
    validation, validation_audit = _prepare_split(frame, "validation", config)
    categorical = tuple(
        feature for feature in ("brand_code", "color_code") if feature in FEATURE_NAMES
    )
    manifest = TrainingPopulationManifest(
        population_version=config.population_version,
        feature_names=FEATURE_NAMES,
        categorical_features=categorical,
        minimum_group_rows=config.min_group_rows,
        minimum_distinct_labels=config.min_distinct_labels,
        train=train.summary,
        validation=validation.summary,
    )
    audit = pl.DataFrame(train_audit + validation_audit).sort("project_split", "query_id")
    return PreparedPopulation(train=train, validation=validation, audit=audit, manifest=manifest)


def stable_rank_predictions(
    product_ids: tuple[str, ...], predictions: np.ndarray[tuple[int], np.dtype[np.float64]]
) -> tuple[int, ...]:
    """Return one-based ranks with descending-score/product-ID deterministic ordering."""
    if len(product_ids) != len(predictions) or len(product_ids) != len(set(product_ids)):
        raise TrainingPopulationError("prediction keys must be aligned and unique")
    if any(not product_id for product_id in product_ids) or not np.isfinite(predictions).all():
        raise TrainingPopulationError("prediction keys and scores must be valid and finite")
    ordered = sorted(
        range(len(product_ids)), key=lambda index: (-predictions[index], product_ids[index])
    )
    ranks = [0] * len(product_ids)
    for rank, index in enumerate(ordered, start=1):
        ranks[index] = rank
    return tuple(ranks)


def stable_grouped_prediction_ranks(
    query_ids: tuple[int, ...],
    product_ids: tuple[str, ...],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> tuple[int, ...]:
    """Rank predictions independently inside each contiguous query group."""
    if len(query_ids) != len(product_ids) or len(query_ids) != len(predictions):
        raise TrainingPopulationError("grouped prediction arrays must be aligned")
    ranks = [0] * len(query_ids)
    positions_by_query: dict[int, list[int]] = {}
    for position, query_id in enumerate(query_ids):
        positions_by_query.setdefault(query_id, []).append(position)
    for positions in positions_by_query.values():
        group_products = tuple(product_ids[position] for position in positions)
        group_predictions = np.asarray(
            [predictions[position] for position in positions], dtype=np.float64
        )
        group_ranks = stable_rank_predictions(group_products, group_predictions)
        for position, rank in zip(positions, group_ranks, strict=True):
            ranks[position] = rank
    return tuple(ranks)


__all__ = [
    "OFFICIAL_GAIN_MAPPING",
    "PopulationSplitSummary",
    "PreparedPopulation",
    "PreparedSplit",
    "TrainingPopulationError",
    "TrainingPopulationManifest",
    "prepare_training_population",
    "stable_grouped_prediction_ranks",
    "stable_rank_predictions",
]
