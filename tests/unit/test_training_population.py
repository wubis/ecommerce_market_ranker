"""Exact grouped training-population and prediction-order tests for Goldfish 011."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from market_rank.config import RankerTrainingConfig
from market_rank.features.registry import FEATURE_NAMES, ltr_core_v1_registry
from market_rank.ranking.population import (
    TrainingPopulationError,
    prepare_training_population,
    stable_grouped_prediction_ranks,
    stable_rank_predictions,
)


def _row(
    query_id: int,
    product_id: str,
    split: str,
    label_id: int,
) -> dict[str, object]:
    gains = {0: 0.0, 1: 0.01, 2: 0.1, 3: 1.0}
    row: dict[str, object] = {
        "population": "closed_judged",
        "profile": "portfolio",
        "query_id": query_id,
        "normalized_query_sha256": f"{query_id:064x}",
        "project_split": split,
        "locale": "us",
        "product_id": product_id,
        "feature_set_id": "ltr_core_v1",
        "label_id": label_id,
        "gain": gains[label_id],
    }
    row.update({name: float(index + query_id) for index, name in enumerate(FEATURE_NAMES)})
    row["brand_code"] = 2
    row["color_code"] = 3
    row["locale_code"] = 1
    return row


def _frame(*, reverse: bool = False) -> pl.DataFrame:
    rows = [
        _row(1, "p2", "train", 3),
        _row(1, "p1", "train", 0),
        _row(2, "p3", "train", 1),
        _row(3, "p4", "train", 2),
        _row(3, "p5", "train", 2),
        _row(4, "p7", "validation", 2),
        _row(4, "p6", "validation", 1),
    ]
    return _typed_rows(list(reversed(rows)) if reverse else rows)


def test_population_keeps_complete_multilabel_groups_and_audits_exclusions() -> None:
    population = prepare_training_population(
        _frame(), ltr_core_v1_registry(), RankerTrainingConfig()
    )

    assert population.train.query_ids == (1,)
    assert population.train.group_sizes == (2,)
    assert population.train.frame["product_id"].to_list() == ["p1", "p2"]
    assert population.train.summary.observed_query_groups == 3
    assert population.train.summary.excluded_too_few_rows == 1
    assert population.train.summary.excluded_single_label == 1
    assert population.validation.query_ids == (4,)
    assert population.validation.group_sizes == (2,)
    assert population.train.features.dtype == np.float32
    assert population.train.labels.dtype == np.int32
    assert population.manifest.official_gain_mapping == (
        (0, 0.0),
        (1, 0.01),
        (2, 0.1),
        (3, 1.0),
    )
    assert set(population.audit["exclusion_reason"]) == {
        "eligible",
        "too_few_rows",
        "single_label",
    }


def test_raw_order_does_not_change_population_matrices_or_hashes() -> None:
    first = prepare_training_population(_frame(), ltr_core_v1_registry(), RankerTrainingConfig())
    reordered = prepare_training_population(
        _frame(reverse=True), ltr_core_v1_registry(), RankerTrainingConfig()
    )

    assert first.manifest == reordered.manifest
    assert first.train.frame.equals(reordered.train.frame)
    assert np.array_equal(first.train.features, reordered.train.features)
    assert np.array_equal(first.validation.features, reordered.validation.features)


def test_official_gain_mismatch_and_test_rows_are_rejected() -> None:
    bad_gain = _frame().with_columns(
        pl.when((pl.col("query_id") == 1) & (pl.col("product_id") == "p1"))
        .then(pl.lit(0.5))
        .otherwise(pl.col("gain"))
        .alias("gain")
    )
    with pytest.raises(TrainingPopulationError, match="official mapping"):
        prepare_training_population(bad_gain, ltr_core_v1_registry(), RankerTrainingConfig())

    test_row = pl.concat((_frame(), _typed_rows([_row(9, "test", "test", 3)])))
    with pytest.raises(TrainingPopulationError, match="exclude project test"):
        prepare_training_population(test_row, ltr_core_v1_registry(), RankerTrainingConfig())


def _typed_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    frame = pl.DataFrame(rows)
    type_map: dict[str, type[pl.DataType]] = {
        "float32": pl.Float32,
        "uint8": pl.UInt8,
        "uint32": pl.UInt32,
    }
    return frame.with_columns(
        pl.col("query_id").cast(pl.Int64),
        pl.col("label_id").cast(pl.UInt8),
        pl.col("gain").cast(pl.Float32),
        *(
            pl.col(feature.name).cast(type_map[feature.dtype])
            for feature in ltr_core_v1_registry().features
        ),
    )


def test_population_rejects_incompatible_dtypes_and_null_targets() -> None:
    wrong_dtype = _frame().with_columns(pl.col("brand_code").cast(pl.Float32))
    with pytest.raises(TrainingPopulationError, match="incompatible dtypes"):
        prepare_training_population(wrong_dtype, ltr_core_v1_registry(), RankerTrainingConfig())

    null_gain = _frame().with_columns(
        pl.when(pl.col("query_id") == 1)
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(pl.col("gain"))
        .alias("gain")
    )
    with pytest.raises(TrainingPopulationError, match="cannot be null"):
        prepare_training_population(null_gain, ltr_core_v1_registry(), RankerTrainingConfig())


def test_normalized_query_groups_cannot_cross_train_and_validation() -> None:
    frame = _frame().with_columns(
        pl.when(pl.col("query_id") == 4)
        .then(pl.lit(f"{1:064x}"))
        .otherwise(pl.col("normalized_query_sha256"))
        .alias("normalized_query_sha256")
    )

    with pytest.raises(TrainingPopulationError, match="cross train and validation"):
        prepare_training_population(frame, ltr_core_v1_registry(), RankerTrainingConfig())


def test_prediction_ranks_use_product_ties_and_reset_inside_each_query() -> None:
    predictions = np.asarray([0.5, 0.5, 0.2, 0.9], dtype=np.float64)

    assert stable_rank_predictions(("p2", "p1", "p3", "p4"), predictions) == (3, 2, 4, 1)
    assert stable_grouped_prediction_ranks((1, 1, 2, 2), ("p2", "p1", "p3", "p4"), predictions) == (
        2,
        1,
        2,
        1,
    )


def test_population_row_limits_block_without_sampling_groups() -> None:
    config = RankerTrainingConfig(max_train_rows=1)

    with pytest.raises(TrainingPopulationError, match="exceed configured maximum"):
        prepare_training_population(_frame(), ltr_core_v1_registry(), config)
