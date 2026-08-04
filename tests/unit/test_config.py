"""Tests for strict layered configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from market_rank.config import (
    ConfigFileError,
    ConfigOverrideError,
    ConfigValidationError,
    load_config,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
BASE_CONFIG = REPOSITORY_ROOT / "configs" / "base.yaml"


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_checked_in_base_config_loads_deterministically() -> None:
    first = load_config([BASE_CONFIG])
    second = load_config([BASE_CONFIG])

    assert first.config.runtime.max_threads == 4
    assert first.config.runtime.rss_limit_mb == 5632
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert first.short_hash == first.sha256[:12]
    assert first.source_paths == (BASE_CONFIG.resolve(),)


def test_layers_and_dotted_overrides_follow_precedence(tmp_path: Path) -> None:
    layer = _write_yaml(
        tmp_path / "development.yaml",
        """
runtime:
  max_threads: 3
logging:
  level: DEBUG
""",
    )

    resolved = load_config(
        [BASE_CONFIG, layer],
        overrides={"runtime.max_threads": 2, "runtime.seed": 7},
    )

    assert resolved.config.runtime.max_threads == 2
    assert resolved.config.runtime.seed == 7
    assert resolved.config.logging.level == "DEBUG"
    assert resolved.config.paths.data_dir == Path("data")


def test_semantically_equal_documents_have_the_same_hash(tmp_path: Path) -> None:
    first = _write_yaml(
        tmp_path / "first.yaml",
        """
runtime:
  seed: 11
schema_version: 1
""",
    )
    second = _write_yaml(
        tmp_path / "second.yaml",
        """
schema_version: 1
runtime:
  seed: 11
""",
    )

    first_resolved = load_config([first])
    second_resolved = load_config([second])

    assert first_resolved.canonical_json == second_resolved.canonical_json
    assert first_resolved.sha256 == second_resolved.sha256
    assert first_resolved.source_paths != second_resolved.source_paths


def test_semantic_change_changes_hash() -> None:
    default = load_config([BASE_CONFIG])
    changed = load_config([BASE_CONFIG], overrides={"runtime.seed": 1})

    assert default.sha256 != changed.sha256


def test_dataset_profile_targets_must_be_nested() -> None:
    with pytest.raises(ConfigValidationError, match="development_query_groups"):
        load_config(
            [BASE_CONFIG],
            overrides={
                "dataset.development_query_groups": 101,
                "dataset.portfolio_query_groups": 100,
            },
        )


def test_sparse_default_top_k_must_not_exceed_hard_maximum() -> None:
    with pytest.raises(ConfigValidationError, match="default_top_k"):
        load_config(
            [BASE_CONFIG],
            overrides={
                "retrieval.sparse.default_top_k": 101,
                "retrieval.sparse.max_top_k": 100,
            },
        )


def test_dense_defaults_pin_minilm_and_m3_batch_size() -> None:
    dense = load_config([BASE_CONFIG]).config.retrieval.dense

    assert dense.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert dense.model_revision == "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    assert dense.embedding_dimension == 384
    assert dense.embedding_batch_size == 16


def test_dense_default_top_k_must_not_exceed_hard_maximum() -> None:
    with pytest.raises(ConfigValidationError, match="default_top_k"):
        load_config(
            [BASE_CONFIG],
            overrides={
                "retrieval.dense.default_top_k": 101,
                "retrieval.dense.max_top_k": 100,
            },
        )


def test_hybrid_and_evaluation_defaults_are_bounded_for_fair_comparison() -> None:
    resolved = load_config([BASE_CONFIG]).config

    assert resolved.retrieval.hybrid.rrf_constant == 60
    assert resolved.retrieval.hybrid.sparse_top_k == 150
    assert resolved.retrieval.hybrid.dense_top_k == 150
    assert resolved.retrieval.hybrid.union_top_k == 200
    assert resolved.evaluation.cutoffs == (10, 100)
    assert resolved.evaluation.bootstrap_replicates == 1000


def test_evaluation_cutoff_cannot_exceed_any_compared_retrieval_depth() -> None:
    with pytest.raises(ConfigValidationError, match="evaluation cutoffs"):
        load_config(
            [BASE_CONFIG],
            overrides={
                "evaluation.cutoffs": [10, 151],
                "retrieval.hybrid.union_top_k": 200,
            },
        )


def test_hybrid_source_depth_cannot_exceed_source_index_maximum() -> None:
    with pytest.raises(ConfigValidationError, match="sparse_top_k"):
        load_config(
            [BASE_CONFIG],
            overrides={
                "retrieval.sparse.max_top_k": 100,
                "retrieval.sparse.default_top_k": 100,
                "retrieval.hybrid.sparse_top_k": 101,
            },
        )


def test_query_and_ranking_feature_defaults_are_m3_bounded() -> None:
    resolved = load_config([BASE_CONFIG]).config

    assert resolved.query_understanding.parser_version == "query-parser-v1"
    assert resolved.query_understanding.max_query_chars == 512
    assert resolved.query_understanding.max_query_tokens == 64
    assert resolved.ranking_features.feature_set_id == "ltr_core_v1"
    assert resolved.ranking_features.query_batch_size == 64
    assert resolved.ranking_features.matrix_partition_rows == 100000
    assert resolved.ranking_features.max_rows_per_query == 200


def test_feature_partition_must_fit_one_complete_query_group() -> None:
    with pytest.raises(ConfigValidationError, match="max_rows_per_query"):
        load_config(
            [BASE_CONFIG],
            overrides={
                "ranking_features.matrix_partition_rows": 100,
                "ranking_features.max_rows_per_query": 200,
            },
        )


def test_feature_row_bound_must_cover_hybrid_union() -> None:
    with pytest.raises(ConfigValidationError, match="feature row bound"):
        load_config(
            [BASE_CONFIG],
            overrides={
                "retrieval.hybrid.union_top_k": 200,
                "ranking_features.max_rows_per_query": 199,
            },
        )


@pytest.mark.parametrize(
    "content",
    [
        "unknown_section: true\n",
        "runtime:\n  unknown_field: 1\n",
        'runtime:\n  offline: "true"\n',
    ],
)
def test_unknown_fields_and_invalid_types_are_rejected(
    tmp_path: Path,
    content: str,
) -> None:
    invalid = _write_yaml(tmp_path / "invalid.yaml", content)

    with pytest.raises(ConfigValidationError):
        load_config([BASE_CONFIG, invalid])


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    duplicate = _write_yaml(
        tmp_path / "duplicate.yaml",
        """
runtime:
  max_threads: 2
  max_threads: 3
""",
    )

    with pytest.raises(ConfigFileError, match="duplicate key"):
        load_config([duplicate])


@pytest.mark.parametrize("content", ["- one\n- two\n", "42\n"])
def test_non_mapping_root_is_rejected(tmp_path: Path, content: str) -> None:
    invalid = _write_yaml(tmp_path / "invalid-root.yaml", content)

    with pytest.raises(ConfigFileError, match="root must be a mapping"):
        load_config([invalid])


def test_empty_path_sequence_is_rejected() -> None:
    with pytest.raises(ConfigFileError, match="at least one"):
        load_config([])


def test_invalid_dotted_override_is_rejected() -> None:
    with pytest.raises(ConfigOverrideError, match="invalid dotted"):
        load_config([BASE_CONFIG], overrides={"runtime..max_threads": 2})


def test_override_cannot_descend_through_scalar() -> None:
    with pytest.raises(ConfigOverrideError, match="is not a mapping"):
        load_config([BASE_CONFIG], overrides={"schema_version.value": 2})


def test_resolved_models_are_immutable() -> None:
    resolved = load_config([BASE_CONFIG])

    with pytest.raises(ValidationError):
        resolved.config.runtime.max_threads = 8
