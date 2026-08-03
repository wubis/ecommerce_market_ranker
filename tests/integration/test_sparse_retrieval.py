"""Persist/build/load/search/pair parity tests for Goldfish 007 sparse retrieval."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import market_rank.cli as cli_module
import market_rank.retrieval.sparse as sparse_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.evaluation.metrics import RETRIEVAL_PROTOCOL, Judgment, evaluate_ranked_products
from market_rank.retrieval.sparse import (
    DOCUMENT_FREQUENCIES_FILENAME,
    DOCUMENT_LENGTHS_FILENAME,
    DOCUMENT_MAP_FILENAME,
    INVERSE_DOCUMENT_FREQUENCIES_FILENAME,
    POSTING_DOC_IDS_FILENAME,
    POSTING_TFS_FILENAME,
    POSTINGS_OFFSETS_FILENAME,
    SPARSE_METADATA_FILENAME,
    VOCABULARY_FILENAME,
    SparseBuildResult,
    SparseQueryError,
    SparseResourceError,
    build_sparse_index,
    load_sparse_index,
    load_sparse_metadata,
    tokenize,
)
from tests.integration.test_esci_foundation import (
    _build as _build_foundation,
)
from tests.integration.test_esci_foundation import (
    _prepare_foundation,
)


def _prepare_sparse(
    tmp_path: Path,
    *,
    reverse: bool = False,
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore]:
    prepared = _prepare_foundation(tmp_path, reverse=reverse)
    _build_foundation(prepared)
    release, config, _, store = prepared
    return release, config, store


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore],
) -> SparseBuildResult:
    release, config, store = prepared
    return build_sparse_index(
        release,
        config,
        code_revision="fixture",
        artifact_store=store,
    )


def test_tokenizer_is_nfkc_casefolded_unicode_and_deterministic() -> None:
    assert tokenize("  CAFÉ usb-C _Hidden model_42  ") == (
        "café",
        "usb-c",
        "hidden",
        "model",
        "42",
    )
    assert tokenize("Straße") == ("strasse",)


def test_build_persists_typed_index_with_exact_foundation_parent(tmp_path: Path) -> None:
    result = _build(_prepare_sparse(tmp_path))
    expected_files = {
        DOCUMENT_MAP_FILENAME,
        VOCABULARY_FILENAME,
        POSTINGS_OFFSETS_FILENAME,
        POSTING_DOC_IDS_FILENAME,
        POSTING_TFS_FILENAME,
        DOCUMENT_LENGTHS_FILENAME,
        DOCUMENT_FREQUENCIES_FILENAME,
        INVERSE_DOCUMENT_FREQUENCIES_FILENAME,
        SPARSE_METADATA_FILENAME,
    }

    assert not result.reused
    assert result.artifact.manifest.artifact_type == "sparse-index"
    assert result.artifact.manifest.dependencies[0].artifact_id == (
        result.metadata.foundation_artifact_id
    )
    assert {item.relative_path for item in result.artifact.manifest.files} == expected_files
    assert result.metadata.document_count == 17
    assert result.metadata.vocabulary_size > 0
    assert result.metadata.posting_count >= result.metadata.document_count
    assert load_sparse_metadata(result.artifact.path / SPARSE_METADATA_FILENAME) == result.metadata


def test_search_returns_positive_bm25_scores_and_stable_ties(tmp_path: Path) -> None:
    result = _build(_prepare_sparse(tmp_path))
    store = ArtifactStore(tmp_path / "artifacts")
    with load_sparse_index(store, result.artifact.manifest.artifact_id) as index:
        mouse = index.search("wireless mouse", top_k=3)
        ties = index.search("fixture", top_k=5)

    assert mouse[0].product_id == "p3-0"
    assert mouse[0].raw_score > 0.0
    assert tuple(item.one_based_rank for item in mouse) == tuple(range(1, len(mouse) + 1))
    assert tuple(item.product_id for item in ties) == tuple(
        sorted(item.product_id for item in ties)
    )


def test_empty_or_unknown_query_returns_empty_candidates(tmp_path: Path) -> None:
    result = _build(_prepare_sparse(tmp_path))
    with load_sparse_index(
        ArtifactStore(tmp_path / "artifacts"), result.artifact.manifest.artifact_id
    ) as index:
        assert index.search("") == ()
        assert index.search("term-not-in-catalog") == ()


def test_explicit_pair_scoring_returns_every_requested_product_in_input_order(
    tmp_path: Path,
) -> None:
    result = _build(_prepare_sparse(tmp_path))
    with load_sparse_index(
        ArtifactStore(tmp_path / "artifacts"), result.artifact.manifest.artifact_id
    ) as index:
        scores = index.score_pairs("wireless mouse", ("p3-0", "p5-0"))

    assert tuple(item.product_id for item in scores) == ("p3-0", "p5-0")
    assert scores[0].raw_score > 0.0
    assert scores[1].raw_score == 0.0
    assert all(item.index_id == result.artifact.manifest.artifact_id for item in scores)


def test_pair_scoring_rejects_unknown_or_duplicate_products(tmp_path: Path) -> None:
    result = _build(_prepare_sparse(tmp_path))
    with load_sparse_index(
        ArtifactStore(tmp_path / "artifacts"), result.artifact.manifest.artifact_id
    ) as index:
        with pytest.raises(SparseQueryError, match="outside the catalog"):
            index.score_pairs("mouse", ("p4-0",))
        with pytest.raises(SparseQueryError, match="unique"):
            index.score_pairs("mouse", ("p3-0", "p3-0"))


def test_top_k_and_query_bounds_are_enforced(tmp_path: Path) -> None:
    result = _build(_prepare_sparse(tmp_path))
    with load_sparse_index(
        ArtifactStore(tmp_path / "artifacts"), result.artifact.manifest.artifact_id
    ) as index:
        with pytest.raises(SparseQueryError, match="top_k"):
            index.search("mouse", top_k=0)
        with pytest.raises(SparseQueryError, match="4096"):
            index.search("x" * 4097)


def test_cold_reload_preserves_search_and_pair_scores_exactly(tmp_path: Path) -> None:
    result = _build(_prepare_sparse(tmp_path))
    store = ArtifactStore(tmp_path / "artifacts")
    with load_sparse_index(store, result.artifact.manifest.artifact_id) as first:
        first_search = first.search("wireless mouse", top_k=10)
        first_pairs = first.score_pairs("wireless mouse", ("p3-0", "p5-0"))
    with load_sparse_index(store, result.artifact.manifest.artifact_id) as second:
        assert second.search("wireless mouse", top_k=10) == first_search
        assert second.score_pairs("wireless mouse", ("p3-0", "p5-0")) == first_pairs


def test_raw_row_reordering_produces_identical_index_payloads_and_results(tmp_path: Path) -> None:
    first = _build(_prepare_sparse(tmp_path / "first"))
    reordered = _build(_prepare_sparse(tmp_path / "reordered", reverse=True))
    deterministic_files = (
        DOCUMENT_MAP_FILENAME,
        VOCABULARY_FILENAME,
        POSTINGS_OFFSETS_FILENAME,
        POSTING_DOC_IDS_FILENAME,
        POSTING_TFS_FILENAME,
        DOCUMENT_LENGTHS_FILENAME,
        DOCUMENT_FREQUENCIES_FILENAME,
        INVERSE_DOCUMENT_FREQUENCIES_FILENAME,
    )

    for filename in deterministic_files:
        assert (first.artifact.path / filename).read_bytes() == (
            reordered.artifact.path / filename
        ).read_bytes()
    with load_sparse_index(
        ArtifactStore(tmp_path / "first" / "artifacts"), first.artifact.manifest.artifact_id
    ) as first_index:
        first_result = first_index.search("wireless mouse", 10)
    with load_sparse_index(
        ArtifactStore(tmp_path / "reordered" / "artifacts"),
        reordered.artifact.manifest.artifact_id,
    ) as reordered_index:
        assert reordered_index.search("wireless mouse", 10) == first_result


def test_catalog_retrieval_output_integrates_with_protocol_safe_metrics(tmp_path: Path) -> None:
    result = _build(_prepare_sparse(tmp_path))
    with load_sparse_index(
        ArtifactStore(tmp_path / "artifacts"), result.artifact.manifest.artifact_id
    ) as index:
        candidates = index.search("wireless mouse", top_k=5)
    records = evaluate_ranked_products(
        RETRIEVAL_PROTOCOL,
        tuple(item.product_id for item in candidates),
        (
            Judgment(product_id="p3-0", label="E", gain=1.0),
            Judgment(product_id="p3-1", label="S", gain=0.1),
        ),
        k=5,
    )
    values = {record.metric: record.value for record in records}

    assert values["exact_hit"] == 1.0
    assert values["judged_mrr"] == 1.0


def test_resource_measurement_records_artifact_bytes_rss_and_gate(tmp_path: Path) -> None:
    resource = _build(_prepare_sparse(tmp_path)).metadata.resource

    assert resource.passed
    assert resource.build_seconds >= 0.0
    assert resource.index_payload_bytes > 0
    assert 0 < resource.peak_rss_bytes <= resource.rss_limit_bytes


def test_observed_rss_over_limit_blocks_sparse_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_sparse(tmp_path)
    _, config, _ = prepared
    observed = config.config.runtime.rss_limit_mb * 1024 * 1024 + 1
    monkeypatch.setattr(sparse_module, "_peak_rss_bytes", lambda: observed)

    with pytest.raises(SparseResourceError) as caught:
        _build(prepared)

    assert not caught.value.measurement.passed
    assert caught.value.measurement.peak_rss_bytes == observed
    assert not any((tmp_path / "artifacts" / "sparse-index").rglob("_SUCCESS"))


def test_compatible_sparse_artifact_is_reused_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_sparse(tmp_path)
    first = _build(prepared)

    def forbid_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("compatible sparse artifact should be reused")

    monkeypatch.setattr(sparse_module, "_build_index_payload", forbid_build)
    second = _build(prepared)

    assert second.reused
    assert second.artifact == first.artifact
    assert second.metadata == first.metadata


def test_cli_outputs_concise_sparse_build_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _build(_prepare_sparse(tmp_path))

    def fake_build(
        release: ResolvedReleaseManifest,
        config: ResolvedConfig,
        *,
        code_revision: str,
    ) -> SparseBuildResult:
        del release, config, code_revision
        return result

    monkeypatch.setattr(cli_module, "build_sparse_index", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")

    exit_code = cli_module.main(["retrieval", "build-bm25"])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "BM25 index: 17 documents" in output.out
    assert "resource:" in output.out
    assert "sparse index:" in output.out
    assert output.err == ""


def test_cli_sparse_failure_is_one_line_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(arguments: Namespace) -> int:
        del arguments
        raise SparseQueryError("fixture sparse failure")

    monkeypatch.setattr(cli_module, "_run_build_bm25", fail)
    exit_code = cli_module.main(["retrieval", "build-bm25"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err.strip() == "error: fixture sparse failure"
    assert "Traceback" not in output.err
