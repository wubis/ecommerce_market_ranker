"""Checkpoint/build/load/search/pair tests for Goldfish 008 dense retrieval."""

from __future__ import annotations

import gc
import weakref
from argparse import Namespace
from hashlib import sha256
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import market_rank.cli as cli_module
import market_rank.retrieval.dense as dense_module
from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.evaluation.metrics import RETRIEVAL_PROTOCOL, Judgment, evaluate_ranked_products
from market_rank.retrieval.dense import (
    DENSE_DOCUMENT_MAP_FILENAME,
    DENSE_METADATA_FILENAME,
    EMBEDDINGS_FILENAME,
    FAISS_INDEX_FILENAME,
    DenseBuildError,
    DenseBuildResult,
    DenseQueryError,
    DenseResourceError,
    build_dense_index,
    cache_dense_model,
    load_dense_index,
    load_dense_metadata,
)
from tests.integration.test_esci_foundation import _build as _build_foundation
from tests.integration.test_esci_foundation import _prepare_foundation

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DIMENSION = 384


class HashEncoder:
    """Small deterministic unit-vector encoder; never loads a model or network."""

    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    dimension = DIMENSION

    @staticmethod
    def _one(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        vector = np.zeros(DIMENSION, dtype=np.float32)
        tokens = text.casefold().split()
        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "little") % DIMENSION
            vector[index] += 1.0 if digest[2] % 2 else -1.0
        if not np.any(vector):
            vector[0] = 1.0
        vector /= np.linalg.norm(vector)
        return vector

    def encode_documents(
        self, documents: tuple[str, ...]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return np.stack([self._one(document) for document in documents]).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        return self._one(query)


class ConstantEncoder(HashEncoder):
    @staticmethod
    def _one(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        del text
        vector = np.zeros(DIMENSION, dtype=np.float32)
        vector[0] = 1.0
        return vector


class InvalidNormEncoder(HashEncoder):
    def encode_documents(
        self, documents: tuple[str, ...]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return super().encode_documents(documents) * np.float32(2.0)


class WrongRevisionEncoder(HashEncoder):
    model_revision = "0" * 40


class InterruptingEncoder(HashEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def encode_documents(
        self, documents: tuple[str, ...]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("fixture interruption")
        return super().encode_documents(documents)


def _prepare_dense(
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
    encoder: HashEncoder | None = None,
) -> DenseBuildResult:
    release, config, store = prepared
    return build_dense_index(
        release,
        config,
        code_revision="fixture",
        artifact_store=store,
        encoder=encoder or HashEncoder(),
    )


def test_build_persists_normalized_vectors_faiss_and_exact_foundation_parent(
    tmp_path: Path,
) -> None:
    result = _build(_prepare_dense(tmp_path))

    assert not result.reused
    assert result.artifact.manifest.artifact_type == "dense-index"
    assert result.artifact.manifest.dependencies[0].artifact_id == (
        result.metadata.foundation_artifact_id
    )
    assert {item.relative_path for item in result.artifact.manifest.files} == {
        DENSE_DOCUMENT_MAP_FILENAME,
        DENSE_METADATA_FILENAME,
        EMBEDDINGS_FILENAME,
        FAISS_INDEX_FILENAME,
    }
    assert result.metadata.document_count == 17
    assert result.metadata.embedding_dimension == DIMENSION
    assert result.metadata.index_type == "IndexFlatIP"
    assert result.metadata.minimum_norm == pytest.approx(1.0, abs=1e-6)
    assert result.metadata.maximum_norm == pytest.approx(1.0, abs=1e-6)
    assert load_dense_metadata(result.artifact.path / DENSE_METADATA_FILENAME) == result.metadata


def test_search_uses_exact_scores_contiguous_ranks_and_product_id_ties(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)
    result = _build(prepared, ConstantEncoder())
    with load_dense_index(
        prepared[2], result.artifact.manifest.artifact_id, encoder=ConstantEncoder()
    ) as index:
        candidates = index.search("anything", top_k=5)

    assert tuple(item.product_id for item in candidates) == tuple(
        sorted(item.product_id for item in candidates)
    )
    assert tuple(item.one_based_rank for item in candidates) == (1, 2, 3, 4, 5)
    assert all(item.raw_score == pytest.approx(1.0) for item in candidates)
    assert all(item.latency_ms >= 0.0 for item in candidates)


def test_empty_query_and_explicit_pair_scoring_are_complete(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)
    result = _build(prepared)
    with load_dense_index(
        prepared[2], result.artifact.manifest.artifact_id, encoder=HashEncoder()
    ) as index:
        assert index.search("  ") == ()
        empty_scores = index.score_pairs("", ("p3-0", "p5-0"))
        scores = index.score_pairs("wireless mouse", ("p3-0", "p5-0"))

    assert tuple(item.product_id for item in scores) == ("p3-0", "p5-0")
    assert all(np.isfinite(item.raw_score) for item in scores)
    assert tuple(item.raw_score for item in empty_scores) == (0.0, 0.0)


def test_pair_scoring_and_top_k_reject_invalid_requests(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)
    result = _build(prepared)
    with load_dense_index(
        prepared[2], result.artifact.manifest.artifact_id, encoder=HashEncoder()
    ) as index:
        with pytest.raises(DenseQueryError, match="outside the catalog"):
            index.score_pairs("mouse", ("unknown",))
        with pytest.raises(DenseQueryError, match="unique"):
            index.score_pairs("mouse", ("p3-0", "p3-0"))
        with pytest.raises(DenseQueryError, match="top_k"):
            index.search("mouse", 0)
        with pytest.raises(DenseQueryError, match="4096"):
            index.search("x" * 4097)


def test_cold_reload_preserves_dense_search_and_pair_scores_exactly(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)
    result = _build(prepared)
    with load_dense_index(
        prepared[2], result.artifact.manifest.artifact_id, encoder=HashEncoder()
    ) as first:
        first_search = first.search("wireless mouse", 10)
        first_pairs = first.score_pairs("wireless mouse", ("p3-0", "p5-0"))
    with load_dense_index(
        prepared[2], result.artifact.manifest.artifact_id, encoder=HashEncoder()
    ) as second:
        second_search = second.search("wireless mouse", 10)
        second_pairs = second.score_pairs("wireless mouse", ("p3-0", "p5-0"))

    assert tuple(
        (item.product_id, item.raw_score, item.one_based_rank) for item in second_search
    ) == (tuple((item.product_id, item.raw_score, item.one_based_rank) for item in first_search))
    assert second_pairs == first_pairs


def test_raw_row_reordering_produces_identical_vectors_and_results(tmp_path: Path) -> None:
    first_prepared = _prepare_dense(tmp_path / "first")
    second_prepared = _prepare_dense(tmp_path / "reordered", reverse=True)
    first = _build(first_prepared)
    reordered = _build(second_prepared)

    assert (first.artifact.path / EMBEDDINGS_FILENAME).read_bytes() == (
        reordered.artifact.path / EMBEDDINGS_FILENAME
    ).read_bytes()
    with load_dense_index(
        first_prepared[2], first.artifact.manifest.artifact_id, encoder=HashEncoder()
    ) as first_index:
        first_result = first_index.search("wireless mouse", 10)
    with load_dense_index(
        second_prepared[2], reordered.artifact.manifest.artifact_id, encoder=HashEncoder()
    ) as reordered_index:
        reordered_result = reordered_index.search("wireless mouse", 10)

    assert tuple((item.product_id, item.raw_score) for item in reordered_result) == tuple(
        (item.product_id, item.raw_score) for item in first_result
    )


def test_interrupted_embedding_build_resumes_only_unfinished_rows(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)
    interrupting = InterruptingEncoder()

    with pytest.raises(DenseBuildError, match="16 completed documents"):
        _build(prepared, interrupting)

    result = _build(prepared)
    assert result.metadata.resumed_documents == 16
    assert not any((tmp_path / "artifacts" / ".dense-build").rglob("checkpoint.json"))


def test_non_unit_vectors_fail_without_promotion(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)

    with pytest.raises(DenseBuildError, match="non-unit"):
        _build(prepared, InvalidNormEncoder())

    assert not any((tmp_path / "artifacts" / "dense-index").rglob("_SUCCESS"))


def test_incompatible_encoder_identity_fails_before_embedding(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)

    with pytest.raises(dense_module.DenseModelError, match="identity"):
        _build(prepared, WrongRevisionEncoder())

    assert not (tmp_path / "artifacts" / ".dense-build").exists()


def test_releases_redundant_foundation_frames_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_dense(tmp_path)
    original_align = dense_module._aligned_documents
    frame_refs: list[weakref.ReferenceType[pl.DataFrame]] = []

    def observe_frames(membership: pl.DataFrame, documents: pl.DataFrame) -> pl.DataFrame:
        frame_refs.extend((weakref.ref(membership), weakref.ref(documents)))
        return original_align(membership, documents)

    class ObservingEncoder(HashEncoder):
        def __init__(self, config: ResolvedConfig) -> None:
            del config
            gc.collect()
            assert frame_refs and all(reference() is None for reference in frame_refs)

    monkeypatch.setattr(dense_module, "_aligned_documents", observe_frames)
    monkeypatch.setattr(dense_module, "SentenceTransformerEncoder", ObservingEncoder)

    result = build_dense_index(
        prepared[0],
        prepared[1],
        code_revision="fixture",
        artifact_store=prepared[2],
    )

    assert result.metadata.document_count == 17


def test_model_cache_uses_exact_revision_and_explicit_network_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_dense(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    observed: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        observed.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(dense_module, "_snapshot_download", snapshot_download)

    assert cache_dense_model(prepared[1], allow_network=False) == snapshot
    assert observed == [
        {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "cache_dir": "models/huggingface",
            "local_files_only": True,
        }
    ]


def test_resource_and_latency_measurements_are_persisted(tmp_path: Path) -> None:
    metadata = _build(_prepare_dense(tmp_path)).metadata

    assert metadata.resource.passed
    assert metadata.resource.embedding_bytes > 0
    assert metadata.resource.faiss_index_bytes > 0
    assert metadata.resource.peak_rss_bytes <= metadata.resource.rss_limit_bytes
    assert metadata.latency.sample_queries > 0
    assert metadata.latency.p50_ms <= metadata.latency.p95_ms <= metadata.latency.maximum_ms


def test_observed_rss_over_limit_blocks_promotion_and_retains_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_dense(tmp_path)
    observed = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024 + 1
    monkeypatch.setattr(dense_module, "_peak_rss_bytes", lambda: observed)

    with pytest.raises(DenseResourceError) as caught:
        _build(prepared)

    assert not caught.value.measurement.passed
    assert caught.value.measurement.completed_documents == 16
    assert any((tmp_path / "artifacts" / ".dense-build").rglob("checkpoint.json"))
    assert not any((tmp_path / "artifacts" / "dense-index").rglob("_SUCCESS"))


def test_compatible_dense_artifact_is_reused_without_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepare_dense(tmp_path)
    first = _build(prepared)

    def forbid_encoder(config: ResolvedConfig) -> object:
        del config
        raise AssertionError("compatible dense artifact should be reused")

    monkeypatch.setattr(dense_module, "SentenceTransformerEncoder", forbid_encoder)
    second = build_dense_index(
        prepared[0],
        prepared[1],
        code_revision="fixture",
        artifact_store=prepared[2],
    )

    assert second.reused
    assert second.artifact == first.artifact
    assert second.metadata == first.metadata


def test_catalog_retrieval_output_integrates_with_protocol_safe_metrics(tmp_path: Path) -> None:
    prepared = _prepare_dense(tmp_path)
    result = _build(prepared)
    with load_dense_index(
        prepared[2], result.artifact.manifest.artifact_id, encoder=HashEncoder()
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

    assert {record.metric for record in records} == {
        "exact_hit",
        "judged_mrr",
        "judged_recall",
        "known_judgment_coverage",
        "unjudged_rate",
    }


def test_cli_outputs_dense_summary_and_cache_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _build(_prepare_dense(tmp_path))

    def fake_build(
        release: ResolvedReleaseManifest,
        config: ResolvedConfig,
        *,
        code_revision: str,
    ) -> DenseBuildResult:
        del release, config, code_revision
        return result

    monkeypatch.setattr(cli_module, "build_dense_index", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")
    assert cli_module.main(["retrieval", "build-dense"]) == 0
    output = capsys.readouterr()
    assert "dense index: 17 documents x 384" in output.out
    assert "warm dense latency:" in output.out
    assert output.err == ""

    observed: list[bool] = []

    def fake_cache(config: ResolvedConfig, *, allow_network: bool) -> Path:
        del config
        observed.append(allow_network)
        return tmp_path / "snapshot"

    monkeypatch.setattr(cli_module, "cache_dense_model", fake_cache)
    assert cli_module.main(["retrieval", "cache-minilm", "--allow-network"]) == 0
    assert observed == [True]


def test_cli_dense_failure_is_one_line_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(arguments: Namespace) -> int:
        del arguments
        raise DenseQueryError("fixture dense failure")

    monkeypatch.setattr(cli_module, "_run_build_dense", fail)
    exit_code = cli_module.main(["retrieval", "build-dense"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err.strip() == "error: fixture dense failure"
    assert "Traceback" not in output.err
