"""Goldfish 013 bundle, runtime, degradation, API, and CLI integration tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import market_rank.cli as cli_module
import market_rank.serving.bundle as bundle_module
import market_rank.serving.orchestrator as orchestrator_module
from market_rank.artifacts import ArtifactStore, ArtifactValidationError
from market_rank.config import ResolvedConfig
from market_rank.data.esci_raw import ResolvedReleaseManifest
from market_rank.ranking.training import load_rankers as load_rankers_original
from market_rank.serving.api import create_app
from market_rank.serving.bundle import (
    PRODUCT_STORE_FILENAME,
    SERVING_BUNDLE_FILENAME,
    ServingBundleBuildResult,
    ServingBundleResourceError,
    ServingBundleValidationError,
    build_serving_bundle,
    load_product_store,
    load_serving_bundle_manifest,
)
from market_rank.serving.orchestrator import SearchRequest, load_serving_runtime
from tests.integration.test_dense_retrieval import HashEncoder
from tests.integration.test_ranking_evaluation import _build as _build_evaluation
from tests.integration.test_ranking_evaluation import _prepare as _prepare_evaluation


def _prepare(
    tmp_path: Path,
) -> tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore]:
    prepared = _prepare_evaluation(tmp_path)
    _build_evaluation(prepared)
    return prepared


def _build(
    prepared: tuple[ResolvedReleaseManifest, ResolvedConfig, ArtifactStore],
) -> ServingBundleBuildResult:
    return build_serving_bundle(
        prepared[0],
        prepared[1],
        code_revision="fixture",
        profile="portfolio",
        artifact_store=prepared[2],
    )


def test_bundle_pins_complete_lineage_and_safe_product_projection(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    result = _build(prepared)
    manifest = result.manifest

    assert not result.reused
    assert result.artifact.manifest.artifact_type == "serving-bundle"
    assert {item.relative_path for item in result.artifact.manifest.files} == {
        SERVING_BUNDLE_FILENAME,
        PRODUCT_STORE_FILENAME,
    }
    assert tuple(item.component for item in manifest.components) == (
        "foundation",
        "sparse",
        "dense",
        "features",
        "rankers",
        "ranking_evaluation",
    )
    assert manifest.bundle_id_policy == "explicit-only-no-latest-v1"
    assert manifest.offline_startup_required
    assert manifest.active_relevance.test_evaluated is False
    assert manifest.resource.passed
    assert load_serving_bundle_manifest(result.artifact.path / SERVING_BUNDLE_FILENAME) == manifest
    with load_product_store(result.artifact, manifest) as products:
        rows = products.fetch(("p3-0", "p5-0"))
    assert tuple(sorted(rows)) == ("p3-0", "p5-0")
    assert all(record.locale == "us" for record in rows.values())


def test_bundle_reuses_before_product_projection_and_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepare(tmp_path)
    first = _build(prepared)

    def forbid_projection(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise AssertionError("compatible bundle should reuse before product projection")

    monkeypatch.setattr(bundle_module, "_create_product_store", forbid_projection)
    second = _build(prepared)
    assert second.reused
    assert second.manifest == first.manifest

    path = first.artifact.path / PRODUCT_STORE_FILENAME
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactValidationError, match="integrity"):
        prepared[2].load(first.artifact.manifest.artifact_id)


def test_bundle_resource_overage_rolls_back_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepare(tmp_path)
    over = prepared[1].config.runtime.rss_limit_mb * 1024 * 1024 + 1
    monkeypatch.setattr(bundle_module, "_peak_rss_bytes", lambda: over)

    with pytest.raises(ServingBundleResourceError):
        _build(prepared)

    assert not any((tmp_path / "artifacts" / "serving-bundle").rglob("_SUCCESS"))


def test_runtime_serves_all_promoted_stages_with_bounded_debug_and_fallbacks(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    bundle = _build(prepared)
    with load_serving_runtime(
        prepared[2], bundle.artifact.manifest.artifact_id, prepared[1], encoder=HashEncoder()
    ) as runtime:
        assert runtime.ready
        active = runtime.search(SearchRequest(query="wireless mouse", top_k=5))
        lambdamart = runtime.search(
            SearchRequest(query="wireless mouse", top_k=5, mode="lambdamart", debug=True)
        )
        optional = runtime.search(
            SearchRequest(query="wireless mouse", top_k=3, neural_rerank=True, diversify=True)
        )

    assert active.promoted_stage == bundle.manifest.active_relevance.selected_stage
    assert active.resolved_stage == active.promoted_stage
    assert tuple(item.rank for item in active.results) == tuple(range(1, len(active.results) + 1))
    assert len({item.product_id for item in active.results}) == len(active.results) <= 5
    assert lambdamart.resolved_stage == "lambdamart"
    assert all(item.debug is not None for item in lambdamart.results)
    assert all(
        tuple(name for name, _ in item.debug.feature_values) == bundle.manifest.feature_names
        for item in lambdamart.results
        if item.debug is not None
    )
    assert {event.component for event in optional.fallbacks} >= {
        "neural_reranker",
        "diversifier",
    }
    assert optional.degraded
    assert active.timings.total_ms >= active.timings.parse_ms


def test_runtime_degrades_to_sparse_and_requires_explicit_bundle_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepare(tmp_path)
    bundle = _build(prepared)

    def fail_rankers(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("fixture ranker load failure")

    monkeypatch.setattr(orchestrator_module, "load_rankers", fail_rankers)
    with load_serving_runtime(
        prepared[2], bundle.artifact.manifest.artifact_id, prepared[1], encoder=HashEncoder()
    ) as runtime:
        ranker_fallback = runtime.search(SearchRequest(query="wireless mouse", mode="lambdamart"))
    assert ranker_fallback.resolved_stage == "rrf"
    assert any(event.reason_code == "ranker_unavailable" for event in ranker_fallback.fallbacks)
    monkeypatch.setattr(orchestrator_module, "load_rankers", load_rankers_original)

    def fail_dense(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("fixture dense load failure")

    monkeypatch.setattr(orchestrator_module, "load_dense_index", fail_dense)
    with load_serving_runtime(
        prepared[2], bundle.artifact.manifest.artifact_id, prepared[1], encoder=HashEncoder()
    ) as runtime:
        response = runtime.search(SearchRequest(query="wireless mouse", mode="active"))

    assert runtime.degraded
    assert response.resolved_stage == "rrf"
    assert any(event.reason_code == "dense_unavailable" for event in response.fallbacks)

    def fail_sparse(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("fixture sparse load failure")

    monkeypatch.setattr(orchestrator_module, "load_sparse_index", fail_sparse)
    with pytest.raises(orchestrator_module.ServingUnavailableError, match="no retriever"):
        load_serving_runtime(
            prepared[2],
            bundle.artifact.manifest.artifact_id,
            prepared[1],
            encoder=HashEncoder(),
        )
    with pytest.raises(ServingBundleValidationError, match="explicit immutable"):
        load_serving_runtime(prepared[2], "latest", prepared[1], encoder=HashEncoder())


def test_fastapi_contracts_readiness_validation_and_no_path_disclosure(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    bundle = _build(prepared)
    runtime = load_serving_runtime(
        prepared[2], bundle.artifact.manifest.artifact_id, prepared[1], encoder=HashEncoder()
    )
    app = create_app(prepared[1], bundle.artifact.manifest.artifact_id, runtime=runtime)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/health/live").json() == {"status": "live"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True
        search = client.post("/v1/search", json={"query": "wireless mouse", "top_k": 3})
        assert search.status_code == 200
        assert len(search.json()["results"]) <= 3
        debug = client.post("/v1/debug/explain", json={"query": "wireless mouse", "top_k": 2})
        assert debug.status_code == 200
        assert debug.json()["results"][0]["debug"] is not None
        assert client.post("/v1/search", json={"query": "   "}).status_code == 422
        assert client.post("/v1/search", json={"query": "mouse", "top_k": 51}).status_code == 422
        artifact_info = client.get("/v1/artifact-info")
        assert artifact_info.status_code == 200
        assert str(tmp_path) not in artifact_info.text
        assert client.get("/v1/model-info").status_code == 200
        hostile_host = client.get("/health/live", headers={"Host": "attacker.example"})
        assert hostile_host.status_code == 400
        assert str(tmp_path) not in hostile_host.text
        oversized = "x" * prepared[1].config.serving.max_request_body_bytes
        assert (
            client.post("/v1/search", json={"query": "mouse", "padding": oversized}).status_code
            == 413
        )
    runtime.close()

    unavailable = create_app(prepared[1], "serving-bundle/missing/explicit/id")
    with TestClient(unavailable, base_url="http://127.0.0.1") as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/search", json={"query": "mouse"}).status_code == 503


def test_serving_cli_promotes_prints_bundle_and_bounds_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _build(_prepare(tmp_path))

    def fake_build(
        release: ResolvedReleaseManifest,
        config: ResolvedConfig,
        *,
        code_revision: str,
        profile: bundle_module.Profile | None,
    ) -> ServingBundleBuildResult:
        del release, config, code_revision, profile
        return result

    monkeypatch.setattr(cli_module, "build_serving_bundle", fake_build)
    monkeypatch.setattr(cli_module, "_resolve_code_revision", lambda args, path: "fixture")
    assert cli_module.main(["serving", "promote", "--profile", "portfolio"]) == 0
    output = capsys.readouterr()
    assert "active stage" in output.out
    assert "serving bundle:" in output.out
    assert output.err == ""

    def fail(arguments: Namespace) -> int:
        del arguments
        raise bundle_module.ServingBundleBuildError("fixture promotion failure")

    monkeypatch.setattr(cli_module, "_run_promote_serving", fail)
    assert cli_module.main(["serving", "promote"]) == 1
    failed = capsys.readouterr()
    assert failed.err.strip() == "error: fixture promotion failure"
    assert "Traceback" not in failed.err
