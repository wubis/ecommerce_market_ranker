"""Goldfish 014 API-client, presentation, and CLI unit tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx2 as httpx
import pytest

import market_rank.cli as cli_module
from market_rank.demo.client import DemoApiClient, DemoClientError
from market_rank.demo.presentation import (
    DemoPresentationError,
    compare_responses,
    compute_list_metrics,
)
from market_rank.serving.contracts import SearchMode, SearchRequest, SearchResponse


def test_demo_import_does_not_load_model_or_artifact_runtime() -> None:
    script = """
import sys
import market_rank.demo.client
import market_rank.demo.presentation
for forbidden in (
    'market_rank.artifacts',
    'market_rank.ranking.training',
    'market_rank.retrieval.dense',
    'market_rank.serving.orchestrator',
):
    assert forbidden not in sys.modules, forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _response(
    mode: SearchMode,
    products: tuple[tuple[str, str, str], ...],
    *,
    query_hash: str = "a" * 64,
) -> SearchResponse:
    return SearchResponse.model_validate(
        {
            "query_sha256": query_hash,
            "bundle_id": "serving-bundle/dataset/portfolio/v1/hash",
            "catalog_id": "esci_task1_us_catalog_v1",
            "config_sha256": "b" * 64,
            "requested_mode": mode,
            "promoted_stage": "rrf",
            "resolved_stage": "rrf" if mode in {"active", "hybrid"} else mode,
            "score_field": "hybrid_rrf_score",
            "score_comparable_with_promoted_stage": mode in {"active", "hybrid"},
            "degraded": False,
            "fallbacks": [],
            "candidate_count": len(products),
            "results": [
                {
                    "product_id": product_id,
                    "locale": "us",
                    "rank": rank,
                    "score": 1.0 / rank,
                    "score_field": "hybrid_rrf_score",
                    "title": title,
                    "brand": brand,
                    "color": "black",
                    "bullets": "",
                    "description_snippet": "fixture description",
                    "provenance": {
                        "bm25_score": 1.0 / rank,
                        "bm25_rank": rank,
                        "sparse_retriever_id": "bm25:fixture",
                        "sparse_index_id": "sparse/fixture",
                        "dense_score": 1.0 / rank,
                        "dense_rank": rank,
                        "dense_retriever_id": "dense:fixture",
                        "dense_index_id": "dense/fixture",
                        "rrf_score": 1.0 / rank,
                        "rrf_rank": rank,
                        "source_count": 2,
                    },
                    "debug": None,
                }
                for rank, (product_id, title, brand) in enumerate(products, start=1)
            ],
            "timings": {
                "parse_ms": 1.0,
                "sparse_ms": 2.0,
                "dense_ms": 3.0,
                "fusion_ms": 1.0,
                "features_ms": 0.0,
                "ranker_ms": 0.0,
                "product_lookup_ms": 1.0,
                "total_ms": 8.0,
            },
        }
    )


def test_client_accepts_only_loopback_and_validates_response_contracts() -> None:
    with pytest.raises(DemoClientError, match="loopback"):
        DemoApiClient("https://example.com", timeout_seconds=1.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/ready":
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "ready": True,
                    "degraded": False,
                    "bundle_id": "bundle/fixture",
                    "active_stage": "rrf",
                    "components": [{"component": "bundle", "state": "ready", "detail": "verified"}],
                },
            )
        payload = json.loads(request.content)
        mode = payload["mode"]
        return httpx.Response(
            200,
            json=_response(mode, (("p1", "Mouse", "Acme"),)).model_dump(mode="json"),
        )

    with DemoApiClient(
        "http://127.0.0.1:8000",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.ready().active_stage == "rrf"
        compared = client.compare(
            SearchRequest(query="wireless mouse", top_k=1),
            ("active", "lambdamart"),
            max_modes=2,
        )
    assert tuple(response.requested_mode for response in compared) == (
        "active",
        "lambdamart",
    )


def test_client_bounds_modes_and_maps_api_errors_without_response_dump() -> None:
    def unavailable(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"detail": {"code": "not_ready", "message": "no relevance path"}},
        )

    with DemoApiClient(
        "http://localhost:8000",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(unavailable),
    ) as client:
        with pytest.raises(DemoClientError, match="no relevance path") as caught:
            client.ready()
        assert caught.value.status_code == 503
        assert caught.value.retryable
        with pytest.raises(DemoClientError, match="unique"):
            client.compare(
                SearchRequest(query="mouse"),
                ("active", "active"),
                max_modes=2,
            )


def test_list_metrics_and_rank_changes_are_transparent_and_signed() -> None:
    baseline = _response(
        "active",
        (
            ("p1", "Wireless Mouse Black", "Acme"),
            ("p2", "Wireless Mouse Blue", "Acme"),
            ("p3", "Mechanical Keyboard", "Beta"),
        ),
    )
    treatment = _response(
        "lambdamart",
        (
            ("p2", "Wireless Mouse Blue", "Acme"),
            ("p1", "Wireless Mouse Black", "Acme"),
            ("p3", "Mechanical Keyboard", "Beta"),
        ),
    )

    metrics = compute_list_metrics(baseline)
    report = compare_responses((baseline, treatment))
    changes = {item.product_id: item for item in report.rank_changes}

    assert metrics.unique_brand_count == 2
    assert metrics.maximum_brand_concentration == pytest.approx(2 / 3)
    assert metrics.brand_entropy_bits == pytest.approx(0.918295834)
    assert 0.0 < metrics.title_token_ild <= 1.0
    assert changes["p1"].positions[1].change_from_baseline == -1
    assert changes["p2"].positions[1].change_from_baseline == 1
    assert report.summaries[0].total_ms == 8.0

    with pytest.raises(DemoPresentationError, match="incompatible"):
        compare_responses((baseline, _response("dense", (), query_hash="c" * 64)))


def test_demo_cli_check_and_run_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            assert base_url == "http://127.0.0.1:8000"
            assert timeout_seconds == 5.0

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def ready(self) -> object:
            from market_rank.serving.api import ReadinessResponse

            return ReadinessResponse(
                status="ready",
                ready=True,
                degraded=False,
                bundle_id="bundle/fixture",
                active_stage="rrf",
                components=(),
            )

    monkeypatch.setattr(cli_module, "DemoApiClient", FakeClient)
    assert cli_module.main(["demo", "check"]) == 0
    assert "demo API: ready" in capsys.readouterr().out

    observed: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert not check
        observed.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cli_module.main(["demo", "run"]) == 0
    assert observed[0][1:4] == ["-m", "streamlit", "run"]
    assert "--server.address" in observed[0]
    assert "127.0.0.1" in observed[0]
    assert str(Path("configs/base.yaml")) == observed[0][-1]
