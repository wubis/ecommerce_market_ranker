"""Strict localhost HTTP client for the MarketRank portfolio demo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self, TypeVar
from urllib.parse import urlsplit

import httpx2 as httpx
from pydantic import BaseModel, ValidationError

from market_rank.serving.contracts import (
    ArtifactInfoResponse,
    LivenessResponse,
    ModelInfoResponse,
    ReadinessResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class DemoClientError(RuntimeError):
    """Safe API-client failure suitable for a local UI or bounded CLI message."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.status_code = status_code
        self.retryable = retryable


class DemoApiClient:
    """Bounded client that accepts only a loopback MarketRank API origin."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise DemoClientError(
                "demo API origin must be an uncredentialed loopback HTTP URL",
                reason_code="invalid_api_origin",
            )
        if not 0.1 <= timeout_seconds <= 30.0:
            raise DemoClientError(
                "demo API timeout must be between 0.1 and 30 seconds",
                reason_code="invalid_timeout",
            )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return "MarketRank API returned a non-JSON error"
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                return str(detail["message"])[:240]
            if isinstance(detail, str):
                return detail[:240]
        return "MarketRank API request failed"

    def _request(self, method: str, path: str, *, json: object | None = None) -> httpx.Response:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise DemoClientError(
                "MarketRank API timed out",
                reason_code="api_timeout",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise DemoClientError(
                "MarketRank API is not reachable on localhost",
                reason_code="api_unreachable",
                retryable=True,
            ) from exc
        if response.status_code != 200:
            raise DemoClientError(
                self._message(response),
                reason_code=f"api_status_{response.status_code}",
                status_code=response.status_code,
                retryable=response.status_code in {429, 502, 503, 504},
            )
        return response

    @staticmethod
    def _validate(model: type[ModelT], response: httpx.Response) -> ModelT:
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise DemoClientError(
                "MarketRank API response violates the demo contract",
                reason_code="invalid_api_response",
            ) from exc

    def live(self) -> LivenessResponse:
        response = self._request("GET", "/health/live")
        return self._validate(LivenessResponse, response)

    def ready(self) -> ReadinessResponse:
        response = self._request("GET", "/health/ready")
        return self._validate(ReadinessResponse, response)

    def model_info(self) -> ModelInfoResponse:
        response = self._request("GET", "/v1/model-info")
        return self._validate(ModelInfoResponse, response)

    def artifact_info(self) -> ArtifactInfoResponse:
        response = self._request("GET", "/v1/artifact-info")
        return self._validate(ArtifactInfoResponse, response)

    def search(self, request: SearchRequest, *, explain: bool = False) -> SearchResponse:
        path = "/v1/debug/explain" if explain else "/v1/search"
        payload = request.model_copy(update={"debug": explain}).model_dump(mode="json")
        response = self._request("POST", path, json=payload)
        return self._validate(SearchResponse, response)

    def compare(
        self,
        request: SearchRequest,
        modes: Sequence[SearchMode],
        *,
        max_modes: int,
        explain: bool = False,
    ) -> tuple[SearchResponse, ...]:
        selected = tuple(modes)
        if (
            not selected
            or len(selected) > max_modes
            or len(set(selected)) != len(selected)
            or not 1 <= max_modes <= 6
        ):
            raise DemoClientError(
                "comparison modes must be unique and within the configured bound",
                reason_code="invalid_comparison_modes",
            )
        return tuple(
            self.search(request.model_copy(update={"mode": mode}), explain=explain)
            for mode in selected
        )


__all__ = ["DemoApiClient", "DemoClientError"]
