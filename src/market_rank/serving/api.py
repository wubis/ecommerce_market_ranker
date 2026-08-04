"""FastAPI application factory for one explicit MarketRank serving bundle."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from market_rank.artifacts import ArtifactStore
from market_rank.config import ResolvedConfig
from market_rank.query.parser import QueryParserError
from market_rank.retrieval.dense import DenseEncoder
from market_rank.serving.contracts import (
    ArtifactComponentResponse,
    ArtifactInfoResponse,
    LivenessResponse,
    ModelInfoResponse,
    ReadinessResponse,
    SearchRequest,
    SearchResponse,
)
from market_rank.serving.orchestrator import (
    ServingBusyError,
    ServingRequestError,
    ServingRuntime,
    ServingUnavailableError,
    load_serving_runtime,
)


@dataclass(slots=True)
class _ApplicationState:
    runtime: ServingRuntime | None = None
    startup_error: str | None = None


def _not_ready(state: _ApplicationState) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "not_ready", "message": "no valid relevance path is loaded"},
    )


def _runtime_or_503(state: _ApplicationState) -> ServingRuntime:
    if state.runtime is None or not state.runtime.ready:
        raise _not_ready(state)
    return state.runtime


def _execute(runtime: ServingRuntime, request: SearchRequest) -> SearchResponse:
    try:
        return runtime.search(request)
    except ServingBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "busy", "message": str(exc)},
            headers={"Retry-After": "1"},
        ) from exc
    except ServingRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_request", "message": str(exc)},
        ) from exc
    except QueryParserError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_query", "message": str(exc)},
        ) from exc
    except ServingUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "search_unavailable", "message": str(exc)},
        ) from exc


def create_app(
    config: ResolvedConfig,
    bundle_id: str,
    *,
    artifact_store: ArtifactStore | None = None,
    encoder: DenseEncoder | None = None,
    runtime: ServingRuntime | None = None,
) -> FastAPI:
    """Create an app whose lifespan loads, but never creates, local artifacts."""
    application_state = _ApplicationState()
    injected_runtime = runtime

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if injected_runtime is not None:
            application_state.runtime = injected_runtime
        else:
            try:
                store = artifact_store or ArtifactStore(config.config.paths.artifacts_dir)
                application_state.runtime = load_serving_runtime(
                    store, bundle_id, config, encoder=encoder
                )
            except Exception as exc:
                application_state.startup_error = type(exc).__name__
        yield
        if application_state.runtime is not None and injected_runtime is None:
            application_state.runtime.close()

    app = FastAPI(
        title="MarketRank",
        version="0.13.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def enforce_request_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": {"code": "invalid_content_length"}},
                )
            if declared_size > config.config.serving.max_request_body_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": {"code": "request_body_too_large"}},
                )
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > config.config.serving.max_request_body_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": {"code": "request_body_too_large"}},
                )
        return await call_next(request)

    @app.get("/health/live", response_model=LivenessResponse)
    def live() -> LivenessResponse:
        return LivenessResponse()

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: {"description": "No valid relevance path is loaded"}},
    )
    def ready() -> ReadinessResponse:
        current = _runtime_or_503(application_state)
        info = current.info()
        return ReadinessResponse(
            status="ready",
            ready=True,
            degraded=info.degraded,
            bundle_id=info.bundle_id,
            active_stage=info.active_stage,
            components=info.components,
        )

    @app.post(
        "/v1/search",
        response_model=SearchResponse,
        responses={429: {"description": "Concurrency bound reached"}, 503: {}},
    )
    def search(search_request: SearchRequest) -> SearchResponse:
        return _execute(_runtime_or_503(application_state), search_request)

    @app.get("/v1/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        current = _runtime_or_503(application_state)
        manifest = current.manifest
        active = manifest.active_relevance
        return ModelInfoResponse(
            bundle_id=manifest.artifact_id,
            active_stage=active.selected_stage,
            active_score_field=active.active_score_field,
            active_score_comparable=active.active_score_comparable,
            fallback_contract=active.fallback_contract,
            feature_set_id=manifest.feature_set_id,
            feature_names=manifest.feature_names,
            parser_state_sha256=manifest.parser_state_sha256,
            ranking_models_artifact_id=active.ranking_models_artifact_id,
        )

    @app.get("/v1/artifact-info", response_model=ArtifactInfoResponse)
    def artifact_info() -> ArtifactInfoResponse:
        manifest = _runtime_or_503(application_state).manifest
        return ArtifactInfoResponse(
            bundle_id=manifest.artifact_id,
            dataset_version=manifest.dataset_version,
            profile=manifest.profile,
            catalog_id=manifest.catalog_id,
            catalog_membership_sha256=manifest.catalog_membership_sha256,
            config_sha256=manifest.config_sha256,
            offline_startup_required=manifest.offline_startup_required,
            components=tuple(
                ArtifactComponentResponse(
                    component=item.component,
                    artifact_id=item.artifact_id,
                    manifest_sha256=item.manifest_sha256,
                )
                for item in manifest.components
            ),
        )

    @app.post("/v1/debug/explain", response_model=SearchResponse)
    def debug_explain(search_request: SearchRequest) -> SearchResponse:
        if not config.config.serving.debug_enabled:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
        return _execute(
            _runtime_or_503(application_state),
            search_request.model_copy(update={"debug": True}),
        )

    return app


__all__ = [
    "ArtifactInfoResponse",
    "LivenessResponse",
    "ModelInfoResponse",
    "ReadinessResponse",
    "create_app",
]
