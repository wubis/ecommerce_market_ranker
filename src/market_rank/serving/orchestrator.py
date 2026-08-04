"""Offline-only serving runtime and bounded search orchestration."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Self, cast

import numpy as np

from market_rank.artifacts import ArtifactStore, LoadedArtifact
from market_rank.config import ResolvedConfig
from market_rank.features.artifact import (
    FEATURE_STATE_FILENAME,
    FeatureState,
    load_feature_state,
)
from market_rank.features.core import ProductFeatureView, compute_core_features, rank_fractions
from market_rank.features.registry import FEATURE_NAMES
from market_rank.query.parser import ParsedQuery, QueryParser
from market_rank.ranking.training import LoadedRankers, load_rankers
from market_rank.retrieval.dense import (
    DenseCandidate,
    DenseEncoder,
    DenseIndex,
    DensePairScore,
    SentenceTransformerEncoder,
    load_dense_index,
)
from market_rank.retrieval.hybrid import HybridCandidate, fuse_rrf
from market_rank.retrieval.sparse import (
    SparseCandidate,
    SparseIndex,
    SparsePairScore,
    load_sparse_index,
)
from market_rank.serving.bundle import (
    SERVING_BUNDLE_FILENAME,
    ProductRecord,
    ProductStore,
    ServingBundleManifest,
    ServingBundleValidationError,
    load_product_store,
    load_serving_bundle_manifest,
)
from market_rank.serving.contracts import (
    ComponentStatus,
    FallbackEvent,
    ResolvedStage,
    ResultDebug,
    RetrievalProvenance,
    RuntimeInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchTimings,
)


class ServingRuntimeError(RuntimeError):
    """Base error for runtime loading and request orchestration."""


class ServingUnavailableError(ServingRuntimeError):
    """Raised when no valid relevance path can serve a request."""


class ServingBusyError(ServingRuntimeError):
    """Raised when the bounded request concurrency budget is exhausted."""


class ServingRequestError(ServingRuntimeError):
    """Raised when a request violates a configuration-dependent bound."""


@dataclass(frozen=True, slots=True)
class _FeatureRows:
    matrix: np.ndarray
    by_product: dict[str, tuple[tuple[str, float], ...]]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _component_id(manifest: ServingBundleManifest, name: str) -> str:
    return next(item.artifact_id for item in manifest.components if item.component == name)


def _product_view(record: ProductRecord) -> ProductFeatureView:
    return ProductFeatureView(
        locale=record.locale,
        title=record.title,
        brand=record.brand,
        color=record.color,
        bullets=record.bullets,
        description=record.description,
        normalized_brand=record.normalized_brand,
        normalized_color=record.normalized_color,
        title_missing=record.title_missing,
        brand_missing=record.brand_missing,
        color_missing=record.color_missing,
        bullets_missing=record.bullets_missing,
        description_missing=record.description_missing,
    )


class ServingRuntime:
    """Loaded immutable serving state; startup never builds or downloads."""

    def __init__(
        self,
        *,
        config: ResolvedConfig,
        artifact: LoadedArtifact,
        manifest: ServingBundleManifest,
        product_store: ProductStore,
        feature_state: FeatureState,
        sparse: SparseIndex | None,
        dense: DenseIndex | None,
        rankers: LoadedRankers | None,
        component_statuses: tuple[ComponentStatus, ...],
    ) -> None:
        self.config = config
        self.artifact = artifact
        self.manifest = manifest
        self.product_store = product_store
        self.feature_state = feature_state
        self.parser = QueryParser(feature_state.parser_state, config.config.query_understanding)
        self._brand_codes = dict(feature_state.brand_codes)
        self._color_codes = dict(feature_state.color_codes)
        self.sparse = sparse
        self.dense = dense
        self.rankers = rankers
        self.component_statuses = component_statuses
        self._slots = threading.BoundedSemaphore(config.config.serving.max_concurrency)
        self._closed = False

    @property
    def ready(self) -> bool:
        return not self._closed and (self.sparse is not None or self.dense is not None)

    @property
    def degraded(self) -> bool:
        return any(status.state != "ready" for status in self.component_statuses)

    def info(self) -> RuntimeInfo:
        return RuntimeInfo(
            ready=self.ready,
            degraded=self.degraded,
            bundle_id=self.manifest.artifact_id,
            catalog_id=self.manifest.catalog_id,
            config_sha256=self.manifest.config_sha256,
            active_stage=self.manifest.active_relevance.selected_stage,
            components=self.component_statuses,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self.sparse is not None:
            self.sparse.close()
        if self.dense is not None:
            self.dense.close()
        self.product_store.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _retrieve(
        self, query: str
    ) -> tuple[
        tuple[SparseCandidate, ...],
        tuple[DenseCandidate, ...],
        float,
        float,
        list[FallbackEvent],
    ]:
        retrieval = self.config.config.retrieval
        fallbacks: list[FallbackEvent] = []
        sparse_candidates: tuple[SparseCandidate, ...] = ()
        dense_candidates: tuple[DenseCandidate, ...] = ()
        sparse_succeeded = False
        dense_succeeded = False
        sparse_started = time.perf_counter()
        if self.sparse is not None:
            try:
                sparse_candidates = self.sparse.search(query, retrieval.hybrid.sparse_top_k)
                sparse_succeeded = True
            except Exception:
                fallbacks.append(
                    FallbackEvent(
                        component="sparse",
                        requested_stage="bm25",
                        resolved_stage="dense",
                        reason_code="sparse_query_failed",
                    )
                )
        else:
            fallbacks.append(
                FallbackEvent(
                    component="sparse",
                    requested_stage="bm25",
                    resolved_stage="dense",
                    reason_code="sparse_unavailable",
                )
            )
        sparse_ms = _elapsed_ms(sparse_started)
        dense_started = time.perf_counter()
        if self.dense is not None:
            try:
                dense_candidates = self.dense.search(query, retrieval.hybrid.dense_top_k)
                dense_succeeded = True
            except Exception:
                fallbacks.append(
                    FallbackEvent(
                        component="dense",
                        requested_stage="dense",
                        resolved_stage="bm25",
                        reason_code="dense_query_failed",
                    )
                )
        else:
            fallbacks.append(
                FallbackEvent(
                    component="dense",
                    requested_stage="dense",
                    resolved_stage="bm25",
                    reason_code="dense_unavailable",
                )
            )
        dense_ms = _elapsed_ms(dense_started)
        if not sparse_succeeded and not dense_succeeded:
            raise ServingUnavailableError("both candidate retrieval paths failed")
        if fallbacks and not self.config.config.serving.allow_degraded_retrieval:
            raise ServingUnavailableError("a candidate retrieval path failed")
        return sparse_candidates, dense_candidates, sparse_ms, dense_ms, fallbacks

    def _feature_rows(
        self,
        query: str,
        parsed: ParsedQuery,
        candidates: tuple[HybridCandidate, ...],
        products: dict[str, ProductRecord],
        *,
        include_debug: bool,
    ) -> _FeatureRows:
        product_ids = tuple(candidate.product_id for candidate in candidates)
        sparse_pairs = (
            self.sparse.score_pairs(query, product_ids)
            if self.sparse is not None
            else tuple(
                SparsePairScore(product_id, "us", 0.0, "unavailable", "unavailable")
                for product_id in product_ids
            )
        )
        dense_pairs = (
            self.dense.score_pairs(query, product_ids)
            if self.dense is not None
            else tuple(
                DensePairScore(product_id, "us", 0.0, "unavailable", "unavailable")
                for product_id in product_ids
            )
        )
        sparse_scores = {item.product_id: item.raw_score for item in sparse_pairs}
        dense_scores = {item.product_id: item.raw_score for item in dense_pairs}
        _, sparse_fractions = rank_fractions(sparse_scores)
        _, dense_fractions = rank_fractions(dense_scores)
        rrf_scores = {item.product_id: item.rrf_score for item in candidates}
        _, rrf_fractions = rank_fractions(rrf_scores)
        idfs = self.sparse.query_idf_values(query) if self.sparse is not None else ()
        specificity = sum(idfs) / len(idfs) if idfs else 0.0
        rows: list[list[float]] = []
        debug: dict[str, tuple[tuple[str, float], ...]] = {}
        for candidate in candidates:
            product_id = candidate.product_id
            values = compute_core_features(
                parsed,
                _product_view(products[product_id]),
                lexical_specificity=specificity,
                brand_codes=self._brand_codes,
                color_codes=self._color_codes,
                bm25_score=sparse_scores[product_id],
                bm25_rank_fraction=sparse_fractions[product_id],
                dense_score=dense_scores[product_id],
                dense_rank_fraction=dense_fractions[product_id],
                rrf_score=rrf_scores[product_id],
                rrf_rank_fraction=rrf_fractions[product_id],
            )
            row = [float(values[name]) for name in FEATURE_NAMES]
            rows.append(row)
            if include_debug:
                debug[product_id] = tuple(zip(FEATURE_NAMES, row, strict=True))
        matrix = np.asarray(rows, dtype=np.float32, order="C")
        return _FeatureRows(matrix, debug)

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute one bounded query with explicit stage and fallback provenance."""
        if not self.ready:
            raise ServingUnavailableError("serving runtime is not ready")
        serving = self.config.config.serving
        top_k = request.top_k or serving.default_top_k
        deadline_ms = request.deadline_ms or serving.default_deadline_ms
        if top_k > serving.max_response_top_k:
            raise ServingRequestError("top_k exceeds the configured serving maximum")
        if deadline_ms > serving.max_deadline_ms:
            raise ServingRequestError("deadline_ms exceeds the configured serving maximum")
        if request.debug and not serving.debug_enabled:
            raise ServingRequestError("debug output is disabled")
        if not self._slots.acquire(blocking=False):
            raise ServingBusyError("serving concurrency limit reached")
        started = time.perf_counter()
        try:
            parse_started = time.perf_counter()
            parsed = self.parser.parse(request.query)
            parse_ms = _elapsed_ms(parse_started)
            sparse_raw, dense_raw, sparse_ms, dense_ms, fallbacks = self._retrieve(request.query)
            fusion_started = time.perf_counter()
            hybrid = fuse_rrf(
                sparse_raw,
                dense_raw,
                rrf_constant=self.feature_state.rrf_constant,
                top_k=self.config.config.retrieval.hybrid.union_top_k,
                max_top_k=self.config.config.retrieval.hybrid.max_union_top_k,
            )
            fusion_ms = _elapsed_ms(fusion_started)
            candidates = hybrid.candidates
            product_started = time.perf_counter()
            products = (
                self.product_store.fetch(tuple(candidate.product_id for candidate in candidates))
                if candidates
                else {}
            )
            product_lookup_ms = _elapsed_ms(product_started)

            promoted = self.manifest.active_relevance.selected_stage
            requested_stage = promoted if request.mode == "active" else request.mode
            sparse_available = self.sparse is not None and not any(
                event.reason_code == "sparse_query_failed" for event in fallbacks
            )
            dense_available = self.dense is not None and not any(
                event.reason_code == "dense_query_failed" for event in fallbacks
            )
            resolved: ResolvedStage
            ordered = candidates
            scores = {candidate.product_id: candidate.rrf_score for candidate in candidates}
            score_field = "hybrid_rrf_score"
            features_ms = 0.0
            ranker_ms = 0.0
            feature_rows: _FeatureRows | None = None

            if requested_stage == "bm25":
                if sparse_available:
                    sparse_order = {item.product_id: index for index, item in enumerate(sparse_raw)}
                    ordered = tuple(
                        sorted(
                            (item for item in candidates if item.product_id in sparse_order),
                            key=lambda item: (
                                sparse_order.get(item.product_id, len(sparse_order)),
                                item.product_id,
                            ),
                        )
                    )
                    scores = {item.product_id: float(item.raw_score) for item in sparse_raw}
                    resolved = "bm25"
                    score_field = "bm25_score"
                else:
                    resolved = "dense"
                    dense_order = {item.product_id: index for index, item in enumerate(dense_raw)}
                    ordered = tuple(
                        sorted(
                            (item for item in candidates if item.product_id in dense_order),
                            key=lambda item: (
                                dense_order.get(item.product_id, len(dense_order)),
                                item.product_id,
                            ),
                        )
                    )
                    scores = {item.product_id: float(item.raw_score) for item in dense_raw}
                    score_field = "dense_score"
            elif requested_stage == "dense":
                if dense_available:
                    dense_order = {item.product_id: index for index, item in enumerate(dense_raw)}
                    ordered = tuple(
                        sorted(
                            (item for item in candidates if item.product_id in dense_order),
                            key=lambda item: (
                                dense_order.get(item.product_id, len(dense_order)),
                                item.product_id,
                            ),
                        )
                    )
                    scores = {item.product_id: float(item.raw_score) for item in dense_raw}
                    resolved = "dense"
                    score_field = "dense_score"
                else:
                    resolved = "bm25"
                    sparse_order = {item.product_id: index for index, item in enumerate(sparse_raw)}
                    ordered = tuple(
                        sorted(
                            (item for item in candidates if item.product_id in sparse_order),
                            key=lambda item: (
                                sparse_order.get(item.product_id, len(sparse_order)),
                                item.product_id,
                            ),
                        )
                    )
                    scores = {item.product_id: float(item.raw_score) for item in sparse_raw}
                    score_field = "bm25_score"
            elif requested_stage in ("pointwise", "lambdamart"):
                elapsed = _elapsed_ms(started)
                if (
                    self.rankers is None
                    or self.sparse is None
                    or self.dense is None
                    or elapsed >= deadline_ms
                ):
                    resolved = "rrf"
                    if self.rankers is None:
                        reason = "ranker_unavailable"
                    elif self.sparse is None or self.dense is None:
                        reason = "model_requires_both_retrievers"
                    else:
                        reason = "deadline_exhausted"
                    fallbacks.append(
                        FallbackEvent(
                            component="rankers",
                            requested_stage=requested_stage,
                            resolved_stage="rrf",
                            reason_code=reason,
                        )
                    )
                else:
                    try:
                        feature_started = time.perf_counter()
                        feature_rows = self._feature_rows(
                            request.query,
                            parsed,
                            candidates,
                            products,
                            include_debug=request.debug,
                        )
                        features_ms = _elapsed_ms(feature_started)
                        if _elapsed_ms(started) >= deadline_ms:
                            raise TimeoutError("request deadline exhausted before model scoring")
                        ranker_started = time.perf_counter()
                        predictions = self.rankers.rank(
                            requested_stage,
                            tuple(candidate.product_id for candidate in candidates),
                            feature_rows.matrix,
                        )
                        ranker_ms = _elapsed_ms(ranker_started)
                        prediction_by_id = {item.product_id: item for item in predictions}
                        ordered = tuple(
                            sorted(
                                candidates,
                                key=lambda item: prediction_by_id[item.product_id].one_based_rank,
                            )
                        )
                        scores = {item.product_id: item.score for item in predictions}
                        resolved = cast(ResolvedStage, requested_stage)
                        score_field = f"{requested_stage}_score"
                    except Exception:
                        if not serving.allow_ranker_fallback:
                            raise ServingUnavailableError("ranker scoring failed") from None
                        resolved = "rrf"
                        ordered = candidates
                        scores = {
                            candidate.product_id: candidate.rrf_score for candidate in candidates
                        }
                        fallbacks.append(
                            FallbackEvent(
                                component="rankers",
                                requested_stage=requested_stage,
                                resolved_stage="rrf",
                                reason_code="ranker_scoring_failed",
                            )
                        )
            else:
                resolved = "rrf"

            if request.neural_rerank:
                fallbacks.append(
                    FallbackEvent(
                        component="neural_reranker",
                        requested_stage="neural_rerank",
                        resolved_stage=resolved,
                        reason_code="not_promoted",
                    )
                )
            if request.diversify:
                fallbacks.append(
                    FallbackEvent(
                        component="diversifier",
                        requested_stage="diversify",
                        resolved_stage=resolved,
                        reason_code="not_promoted",
                    )
                )

            evidence = {candidate.product_id: candidate for candidate in candidates}
            results: list[SearchResult] = []
            for rank, candidate in enumerate(ordered[:top_k], start=1):
                product = products[candidate.product_id]
                source = evidence[candidate.product_id]
                debug = None
                if request.debug and rank <= serving.max_debug_candidates:
                    if feature_rows is None:
                        feature_rows = self._feature_rows(
                            request.query,
                            parsed,
                            candidates,
                            products,
                            include_debug=True,
                        )
                    debug = ResultDebug(
                        feature_values=feature_rows.by_product[candidate.product_id]
                    )
                results.append(
                    SearchResult(
                        product_id=candidate.product_id,
                        locale="us",
                        rank=rank,
                        score=scores[candidate.product_id],
                        score_field=score_field,
                        title=product.title,
                        brand=product.brand,
                        color=product.color,
                        bullets=product.bullets,
                        description_snippet=product.description[
                            : serving.description_snippet_chars
                        ],
                        provenance=RetrievalProvenance(
                            bm25_score=source.sparse_score,
                            bm25_rank=source.sparse_rank,
                            sparse_retriever_id=source.sparse_retriever_id,
                            sparse_index_id=source.sparse_index_id,
                            dense_score=source.dense_score,
                            dense_rank=source.dense_rank,
                            dense_retriever_id=source.dense_retriever_id,
                            dense_index_id=source.dense_index_id,
                            rrf_score=source.rrf_score,
                            rrf_rank=source.one_based_rank,
                            source_count=source.source_count,
                        ),
                        debug=debug,
                    )
                )
            total_ms = _elapsed_ms(started)
            return SearchResponse(
                query_sha256=parsed.query_sha256,
                bundle_id=self.manifest.artifact_id,
                catalog_id=self.manifest.catalog_id,
                config_sha256=self.manifest.config_sha256,
                requested_mode=request.mode,
                promoted_stage=promoted,
                resolved_stage=resolved,
                score_field=score_field,
                score_comparable_with_promoted_stage=resolved == promoted,
                degraded=bool(fallbacks) or self.degraded,
                fallbacks=tuple(fallbacks),
                candidate_count=len(candidates),
                results=tuple(results),
                timings=SearchTimings(
                    parse_ms=parse_ms,
                    sparse_ms=sparse_ms,
                    dense_ms=dense_ms,
                    fusion_ms=fusion_ms,
                    features_ms=features_ms,
                    ranker_ms=ranker_ms,
                    product_lookup_ms=product_lookup_ms,
                    total_ms=total_ms,
                ),
            )
        finally:
            self._slots.release()


def load_serving_runtime(
    store: ArtifactStore,
    bundle_id: str,
    config: ResolvedConfig,
    *,
    encoder: DenseEncoder | None = None,
) -> ServingRuntime:
    """Load one explicit verified bundle using local artifacts only."""
    if not bundle_id or bundle_id.strip() != bundle_id or "latest" in bundle_id.casefold():
        raise ServingBundleValidationError("an explicit immutable bundle ID is required")
    artifact = store.load(bundle_id)
    if artifact.manifest.artifact_type != "serving-bundle":
        raise ServingBundleValidationError("artifact is not a serving bundle")
    manifest = load_serving_bundle_manifest(artifact.path / SERVING_BUNDLE_FILENAME)
    if (
        manifest.artifact_id != bundle_id
        or manifest.config_sha256 != config.sha256
        or artifact.manifest.config_sha256 != config.sha256
    ):
        raise ServingBundleValidationError("bundle identity differs from resolved configuration")
    feature_artifact = store.load(_component_id(manifest, "features"))
    state = load_feature_state(feature_artifact.path / FEATURE_STATE_FILENAME)
    if (
        state.registry_sha256 != manifest.feature_registry_sha256
        or state.parser_state.state_sha256 != manifest.parser_state_sha256
        or state.sparse_retriever_id != manifest.sparse_retriever_id
        or state.dense_retriever_id != manifest.dense_retriever_id
    ):
        raise ServingBundleValidationError("bundle feature state is incompatible")
    product_store = load_product_store(artifact, manifest)
    statuses: list[ComponentStatus] = [
        ComponentStatus(component="bundle", state="ready", detail="manifest verified"),
        ComponentStatus(
            component="product_store", state="ready", detail="read-only projection verified"
        ),
    ]
    sparse: SparseIndex | None = None
    dense: DenseIndex | None = None
    rankers: LoadedRankers | None = None
    try:
        sparse = load_sparse_index(store, _component_id(manifest, "sparse"))
        statuses.append(
            ComponentStatus(component="sparse", state="ready", detail="BM25 memory maps loaded")
        )
    except Exception:
        statuses.append(
            ComponentStatus(
                component="sparse", state="unavailable", detail="BM25 runtime load failed"
            )
        )
    try:
        selected_encoder = cast(DenseEncoder, encoder or SentenceTransformerEncoder(config))
        dense = load_dense_index(
            store,
            _component_id(manifest, "dense"),
            encoder=selected_encoder,
            max_threads=config.config.runtime.max_threads,
        )
        statuses.append(
            ComponentStatus(component="dense", state="ready", detail="FAISS and encoder loaded")
        )
    except Exception:
        statuses.append(
            ComponentStatus(
                component="dense",
                state="unavailable",
                detail="dense runtime or local encoder load failed",
            )
        )
    try:
        rankers = load_rankers(store, _component_id(manifest, "rankers"))
        statuses.append(
            ComponentStatus(
                component="rankers", state="ready", detail="both LightGBM models loaded"
            )
        )
    except Exception:
        statuses.append(
            ComponentStatus(
                component="rankers", state="unavailable", detail="ranking model load failed"
            )
        )
    if sparse is None and dense is None:
        product_store.close()
        raise ServingUnavailableError("no retriever could be loaded from the verified bundle")
    if not config.config.serving.allow_degraded_retrieval and (sparse is None or dense is None):
        if sparse is not None:
            sparse.close()
        if dense is not None:
            dense.close()
        product_store.close()
        raise ServingUnavailableError("degraded retrieval is disabled")
    return ServingRuntime(
        config=config,
        artifact=artifact,
        manifest=manifest,
        product_store=product_store,
        feature_state=state,
        sparse=sparse,
        dense=dense,
        rankers=rankers,
        component_statuses=tuple(statuses),
    )


__all__ = [
    "ComponentStatus",
    "FallbackEvent",
    "RuntimeInfo",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchTimings",
    "ServingBusyError",
    "ServingRequestError",
    "ServingRuntime",
    "ServingRuntimeError",
    "ServingUnavailableError",
    "load_serving_runtime",
]
