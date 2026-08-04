# Goldfish 013 — Serving Bundle, Orchestration, and FastAPI

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 013 |
| Parent design | `ELEPHANT.md`, Sections 9, 11, 14, 17, 22, 29, 34–35, 38, 41–42, 47 |
| Milestone | M10 — Explicit offline relevance runtime and validated local API |

## Objective

Promote one immutable, explicitly addressed relevance bundle from compatible Goldfish 006–012
artifacts, load it without building or downloading, execute the shared online ranking formulas,
and expose a bounded local FastAPI service with readiness and degradation contracts.

```bash
uv run market-rank serving promote --profile portfolio
uv run market-rank serving run --bundle-id \
  serving-bundle/<dataset-version>/portfolio/serving-bundle-v1/<config-sha256>
```

The run command requires the complete bundle ID. `latest`, discovery aliases, startup builds,
and startup downloads are prohibited.

## Immutable Serving Bundle

The direct bundle dependencies are the exact foundation, sparse index, dense index, ranking
feature, ranking-model, and ranking-evaluation artifacts. Recursive artifact verification still
covers their complete parent graph. `serving-bundle.json` repeats the serving-critical facts:

- catalog membership and product-document identities;
- ordered `ltr_core_v1` features plus registry, feature-state, and parser-state hashes;
- exact sparse and dense retriever identities;
- the complete Goldfish 012 active-relevance and RRF-fallback contract;
- explicit-only addressing, offline-startup, readiness, and degradation policies;
- measured promotion RSS and successful compatibility checks.

`products.sqlite3` is an indexed projection of only fixed-catalog US products. Null catalog
attributes become empty display/feature strings while independent missingness flags are
preserved. Runtime reads it in SQLite immutable/query-only mode with bounded candidate-key
lookups. No pickle or full product-table startup collection is permitted.

```text
serving-bundle/<dataset-version>/<profile>/serving-bundle-v1/<config-sha256>/
├── products.sqlite3
├── serving-bundle.json
├── manifest.json
└── _SUCCESS
```

Compatible promotion reuses the bundle before reading product rows. Initial or post-projection
RSS overage rolls staging back and cannot publish `_SUCCESS`.

## Runtime Orchestration

Startup recursively verifies the explicit bundle, opens the product projection, restores the
persisted parser/category state, memory-maps BM25, loads FAISS with an exact locally cached
MiniLM encoder, and cold-loads both LightGBM models. It invokes no builder and allows no network
fallback.

Each request performs strict parsing, independent bounded BM25 and dense retrieval,
deterministic RRF union, bounded product lookup, direct pair feature computation when needed,
stable model scoring or declared fallback, and bounded display projection. Modes are `active`,
`bm25`, `dense`, `hybrid`, `pointwise`, and `lambdamart`. `active` resolves to exactly the
Goldfish 012 promoted stage.

A learned stage runs only with both retrieval feature sources and a valid ranker. Ranker
failure, missing model/source state, or a deadline exhausted before scoring resolves to RRF when
configured. One failed retriever leaves the runtime ready in degraded mode and uses the other
source; two failed retrievers block readiness. Neural reranking and diversification flags
produce explicit `not_promoted` fallback events without silently changing results.

Responses distinguish requested mode, promoted stage, resolved stage, active-score
comparability, degradation, fallback reasons, candidate count, display fields, source
scores/ranks/index IDs, RRF evidence, and stage timings. Raw queries, local paths, traces, labels,
and test judgments are absent.

## FastAPI Contract

The local server binds only to `127.0.0.1` by checked-in configuration and exposes:

| Endpoint | Contract |
|---|---|
| `GET /health/live` | Process liveness even when startup loading fails |
| `GET /health/ready` | `200` with component state or `503` without a relevance path |
| `POST /v1/search` | Validated bounded search request and typed result response |
| `GET /v1/model-info` | Active contract and feature identities without paths |
| `GET /v1/artifact-info` | Bundle/component lineage without filesystem details |
| `POST /v1/debug/explain` | Local, bounded ordered feature values when enabled |

Pydantic rejects unknown fields and invalid modes/types. Configuration caps response top-K,
query characters/bytes/tokens, request-body bytes, deadline, concurrent work, debug candidates,
description snippets, and product-store batch rows. Saturation returns `429`; semantic query
errors return `422`; oversized declared bodies return `413`; unavailable relevance returns
`503`. Startup failures become safe readiness state rather than paths or tracebacks.

## M3/8 GB Resource Contract

The serving bundle and runtime retain the 5,632 MiB process RSS ceiling. BM25 uses memory maps,
dense vectors use NumPy memory mapping, FAISS is CPU/thread bounded, the product store is queried
by candidate key, response top-K defaults to 10 and caps at 50, and request concurrency defaults
to two. Goldfish 015 owns measured release-candidate latency and memory qualification on the
physical Apple M3/8 GB reference machine.

## Out of Scope

- Streamlit UI, screenshots, and browser demonstration workflow;
- production auth, TLS, public binding, multi-worker deployment, or remote artifact storage;
- optional neural reranking or diversity algorithms;
- project-test evaluation, frozen portfolio claims, or final performance qualification;
- implicit model/data acquisition or runtime artifact repair.

## Acceptance Criteria

1. Promotion pins six exact compatible components and the complete active contract.
2. The bundle is immutable, explicit-only, recursively verified, and has no `latest` path.
3. Product projection contains exactly fixed-catalog products and preserves missingness semantics.
4. Compatible promotion reuses before product reads; corruption and RSS failure fail closed.
5. Startup performs no artifact build, training, embedding generation, or download.
6. Online model features use authoritative parser state, category codes, order, and pair formulas.
7. All six ranking modes have deterministic score/rank and product-ID tie behavior.
8. Active mode resolves to the exact validation-selected Goldfish 012 stage.
9. One retriever may degrade; two unavailable retrievers cannot become ready.
10. Learned-stage failure follows persisted RRF fallback and marks scores incomparable.
11. Optional unimplemented stages are explicit fallback events rather than silent no-ops.
12. API schemas bound bodies, queries, top-K, deadlines, concurrency, and debug payloads.
13. Health, search, metadata, and local debug endpoints return no paths, traces, or labels.
14. Fixture tests cover bundle, runtime, degradation, API, corruption, resources, reuse, and CLI.
15. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove the `serving` configuration, serving package/CLI/tests, FastAPI/Uvicorn dependencies, and
this document. Goldfish 006–012 remain intact, although configuration-hashed artifacts must be
rebuilt after removing the Goldfish 013 fields.
