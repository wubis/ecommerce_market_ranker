# Goldfish 014 — Streamlit Portfolio Demo

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 014 |
| Parent design | `ELEPHANT.md`, Sections 8.6, 33–34, 41, 47, 51, D-012 |
| Milestone | M13 — Interactive, API-backed local portfolio demo |

## Objective

Provide a reliable local Streamlit workflow that compares bounded MarketRank search modes while
preserving the FastAPI architecture boundary. The UI is a thin localhost client: it validates
typed responses, computes presentation-only list diagnostics, and never imports or invokes model,
index, artifact, training, or builder code.

With one explicit Goldfish 013 bundle already running, launch the demo from a second terminal:

```bash
uv run market-rank demo check
uv run market-rank demo run
```

The API remains on `127.0.0.1:8000`; Streamlit binds to `127.0.0.1:8501`. Neither service is
published to the LAN or internet.

## Architecture Boundary

The serving request, response, health, and metadata schemas now live in the lightweight
`market_rank.serving.contracts` module. FastAPI, the serving orchestrator, and the demo client
share those strict Pydantic contracts without forcing the client process to import FAISS,
LightGBM, sentence-transformers, artifacts, or runtime builders.

`DemoApiClient` accepts only uncredentialed loopback HTTP origins, uses a bounded timeout,
disables redirects, maps connection/status/schema failures to safe reason codes, and limits a
comparison to unique configured modes. It exposes health, model/artifact metadata, normal search,
and bounded debug search calls, but it contains no relevance logic.

The Streamlit process performs no startup build, model download, artifact discovery, implicit
bundle selection, or direct filesystem read of serving artifacts. The API owns all runtime state.

## Demonstration Contract

The page provides:

- checked-in ESCI-compatible examples plus a custom search box;
- bounded top-K and comparison-mode selection;
- active, BM25, dense, hybrid RRF, pointwise, and LambdaMART modes;
- optional neural/diversity request flags with explicit API fallback explanations;
- official product ID, title, description snippet, brand, and color cards;
- retrieval rank/score provenance, resolved stage, fallback reason, and latency visibility;
- signed movement from the first selected mode, where positive delta means movement toward rank 1;
- optional bounded model feature values through `/v1/debug/explain`;
- bundle, catalog, config, and query identifiers for reproduction;
- readiness/degraded banners and an always-available limitations note.

The comparison requires equal query, bundle, catalog, and config identities. Incompatible or
duplicate responses fail closed instead of producing misleading cross-run rank deltas.

## Honest Presentation Metrics

Unique-brand count, missing-brand count, maximum brand concentration, brand entropy, and pairwise
title-token intra-list dissimilarity describe only the displayed result list. They are not
relevance judgments, semantic-quality estimates, fairness measurements, or business outcomes.
Goldfish 014 shows no arbitrary-query relevance metric because ESCI judgments are bounded and
unknown products must not be treated as irrelevant.

The limitations panel also states that the fixed ESCI research catalog is not live Amazon search
and lacks authoritative price, inventory, seller, shipping, review, sponsorship, conversion,
returns, margin, product-age, and canonical-category fields.

## Configuration and Resource Contract

`DemoConfig` pins the API and UI loopback addresses, five-second HTTP timeout, default top-K 10,
at most four compared modes, at most twelve product cards, and a small example-query set. Root
validation prevents demo result/card bounds from exceeding the serving response bound.

The client retains only bounded JSON responses in Streamlit session state. It does not duplicate
the API's indexes or models, making the second process suitable for the Apple M3/8 GB reference
machine. Goldfish 015 still owns measured combined latency and peak-RSS qualification.

## Verification

- Unit tests cover loopback enforcement, typed response validation, status mapping, comparison
  bounds, transparent list metrics, signed rank changes, incompatible lineage, and CLI launch.
- An isolated import test proves the demo client does not load artifact, dense-retrieval, ranking,
  or serving-orchestrator modules.
- Streamlit's headless application harness verifies the API-unavailable page, controls, disabled
  submit behavior, and always-visible access to limitations.
- Existing API schemas remain import-compatible through `market_rank.serving.api` and
  `market_rank.serving.orchestrator` while sharing the new contract source.

## Out of Scope

- public hosting, authentication, TLS, LAN binding, telemetry, or multi-user operation;
- direct model/index/artifact access from Streamlit;
- product images or fields absent from the serving contract;
- generated relevance labels or claims about arbitrary searches;
- final screenshots, measured M3/8 GB qualification, or frozen portfolio results;
- implementation of neural reranking or diversity algorithms.

## Acceptance Criteria

1. The demo launches through a checked-in CLI command and depends only on the local API.
2. API and UI origins are loopback-only and request/mode/result state is bounded.
3. Ranking modes, rank changes, provenance, fallbacks, timings, and lineage are inspectable.
4. Cards use only fields supplied by the validated serving response.
5. Debug feature values are opt-in and bounded by the API contract.
6. Presentation metrics are labeled as non-relevance diagnostics.
7. Dataset limitations remain accessible even when the API is unavailable.
8. The UI handles unavailable, degraded, invalid-response, and search-error states safely.
9. Import isolation prevents Streamlit from loading the model/artifact runtime.
10. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove the `demo` package/configuration/CLI/tests, Streamlit and HTTP-client runtime dependencies,
the shared contracts extraction, and this document. Restore the equivalent schema definitions in
the API/orchestrator modules. Goldfish 013 remains independently operable, although configuration-
hashed artifacts must be rebuilt after removing the Goldfish 014 fields.
