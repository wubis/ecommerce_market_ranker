# Consolidated Goldfish Roadmap

## Purpose

This document defines the preferred Goldfish sizing and remaining implementation sequence for
MarketRank. It supplements the milestone order in `ELEPHANT.md` without changing the end-state
architecture.

The project is intentionally moving from small infrastructure Goldfishes toward larger,
subsystem-sized Goldfishes. Each remaining Goldfish should still leave the repository working,
tested, documented, reproducible, and independently reviewable.

## Recommended Finish Line

| Scope | Target |
|---|---:|
| Current implemented sequence | Through Goldfish 014 |
| Core portfolio-ready project | Goldfish 016 |
| Remaining core Goldfishes | Approximately 2 |
| Optional neural and diversity extensions | Goldfish 017–018 |

Goldfish 004A is an additional completed extension even though it does not advance the primary
integer numbering. The recommended plan therefore concerns the remaining numbered stages and
does not attempt to renumber prior work.

The expanded plan targeted roughly ten subsystem-sized stages beginning with Goldfish 007. Eight
of those stages are now implemented and two core stages remain. That sizing keeps dense
retrieval, features, training, evaluation, serving, demo, and qualification independently
reviewable without returning to the earlier fragmentation.

## Remaining Core Sequence

### Goldfish 007 — Sparse Retrieval and Evaluation Foundation

**Status: Implemented.**

Implement persisted BM25 indexing, deterministic tokenization, document/catalog alignment,
reload parity, top-K retrieval, explicit-pair scoring, resource measurement, and the
protocol-safe ranking/retrieval metric primitives needed to evaluate the baseline.

### Goldfish 008 — Dense Retrieval and FAISS

**Status: Implemented.**

Implement pinned MiniLM product embeddings, memory-bounded batch generation, normalized vector
storage, ordered ID alignment, FAISS CPU build/load/search, explicit-pair scoring, latency and
RSS measurements, and restart/reload parity.

### Goldfish 009 — Hybrid Retrieval and Retrieval Evaluation

**Status: Implemented.**

Implement deterministic RRF union/deduplication, source score/rank/provenance preservation,
fixed-catalog candidate generation, fair sparse/dense/hybrid comparisons, query-level
confidence intervals, retrieval slices, and the combined sparse+dense memory gate.

### Goldfish 010 — Query Understanding and Ranking Features

**Status: Implemented.**

Implement the deterministic query parser, versioned feature registry, `ltr_core_v1`, bounded
candidate/pool feature materialization, leakage checks, distribution reports, persisted feature
state, and offline/online formula parity fixtures.

### Goldfish 011 — Pointwise and LambdaMART Rankers

**Status: Implemented.**

Construct the exact judged training populations and group arrays, train pointwise and
LambdaMART models, enforce official gains, support early stopping, persist models and training
lineage, and verify serialization/reload prediction parity.

### Goldfish 012 — Ranking Evaluation and Champion Selection

**Status: Implemented.**

Implement closed-pool and end-to-end ranking evaluation, required ablations, query-level
bootstrap intervals, slices and failure analysis, protocol identifiers, experiment records,
and validation-only champion selection with one promoted active-relevance contract.

### Goldfish 013 — Serving Bundle, Orchestration, and FastAPI

**Status: Implemented.**

Promote an explicit compatible relevance bundle, implement the online search orchestrator,
load persisted assets without rebuilding or downloading, expose validated FastAPI contracts,
support readiness and degraded modes, and add bounded API integration tests.

### Goldfish 014 — Streamlit Portfolio Demo

**Status: Implemented.**

Implement the API-backed Streamlit client, ranking-mode comparisons, product cards, bounded
debug explanations, rank-change/provenance displays, latency visibility, limitations, and a
reliable local demonstration workflow.

### Goldfish 015 — Hardening and M3/8 GB Qualification

Complete regression, corruption, security, offline-startup, and end-to-end tests; measure local
latency and peak RSS; resolve resource blockers; document runbooks, recovery, limitations, and
reproduction; and prepare a release candidate on the reference Apple M3 with 8 GB memory.

### Goldfish 016 — Frozen Portfolio Experiments and Final Report

Run the frozen development/portfolio workflow, produce final baseline and ablation tables,
record hardware/config/code/artifact lineage, capture demo screenshots, write the scientific
and engineering narrative, disclose negative results honestly, and verify clean reproduction.

## Optional Extensions

### Goldfish 017 — Neural Reranking

Add a bounded pinned cross-encoder, cache and batch controls, head-only scoring, explicit active
relevance promotion, comparability rules, latency/resource bounds, failure fallback, and the
required ablation. This stage must remain optional and cannot destabilize the core champion.

### Goldfish 018 — Diversity Reranking

Add deterministic relevance-guarded brand/semantic diversification, missing-brand behavior,
cap relaxation, complete rank lineage, disabled-mode exact parity, relevance-loss guardrails,
diversity evaluation, and an optional promoted configuration.

## Goldfish Sizing Rules

A larger Goldfish may combine work when all parts share one coherent artifact boundary and can
be accepted together. Good combinations include index build plus reload/search/pair parity, or
model training plus serialization parity. Avoid combining unrelated lifecycle boundaries merely
to reduce the task count.

Each Goldfish must still:

1. cite the relevant Elephant decisions and define explicit in/out-of-scope behavior;
2. use typed configuration and reusable package modules with a thin CLI;
3. persist outputs through immutable artifact lineage where applicable;
4. include deterministic fixture tests and failure-path coverage;
5. remain viable on the Apple M3 with 8 GB memory;
6. perform no implicit downloads or startup-time builds;
7. run the lock, Ruff, strict mypy, pytest, and pre-commit gates;
8. leave the repository usable and changes uncommitted unless explicitly requested otherwise.

## Planning Guidance

Goldfish numbers are planning boundaries, not a mandate to force predetermined scope. If one
stage reveals a genuine resource or scientific blocker, split it only where an independently
useful, testable artifact boundary exists. Conversely, combine adjacent tasks when their
separation would create temporary formats or duplicated validation that have no enduring role.

The default target is therefore **Goldfish 016 for the core portfolio-ready system**, with
**Goldfish 017–018 treated as optional enhancements**.
