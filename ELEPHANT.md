# MarketRank Technical Design Document

| Field | Value |
|---|---|
| Document status | **Proposed — Elephant review required before implementation** |
| Intended audience | ML engineers, applied scientists, software engineers, reviewers, and future Goldfish authors |
| Authors | _TBD_ |
| Last updated | _YYYY-MM-DD_ |
| Repository | `ecommerce_market_ranker` |
| System | MarketRank: Multi-Stage Marketplace Search and Ranking System |
| Required reference hardware | Apple M3 Mac with 8 GB unified memory; CPU-first, no CUDA |
| Decision authority | This document is the end-state technical blueprint. Approved Goldfish documents may clarify implementation details but must not silently redesign it. |

> **Truth and provenance statement.** ESCI query, product, locale, and relevance fields are real dataset fields. Marketplace attributes and outcomes in this project are deterministic simulations for systems demonstration only. They are not Amazon seller, price, conversion, inventory, margin, shipping, or risk data and must never be described as such.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Motivation](#2-background-and-motivation)
3. [Scope](#3-scope)
4. [Non-Goals](#4-non-goals)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Success Criteria](#7-success-criteria)
8. [Data Sources](#8-data-sources)
9. [Data Model](#9-data-model)
10. [Synthetic Marketplace Metadata Design](#10-synthetic-marketplace-metadata-design)
11. [Data Splitting Strategy](#11-data-splitting-strategy)
12. [Data Processing Pipeline](#12-data-processing-pipeline)
13. [Query Understanding](#13-query-understanding)
14. [Candidate Retrieval](#14-candidate-retrieval)
15. [Hybrid Retrieval](#15-hybrid-retrieval)
16. [Embedding Pipeline](#16-embedding-pipeline)
17. [Feature Engineering](#17-feature-engineering)
18. [Baseline Ranking Methods](#18-baseline-ranking-methods)
19. [Learning-to-Rank Design](#19-learning-to-rank-design)
20. [Training Strategy](#20-training-strategy)
21. [Optional Neural Reranking](#21-optional-neural-reranking)
22. [Marketplace Optimization](#22-marketplace-optimization)
23. [Marketplace Constraints](#23-marketplace-constraints)
24. [Personalization](#24-personalization)
25. [Offline Evaluation](#25-offline-evaluation)
26. [Evaluation Methodology](#26-evaluation-methodology)
27. [Required Ablation Studies](#27-required-ablation-studies)
28. [Position Bias and Counterfactual Evaluation](#28-position-bias-and-counterfactual-evaluation)
29. [Explainability](#29-explainability)
30. [Local Storage and Artifact Management](#30-local-storage-and-artifact-management)
31. [Experiment Tracking](#31-experiment-tracking)
32. [Configuration Management](#32-configuration-management)
33. [API Design](#33-api-design)
34. [Streamlit Demo](#34-streamlit-demo)
35. [Repository Structure](#35-repository-structure)
36. [Module and Interface Design](#36-module-and-interface-design)
37. [Testing Strategy](#37-testing-strategy)
38. [Data Validation](#38-data-validation)
39. [Observability](#39-observability)
40. [Memory Management](#40-memory-management)
41. [Latency Targets](#41-latency-targets)
42. [macOS Compatibility](#42-macos-compatibility)
43. [Failure Modes and Fallbacks](#43-failure-modes-and-fallbacks)
44. [Security and Privacy](#44-security-and-privacy)
45. [Production-Scale Evolution](#45-production-scale-evolution)
46. [Alternatives Considered](#46-alternatives-considered)
47. [Risks and Mitigations](#47-risks-and-mitigations)
48. [Implementation Milestones](#48-implementation-milestones)
49. [Goldfish Decomposition Strategy](#49-goldfish-decomposition-strategy)
50. [Definition of Done](#50-definition-of-done)
51. [Portfolio Presentation](#51-portfolio-presentation)
52. [Future Work](#52-future-work)
53. [Open Questions](#53-open-questions)
54. [Decision Log](#54-decision-log)
55. [Recommended First Goldfish](#recommended-first-goldfish)

---

## 1. Executive Summary

MarketRank is a local, production-shaped search system that accepts a shopping query such as “wireless gaming mouse under $80” and returns a ranked, explainable list of products. The user problem is intent satisfaction: words, constraints, brands, attributes, and acceptable substitutes must be interpreted together. The marketplace problem is that a pure relevance order can over-concentrate sellers, expose unavailable inventory, ignore fulfillment quality, and starve new products. The ML problem is to learn a ranking function from grouped relevance judgments while preventing leakage and preserving a measurable boundary between relevance quality and simulated business policy.

The proposed system uses the Amazon Shopping Queries ESCI dataset as the authoritative relevance source. A lightweight parser extracts structured query signals. BM25 and compact Sentence Transformer retrieval independently generate candidates; reciprocal rank fusion (RRF) merges them. A feature builder combines query, product, retrieval, interaction, and clearly marked synthetic marketplace fields. LightGBM LambdaMART produces the primary learned ranking. A compact cross-encoder may rerank only the top 10–30 results and is optional. Finally, a deterministic greedy policy removes ineligible products and trades a bounded amount of relevance for seller diversity, quality, inventory, and new-product exposure.

The local implementation is deliberately scaled to an Apple M3 Mac with 8 GB unified memory: English-only query-group samples, Parquet and DuckDB, a persisted sparse index, `all-MiniLM-L6-v2`, FAISS CPU, compact feature matrices, bounded candidate pools, local MLflow files, FastAPI, and Streamlit. Offline stages run sequentially, use memory mapping and conservative batches, and do not assume that training and serving artifacts can coexist in memory. The architecture preserves production concepts—offline/online separation, versioned artifacts, stage contracts, fallbacks, observability, evaluation gates—without paid services, CUDA, or transformer fine-tuning.

This is valuable as a portfolio system because it demonstrates more than model fitting. It joins information retrieval, learning-to-rank, deterministic data engineering, multi-objective policy simulation, API serving, UI comparison, experiment design, and honest limitations. The intended final deliverable is a reproducible repository that builds persisted artifacts, runs fixed offline evaluations and ablations, starts a local API without recomputation, and presents results interactively. This document defines that end state; it does not implement it.

### 1.1 End-to-end architecture

```text
                                    OFFLINE PLANE
  ESCI Parquet/CSV ──> Validate ──> Locale + group sampling ──> Normalized tables
          │                                    │                       │
          │                                    ├──> synthetic metadata │
          │                                    │    (seeded, separate) │
          │                                    v                       v
          │                              Product documents ──> BM25 index
          │                                    │             + FAISS index
          │                                    v
          └────────────────────────────> Candidate generation
                                               │
                                               v
                                     Feature Parquet / matrices
                                               │
                                               v
                                   LambdaMART train + evaluation
                                               │
                          experiment logs <────┴────> model/artifact registry

                                     ONLINE PLANE
  User/Streamlit ──HTTP──> FastAPI ──> Query parser ──> BM25 + FAISS
                               │                              │
                               │                         RRF + dedupe
                               │                              │
                               │                     online feature build
                               │                              │
                               │                       LambdaMART score
                               │                              │
                               │                  [optional cross-encoder]
                               │                              │
                               │                   marketplace policy
                               │                              │
                               └──── JSON results <── explanations + metrics
                                      │
                                      └── structured local logs / caches

  Persisted boundary: processed data, indexes, embeddings, encoders, model,
  policy configuration, manifests, reports. API startup loads; it never rebuilds.
```

### 1.2 Architectural principles

1. **Relevance truth is never simulated.** ESCI labels remain the only relevance targets.
2. **Provenance is a first-class field.** Source, derived, simulated, and predicted values have different namespaces and documentation.
3. **Retrieve broadly, rank precisely, optimize transparently.**
4. **Offline and online computations share feature definitions.**
5. **Every optional stage has a deterministic fallback.**
6. **Measured trade-offs replace business claims.**
7. **Artifacts are immutable, content-addressed, and loadable.**
8. **The required reference machine is an Apple M3 Mac with 8 GB unified memory; production scale is only an analogue.**

## 2. Background and Motivation

Marketplace search is difficult because “relevance” is not one relation. Exact products satisfy the expressed need; substitutes satisfy most intent by a different item; complements may be useful but should usually not displace exact items; irrelevant items must be suppressed. Lexical matching catches model numbers and exact brands but misses paraphrases. Semantic matching connects concepts but can blur critical constraints such as “case for” versus “phone,” negation, size, compatibility, and price.

Catalog text is noisy and uneven. Titles may be keyword-heavy, descriptions missing, colors inconsistent, and brands aliased. The same product can appear across query judgments. Queries are short, ambiguous, misspelled, and frequently contain latent category or attribute constraints. Retrieval must keep enough relevant products for downstream ranking, while ranking must compare products only within a query group.

Marketplace objectives introduce another layer. Popular products can dominate exposure, a single seller can occupy the page, cold-start items lack history, and a relevant product may be unavailable or high-risk. However, optimizing synthetic conversion or margin as though it were observed behavior would be misleading. MarketRank therefore treats marketplace reranking as a policy simulation whose effect on ESCI relevance is explicitly measured.

Latency and resource budgets couple all decisions. A cross-encoder over the whole catalog is impossible locally; a compact cross-encoder over 20 finalists is feasible. Exact dense search over a portfolio-sized corpus may be simpler and more reproducible than approximate search. Persisted embeddings and indexes shift work offline.

BM25 alone is insufficient because it cannot naturally represent semantic equivalence and may over-reward repeated terms. Cosine similarity alone is insufficient because it can miss rare exact identifiers, has no marketplace context, and produces a single opaque score. Neither alone learns how title overlap, brand match, dense similarity, price compatibility, and candidate provenance interact. A multi-stage architecture lets each method solve the problem it is suited to and exposes measurable stage-level failures.

## 3. Scope

### 3.1 Required

- Official ESCI file ingestion, license/attribution, schema validation, and documented download instructions.
- Query-group-preserving English (`us`) development and portfolio profiles.
- Normalized query, product, and judgment tables plus denormalized training artifacts.
- Deterministic, separately stored synthetic marketplace metadata.
- Lightweight query normalization and attribute/constraint extraction without an LLM.
- Persisted BM25-compatible sparse retrieval and FAISS CPU dense retrieval.
- Hybrid candidate union, deduplication, RRF, filters, and retrieval evaluation.
- Shared offline/online feature definitions.
- Random, sparse, dense, hybrid, heuristic, pointwise, and LambdaMART baselines.
- CPU-feasible LightGBM LambdaMART training and fixed offline evaluation.
- Deterministic marketplace eligibility and constrained reranking.
- Optional top-10–30 cross-encoder reranking with graceful fallback.
- Query-level confidence intervals, slices, ablations, latency, and memory reports.
- File-backed experiment tracking, optional local MLflow UI, and artifact manifests.
- FastAPI inference and a Streamlit comparison demo.
- Unit, integration, smoke, regression, data, determinism, and API tests.
- macOS-first setup, locked dependencies, quality gates, and complete documentation.

### 3.2 Optional but designed

- Neural reranking.
- MPS acceleration when compatible.
- Full benchmark profile.
- Simulated position-bias/counterfactual notebook.
- Cold-start product stress split.
- Docker image for portability.

### 3.3 Implementation boundary

Goldfish tasks may create the structure specified in Section 35, but this Elephant creates only the design document. No source files, configs, generated artifacts, dataset copies, model weights, or notebooks belong in this design task.

## 4. Non-Goals

The first complete version will not provide:

- real-time distributed search or Amazon-scale indexing;
- claims of representing Amazon’s live search system;
- transformer, dual-encoder, or cross-encoder fine-tuning;
- reinforcement learning, online learning, or learning from live users;
- real A/B tests, revenue optimization, conversion prediction, or seller-fairness guarantees;
- production cloud deployment, Kubernetes, Spark, distributed feature stores, or managed model registries;
- LLM APIs, paid embeddings, hosted vector databases, or paid experiment tracking;
- GPU-required training or inference;
- personalization, because ESCI has no user histories;
- image retrieval, sponsored-auction optimization, checkout logic, or seller payouts;
- exhaustive hyperparameter search;
- inference over products outside the selected local catalog;
- a claim that unjudged catalog products are irrelevant.

## 5. Functional Requirements

### 5.1 Requirements table

| ID | Requirement | Acceptance evidence |
|---|---|---|
| FR-001 | Ingest the three official ESCI files and record checksums, source URL, dataset version, and license. | Validated raw manifest and row/schema report. |
| FR-002 | Normalize queries, products, judgments, and sources without changing authoritative labels. | Key/row reconciliation report. |
| FR-003 | Build seeded development, portfolio, and optional full profiles by complete query groups. | Same config produces identical query IDs and hash. |
| FR-004 | Filter required profiles to `product_locale=us`; reject unsupported mixed-locale joins. | Locale invariant test. |
| FR-005 | Create deterministic product documents with explicit handling of missing fields. | Golden document tests and versioned template. |
| FR-006 | Generate synthetic marketplace metadata independently of ESCI labels. | Determinism and no-label-dependency tests. |
| FR-007 | Parse normalized query text, numbers, price constraints, brand, color, and category hints. | Curated query fixture tests. |
| FR-008 | Build, persist, load, and query a sparse index. | Reload parity and Recall@K report. |
| FR-009 | Encode products offline and persist normalized embeddings plus a FAISS CPU index. | Manifest, dimension/checksum validation, reload test. |
| FR-010 | Compute query embeddings online without re-encoding products. | Startup and request profiling. |
| FR-011 | Retrieve BM25, dense, and hybrid top-K candidates with provenance. | Unique candidate keys and per-source ranks. |
| FR-012 | Deduplicate by product, enforce catalog eligibility, and deterministically break ties. | Invariant and tie fixture tests. |
| FR-013 | Compute versioned query, product, retrieval, interaction, and marketplace features offline and online. | Offline/online parity test. |
| FR-014 | Train and serialize a pointwise baseline and LightGBM LambdaMART with query groups intact. | Model manifest, group validation, held-out metrics. |
| FR-015 | Score candidates and preserve stage-by-stage scores and ranks. | Prediction schema and rank trace. |
| FR-016 | Optionally cross-encode only the configured top 10–30 candidates. | Disabled-path and bounded-count tests. |
| FR-017 | Apply hard eligibility constraints and deterministic marketplace policy reranking. | Constraint audit per query. |
| FR-018 | Compute retrieval, ranking, marketplace, latency, memory, slice, and bootstrap outputs. | Machine-readable and Markdown reports. |
| FR-019 | Track each experiment’s data, code, config, model, metrics, and artifact lineage locally. | Reconstructable run directory. |
| FR-020 | Serve health, search, model, artifact, and debug endpoints from persisted artifacts. | API contract tests. |
| FR-021 | Provide a Streamlit UI comparing ranking modes and clearly label synthetic data. | Demo smoke test and screenshots. |
| FR-022 | Explain non-causally why a result scored/ranked as it did and show candidate provenance. | Explanation response fixtures. |
| FR-023 | Degrade to the highest available relevance pipeline when optional artifacts fail. | Failure injection tests. |
| FR-024 | Reject incompatible artifact bundles before accepting traffic. | Startup compatibility tests. |
| FR-025 | Provide reproducible CLI entry points for each offline stage without notebooks being required. | End-to-end development-profile smoke run. |

## 6. Non-Functional Requirements

| ID | Requirement | Target or verification |
|---|---|---|
| NFR-001 | Zero monetary cost | No credential or paid-service requirement. |
| NFR-002 | CPU-first macOS support | Required workflow passes on the 8 GB Apple M3 reference machine and is documented for other Apple Silicon and Intel Macs. |
| NFR-003 | 8 GB unified-memory envelope | Required development and lower-bound portfolio workflows target peak process RSS at or below 5.5 GB, leaving headroom for macOS; stages run sequentially and avoid unnecessary full joins/copies. Actual peak RSS is reported. |
| NFR-004 | Reproducibility | Seeds, input checksums, config hash, dependency lock, and code revision are recorded. |
| NFR-005 | Determinism | Repeated same-environment runs produce identical sampled IDs, synthetic data, candidates, and metrics within declared floating tolerance. |
| NFR-006 | Modularity | Retrieval, features, ranker, neural stage, policy, and serving depend on typed contracts, not each other’s internals. |
| NFR-007 | Testability | Core logic has toy fixtures; slow/model/download tests are marked separately. |
| NFR-008 | Observability | Structured stage timing, counts, versions, cache status, and errors. |
| NFR-009 | Fast startup | API loads validated artifacts; no download, training, embedding, or index build at startup. |
| NFR-010 | Caching | Query parsing/embeddings and optional cross-encoder scores use bounded version-aware caches. |
| NFR-011 | Persistence | Every expensive stage writes an atomic artifact and `_SUCCESS`/manifest marker. |
| NFR-012 | Local latency | Meet Section 41 benchmark targets on the declared reference machine where feasible; report actuals. |
| NFR-013 | Maintainability | `pyproject.toml`, Ruff, pytest, type checks on public boundaries, pre-commit, and concise docs. |
| NFR-014 | Graceful degradation | Optional component failures never force irrelevant or policy-only ordering. |
| NFR-015 | Auditability | Each final result exposes source ranks, ranker score, policy actions, and provenance classes in debug mode. |
| NFR-016 | Safe artifacts | Avoid arbitrary untrusted pickle loading; allowlist roots and verify manifests/checksums. |
| NFR-017 | Offline operation | After initial dataset/model/dependency downloads, required search, evaluation, API, and demo work without internet. |
| NFR-018 | Configuration discipline | No experiment-defining constant is embedded only in source code. |

## 7. Success Criteria

Completion is comparative, not a promise of invented metric values. A benchmark report must include point estimates, query-level 95% bootstrap confidence intervals, and the exact profile/hardware/config.

| Dimension | Primary measure | Success gate |
|---|---|---|
| Candidate coverage | judged-relevant Recall@100 | Hybrid exceeds or statistically ties the better single retriever and strictly improves at least one primary relevance metric on validation or test. |
| Early retrieval | MRR@10, NDCG@10 | Report BM25, dense, and hybrid under identical judged queries. |
| Learned ranking | NDCG@10 and NDCG@20 | LambdaMART outperforms raw hybrid ordering on at least one primary held-out metric without a material unexplained regression on the other. |
| Robustness | per-query/slice metrics | No result is presented only as a global average; include head/tail, label, and category-proxy slices. |
| Policy diversity | unique sellers@10, HHI@10, max seller share | Policy improves at least one diversity measure on eligible test queries. |
| New products | exposure@10 | Policy moves exposure toward the configured target on feasible queries. |
| Relevance preservation | ΔNDCG@10 vs pre-policy | Mean and worst-slice loss stay within the configured validation-selected budget; exact-item preservation violations are zero unless infeasible and audited. |
| Eligibility | stock and constraint violations | Zero out-of-stock results and zero seller-cap violations after policy, unless a documented feasibility fallback applies. |
| Performance | per-stage and total latency | Actual p50/p95 compared with Section 41 targets; no hidden index rebuild. |
| Memory | peak RSS by workflow | Development and the lower-bound portfolio profile complete within the 8 GB machine envelope, targeting ≤5.5 GB process RSS; larger portfolio variants are optional and must not replace the required result. |
| Reproducibility | artifact/config/result hashes | Two clean same-environment development runs match deterministic artifacts and metric tolerances. |
| Reliability | tests and reload | Required test suites pass; persisted indexes and models reload with prediction parity. |

The report must also disclose failures. If hybrid or LambdaMART does not win, the project is still complete when evaluation is correct, causes are analyzed, and no unsupported claim is made.

## 8. Data Sources

### 8.1 Authoritative source

Use the [Amazon Science ESCI repository](https://github.com/amazon-science/esci-data) and cite the [Shopping Queries Dataset paper](https://arxiv.org/abs/2206.06588). The official repository describes a multilingual dataset with English, Spanish, and Japanese queries, two sizes, up to roughly 40 judged products per query, and an Apache-2.0 license. Its documented files are:

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`
- `shopping_queries_dataset_sources.csv`

The project must pin a retrieval date/release identifier and record SHA-256 checksums. Raw files remain immutable and are excluded from Git.

### 8.2 Official dataset schema

| File | Field | Type after normalization | Meaning / handling |
|---|---|---:|---|
| examples | `example_id` | `int64` | Unique judgment-row identifier; retain for traceability. |
| examples | `query` | `utf8` | Raw query text; canonical query table must verify consistency per `query_id`. |
| examples | `query_id` | `int64` | Ranking group identifier. |
| examples | `product_id` | `utf8` | Product identifier; joins with locale. |
| examples | `product_locale` | categorical/`utf8` | `us`, `es`, or `jp`; required profiles use `us`. |
| examples | `esci_label` | categorical | Authoritative `E`, `S`, `C`, or `I`; never overwritten. |
| examples | `small_version` | integer/bool | Official subset membership indicator; preserve raw semantics. |
| examples | `large_version` | integer/bool | Official larger-set membership indicator; preserve raw semantics. |
| examples | `split` | categorical | Official train/test designation; test remains untouched by model selection. |
| products | `product_id` | `utf8` | Product key component. |
| products | `product_locale` | categorical/`utf8` | Product key component; prevents cross-locale collisions. |
| products | `product_title` | `utf8?` | Source title; missing becomes empty only in derived document. |
| products | `product_description` | `utf8?` | Source description; preserve source null flag. |
| products | `product_bullet_point` | `utf8?` | Source attributes/bullets; preserve source null flag. |
| products | `product_brand` | `utf8?` | Source brand text; normalize only into a derived field. |
| products | `product_color` | `utf8?` | Source color text; normalize only into a derived field. |
| sources | `query_id` | `int64` | Query key. |
| sources | `source` | categorical/`utf8` | Dataset source partition; retain for analysis, never use as a target shortcut without review. |

The official files do **not** supply canonical category, seller, price, inventory, shipping, margin, conversion, return, cancellation, sponsorship, review, or popularity fields. Any such field is derived or simulated and must use the prefixes/conventions in Section 9.

### 8.3 Label semantics and use

Canonical graded mapping is `E=3`, `S=2`, `C=1`, `I=0`. It is config-versioned, used for LambdaMART and graded NDCG, and never described as universal business truth. Retrieval Recall@K defaults to `E` and `S` as relevant; alternate `E`-only and `E/S/C` views are reported because complements behave differently by intent. MRR’s relevant threshold must be named in the metric output.

### 8.4 Deduplication and missing values

- Product uniqueness key is `(product_locale, product_id)`.
- If duplicate product rows are identical after canonical null normalization, collapse and count them. Conflicts quarantine the product and fail strict builds.
- Judgment uniqueness key is `(query_id, product_locale, product_id)`. Conflicting labels are a hard validation failure; exact duplicates collapse only with an audit.
- Missing text fields remain null in normalized source tables. Derived document construction uses empty segments and missingness indicators.
- A product with no usable title, description, bullets, brand, or color is invalid for retrieval and reported.
- Query whitespace may be normalized in derived text, but raw text remains preserved.

### 8.5 Product document

The default versioned template is conceptually:

```text
[TITLE] title [BRAND] brand [COLOR] color [BULLETS] bullet text
[DESCRIPTION] truncated description
```

Field markers prevent accidental concatenation ambiguity. HTML is stripped, Unicode normalized with NFKC, whitespace collapsed, and control characters removed. Case is preserved in stored display text; retriever-specific tokenization may lowercase. Descriptions and bullets have configurable character/token caps to bound index size. No stemming or stopword policy changes the authoritative source fields.

### 8.6 Legal, attribution, and limitations

Retain `LICENSE`/`NOTICE` obligations in the eventual repository and cite Reddy et al. Raw data and pretrained weights are not committed unless their licenses explicitly permit it; provide download instructions. Product text can contain brands and catalog content and should be treated as research data, not republished as a new commercial dataset.

ESCI is a bounded judged-result dataset, not a complete live catalog with exhaustive judgments. Products absent from a query’s judged pool are **unjudged**, not known irrelevant. Open-catalog retrieval Recall@K is therefore computed against known judged relevant products and labeled “judged recall”; precision-like metrics over newly retrieved unjudged items are biased. Ranking comparisons should additionally use a fixed candidate pool drawn from the ESCI judgments to isolate ranker quality.

## 9. Data Model

### 9.1 Provenance classes

| Class | Namespace / metadata | Examples | May train relevance model? |
|---|---|---|---|
| Source | `src_*` or `provenance=esci` | query, title, brand, ESCI label | Yes, except label is target only. |
| Derived | `drv_*` or `provenance=derived` | normalized brand, token overlap, BM25 score | Yes if available online and leakage-reviewed. |
| Simulated | `sim_*` or `provenance=synthetic_vN` | price, seller rating, inventory | Only in named ablations; never called observed. |
| Prediction | `pred_*` or `provenance=model:<id>` | LambdaMART score, cross-encoder score | Downstream stages only; no same-fold target leakage. |
| Policy output | `policy_*` | eligibility reason, diversity penalty | No; audit/final presentation. |

### 9.2 Logical schemas and artifacts

| Entity / artifact | Purpose and key | Important fields and types | Relationships | Local storage / expected profile size | Production analogue |
|---|---|---|---|---|---|
| `queries` | One query; PK `query_id` | raw/normalized text `utf8`, locale `cat`, split `cat`, source `cat`, parser version | 1:N judgments/candidates | Parquet; 5k–50k required | Query/log warehouse |
| `products` | Canonical product; PK `(locale, product_id)` | source text nullable strings, derived document, missing flags | 1:N judgments; 1:1 synthetic metadata | Parquet; tens to hundreds of thousands | Catalog service/table |
| `judgments` | Ground truth; PK `(query_id, locale, product_id)` | `example_id int64`, `esci_label cat`, `grade int8`, official split | N:1 queries/products | Parquet; 50k–500k required | Label store |
| `marketplace_metadata` | Simulated product/seller state; PK product key | seller ID, price, risks, stock, margin, flags; provenance/version | N:1 seller; 1:1 product | Separate Parquet; one row/product | Offer/inventory services |
| `sellers` | Shared latent simulated seller state; PK `seller_id` | quality latent, rating, review count, fulfillment, shipping | 1:N products | Synthetic Parquet; thousands | Seller profile service |
| `query_parse` | Versioned parsed request; PK `(query_id/parser_version)` or request hash | tokens/list, price bounds, brand/color/category hints, confidence | 1:1 query/version | Parquet offline; LRU online | Query understanding service |
| `retrieval_candidates` | Candidate provenance; PK `(run_id, query_id, product_id)` | sparse/dense ranks/scores nullable, RRF, source bitset, union rank | N:1 query/product | Partitioned Parquet; roughly queries × 100–300 | Candidate service logs |
| `ranking_features` | Denormalized model matrix; same candidate key + feature set | compact numeric/categorical feature columns, label only in training artifact | Joins all prior entities | Parquet + NumPy/LightGBM matrix; potentially millions of rows | Offline/online feature store |
| `model_predictions` | Per-stage predictions; PK `(model_id, query_id, product_id)` | score `float32`, rank `int32`, prediction timestamp/version | N:1 candidate/model | Partitioned Parquet | Prediction log |
| `final_rankings` | User-facing order; PK `(request/run, query, final_rank)` | product, relevance rank, final score, policy actions, explanation IDs | Derived from predictions/policy | JSON online; Parquet evaluation | Search response/log |
| `evaluation_outputs` | Metric facts; composite PK `(run, stage, slice, metric, cutoff)` | value, CI bounds, query count, threshold definition | N:1 experiment | Parquet + JSON + Markdown | Metrics warehouse |
| `experiments` | Reproduction metadata; PK `run_id` | config/data/code hashes, seed, hardware, params, status, paths | Parent of run artifacts | JSON/YAML/MLflow file store | Experiment platform |
| `artifact_manifest` | Bundle contract; PK `artifact_id` | type, version, hashes, schema, dependencies, created UTC, code revision | DAG across artifacts | JSON sidecar per artifact | Model/artifact registry |
| `reranker_cache` | Optional score cache; PK `(model_hash, query_hash, product_id, text_hash)` | cross score, created UTC | Version-bound | SQLite | Distributed feature/cache store |

Normalized source tables preserve truth and minimize duplication. Candidate, feature, prediction, and final-ranking artifacts are intentionally denormalized for sequential scans, training, and reproducible evaluation. Labels must not be present in online feature payloads.

## 10. Synthetic Marketplace Metadata Design

### 10.1 Goals and constraints

The simulation exists to demonstrate joins, feature provenance, policy constraints, and multi-objective reporting—not to imitate confidential Amazon behavior. It must:

- depend only on the configured root seed, stable identifiers, source/derived non-label product properties, and documented distribution parameters;
- never read `esci_label`, grade, official split, retrieval score, or ranker output;
- be invariant to input row order, parallelism, and incremental regeneration;
- produce bounded, type-valid fields and a generation audit;
- separate seller-level variables from product-level variables;
- label every UI and report field as simulated.

Use SHA-256-based stable sub-seeds, not Python’s randomized `hash()`. For example, the product stream derives from `SHA256(generator_version || root_seed || locale || product_id)`, while seller streams derive from seller ID. This makes regeneration order-independent.

### 10.2 Generative graph

```text
 stable product ID ──> seller assignment ──> seller latent quality
       │                       │                    ├─ rating
       │                       │                    ├─ review count
       │                       │                    ├─ fulfillment
       │                       │                    └─ shipping
       ├─ category proxy ──> category priors ──> price / margin / return risk
       ├─ age ──> new flag ──> reviews / popularity / exploration eligibility
       └─ popularity latent ──> inventory / simulated conversion proxy

 quality + category + age + popularity ──> cancellation / return / inventory
                                           (never ESCI label)
```

### 10.3 Generation rules

| Field | Type / bounds | Proposed deterministic distribution and correlation | Caveat |
|---|---|---|---|
| `seller_id` | string | Stable hash bucket with a long-tail product-count allocation; seller table generated once. | Artificial identity. |
| `sim_category` | category | Prefer a deterministic, versioned text taxonomy classifier; otherwise stable category-proxy assignment conditioned on title tokens. | Not an ESCI source category. |
| `product_price` | float, `>0` | Category-conditioned log-normal, clipped to documented plausible bounds and rounded to cents. | Synthetic price. |
| `seller_rating` | float `[1,5]` | Transform seller quality latent through a beta-like distribution concentrated above 3; round to 0.1. | Synthetic reputation. |
| `seller_review_count` | int `>=0` | Zero-inflated log-normal/negative-binomial-like draw, scaled by seller age/quality. | Not observed reviews. |
| `seller_fulfillment_rate` | float `[0,1]` | Logistic transform of seller quality plus seeded noise; positively correlated with rating. | Policy proxy. |
| `expected_shipping_days` | int `[1,14]` | Ordered draw decreasing with fulfillment/quality and varying by category. | No geography. |
| `return_probability` | float `[0,1]` | Logistic function of category prior, price z-score, and inverse quality plus noise; clip e.g. `[0.01,0.60]`. | Not a calibrated risk model. |
| `cancellation_probability` | float `[0,1]` | Logistic function of inverse fulfillment, low inventory, and noise; clip. | Synthetic. |
| `inventory_count` | int `>=0` | Zero-inflated log-normal/negative-binomial-like draw, increasing with popularity; explicit stockout mass. | Snapshot, not temporal truth. |
| `profit_margin` | float `[0,1]` | Category beta prior with small seller/product noise; clip to policy range. | Not currency profit. |
| `product_age_days` | int `>=0` | Mixture: recent-products component plus long-tail log-normal/exponential age. | Synthetic age. |
| `is_new_product` | bool | `product_age_days <= configured_days` (default candidate: 90). | Definition is configurable. |
| `is_sponsored` | bool | Low base-rate Bernoulli conditioned on seller/product attributes, but ignored by required relevance/policy scoring. | Demonstrative display only. |
| `historical_popularity` | float `[0,1]` | Percentile of latent demand influenced by category and age; shrink young items toward prior. | Synthetic proxy. |
| `estimated_conversion_rate` | float `[0,1]` | Logistic proxy of popularity, quality, shipping, return risk, and relative category price; no query or ESCI label. | Must be called simulated estimate. |

Correlations must be tested by broad expected direction, not exact sample correlation. The generator writes parameters, package versions, seed, input product checksum, and summary plots. Adding a product must not change existing products’ values.

### 10.4 Leakage controls

- The generator interface does not accept judgments.
- CI includes a source scan/test that prohibited columns are absent.
- Marketplace values are generated before data splitting but from product identity/non-label text only.
- If products overlap splits, the same simulated metadata is expected; it is an item property, not learned behavior.
- Estimated conversion cannot use query text or relevance score.
- Feature ablations report LambdaMART with and without simulated marketplace features. The preferred relevance model should default to real/derived relevance features; synthetic features are experimental.
- Policy tuning uses validation only; test relevance labels cannot set weights or constraints.

## 11. Data Splitting Strategy

Random row splitting is invalid because it fragments a query’s ranked list between train and evaluation, corrupts LightGBM group arrays, and allows nearly identical query context into both sides.

The default strategy:

1. Preserve the official ESCI test query groups as the final test set.
2. Within official training data, group by a leakage key consisting of locale plus normalized query text, not only `query_id`. Deterministically hash the group key and allocate approximately 85% to train and 15% to validation.
3. Ensure each `query_id` occurs in exactly one split and every judgment for it follows.
4. Sample profile query groups within each split using stable seeded hashing, preserving split proportions and useful label stratification at the group level.
5. Freeze test until architecture and policy choices are selected.

Products may overlap train/validation/test in the default split because the main task measures ranking generalization to unseen query groups over a shared catalog. Report product overlap. An optional cold-start stress test moves queries whose judged products meet a product-held-out rule, or masks product-history-like features; its smaller coverage and changed task must be labeled. A pseudo-temporal split is not valid without real timestamps; `product_age_days` is simulated and must not be used to claim temporal evaluation.

Leakage checks include normalized query-text overlap, group disjointness, label distribution by split, product overlap, duplicate documents, feature timestamp/provenance, target-derived aggregate detection, and fit-on-train-only categorical/statistical encoders.

## 12. Data Processing Pipeline

### 12.1 Restartable offline training pipeline

```text
 [00 download + checksums]
              |
 [01 raw schema validation] --failure--> quarantine/report
              |
 [02 normalized queries/products/judgments/sources]
              |
 [03 group split + deterministic profile sampling]
              |
      +-------+------------------+
      |                          |
 [04 product documents]   [05 synthetic metadata]
      |                          |
 [06 sparse index]        [metadata validation]
 [07 embeddings + FAISS]         |
      +------------+-------------+
                   |
 [08 train/val/test candidate generation + judged-pool candidates]
                   |
 [09 feature materialization + offline/online parity fixtures]
                   |
 [10 baseline + LambdaMART training / early stopping]
                   |
 [11 optional reranker scoring]
                   |
 [12 policy tuning on validation]
                   |
 [13 frozen test evaluation, ablations, reports]
                   |
 [14 serving bundle promotion]

 Each numbered stage: config+dependency hash -> temp output -> validate ->
 atomic rename -> manifest + _SUCCESS. A matching successful stage is reused.
```

### 12.2 Stage rules

- **Download:** manual script or instructions fetch only official files. Never silently redownload.
- **Validate:** compare exact expected schema, label set, locale set, keys, nulls, and counts.
- **Normalize:** project columns with Polars lazy scans; create canonical tables without destructive source cleaning.
- **Split/sample:** select complete query groups using stable hashes. Profile manifests list IDs and counts.
- **Text:** build a versioned retrieval document and normalization dictionary from training/catalog data.
- **Categoricals:** fit encoders on training only, reserve unknown/missing values, and persist mappings.
- **Labels:** map ESCI to `int8` grade in a separate target column with mapping metadata.
- **Synthetic:** generate separate seller/product files using stable sub-seeds.
- **Indexes:** build only from the declared catalog snapshot and document version.
- **Features:** materialize candidate-aligned Parquet; do not compute all catalog cross-products.
- **Versioning:** artifact ID includes stage name, semantic schema version, truncated config hash, and input dependency hashes. Creation time is metadata, not part of deterministic content identity.

Interrupted stages leave only a temporary directory and cannot be loaded. A `--force` rebuild writes a new version; it does not mutate a promoted bundle.

## 13. Query Understanding

The parser is deterministic, fast, and intentionally modest:

1. Preserve raw input; enforce UTF-8, length, and control-character limits.
2. NFKC-normalize, lowercase a retrieval view, collapse whitespace, and tokenize with a versioned regex.
3. Extract currency/price patterns: “under/below/less than,” “over/above,” ranges (“$40–80”), and “around.” Conflicts yield low confidence and no hard filter.
4. Match brands against an alias dictionary built from training/catalog brands using longest-boundary match.
5. Match colors against a curated versioned lexicon with common aliases.
6. Match category hints using a small versioned taxonomy/dictionary derived without test labels.
7. Extract numbers, units, model identifiers, and simple attributes such as size/capacity.
8. Apply a small spelling alias table only when unambiguous; retain original tokens.
9. Produce stopword-preserving and stopword-reduced token views. Model numbers, negations, compatibility words, and unit tokens are never dropped.

Output includes normalized text, token arrays, extracted bounds and entities, per-extraction confidence, parser version, warnings, and a deterministic hash. Low-confidence extraction contributes a feature but does not hard-filter. The parser must not require spaCy; a compact local tokenizer is optional. In production this boundary could become a learned query-understanding service.

## 14. Candidate Retrieval

### 14.1 Retrieval contracts

Each retriever accepts parsed query text, `top_k`, locale/catalog version, and optional safe filters. It returns `(query/request_id, product_id, raw_score, rank, retriever_id, index_id, latency_ms)`. Ranks are one-based, scores are finite, product IDs belong to the active catalog, and ties resolve by product ID after score.

### 14.2 Sparse retrieval

**Default local:** `bm25s` (or an equivalent audited SciPy sparse implementation) with persisted vocabulary, tokenization metadata, document map, and sparse score structures. It offers a lighter macOS path and better persistence/memory behavior than retaining `rank-bm25` Python token lists. `rank-bm25` is the smoke/reference implementation because it is simple and useful for parity tests. Pyserini is an optional benchmark when a JDK/Lucene installation is acceptable, not the required path.

Input is the versioned product document; output is top `K_sparse` (default-config candidate 150, tuned on validation). BM25 parameters `k1` and `b` are config-driven. Index persistence is non-pickle where supported; manifest verification guards document order.

Failure modes include long keyword-stuffed documents, misspellings, synonyms, memory growth, and poor semantic recall. Fielded weighting may be approximated by controlled title/brand repetition only as an explicit document-template version; learned interaction features handle fields later.

### 14.3 Dense retrieval

**Default local model:** `sentence-transformers/all-MiniLM-L6-v2`, a compact 384-dimensional model with strong ease-of-installation and CPU portfolio value. Products encode offline in batches; queries encode online. Vectors are L2-normalized, so FAISS inner product equals cosine similarity.

**Default index:** `faiss.IndexFlatIP` for deterministic exact search at the required lower-bound portfolio size. It is simple, has no training stage, and yields a reliable reference. If measured portfolio latency exceeds the budget, `IndexHNSWFlat` is the pre-approved latency alternative, recognizing that it usually consumes more—not less—memory. If dense-index memory exceeds the 8 GB envelope, first reduce the catalog to the declared lower-bound portfolio profile; a compressed IVF/PQ experiment is allowed only after its recall against FlatIP is reported.

Output is top `K_dense` (default-config candidate 150). Failure modes include semantic overgeneralization, identifier mismatch, truncation, model download absence, query/product domain mismatch, and CPU latency.

### 14.4 Filters

Locale and `inventory_count > 0` are safe hard eligibility filters. Brand, color, category, and price parsing are fallible; default behavior uses them as ranking features, not hard filters. The API’s explicit structured price filter may hard-filter synthetic price because the user knowingly requested it and the field is labeled simulated.

### 14.5 Cost comparison

| Option | Strength | Local cost / persistence | Failure or caveat | Role |
|---|---|---|---|---|
| `rank-bm25` | Minimal dependency, transparent | Python object/token memory; custom persistence needed | Slow/larger at portfolio scale | Smoke reference |
| `bm25s`/equivalent | Sparse, fast CPU, persistable | Low-to-moderate RAM; NumPy/SciPy artifacts | Smaller ecosystem than Lucene | **Required local default** |
| Pyserini/Lucene | Mature IR semantics | JDK, subprocess/JVM, larger setup | macOS friction | Optional comparison / production-shaped |
| FAISS FlatIP CPU | Exact, deterministic dense search | `4 × N × 384` bytes plus IDs, ~146 MiB per 100k vectors | Linear scan | **Required dense default** |
| FAISS HNSW | Faster ANN at scale | More index memory and tunable nondeterministic details | Approximation and build cost | Validated fallback |
| OpenSearch | Fielded BM25, filters, operations | Service/JVM overhead | Unnecessary required infrastructure | Production analogue only |

## 15. Hybrid Retrieval

Candidate fusion options:

- **Weighted normalized scores:** can use min-max, z-score, or calibrated transforms but is sensitive to query-specific score distributions and missing sources.
- **RRF:** sums `1/(c + rank)` across retrievers, ignores incompatible score scales, and is robust with little tuning.
- **Other rank fusion:** Borda or learned fusion provides flexibility but adds choices/data needs.
- **Union then learned ranker:** retains source scores/ranks as features and lets LambdaMART learn interactions.

The default is **RRF for the pre-ranker order plus candidate union for LambdaMART**. `c` defaults to a config value such as 60 and is tuned only on validation. A product returned by one retriever receives only that contribution, plus explicit missing/source indicator features. Duplicate product IDs merge into one row retaining both scores/ranks. Non-finite scores fail validation. The union is deterministically sorted by RRF, best source rank, then product ID and truncated to `K_union` (default-config candidate 200).

The judged-pool ranking track is separate: all judged products for a query form the candidate set to isolate ranking quality. The open-catalog retrieval track uses the real index and measures judged recall, acknowledging unjudged items.

## 16. Embedding Pipeline

Product embeddings are always offline:

1. Resolve a locally cached, pinned model revision; initial network download is explicit.
2. Read product IDs/documents in deterministic sorted order.
3. Encode configurable batches conservatively (start at 16–32 on the 8 GB M3 and test up to 64 only after measuring peak memory), with fixed truncation/max sequence behavior.
4. Convert to `float32`, L2-normalize, validate finite values and norm tolerance.
5. Write to a temporary NumPy memory-map and a parallel product-ID array.
6. Checkpoint completed contiguous row ranges so an interrupted run resumes only after verifying model/document hashes.
7. Build and persist FAISS from the completed matrix.
8. Write a manifest and atomically promote.

Artifact names encode catalog profile, document-template version, model slug/revision, dimension, dtype, normalization, and config hash. Expected default dimension is 384, but consumers read it from the manifest and reject mismatches. The raw embedding matrix is retained because it supports index rebuilds and feature lookup without re-encoding.

API startup loads the model from local cache, memory-maps product vectors if interaction similarity needs them, and reads the FAISS index. It never computes product embeddings. Query vectors are cached by `(model_hash, normalized_query)` in a bounded in-process LRU. If the model or index is unavailable, startup marks dense unavailable and hybrid falls back to BM25 if fallback mode is enabled; strict mode fails health/readiness.

## 17. Feature Engineering

### 17.1 Feature policy

Features must have one authoritative definition used by batch and serving adapters. Fit state (vocabularies, category encoders, normalizers) comes from training only. Every feature registry entry declares dtype, default/missing behavior, provenance, online availability, and leakage status. Numeric features use compact `float32`/small integers unless metric precision requires otherwise.

### 17.2 Feature catalog

| Feature(s) | Class | Definition / computation | Rationale | Train + inference availability | Leakage risk / storage |
|---|---|---|---|---|---|
| query char/token count | Query, derived | Counts from parser views | Query specificity/complexity | Both | Low; computed online, optional offline cache |
| query unique-token ratio | Query | unique tokens / tokens | Distinguishes repetition | Both | Low; computed |
| query digit/model-token count | Query | Regex token classes | Exact identifiers favor lexical match | Both | Low; computed |
| detected brand/color/category flags and confidence | Query | Parser entity outputs | Conditions interaction features | Both | Dictionary fitted/versioned without test labels; computed |
| has price bound; lower/upper bound | Query | Parsed currency constraints | Supports compatibility | Both | Parser errors; computed |
| locale | Query/source | Canonical locale categorical | Controls text/catalog behavior | Both | Low; payload |
| title/description/bullet char and token counts | Product/derived | Counts before/after truncation | Completeness and verbosity | Both | Low; product feature Parquet/in-memory projection |
| field missingness indicators | Product/derived | One flag per source text field | Missingness is informative and explicit | Both | Low; Parquet |
| normalized brand/color/category proxy | Product source/derived | Source normalization; category is derived/simulated, never source | Matching and slices | Both | Taxonomy may drift; encoded mapping persisted |
| product completeness | Product/derived | Weighted fraction of nonempty source fields | Catalog quality proxy | Both | Must not use labels; Parquet |
| BM25 score/rank | Retrieval/predicted-stage | Raw sparse score; one-based rank, missing flag | Exact lexical evidence | Both after retrieval | Candidate-selection bias; candidate Parquet/request |
| dense cosine/rank | Retrieval/predicted-stage | Inner product of normalized vectors; rank | Semantic evidence | Both after retrieval | Model-version coupling; candidate payload |
| RRF score/rank | Retrieval/derived | Sum reciprocal ranks; deterministic order | Robust fusion prior | Both | Low; candidate payload |
| source indicators | Retrieval | Sparse-only/dense-only/both bit flags | Missing source is informative | Both | Low; candidate payload |
| title token Jaccard/coverage | Interaction | Query/title intersection over union and over query tokens | Direct title match | Both | Tokenizer-version risk; computed |
| description/bullet token coverage | Interaction | Query-token coverage per field | Evidence beyond title | Both | Low; computed |
| exact phrase in title/document | Interaction | Boundary-aware normalized substring | Strong precision signal | Both | Avoid empty query; computed |
| brand/color match and conflict | Interaction | Parsed entity vs normalized product field | Constraint satisfaction | Both | Parser confidence needed; computed |
| category hint match | Interaction | Query hint vs product category proxy | Intent fit | Both | Proxy is not ground truth; computed |
| price compatibility/distance | Interaction + simulated | In bound flag and normalized distance to bound/center | Query constraint under simulation | Both | Synthetic price; must be tagged; computed |
| semantic similarity | Interaction/retrieval | Dense score or direct normalized dot product | Semantic alignment | Both when dense available | Missing fallback; candidate payload |
| query-title length ratio | Interaction | bounded log ratio | Penalizes mismatched granularity | Both | Low; computed |
| seller rating/reviews/fulfillment/shipping | Marketplace/simulated | Joined product→seller values; log1p reviews | Policy quality proxies | Both when metadata loaded | Synthetic; separate Parquet/in-memory columns |
| return/cancellation probability | Marketplace/simulated | Generated risk values | Risk-aware policy experiment | Both | Synthetic and not calibrated; separate |
| inventory and in-stock | Marketplace/simulated | count and `count>0` | Eligibility and availability | Both | Synthetic snapshot; separate |
| price and category-relative price z-score | Marketplace/simulated | log price; train/catalog category statistics | Affordability context | Both | Stats fit without labels; persisted state |
| profit margin | Marketplace/simulated | Generated ratio | Policy trade-off demonstration | Both | Never call real profit; separate |
| product age and new flag | Marketplace/simulated | log1p days; threshold flag | Exploration/cold start | Both | Synthetic; separate |
| historical popularity | Marketplace/simulated | Generated `[0,1]` proxy | Popularity-bias ablation | Both | Avoid target encoding; separate |
| estimated conversion rate | Marketplace/simulated/model-like | Generated deterministic proxy | Multi-objective demonstration | Both | Must not use query labels; not a real prediction |
| LambdaMART score/rank | Prediction | Serialized ranker output | Input to later stages | Inference and evaluation only | Never feed into same model; prediction artifact |
| cross-encoder score/rank | Prediction/optional | Batched pair score and normalized form | Fine semantic interaction | Only when enabled | Cache/model coupling; SQLite/request |

No aggregate such as “historical relevance by product/query” is allowed by default because it can encode target labels and exploit product/query overlap. Any future aggregate requires out-of-fold computation and an explicit leakage review.

## 18. Baseline Ranking Methods

| Model | Inputs | Purpose | Expected limitation |
|---|---|---|---|
| Seeded random | candidate IDs | Metric sanity floor and deterministic test | No relevance signal |
| BM25 | sparse score/rank | Strong lexical baseline | Semantic gaps |
| Dense | cosine/rank | Semantic baseline | Exact-token/constraint failures |
| Hybrid RRF | source ranks | Fusion baseline | Fixed weights, no rich interactions |
| Weighted heuristic | normalized retrieval + selected match features | Interpretable non-ML comparison and fallback | Manual tuning |
| Pointwise LightGBM classifier/regressor | same leakage-safe feature set | Tests value of supervised features without ranking objective | Ignores within-query list structure |
| LightGBM LambdaMART | grouped features and graded labels | Required primary ranker | Depends on candidate coverage and valid groups |
| Optional cross-encoder fusion | top LambdaMART candidates | Tests local neural interaction value | CPU latency |

All baselines use identical split definitions and, for ranking comparisons, identical candidate pools. Random ordering uses a stable `(run_seed, query_id, product_id)` key rather than process randomness.

## 19. Learning-to-Rank Design

Pointwise methods predict each row independently and do not directly optimize relative order. Pairwise methods learn preferred item pairs and align better with ranking but can overweight large groups. Listwise methods optimize list-level objectives; LambdaMART uses gradient-boosted trees with lambda gradients that approximate ranking-metric improvements.

LightGBM `LGBMRanker` with `objective=lambdarank` is the primary model because it handles nonlinear mixed tabular features, missing values, CPU training, query groups, feature importance, and fast inference without GPU or fine-tuning. The grade mapping is config-versioned (`I=0,C=1,S=2,E=3`) with `label_gain` explicitly matching the chosen gains. Training rows are sorted by query group and stable candidate key; the group-size array must sum exactly to row count and align contiguously.

Categoricals use LightGBM native categorical columns when stable and low-cardinality; brand/product IDs are not naively one-hot encoded. Unknown and missing categories have explicit codes. Missing retriever values remain missing plus source indicators. Optional monotonic constraints may enforce sensible directions only for unambiguous simulated risk/quality features after an ablation; they are not default because relevance interactions can legitimately be non-monotone.

Training uses early stopping on validation NDCG cutoffs, controlled thread count, shallow/moderate trees, feature subsampling, and seeded determinism settings. Limited search covers learning rate, leaves, minimum child samples, estimators, feature/bagging fractions, and regularization. Test data is evaluated once after selection.

Model artifacts include LightGBM’s text model format, feature schema/order, categorical mappings, gains, training config, dependency versions, input hashes, validation history, and compatibility version. Inference validates feature names/types, predicts candidates as a batch, and sorts score descending with deterministic tie breaks.

Feature importance includes gain/split importance and optional TreeSHAP on a bounded stratified query sample. Explanations are associational, not causal. Compare with:

- **XGBoost Ranker:** credible alternative, similar CPU tree ranking; keep as fallback benchmark, not duplicate required infrastructure.
- **CatBoost ranking:** attractive categorical handling but another dependency and potentially different ranking workflow.
- **Neural rankers:** better text interaction potential but higher CPU cost and less transparent feature/policy integration.

## 20. Training Strategy

| Level | Data/profile | Candidate/features | Search and controls | Purpose |
|---|---|---|---|---|
| Smoke | Tiny deterministic fixture or 100–500 queries | Judged pool; minimal and full-schema subsets | Fixed params, 1–2 threads, tens of trees | Validate contracts in minutes and CI-friendly paths |
| Development | 5k–10k query groups / 50k–100k judgments | Hybrid candidates plus judged-pool track; cached full feature set | Early stopping; up to ~5 seeded trials; 2–4 threads | Feature iteration and debugging |
| Portfolio | Required default at the lower bound: ~20k groups / ~200k judgments; scale toward 50k/500k only if measured memory allows | Frozen feature version and candidate config | Validation-selected config; ~5–10 trials on the 8 GB M3; sequential stages and controlled threads | Final ablations and report |

Exact counts are configuration targets, not promises; complete groups take precedence. The 8 GB reference workflow must demonstrate the portfolio profile at its lower bound rather than silently substituting the development profile. Trial count may be lowered when runtime or memory is excessive. Optuna is optional; a seeded parameter list/random search is sufficient and easier to reproduce. Feature Parquet is cached, model matrices use compact dtypes, and model trials run one at a time. Training records wall time, CPU/thread settings, peak RSS, machine details, and random seeds. No full-benchmark run is required.

## 21. Optional Neural Reranking

The default optional model is `cross-encoder/ms-marco-TinyBERT-L-2-v2` because its small size and CPU latency make the feature demonstrable on a MacBook. `cross-encoder/ms-marco-MiniLM-L-6-v2` is a quality-oriented alternative that must be benchmarked before becoming a profile default.

For the top `K_neural` LambdaMART results (configurable 10–30, default candidate 20), format input as `(normalized query, versioned product text with field markers)`. Batch inference uses a small configurable batch, truncation metadata, `torch.no_grad`, and controlled threads. Scores are normalized within query using rank or validation-fitted standardization, then fused with LambdaMART through a validation-selected weighted sum or a tiny downstream linear calibration. The core LambdaMART ranking is retained for all remaining candidates.

Cache keys include model revision, query hash, product ID, product-text hash, and preprocessing version. Cache storage is SQLite with a size/TTL maintenance policy. The model download is explicit; weights are not fetched at startup. Missing model, timeout, non-finite scores, or memory error emits a warning and returns LambdaMART order. Model/version, input count, batch latency, cache hit rate, and rank changes are logged.

This stage is complete when disabled: all required endpoints, tests, ranking, policy, and demo modes continue to work.

## 22. Marketplace Optimization

Relevance ranking estimates which products satisfy the query according to ESCI. Marketplace optimization is a post-ranking **simulated policy** operating on predicted relevance and synthetic metadata. It cannot claim improved revenue, conversion, user welfare, or fairness.

The policy first applies hard eligibility, then greedily selects results using a validation-configured utility:

```text
utility(p | selected) =
    normalized_relevance(p)
  + w_quality     * seller_quality_proxy(p)
  + w_fulfillment * fulfillment_proxy(p)
  - w_return      * return_risk(p)
  + w_inventory   * inventory_health(p)
  + w_explore     * new_product_bonus(p, selected)
  + w_margin      * simulated_margin(p)
  - w_seller_dup  * seller_duplication(p, selected)
  - w_category_dup* category_duplication(p, selected)
```

All terms are bounded and named in an audit. Predicted engagement/conversion is off by default in the required policy because it is simulated; an ablation may add it. Sponsored status never improves required organic order.

The policy config includes objective weights, normalization method, hard caps, new-product target, relevance guardrail, exact-item protection, fallback rules, and version. Every final row records pre-policy rank, post-policy rank, utility components, constraints triggered, and simulated-field disclaimer.

## 23. Marketplace Constraints

Hard constraints:

- Remove `inventory_count <= 0`.
- Respect explicit API price bounds against clearly labeled synthetic price.
- No duplicate product IDs.
- Cap a seller at configurable `M` products in top `K` where enough alternatives exist.
- Preserve the top predicted/high-confidence Exact-like results via a conservative relevance threshold; evaluation uses real Exact labels only for audit, never online decisions.

Soft/guarded constraints:

- Target minimum new-product exposure when enough eligible new products exist.
- Penalize low fulfillment/high cancellation/return risk.
- Limit category-proxy repetition.
- Keep query-level and aggregate relevance loss within a validation-selected NDCG budget during offline tuning.

Real ESCI labels are unavailable online, so the online relevance guard uses normalized LambdaMART score gaps and protected top ranks. Offline reports separately audit true Exact displacement and ΔNDCG.

| Method | Advantages | Weaknesses | Decision |
|---|---|---|---|
| Weighted score | Simple and fast | Cannot guarantee caps/exposure | Component of utility |
| Greedy constrained reranking | Deterministic, auditable, `O(KC)`, supports feasibility logic | Not globally optimal | **Required default** |
| MMR | Natural diversity/relevance trade-off | Requires similarity definition; weak hard constraints | Optional category/product diversity term |
| Linear programming | Global continuous optimum | Awkward ordering and nonlinear diversity | Deferred |
| Integer optimization | Expressive hard constraints | Solver dependency, latency, infeasibility handling | Production/offline research option |

If a constraint is infeasible, the algorithm follows a declared relaxation order: retain stock constraint always; relax new-product minimum; relax category cap; relax seller cap only if fewer than `K` results would remain; never insert an out-of-stock product. Relaxations are returned in the policy audit.

## 24. Personalization

Personalization is future work because ESCI has no user/session histories. A production extension could add session queries/clicks, user or session embeddings, category preferences, brand affinity, price sensitivity, and recency features. It would require feature freshness contracts, consent and retention controls, anonymous/session cold-start fallbacks, position-bias correction, and evaluation beyond static relevance. New users fall back to contextual/global ranking. Sensitive attributes should not be inferred. Personalized features must not be simulated and presented as observed in the MVP.

## 25. Offline Evaluation

### 25.1 Metric definitions and ownership

| Metric | Definition | Stage / interpretation |
|---|---|---|
| Recall@K | relevant judged products retrieved in top K / all relevant judged products | Candidate retrieval; report thresholds `E`, `E+S`, optionally `E+S+C`. |
| Precision@K | relevant judged results / judged results evaluated at K | Retrieval/ranking only on fully judged/fixed pools; open-catalog unjudged items make naive precision invalid. |
| MRR@K | mean reciprocal rank of first relevant item by named threshold | Early retrieval/ranking. |
| MAP@K | mean average precision over binary relevance threshold | Ranking; threshold must be explicit. |
| NDCG@K | graded discounted gain normalized by ideal list | Primary ranking/reranking relevance metric; mapping/gains versioned. |
| Unique sellers@K | mean distinct simulated sellers | Marketplace diversity. |
| Seller HHI@K | sum of squared seller exposure shares | Concentration; lower means less concentration. |
| Max seller share@K | largest seller count/K | Constraint/audit. |
| Catalog coverage@K | unique products exposed across queries / eligible catalog | Aggregate coverage; synthetic policy context. |
| New-product exposure@K | fraction of top-K items with simulated `is_new_product` | Exploration. |
| Category diversity@K | unique category proxies or entropy | Product-list diversity; proxy caveat. |
| Expected synthetic margin | exposure-weighted mean of simulated margin | Policy simulation only. |
| Expected synthetic return risk | exposure-weighted mean simulated probability | Policy simulation only. |
| Stock violations | count of results with zero simulated inventory | Must be zero post-policy. |
| Relevance loss | NDCG after policy − before policy | Guardrail; report mean, CI, and slices. |
| Stage latency | wall-clock p50/p95/p99 after warmup | System evaluation. |
| Peak RSS/artifact bytes | process memory and file sizes | Resource evaluation. |

Candidate retrieval and open-catalog metrics use known judgments but distinguish unjudged. Ranker evaluation uses a fixed judged pool as the clearest supervised comparison and an end-to-end retrieved pool to measure actual pipeline loss. Marketplace metrics compare the identical pre-policy candidate ranking to post-policy output.

## 26. Evaluation Methodology

- Freeze one test query set and all artifact versions before final evaluation.
- Use identical candidates when comparing ranking methods; use separate end-to-end evaluation when comparing retrievers.
- Evaluate every method on the intersection of valid query groups and disclose dropped/empty groups.
- Compute query-level metric values, then paired deltas. Bootstrap query IDs with replacement using a fixed seed for 95% confidence intervals; preserve all products in each resampled query.
- Prefer paired intervals over declaring significance from overlapping unpaired intervals.
- Report mean, median, and failure-tail examples where appropriate.
- Slice by query token-length buckets, head/tail based on non-label catalog/query statistics, brand/price/entity presence, source, category proxy, and candidate relevant-count.
- Report E-first performance, E/S binary retrieval, and behavior on complement-heavy queries.
- Keep category proxy slices labeled derived/synthetic.
- Measure cold and warm startup separately. Run warmup, repeated queries, single-threaded latency for comparability, plus modest-concurrency API tests.
- Measure RSS with a platform-compatible method and record background conditions/hardware.
- Manually inspect a seeded sample of wins, losses, parser failures, and large policy moves.

No test-set result may select features, weights, candidate counts, or hyperparameters. Any post-test fix creates a new named evaluation generation and must be disclosed.

## 27. Required Ablation Studies

| ID | Comparison | Fixed controls | Purpose |
|---|---|---|---|
| ABL-01 | BM25 only | catalog, queries, K, evaluator | Establish lexical baseline. |
| ABL-02 | Dense only | same as ABL-01 | Isolate semantic retrieval. |
| ABL-03 | Hybrid RRF vs best single | union cap and evaluation queries | Test complementary candidate coverage. |
| ABL-04 | Hybrid ordering vs hybrid + LambdaMART | identical hybrid candidate pool | Measure learned ranker value. |
| ABL-05 | LambdaMART without simulated marketplace features | same real/derived features and tuning budget | Establish relevance-first model. |
| ABL-06 | LambdaMART with simulated marketplace features | same candidates/splits | Test whether synthetic signals change relevance; prevent hidden claims. |
| ABL-07 | Before vs after marketplace policy | identical predicted ranking | Quantify relevance/diversity/risk trade-off. |
| ABL-08 | Policy without vs with new-product/diversity terms | same constraints and pool | Attribute policy effects. |
| ABL-09 | LambdaMART vs LambdaMART + optional cross-encoder | same top-K and downstream policy off first | Measure neural quality/latency trade. |
| ABL-10 | Pointwise baseline vs LambdaMART | same features/candidates/training queries | Justify ranking objective. |

Every ablation records metric deltas, CIs, latency, memory/artifact costs, config hashes, and query counts. Marketplace policy is evaluated both with policy off and on; otherwise neural/ranker gains could be confounded by changing constraints.

## 28. Position Bias and Counterfactual Evaluation

This is an advanced optional module. Real impression/click logs would include request, displayed position, item, policy propensity, examination/click/conversion indicators, context, and time. Position bias occurs because higher items are examined more often independent of relevance.

An educational simulator may generate examination probabilities by position and clicks conditional on examination plus a declared synthetic relevance response. It must live outside required evaluation and state that the log is simulated. Inverse propensity scoring weights outcomes by inverse logging-policy propensity but can have high variance; clipping and effective sample size are required. Self-normalized IPS improves stability. Doubly robust estimation combines a response model and propensity correction and is robust if either nuisance model is correct under assumptions.

Counterfactual claims require positivity/support, consistent outcomes, known or estimated propensities, and no unmeasured confounding—assumptions not satisfied by ESCI alone. Therefore no counterfactual metric is used for the MVP completion gate.

## 29. Explainability

User-facing reason templates use verified, non-causal facts:

- “Strong title-term match”
- “Brand matches your query”
- “Within your requested price range (simulated price)”
- “Category hint matches”
- “High simulated seller fulfillment”
- “Short simulated shipping estimate”

Reasons are emitted only when their predicate and confidence pass a threshold. The UI must not say “you will like,” “best,” or “caused by.”

Developer debug output includes source candidates/ranks/scores; normalized feature values; LambdaMART score/rank; optional cross score; final utility components; constraint actions; stage-by-stage rank changes; artifact/model versions; and bounded TreeSHAP contributions on request when enabled. SHAP is never on the normal latency path. Sensitive raw descriptions may be truncated in logs.

## 30. Local Storage and Artifact Management

### 30.1 Artifact dependency flow

```text
 raw checksums
      |
 normalized tables ---> split/profile manifest
      |                         |
      +--> product documents <--+
      |          |              |
      |          +--> BM25 index|
      |          +--> embeddings --> FAISS index
      |                               |
      +--> synthetic metadata         |
                  \                   /
                   --> candidates <---
                          |
                    feature schema/state
                          |
                    feature matrices
                          |
                    ranker model
                          |
           optional reranker + policy config
                          |
                    serving bundle
                          |
               evaluations + demo/API

 A consumer is loadable only when every parent hash matches its manifest.
```

### 30.2 Artifact inventory

| Artifact | Format | Required metadata | Loaded when |
|---|---|---|---|
| Raw manifest | JSON | URL, license, size, SHA-256, retrieved UTC | Ingestion |
| Normalized tables | Parquet | schema version, counts, source hash | Most offline stages |
| Profile/split manifest | JSON + Parquet IDs | seed, group hash rule, counts/distributions | All profile workflows |
| Synthetic tables | Parquet | generator/parameter version, seed, input hash, provenance banner | Features/policy/API |
| Product documents | Parquet | template/tokenizer version, truncation, catalog hash | Index build/features |
| Sparse index | Library-native + NumPy/JSON metadata | tokenizer, BM25 params, document map hash | API/retrieval evaluation |
| Embeddings | `.npy` memory map + ID array | model revision, dimension, dtype, normalized flag | FAISS build/features/API as needed |
| FAISS index | `.faiss` + manifest | index type/params, vector/ID hashes | API/retrieval evaluation |
| Candidates | Partitioned Parquet | retriever/index IDs, K, fusion config | Feature generation/evaluation |
| Feature registry/state | JSON/Parquet | names, dtypes, provenance, defaults, encoders | Training/API |
| Feature matrix | Parquet, optional binary matrix | candidate/data/feature hashes, label inclusion | Training/evaluation |
| Models | LightGBM text + JSON manifest | features, params, code/dependency versions, metrics | Evaluation/API |
| Reranker cache | SQLite | model/input versions, size policy | Optional evaluation/API |
| Policy config | YAML + validation report | weights, constraints, guardrail, validation run | Evaluation/API |
| Experiment run | JSON/Parquet + optional MLflow files | lineage, config, hardware, metrics, status | Reporting |
| Evaluation report | Parquet/JSON/Markdown/PNG | query set and every compared artifact hash | Portfolio/reporting |
| Serving bundle | Directory manifest/pointers | full compatibility matrix and readiness mode | API startup |

Paths follow `artifact_type/dataset_version/profile/component_version/config_hash/`. Manifests include created UTC and Git commit or `dirty:<diff-hash>`. Writes are temporary then atomic. “Latest” may be a human-readable pointer but services load an explicit immutable bundle ID. Large/data artifacts are Git-ignored.

## 31. Experiment Tracking

MLflow uses a local file or SQLite backend and local artifact directory; no server is required for runs. A lightweight canonical `run.json` plus long-form `metrics.parquet` is always written so reproduction does not depend on MLflow. Starting the MLflow UI is optional.

Track dataset release/checksums/profile/split, feature set, code revision and dirty state, model/retriever/index versions, hyperparameters, seeds, thread counts, candidate counts, all metric definitions/values/CIs, policy metrics, latency distributions, peak RSS, artifact paths/hashes, machine/OS/Python/package versions, and run status. Nested trials link to a parent study. Failed runs retain config and error summary but are never promoted.

Run comparison scripts read canonical files, not UI state. A successful final report names exactly one champion relevance bundle and one champion policy config selected on validation.

## 32. Configuration Management

YAML is the user-facing declarative format; strict typed validation rejects unknown keys. Suggested layered configs:

- `base.yaml`: paths, seed, locale, logging, thread counts.
- `profiles/{smoke,development,portfolio,full}.yaml`: query/judgment targets and sampling.
- `retrieval/*.yaml`: tokenization, BM25, model revision, K values, FAISS type, RRF.
- `features/*.yaml`: feature groups and registry version.
- `ranking/*.yaml`: gains, LightGBM params, search space, early stopping.
- `reranker/*.yaml`: enabled/model/revision/top-K/batch/cache/fusion.
- `marketplace/*.yaml`: generator and policy weights/constraints.
- `evaluation/*.yaml`: cutoffs, thresholds, slices, bootstrap seed/count.
- `serving/*.yaml`: explicit bundle ID, cache sizes, timeouts, strict/fallback readiness.

Resolution order is base → profile/component → explicit CLI overrides. The resolved config is validated, canonicalized with sorted keys, hashed, and copied into the run. Environment variables may override paths and ports but not silently change model semantics. Secrets are unnecessary.

## 33. API Design

### 33.1 Startup and online inference pipeline

```text
 STARTUP
 explicit bundle ID -> verify manifest DAG/checksums/schema
                    -> load product display projection + metadata
                    -> load BM25 + FAISS + query encoder
                    -> load feature state + LambdaMART
                    -> optionally load cross-encoder
                    -> warm bounded probes -> readiness=true

 REQUEST
 JSON query -> validate -> parse -> cache lookup
            -> sparse || dense retrieval (bounded top-K)
            -> union/RRF -> eligibility/dedupe
            -> online features -> LambdaMART
            -> [cross-encoder top 10–30]
            -> marketplace constraints/policy
            -> explanation assembly -> response + structured timings
```

### 33.2 Endpoints

| Endpoint | Purpose | Request | Response | Errors/cache |
|---|---|---|---|---|
| `GET /health/live` | Process liveness | none | status, timestamp | Always cheap; no artifact details. |
| `GET /health/ready` | Serving readiness | none | loaded component status and degraded flags | `503` if no acceptable relevance path. |
| `POST /v1/search` | Ranked search | query string, top_k (bounded), mode, optional explicit price bounds, policy/neural flags, debug=false | request ID, results, timings, mode/fallbacks, synthetic disclaimer | `422` invalid; `503` no retriever/model; version-aware query cache. |
| `GET /v1/model-info` | Model/config summary | none | safe model IDs, feature/policy versions, optional stage availability | Cached static; no local paths. |
| `GET /v1/artifact-info` | Reproducibility summary | none | bundle/data/index hashes and creation metadata | Cached; hide unsafe filesystem details. |
| `POST /v1/debug/explain` | Bounded developer rank trace | search request plus max candidates/result IDs | source ranks, feature/score/policy breakdown, warnings | Local-only by default; rate/size limited; SHAP optional and uncached separately. |

Search modes are `bm25`, `dense`, `hybrid`, `lambdamart`; unsupported/unavailable requested modes return a clear error unless the request explicitly permits fallback. Results contain product ID, display fields, synthetic fields nested under `simulated_marketplace`, stage scores/ranks appropriate to debug level, reason codes, and final rank. ESCI labels appear only for known offline/demo queries and are marked ground truth; arbitrary user requests do not imply labels.

API limits include query length, top-K maximum, body size, concurrent neural requests, and timeouts. Query/result caches include bundle and mode in keys and are bounded LRU/TTL. Startup does not download or build.

## 34. Streamlit Demo

The demo is a thin API client, not a second inference implementation. It provides:

- query box with examples and optional explicit price bounds;
- top-K selector;
- BM25/dense/hybrid/LambdaMART mode selector;
- side-by-side BM25 versus selected mode;
- marketplace policy toggle and optional neural toggle, disabled with an explanation when unavailable;
- product cards with title, brand, color, text snippet, and nested **Simulated marketplace data** panel;
- ESCI labels only for known evaluation queries;
- score/reason breakdown and stage timings;
- unique sellers, HHI, new-product exposure, and relevance delta when ground truth exists;
- a rank-change slope/table visualization from retrieval → ranker → neural → policy;
- artifact/model IDs and degraded-mode banner.

The UI cannot silently call models directly, rebuild artifacts, or claim live Amazon results. Screenshots in the portfolio must visibly include the simulation disclaimer.

## 35. Repository Structure

```text
ecommerce_market_ranker/
├── ELEPHANT.md                  # Approved end-state design
├── README.md                    # Setup, results, demo, limitations
├── pyproject.toml               # Package, tools, dependencies
├── lockfile                     # Deterministic platform-aware lock
├── configs/
│   ├── profiles/               # Dataset execution profiles
│   ├── retrieval/              # Sparse/dense/hybrid configs
│   ├── ranking/                # Features and ranker configs
│   ├── marketplace/            # Synthetic generator and policy
│   ├── evaluation/             # Metrics/slices/bootstraps
│   └── serving/                # Explicit serving bundles/runtime
├── data/
│   ├── raw/                    # Immutable downloaded inputs (Git-ignored)
│   ├── interim/                # Validated normalized intermediates
│   ├── processed/              # Canonical tables
│   ├── samples/                # Profile query IDs/tables
│   └── synthetic/              # Clearly isolated generated metadata
├── artifacts/
│   ├── embeddings/             # Memory-mapped vectors and ID maps
│   ├── indexes/                # BM25 and FAISS artifacts
│   ├── features/               # Registry/state/matrices
│   ├── models/                 # Ranker/reranker metadata
│   └── evaluations/            # Machine-readable metrics/reports
├── src/market_rank/
│   ├── data/                   # Load, validate, sample, text build
│   ├── query/                  # Query parser
│   ├── retrieval/              # Sparse, dense, hybrid contracts
│   ├── features/               # Shared feature definitions/state
│   ├── ranking/                # Baselines, LambdaMART train/score
│   ├── reranking/              # Optional cross-encoder
│   ├── marketplace/            # Generator and policy optimizer
│   ├── evaluation/             # Metrics, slices, bootstraps
│   ├── serving/                # FastAPI schemas/lifecycle/routes
│   └── utils/                  # Config, hashes, logging, manifests
├── scripts/                    # Thin deterministic CLI entry points
├── app/                        # Streamlit API client and presentation
├── tests/
│   ├── unit/                   # Pure/fast toy tests
│   ├── integration/            # Cross-component and reload tests
│   ├── smoke/                  # Tiny end-to-end profile
│   ├── regression/             # Golden metrics/contracts
│   └── fixtures/               # Small redistributable synthetic fixtures
├── notebooks/                  # Exploration only; imports package code
├── experiments/                # Resolved run configs/metadata (large ignored)
├── docs/                       # Goldfish docs, ADRs, operations
└── reports/                    # Final tracked summaries/plots/screenshots
```

Reusable logic belongs under `src/market_rank`; scripts validate arguments and call it. Notebooks may visualize or explore but cannot contain the only implementation of ingestion, features, metrics, training, or serving. `data/` is data lifecycle; `artifacts/` is expensive derived model/index state; `experiments/` is run lineage; `reports/` is curated communication.

## 36. Module and Interface Design

Interface sketches are illustrative contracts, not implementation code.

| Module | Responsibility / conceptual interface | Inputs → outputs | Persistence / dependencies | Errors and tests |
|---|---|---|---|---|
| `DatasetLoader` | Load official files with projection and schema contract | paths/profile → lazy frames | Raw manifest, Polars/Parquet | Missing/checksum/schema errors; fixture schemas |
| `DatasetSampler` | Split/sample complete query groups | judgments, seed, targets → query IDs/manifest | Profile artifact | Impossible targets/group leakage; determinism tests |
| `ProductTextBuilder` | Build versioned documents/display text | normalized products + template → documents | Document Parquet | Empty/oversize/Unicode; golden strings |
| `MarketplaceMetadataGenerator` | Stable per-ID seller/product simulation | products, generator config → seller/product tables | Separate synthetic Parquet | Bounds/prohibited dependency; order invariance |
| `QueryParser` | Normalize and extract entities/constraints | raw query → `ParsedQuery` | Parser/dictionary artifact | Never throws on valid bounded UTF-8; curated fixtures |
| `SparseRetriever` | Build/load/search BM25 | documents or parsed query → index/candidates | Sparse index + ID map | Missing/corrupt index; reload/rank parity |
| `DenseRetriever` | Build/load/search embeddings/FAISS | documents/query → vectors/index/candidates | `.npy`, FAISS, model revision | Dimension/model mismatch; tiny exact-search fixtures |
| `HybridRetriever` | Merge/dedupe/truncate candidates | candidate lists + fusion config → candidates | Candidate Parquet offline | Non-finite/duplicate; RRF toy examples |
| `FeatureBuilder` | Shared candidate feature contract | parsed query, products, candidates, metadata → frame | Registry/state/matrices | Missing required values/type/order; offline-online parity |
| `Ranker` | Train/load/predict grouped ranks | features, labels, groups → model/predictions | LightGBM text + manifest | Invalid groups/features; serialization parity |
| `NeuralReranker` | Optional bounded cross scoring | query + top products → scores/order | Model metadata + SQLite cache | Timeout/unavailable fallback; top-K bound/cache tests |
| `MarketplaceOptimizer` | Eligibility and constrained greedy order | scored candidates + policy → final ranking/audit | Policy config/report | Infeasibility/metadata absence; constraint fixtures |
| `RankingEvaluator` | Metrics, slices, CIs, comparisons | rankings + judgments + configs → metric facts/reports | Evaluation artifacts | Unjudged/empty group handling; known toy metrics |
| `ArtifactRegistry` | Manifest DAG, compatibility, promotion | artifact paths/IDs → verified bundle | JSON manifests | Hash/schema/dependency mismatch; corruption tests |
| `ExperimentTracker` | Canonical local run state | config/metrics/artifacts → run directory | JSON/Parquet + optional MLflow | Partial run/failure; round-trip tests |
| `SearchOrchestrator` | Compose online stages and fallbacks | search request + loaded bundle → response | Query caches/logs | Deadline/component failures; integration tests |

Public objects should be immutable dataclasses or validated models where practical. Boundary methods use explicit IDs/manifests rather than ambient global paths. Domain exceptions distinguish invalid data, missing artifact, incompatibility, unavailable optional stage, and request validation.

## 37. Testing Strategy

- **Unit:** parser rules, stable hashing, document templates, RRF, feature formulas, gains, policy selection, metric math, manifests.
- **Integration:** normalize→sample, build/load/query each index, candidate→features→model, policy after ranking, API with a tiny persisted bundle.
- **Smoke:** one command builds a tiny fully synthetic fixture bundle and searches it; a gated ESCI smoke run validates real schema when data exists.
- **Regression:** golden product document, candidate ranks, feature rows, known toy metric values, and serving response schema. Golden numbers include tolerance and version.
- **Data validation:** all Section 38 checks per profile.
- **Determinism:** shuffle input rows and vary chunking; stable sample and synthetic output must match. Same index/model settings must match declared tolerance.
- **Artifact loading:** cold reload, checksum corruption, wrong vector dimension, wrong feature schema, old model version.
- **API:** status codes, limits, degraded modes, caches, debug redaction, concurrency bound.

Critical invariants:

1. Every candidate key contains its request/query and active-catalog product.
2. No query group crosses train/validation/test.
3. Group sizes are positive, contiguous, and sum to matrix rows.
4. Top-K contains no duplicate product IDs.
5. Out-of-stock simulated products never survive post-policy.
6. Seller caps and relaxation audits agree.
7. Seeded marketplace generation is order- and chunk-invariant.
8. Labels are absent from online features and generator inputs.
9. Persisted indexes reload with candidate parity.
10. Metrics match hand-computed toy examples including empty/no-relevant cases.
11. API startup performs no download/index/embedding/training operation.
12. Optional neural disable/failure preserves LambdaMART ordering before policy.

Network/model-download tests are opt-in. Performance tests record distributions and do not use brittle single-run thresholds in ordinary CI.

## 38. Data Validation

Use Polars expressions, DuckDB SQL assertions, and lightweight custom validators; Pandera is optional for small frames. Required checks:

- exact schema names, compatible dtypes, source checksum, and nonzero rows;
- primary-key uniqueness and query-text consistency per query ID;
- expected locale and label domains;
- official split validity and no normalized-query group overlap after project split;
- null rates and changes versus a stored expectation range;
- duplicate/conflicting product and judgment handling;
- nonempty usable product documents and nonempty queries;
- every judgment/candidate product joins to exactly one product;
- every candidate belongs to its query/request and active catalog;
- query group counts/sizes are valid before model training;
- price finite and positive; probabilities in `[0,1]`; ratings `[1,5]`; counts nonnegative;
- synthetic metadata covers every product exactly once and seller joins resolve;
- prohibited label columns absent from synthetic generator/online feature inputs;
- embeddings are finite, dimension-consistent, normalized within tolerance, and aligned with IDs;
- feature schema/order/dtypes match model manifest.

Reports include pass/fail, severity, counts, sample offending keys, and artifact version. Hard invariant failures stop promotion; warnings such as high null rate require explicit acknowledgment in the run.

## 39. Observability

Emit structured JSON logs with UTC timestamp, level, event, request/run ID, bundle/model/index versions, mode, stage, duration, candidate input/output counts, cache hit, degraded/fallback reason, policy relaxation, and exception class. User query logging defaults to a salted hash plus length/parser flags; raw query logging is opt-in for local debugging.

Online timers cover parse, sparse, query embedding, dense, fusion, lookup/features, ranker, neural, policy, serialization, and total. Offline timers cover rows/second, batches, peak RSS checkpoints, artifact bytes, and cache/reuse decisions. A local metrics endpoint is optional; canonical request logs must be analyzable with DuckDB/Polars. Health exposes component status without stack traces or filesystem secrets.

## 40. Memory Management

| Workflow | Loaded / streamed | Controls |
|---|---|---|
| Ingestion | Projected Parquet scans, chunked joins, small validation aggregates | Polars lazy scan, predicate pushdown, DuckDB, avoid full pandas copies |
| Index build | Product ID/document columns; sparse chunks or embedding batches | Truncation, batch encoding, temporary memmap, controlled threads |
| Training | Candidate-aligned compact features and group arrays | Column projection, `float32`/small ints, categorical codes, cached matrices |
| Evaluation | One method/stage partition plus query metrics | Streaming partitions, long-form outputs, bounded bootstrap arrays |
| API startup | Display/product lookup projection, required metadata columns, sparse index, FAISS, ranker, parser dictionaries; optional models | Explicit bundle, do not also map the full embedding matrix unless a required feature cannot use returned dense scores, lazy-load the neural model, and avoid duplicate DataFrame copies |
| Search request | Bounded source candidates, union ≤ configured cap, top neural subset | No full-catalog DataFrame copies; vectorized/batched feature computation |

Parquet is the canonical tabular format; DuckDB supports ad hoc joins/reports. NumPy memory maps avoid copying the full embedding matrix. Product display data may be a compact Arrow/Parquet lookup or DuckDB table rather than a large Python dictionary if profiling shows overhead. Intermediate objects are released between offline stages; process-per-stage execution naturally returns memory to the OS.

## 41. Latency Targets

Targets are warm, single-request design budgets on the 8 GB Apple M3 reference machine and are not guarantees.

| Stage | Target | Measurement notes |
|---|---:|---|
| Query preprocessing | < 50 ms | Includes parsing/entity dictionaries |
| BM25 retrieval | < 200 ms | Index already loaded |
| Query embedding + dense retrieval | < 200 ms | Model/index loaded; report cache hit/miss |
| Candidate fusion/dedupe | < 50 ms | Union cap enforced |
| Feature generation/lookups | < 250 ms | Candidate set only |
| LambdaMART scoring | < 50 ms | Batch of ≤ configured union cap |
| Marketplace reranking | < 100 ms | Greedy top-K |
| Serialization/overhead | < 100 ms | Bounded response/debug off |
| **Total without neural** | **target < 1 s** | Report p50/p95 after warmup |
| Optional cross-encoder | < 2 s | Top 10–30, batch CPU |
| **Total with neural** | **target < 3 s** | Report cache miss and hit separately |
| API cold startup | measured, goal < 30 s | May load local models/indexes; never builds/downloads |

If a target is missed, profile first, reduce to the declared lower-bound portfolio catalog or reduce candidate/top-neural K within approved ranges, switch FlatIP to validated HNSW for latency (not memory), or disable the optional stage. A compressed FAISS index may address memory only after a recall/latency comparison. Do not hide quality changes.

### 41.1 Ranking-stage funnel

```text
 Portfolio catalog:          ~tens/hundreds of thousands products
       BM25 top 150  ─┐
                      ├─ union + dedupe ──> at most 200 candidates
       Dense top 150 ─┘
                                      |
                              LambdaMART: 200
                                      |
                        optional cross-encoder: top 20
                                      |
                       eligibility + policy: pool ≤200
                                      |
                              response: top 10–50

 Counts are configurable validation-selected defaults, not dataset facts.
```

## 42. macOS Compatibility

- The required benchmark machine is an Apple M3 Mac with 8 GB unified memory. Final reports record whether it is a MacBook Air, MacBook Pro, or another M3 Mac, plus macOS version, power mode, and free memory.
- Target Python 3.11 initially; pin an exact supported minor range in `pyproject.toml` after dependency compatibility is verified.
- Use `venv`, `uv`, or Conda consistently; commit a deterministic lock with macOS ARM64 and x86_64 resolution guidance.
- Prefer wheels for Polars, DuckDB, PyArrow, LightGBM, PyTorch, Sentence Transformers, and FAISS CPU.
- Document a Homebrew `libomp` fallback for LightGBM only if the selected wheel/runtime needs it.
- FAISS CPU packaging differs across pip/Conda and architectures; test the locked route on Apple Silicon, document a Conda fallback, and keep a NumPy exact-search smoke fallback for diagnostics—not portfolio operation.
- PyTorch runs CPU by default. MPS on the M3 is optional, explicitly configured, and benchmarked for correctness, latency, and unified-memory pressure; no required code assumes CUDA or MPS.
- Bound `OMP_NUM_THREADS`, BLAS, LightGBM, PyTorch, and tokenizer parallelism to prevent oversubscription. Begin with 2–4 compute threads and tune from measurements rather than using all cores automatically.
- Keep a practical process-RSS target of 5.5 GB for required workflows. Run embedding, training, evaluation, and serving as separate processes/stages so memory returns to macOS between them; do not keep Polars/Pandas training frames alive while loading models or indexes.
- Use `pathlib` and repository/config-relative paths; no Linux-only `/proc`, fork assumptions, or shell-only critical logic.
- Multiprocessing entry points must be guarded for macOS `spawn`; deterministic chunk assignment is required.
- Intel Macs use the same CPU path with potentially smaller batch/profile settings.
- Docker is optional, does not replace native macOS documentation, and is not the required benchmark path on the 8 GB machine because its VM overhead reduces usable memory.

## 43. Failure Modes and Fallbacks

| Failure | Detection | Behavior |
|---|---|---|
| Dense model unavailable | Startup model/cache check | Mark dense unavailable; hybrid→BM25 when fallback allowed. |
| FAISS missing/corrupt | Manifest/checksum/load probe | Same fallback; readiness degraded; never rebuild online. |
| Sparse index unavailable | Startup probe | Dense-only allowed if explicitly configured; otherwise not ready. |
| Cross-encoder disabled/missing/timeout | Config/load/request deadline | Skip stage and preserve LambdaMART order. |
| Marketplace metadata missing | Bundle validation | Relevance-only results with policy disabled and visible warning; strict bundle may fail readiness. |
| Query parser rule fails | Caught domain error | Use minimal normalized query with empty structured attributes; log warning. |
| No candidates | Source results empty after eligible filtering | Try allowed alternate retriever/relax only soft filters; return empty structured response, never fabricate. |
| Only one retriever returns | Source status/candidate count | RRF works with one list; provenance and degraded flag set. |
| Ranker incompatible | Feature/model manifest mismatch | Do not score; fall back to configured hybrid/heuristic or fail readiness in strict mode. |
| Memory pressure | Startup estimate, 5.5 GB RSS guardrail, allocation exception, RSS monitoring | Avoid optional model, release stage processes, reduce configured batch/K, use memmap and lower-bound portfolio catalog; do not silently substitute the development profile. |
| Policy infeasible | Feasibility audit | Apply documented relaxation order; stock constraint remains hard. |
| Cache corrupt | Read/validation error | Evict affected entry/cache; compute without it. |
| Unexpected model score | finite/range validation | Drop optional stage or fail ranker path; report. |

## 44. Security and Privacy

The system is local and has no real personal user data, but it still:

- validates request schemas, Unicode, query length, top-K, body size, mode, and timeouts;
- resolves artifacts only under allowlisted roots and does not accept request-supplied file paths;
- verifies checksums and avoids loading untrusted pickle/joblib objects where safer native formats exist;
- pins dependencies, runs vulnerability/license review, and never commits downloaded data/model caches;
- binds to localhost by default and documents that public exposure needs authentication, TLS, and rate limiting;
- bounds concurrent neural work to resist resource exhaustion;
- redacts stack traces and local paths from API errors;
- logs hashed queries by default and provides retention/deletion guidance;
- never stores user profiles or claims enterprise-grade security.

## 45. Production-Scale Evolution

### 45.1 Local versus hypothetical production architecture

```text
 LOCAL PORTFOLIO                         HYPOTHETICAL PRODUCTION
 ----------------                         -----------------------
 ESCI + Parquet/DuckDB        ----->      Catalog/label lake + warehouse
 Polars single-node ETL       ----->      Distributed batch/stream processing
 bm25s local index            ----->      OpenSearch/Lucene retrieval tier
 FAISS CPU FlatIP/HNSW        ----->      Sharded ANN/vector service
 Parquet feature artifacts    ----->      Offline + online feature platform
 LightGBM local training      ----->      Managed distributed training jobs
 Files + local MLflow         ----->      Governed experiment/model registry
 In-process caches/SQLite     ----->      Distributed cache/feature services
 One FastAPI process          ----->      Autoscaled multi-region serving
 JSON logs + DuckDB           ----->      Central metrics/traces/log platform
 Streamlit demo               ----->      Marketplace web/mobile clients

 The stage contracts, schemas, lineage, fallbacks, and metrics migrate;
 infrastructure substitutions do not change the ranking semantics silently.
```

| Capability | Local implementation | Production analogue | Migration boundary |
|---|---|---|---|
| Data | Parquet + DuckDB/Polars | Data lake/warehouse + distributed compute | Normalized schemas/manifests |
| Sparse | `bm25s` index | OpenSearch/Lucene shards | Retriever candidate contract |
| Dense | FAISS CPU | Distributed ANN service | Normalized vector/query contract |
| Features | Versioned Parquet + shared Python definitions | Feature store/batch compute | Feature registry and parity tests |
| Training | LightGBM on one host | Scheduled/managed CPU jobs | Matrix/group/model manifest |
| Artifacts | Content-addressed directories | Object store + model registry | Artifact DAG and promotion rules |
| Tracking | JSON/Parquet + local MLflow | Hosted experiment platform | Canonical run schema |
| Serving | FastAPI single process | Autoscaled service mesh | Search API schemas/SLOs |
| Cache | LRU + SQLite | Redis/distributed cache | Versioned cache keys |
| Observability | JSON logs + local analysis | Central logs/metrics/traces | Event schema and stage timers |
| Policy | Greedy in process | Dedicated policy/reranking service or optimizer | Policy input/output/audit contract |

Production would add catalog updates, index shadow builds and atomic swaps, feature freshness, canaries, capacity planning, privacy controls, online experimentation, SLOs, and rollback. None is required locally.

## 46. Alternatives Considered

| Alternative | Why not the required default | Reconsider when |
|---|---|---|
| Elasticsearch/OpenSearch locally | Service/JVM and operations obscure the ML workflow on a MacBook | Fielded retrieval/filters or production parity becomes the primary goal |
| Pyserini required | Strong Lucene base but JDK setup and memory add friction | Reproducible local packaging is proven and BM25 semantics need Lucene |
| Large neural retriever | CPU embedding/serving cost and memory | Hardware budget or quality evidence changes |
| End-to-end transformer ranker | Expensive over candidate sets and weak structured-policy integration | GPU/latency budget and real interaction data exist |
| Transformer fine-tuning | Violates required compute focus; high experiment cost | Optional future environment provides justified compute |
| Hosted vector DB/model endpoint | Cost, network, credentials, offline failure | Never required; production organization may choose it |
| GPU training | Not available and unnecessary for LambdaMART | Optional future neural work |
| Reinforcement learning/bandits | No live rewards or safe exploration environment | Real logged/online feedback and governance exist |
| Graph neural networks | No rich interaction graph; complexity exceeds evidence | Product/co-view/purchase graph becomes available |
| LLM query rewriting | Paid/large local models violate simplicity and latency goals | Small local model shows measurable value |
| Online learning | No real stream, hard reproducibility | Feedback, monitoring, rollback, and bias correction mature |
| Collaborative filtering | No users/histories in ESCI | Privacy-reviewed interaction data exists |
| Weighted normalized hybrid | Score calibration is query-sensitive | Validation shows stable improvement over RRF |
| ANN required from start | FlatIP is simpler and exact at profile scale | Measured dense latency exceeds budget |
| Integer-program policy | Solver/latency/infeasibility complexity | Global constraint value is proven offline |

## 47. Risks and Mitigations

| Risk | Probability | Impact | Mitigation | Contingency |
|---|---|---|---|---|
| No behavioral labels | High | High | Limit claims to ESCI relevance and policy simulation | Add only future real/logged data module |
| Synthetic marketplace signals misunderstood | Medium | High | Provenance namespaces, UI banners, separate tables/reports | Disable marketplace features/policy in default demo |
| Open-catalog judgments incomplete | High | High | Call metric judged recall; fixed judged-pool rank evaluation | Report both tracks and unjudged rate |
| CPU neural latency | High | Medium | Optional top-20 TinyBERT, batching/cache | Disable neural stage |
| Embedding preprocessing time | Medium | Medium | Offline batches, checkpoints, profiles, memmap | Use development profile/resume |
| Memory exceeds the 8 GB unified-memory envelope | High | High | Target ≤5.5 GB process RSS, run stages sequentially, lazy scans, compact dtypes, bounded K, memmaps, lower-bound portfolio default | Reduce to the declared portfolio lower bound; disable optional neural stage; test compressed FAISS only with recall report |
| ESCI target leakage | Medium | High | Prohibited dependencies, registry review, out-of-fold rule | Remove suspect feature and rerun all affected experiments |
| Query-group leakage | Medium | High | Normalized-query grouping and hard split tests | Invalidate artifacts and regenerate |
| Product overlap misinterpreted | Medium | Medium | Report overlap and optional cold-start stress test | Qualify generalization claim |
| Overengineering | Medium | Medium | Milestone acceptance, one default per stage, optional gates | Cut optional modules before core scope |
| Hybrid/LambdaMART weak improvement | Medium | Medium | Fair candidates, feature audits, error slices | Publish honest result and retain simpler champion |
| Unfair metric comparison | Medium | High | Fixed candidates/queries, paired bootstrap, config lineage | Invalidate and rerun comparison |
| macOS FAISS/LightGBM issues | Medium | High | Pin tested wheels; Conda/Homebrew fallbacks | NumPy diagnostic; defer dense only if documented blocker |
| Optional scope expansion | High | Medium | Neural/full/counterfactual feature flags and milestone gates | Ship complete core with options disabled |
| Simulated policy harms Exact results | Medium | High | Score-gap protection, caps, validation NDCG budget, audits | Relevance-only fallback or weaker policy |
| Category proxy is poor | High | Medium | Label it derived, confidence/missing state, slice review | Disable category constraints/features |
| Artifact incompatibility | Medium | High | Manifest DAG, schema versions, startup probes | Fall back to prior promoted bundle |
| Non-deterministic native libraries | Medium | Medium | Threads/seeds/version pin/tolerances | Record variance and deterministic evaluation mode |

## 48. Implementation Milestones

Each milestone ends in a working repository state. Complexity is relative (`S`, `M`, `L`). “Goldfish” lists likely documents, not full tasks.

| Milestone | Objective | Inputs → outputs | Acceptance criteria | Dependencies | Complexity / likely Goldfish |
|---|---|---|---|---|---|
| M0 Environment | Package/tool/config skeleton and fixture conventions | Elephant → installable quality-gated repo | macOS setup, locked deps, tests/linters run | Approved Elephant | M / repo skeleton; config loader; artifact manifest |
| M1 Ingestion/sampling | Official data load and complete-group profiles | raw files → manifests/sample IDs | checksums/schema; deterministic no-split groups | M0, downloaded data | M / downloader docs; loader; sampler |
| M2 Processed data | Normalized Parquet and validation | raw + sample → canonical tables/reports | keys/joins/null/locale/label checks pass | M1 | M / normalizer; validator; text builder |
| M3 BM25 baseline | Persisted sparse retrieval | documents → index/candidates | reload parity, unique top-K, latency/Recall report | M2 | M / tokenizer; sparse build; search |
| M4 Evaluation | Correct reusable retrieval/ranking metrics | candidates + judgments → reports | hand-computed tests and judged caveat | M3 | M / metrics; reports; bootstrap |
| M5 Dense | Offline embeddings and FAISS CPU | documents/model → vectors/index | resumable, finite/normalized, reload/latency | M2, model cached | L / embed pipeline; FAISS; dense retriever |
| M6 Hybrid | RRF union and evaluation | sparse+dense → hybrid candidates | deterministic dedupe; fair comparison | M3–M5 | M / fusion; end-to-end retrieval report |
| M7 Features | Shared feature registry/materialization | candidates + tables → matrices/state | provenance/leakage review; parity fixtures | M2, M6 | L / parser; feature groups; parity |
| M8 LambdaMART | Train pointwise and ranker | grouped matrices → models | valid groups; early stopping; reload parity | M7 | L / baselines; trainer; serializer |
| M9 Ranking eval | Fixed-pool/end-to-end ablations | models/predictions → reports | ABL-01–06/10 metrics, CIs, slices | M8 | M / scorer; ablation runner; explanations |
| M10 Synthetic data | Deterministic seller/product layer | products + seed → separate Parquet | order invariance, bounds, no label access, disclaimer | M2 | M / generator; diagnostics |
| M11 Policy | Marketplace-aware constrained reranking | relevance + metadata → final ranks/audit | stock/cap invariants; relevance/diversity report | M9, M10 | L / utility; constraints; policy tuning |
| M12 Neural optional | Compact top-K cross reranker | model cache + top ranks → scores | bounded latency/cache/fallback; ABL-09 | M9 | M / wrapper; cache; fusion |
| M13 FastAPI | Persisted-bundle serving | promoted artifacts → HTTP API | startup no rebuild; contract/degraded tests | M11; M12 optional | L / lifecycle; schemas/routes; orchestrator |
| M14 Streamlit | Interactive comparison client | API → demo | modes/toggles/cards/labels/metrics render | M13 | M / API client; UI; visualizations |
| M15 Hardening | Tests, profiling, docs, macOS verification | complete system → release candidate | required suites, latency/RSS/failure reports | M0–M14 | L / regression; profiling; runbooks |
| M16 Portfolio | Frozen experiments and presentation | promoted bundle/test → final report | ablations/tables/screenshots/reproduction/limitations | M15 | M / final runs; README/report |

Milestones M5 onward must not weaken earlier tests. M12 is skipped without blocking M13. M16 may conclude with an honest non-winning learned model if methodology is sound.

## 49. Goldfish Decomposition Strategy

A Goldfish addresses one bounded, independently testable change in a short coding session. It cites the Elephant section/decision it implements, names exact files, inputs, outputs, public interfaces, configuration, tests, acceptance commands, failure behavior, and rollback. It does not reconsider architecture unless it files an explicit design-change question. Each Goldfish leaves the default branch installable and tests passing; generated/large artifacts stay out of Git.

Proposed first sequence:

1. **Repository quality skeleton and macOS environment contract** — packaging, tool configuration, minimal test command, directory ownership, no ML logic.
2. **Strict configuration loader and canonical config hashing** — typed schema, layering, unknown-key rejection, deterministic hash.
3. **Artifact manifest and atomic stage-output protocol** — checksum/DAG contract with corruption tests.
4. **ESCI raw manifest and schema validation** — projected loader plus tiny legal fixture.
5. **Query-group split and deterministic profile sampler** — stable hash groups and leakage tests.
6. **Canonical normalized tables** — products/queries/judgments/source with key audits.
7. **Versioned product-document builder** — field markers, Unicode/missing/truncation fixtures.
8. **Core ranking metric library** — Recall/MRR/MAP/NDCG toy goldens and unjudged policy.
9. **Sparse tokenizer and smoke BM25 retriever** — simple in-memory reference.
10. **Persisted sparse portfolio index adapter** — build/load/search parity and benchmark.

Subsequent Goldfish follow milestone order. Avoid one Goldfish that simultaneously downloads ESCI, builds both indexes, trains a model, and serves an API; it would be hard to review and roll back.

## 50. Definition of Done

### 50.1 Checklist

- [ ] Official ESCI data is ingested reproducibly with source, license, version, and checksums.
- [ ] Development and portfolio profiles preserve complete query groups and are deterministic.
- [ ] Normalized query/product/judgment tables validate and retain authoritative ESCI labels.
- [ ] BM25 sparse retrieval builds, persists, reloads, searches, and is evaluated.
- [ ] Offline product embeddings and FAISS CPU build, persist, reload, and search.
- [ ] Hybrid retrieval deduplicates candidates, records provenance, and is evaluated fairly.
- [ ] Ranking metrics pass toy examples and disclose incomplete-judgment limitations.
- [ ] Query parsing and all feature groups have versioned definitions and offline/online parity.
- [ ] Pointwise and LightGBM LambdaMART baselines train with valid query groups and serialize.
- [ ] LambdaMART and required retrieval/ranking ablations have fixed-test metrics and CIs.
- [ ] Synthetic marketplace metadata is reproducible, separate, bounded, and label-independent.
- [ ] Marketplace reranking enforces stock/cap rules and reports relevance/diversity trade-offs.
- [ ] Optional cross-encoder, if included, is bounded and can be disabled without loss of core function.
- [ ] All expensive artifacts persist with manifests and reload with compatibility checks.
- [ ] Local experiment records reconstruct configs, data, code, models, metrics, hardware, and paths.
- [ ] FastAPI loads an explicit bundle and serves ranked results without rebuilding/downloading.
- [ ] Streamlit compares modes and clearly labels all simulated fields.
- [ ] Unit, integration, smoke, regression, determinism, artifact, data, and API tests pass.
- [ ] Latency and peak-memory measurements are published against, not substituted by, targets.
- [ ] Required development and lower-bound portfolio workflows run on the Apple M3/8 GB reference machine without CUDA, MPS dependence, or a paid API.
- [ ] README covers setup, architecture, reproduction, results, limitations, and optional modules.
- [ ] Final claims distinguish real labels, derived features, simulated metadata, and predictions.

Project completion does not require meeting a predeclared quality number, optional neural/full/counterfactual modules, or production deployment. It requires correct, reproducible evidence and an operational local core.

## 51. Portfolio Presentation

The final README should lead with the problem, result summary, and truth statement, then show:

1. one architecture diagram and a 60-second local demo path;
2. exact environment/data/model download and artifact-build commands;
3. a benchmark table for BM25, dense, hybrid, pointwise, and LambdaMART;
4. an ablation table with paired confidence intervals;
5. retrieval/ranking/policy funnel and latency/RSS table;
6. before/after ranking examples with stage provenance;
7. marketplace trade-off plots: NDCG delta versus seller HHI/new exposure/synthetic risk;
8. demo screenshots with synthetic labels visible;
9. reproducibility identifiers and instructions;
10. limitations, failure cases, ethics/provenance, and future work.

Recommended figures include query-level win/loss distribution, Recall@K curves, NDCG by query slice, stage-latency waterfall, rank-change plot, seller exposure Lorenz/HHI comparison, and policy Pareto frontier. Do not cherry-pick examples without also presenting seeded/random error analysis.

Honest resume bullet patterns:

- “Built a CPU-first multi-stage marketplace search prototype combining BM25, MiniLM/FAISS retrieval, LambdaMART, and constrained reranking; evaluated with query-level confidence intervals on Amazon’s public ESCI relevance judgments.”
- “Designed reproducible offline/online feature and artifact contracts, deterministic synthetic policy metadata, FastAPI serving, and a Streamlit rank-trace demo on macOS.”

Actual bullets must substitute measured profile sizes, metric deltas, p95 latency, and test counts only after final runs. Never imply real Amazon revenue, seller, conversion, or inventory modeling.

## 52. Future Work

Future extensions, clearly outside the required system:

- consented session histories and privacy-aware personalization;
- real impression/click/purchase data and calibrated engagement models;
- position-bias correction, propensity estimation, and counterfactual evaluation;
- contextual bandits and guarded online experimentation;
- local small-model query rewriting with measured latency/quality;
- multimodal text/image product embeddings;
- product/substitution/complement and seller graphs;
- temporal inventory, shipping, price, and index updates;
- learned return/cancellation risk on real governed outcomes;
- seller-side objectives with formal marketplace welfare/fairness review;
- production catalog ingestion, sharded retrieval, feature freshness, autoscaling, canaries, and centralized observability.

Every extension requires new data contracts, evaluation, privacy/ethical review, and a new decision record. It must not retroactively relabel synthetic MVP results as real.

## 53. Open Questions

These do not block the architecture but must be resolved by the named milestone:

| Question | Default assumption | Resolve by |
|---|---|---|
| Which official ESCI release/checksums are current at implementation time? | Pin the Amazon Science repository release retrieved for M1. | M1 |
| Does `bm25s` meet install/reload/latency needs on both Mac architectures? | Yes; retain `rank-bm25` reference and Pyserini option. | M3 |
| What exact profile query targets are feasible after preserving groups and splits? | 7.5k development and 20k portfolio are the 8 GB M3 defaults; scale upward only after profiling. | M1/M5 profiling |
| Is FlatIP under 200 ms for the portfolio catalog on the reference Mac? | Use FlatIP unless measured otherwise. | M5 |
| Is derived category quality adequate for a policy cap? | Category constraint off until coverage/precision audit passes. | M7/M11 |
| Should simulated marketplace features enter the champion relevance ranker? | No; default champion is relevance-only unless validation evidence and disclosure justify otherwise. | M9 |
| Which cross-encoder offers acceptable p95 CPU latency? | TinyBERT L-2; neural remains optional. | M12 |
| What relevance-loss guardrail value should policy use? | Select from validation Pareto curve; no invented fixed claim. | M11 |
| Which exact M3 Mac model and macOS version define the final benchmark? | Chip and memory are fixed at Apple M3/8 GB; record Mac model, OS, power state, and available disk in the final report. | M15 |

## 54. Decision Log

Each important decision uses the same fields and may be superseded only by an explicit ADR/Elephant revision.

### D-001 Dataset selection

**Decision:** Use official Amazon Shopping Queries ESCI as the authoritative relevance dataset.

**Context:** The system needs grouped shopping query–product judgments and realistic product text.

**Options considered:** ESCI; generic web IR corpora; synthetic relevance; proprietary click logs.

**Chosen approach:** ESCI English `us` profiles, official test held out, real E/S/C/I labels unchanged.

**Rationale:** Public, large, domain-relevant, graded, and license-documented.

**Trade-offs:** Bounded judged pools, no behavior, price, sellers, inventory, or canonical category.

**Future reconsideration trigger:** A legally usable dataset adds exhaustive catalog judgments or governed user interactions.

### D-002 Sparse retrieval library

**Decision:** Use `bm25s` or an equivalent persisted sparse CPU implementation as local default, with `rank-bm25` as smoke reference.

**Context:** The MacBook path needs simple install, bounded memory, persistence, and fast reload.

**Options considered:** `rank-bm25`, `bm25s`/equivalent, Pyserini/Lucene, local OpenSearch.

**Chosen approach:** Persistable sparse default plus a tiny transparent reference.

**Rationale:** Avoids JVM/service overhead and Python token-list cost while preserving BM25.

**Trade-offs:** Less production parity and a smaller ecosystem than Lucene.

**Future reconsideration trigger:** Cross-architecture install fails, parity is incorrect, or measured performance misses budgets.

### D-003 Dense model

**Decision:** Default to `sentence-transformers/all-MiniLM-L6-v2`.

**Context:** Required inference is CPU-first on an Apple M3 with 8 GB unified memory.

**Options considered:** all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, BGE-small-en-v1.5.

**Chosen approach:** Pin an all-MiniLM-L6-v2 revision and normalize 384-D vectors.

**Rationale:** Compact, widely supported, fast, and portfolio-recognizable.

**Trade-offs:** Not specifically fine-tuned for ESCI and may underperform larger/domain models.

**Future reconsideration trigger:** Fixed-profile evaluation shows another compact model materially improves quality within latency/memory budgets.

### D-004 FAISS index type

**Decision:** Use CPU `IndexFlatIP` first.

**Context:** Exact, reproducible retrieval is preferable at local profile scale.

**Options considered:** FlatIP, HNSW, IVF/PQ, GPU indexes.

**Chosen approach:** Exact inner-product search over normalized vectors at the lower-bound portfolio size; validate HNSW only for latency and compressed IVF/PQ only for memory.

**Rationale:** No training, deterministic reference, simple persistence, manageable memory.

**Trade-offs:** Linear query cost grows with catalog size.

**Future reconsideration trigger:** Measured portfolio p95 exceeds budget or the combined serving bundle approaches the 5.5 GB process-RSS guardrail.

### D-005 Hybrid fusion

**Decision:** RRF for initial order plus union for learned ranking.

**Context:** BM25 and cosine scores have incomparable, query-varying scales.

**Options considered:** Min-max weighted sum, z-score fusion, Borda/rank fusion, RRF, learned fusion.

**Chosen approach:** RRF with validation-configured constant, preserve raw source features for LambdaMART.

**Rationale:** Robust, deterministic, transparent, little calibration burden.

**Trade-offs:** Ignores raw score magnitude in pre-rank order.

**Future reconsideration trigger:** A calibrated/learned fusion wins robustly on fixed validation and slices.

### D-006 Feature storage

**Decision:** Parquet is canonical; compact in-memory/binary matrices are disposable accelerators.

**Context:** Features need inspection, reuse, versioning, and CPU-efficient training.

**Options considered:** CSV, SQLite/DuckDB only, Parquet, online feature store.

**Chosen approach:** Partitioned Parquet with registry/state manifests and DuckDB/Polars reads.

**Rationale:** Columnar projection, types, compression, local tooling, no service.

**Trade-offs:** Point lookups require an adapter and online features still compute per request.

**Future reconsideration trigger:** Real-time feature freshness or multi-service serving becomes required.

### D-007 Primary ranker

**Decision:** LightGBM LambdaMART.

**Context:** Mixed tabular/text-match features, graded labels, query groups, CPU budget.

**Options considered:** Heuristic, pointwise LightGBM, XGBoost Ranker, CatBoost ranker, neural ranker.

**Chosen approach:** LambdaMART with explicit gains and group arrays; pointwise remains baseline.

**Rationale:** Ranking-aware, fast, explainable, mature, and strong on engineered features.

**Trade-offs:** Cannot recover missed candidates and relies on feature quality.

**Future reconsideration trigger:** A competing ranker materially improves fixed-candidate NDCG within operational budgets.

### D-008 Optional reranker

**Decision:** TinyBERT L-2 cross-encoder over only top 10–30; MiniLM L-6 is optional quality alternative.

**Context:** Neural interaction may improve relevance but CPU latency is constrained.

**Options considered:** No neural stage, TinyBERT, MiniLM L-6, large/fine-tuned models.

**Chosen approach:** Optional, cached, bounded, failure-skippable stage.

**Rationale:** Demonstrates neural reranking without making it an availability dependency.

**Trade-offs:** Domain mismatch and up to seconds of latency.

**Future reconsideration trigger:** p95 exceeds budget, quality gain is negligible, or a smaller local model wins.

### D-009 Marketplace optimization

**Decision:** Deterministic hard eligibility plus greedy constrained reranking.

**Context:** Need to simulate diversity/quality/exploration while preserving relevance.

**Options considered:** Weighted score only, greedy constraints, MMR, LP, integer optimization.

**Chosen approach:** Bounded utility with explicit caps, feasibility relaxation, and relevance guard.

**Rationale:** Fast, explainable, easy to test, and supports constraints.

**Trade-offs:** Not globally optimal; ordering depends on chosen normalization/weights.

**Future reconsideration trigger:** Offline optimization proves meaningful global gains and solver latency/reliability is acceptable.

### D-010 Experiment tracking

**Decision:** Canonical JSON/Parquet runs plus optional file-backed MLflow.

**Context:** Reproduction must work without a running paid or local server.

**Options considered:** MLflow-only, flat logs only, Weights & Biases, custom database.

**Chosen approach:** Portable files are source of truth; MLflow mirrors for UI.

**Rationale:** Zero-cost, offline, inspectable, and robust.

**Trade-offs:** Less collaboration/search functionality than hosted platforms.

**Future reconsideration trigger:** Multi-user governance and centralized registry needs emerge.

### D-011 API architecture

**Decision:** One FastAPI service loads an explicit immutable serving bundle; Streamlit is a client.

**Context:** Local inference needs realistic boundaries without microservice overhead.

**Options considered:** Notebook-only, Streamlit direct model calls, FastAPI monolith, separate services.

**Chosen approach:** In-process modular stages behind versioned HTTP schemas.

**Rationale:** Testable, debuggable, production-shaped, simple locally.

**Trade-offs:** One process shares memory and cannot scale stages independently.

**Future reconsideration trigger:** Independent scaling, updates, or fault isolation become necessary.

### D-012 Dataset profile sizes

**Decision:** Three complete-query profiles with target ranges, not exact row slicing.

**Context:** Iteration speed and MacBook constraints differ from full benchmark research.

**Options considered:** One full dataset, random rows, fixed first-N queries, seeded group profiles.

**Chosen approach:** Development 5k–10k queries/50k–100k judgments; portfolio 20k–50k/200k–500k with the 8 GB M3 required default at approximately 20k/200k; full optional.

**Rationale:** Preserves ranking groups, supports quick iteration, and scales final evidence.

**Trade-offs:** Profile results are not full-dataset benchmark results, and the 8 GB reference prioritizes the lower end of the portfolio range over larger local runs.

**Future reconsideration trigger:** Profiling supports larger groups or resource constraints require lower midpoint targets.

---

## Recommended First Goldfish

**Title:** Goldfish 001 — Repository Quality Skeleton and macOS Environment Contract

**Scope:** Create only the minimal installable Python project structure, `pyproject.toml`, deterministic dependency-lock approach, Ruff/pytest/type-check/pre-commit configuration, Git ignore rules for data/artifacts/model caches, a tiny import/version module, and one smoke test. Document the required Apple M3/8 GB environment, additional Apple Silicon and Intel guidance, the 5.5 GB process-RSS target, and the initial-download/offline boundary. Do not ingest ESCI, implement configuration semantics, create retrievers, generate synthetic metadata, or add model code. Acceptance is a clean environment in which formatting, linting, type checks, and tests run successfully and no paid credential is referenced.
