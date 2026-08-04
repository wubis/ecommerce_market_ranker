# MarketRank Technical Design Document

| Field | Value |
|---|---|
| Document status | **Implemented — core roadmap complete through Goldfish 016A** |
| Intended audience | ML engineers, applied scientists, software engineers, reviewers, and future Goldfish authors |
| Authors | Justin Wang |
| Last updated | 2026-08-04 |
| Repository | `ecommerce_market_ranker` |
| System | MarketRank: Multi-Stage E-commerce Search and Ranking System |
| Required reference hardware | Apple M3 Mac with 8 GB unified memory; CPU-first, no CUDA |
| Decision authority | This document is the end-state blueprint. Approved Goldfish documents may clarify implementation details but must not silently redesign it. |

> **Evidence boundary.** MarketRank uses official Amazon ESCI query, product, locale, and relevance fields plus deterministic features derived from those fields and from retrieval/model outputs. ESCI relevance judgments are the only supervised targets. The required system does not manufacture business or offer data.

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
10. [Data Provenance and Derived Metadata](#10-data-provenance-and-derived-metadata)
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
22. [Catalog Diversity Reranking](#22-catalog-diversity-reranking)
23. [Diversity Constraints and Relevance Guardrails](#23-diversity-constraints-and-relevance-guardrails)
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
42. [macOS and Optional Colab Compatibility](#42-macos-and-optional-colab-compatibility)
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

MarketRank is a CPU-first, multi-stage e-commerce search and ranking system evaluated using real Amazon ESCI relevance judgments. It accepts a textual shopping query such as “wireless gaming mouse” or “red running shoes,” retrieves products from a fixed benchmark catalog, constructs explainable query-product features, and returns a ranked list. The core scientific story is lexical and semantic retrieval followed by learning-to-rank—not a notebook-only classifier and not an imitation of unavailable business systems.

The primary benchmark is the English (`us`) ESCI Task 1 reduced subset (`small_version == 1`). Official query-product lists provide all supervised rows. Label IDs are `I=0`, `C=1`, `S=2`, `E=3`; official Task 1 gains are `[0.0, 0.01, 0.1, 1.0]`. A lightweight parser extracts tokens, brands, colors, model identifiers, units, and compatibility terms. BM25 and `all-MiniLM-L6-v2`/FAISS retrieve candidates independently; reciprocal rank fusion (RRF) forms a hybrid union. LightGBM LambdaMART ranks candidates using only source and derived signals. A compact cross-encoder may rerank the head and is optional.

An additional optional catalog-diversity stage can reduce brand repetition and semantic redundancy using real ESCI brand/color/text fields and normalized product embeddings. It begins from the promoted relevance order, is deterministic, records every rank change, and is accepted only when validation-selected relevance guardrails pass. Pure LambdaMART—or LambdaMART plus the optional neural stage—remains a complete system when diversity is disabled.

The required reference machine is an Apple M3 Mac with 8 GB unified memory. The design uses sequential offline stages, lazy Polars/DuckDB processing, Parquet, memory-mapped arrays, conservative embedding batches, bounded candidate sets, and a 5.5 GB process-RSS target. Local execution is authoritative. Free Colab may optionally accelerate isolated batch jobs, but it is neither required nor trusted as the serving or final benchmarking environment.

The final deliverable is a reproducible repository with versioned artifacts, fair retrieval and ranking evaluations, query-level confidence intervals, ablations, a FastAPI service, a Streamlit comparison demo, automated tests, and honest limitations. This Elephant specifies that end state; it does not implement it.

### 1.1 End-to-end architecture

```text
                                    OFFLINE PLANE
  ESCI Parquet/CSV -> Validate -> Task-1 US filter -> group splits/profiles
          |                                         |
          v                                         v
  normalized queries/products/judgments     fixed catalog + judged pools
          |                                         |
          +--------> versioned product documents <--+
                               |                |
                         BM25 index       embeddings + FAISS
                               \                /
                                candidate generation
                                         |
                           pair rescoring + feature matrices
                                         |
                       pointwise baseline + LambdaMART training
                                         |
                       optional neural/diversity evaluation
                                         |
                    reports + experiment/artifact manifests

                                     ONLINE PLANE
  User/Streamlit -> FastAPI -> query parser -> BM25 || FAISS
                           -> RRF + dedupe -> pair features
                           -> LambdaMART
                           -> [optional cross-encoder]
                           -> promoted active-relevance contract
                           -> [optional catalog-diversity reranking]
                           -> explanations + ranked JSON response

  Startup loads one immutable serving bundle. It never downloads models,
  computes product embeddings, builds indexes, or trains a model.
```

### 1.2 Architectural principles

1. **Relevance truth is never manufactured.** ESCI labels are the only supervised targets.
2. **Candidate populations are explicit.** Closed-pool ranking and catalog retrieval answer different questions.
3. **Unjudged is not irrelevant.** Retrieved products outside a query’s official list remain unjudged.
4. **Source, derived, and predicted fields have explicit provenance.**
5. **Retrieve broadly, rank precisely, diversify optionally.**
6. **One promoted relevance contract feeds every optional downstream stage.**
7. **Offline and online feature formulas are identical and parity-tested.**
8. **Artifacts are immutable, content-addressed, and compatibility-checked.**
9. **Local M3/8 GB execution is required; optional remote acceleration cannot hide local infeasibility.**

## 2. Background and Motivation

Product search must reconcile several kinds of match. Exact items satisfy the expressed intent; substitutes satisfy most of it; complements may be useful without deserving the first position; irrelevant products should be suppressed. Lexical methods are strong on brands, rare tokens, and model numbers but miss paraphrases. Semantic methods connect related concepts but can blur critical distinctions such as “case for phone” versus “phone,” compatibility, sizes, and negation.

Catalog text is noisy. Titles may repeat keywords, descriptions may be missing, colors may be inconsistent, and brands may have aliases. Queries are short and ambiguous. The same product can be judged for multiple queries, so row-level splitting leaks ranking context. Downstream learning requires complete query groups and a clear rule for products that were retrieved but never judged for that query.

BM25 alone cannot represent semantic equivalence. Cosine similarity alone can miss exact identifiers and offers no structured interaction model. Neither learns how title overlap, brand conflict, color match, model-number agreement, and retrieval evidence interact. A staged architecture permits broad candidate recall, learned ordering, optional fine text interaction, and stage-local diagnostics.

Result lists can also be repetitive. Several near-duplicate products or one brand may dominate the head even when comparably relevant alternatives exist. A narrowly scoped diversity stage can demonstrate relevance-versus-redundancy trade-offs using observed catalog text and embeddings. It does not make causal, business-value, or fairness claims.

## 3. Scope

### 3.1 Required

- Ingest and validate the three official ESCI files.
- Freeze the primary benchmark to English (`us`) Task 1 rows with `small_version == 1`.
- Preserve complete query groups and official train/test provenance.
- Create deterministic nested development and portfolio profiles.
- Normalize source tables and build versioned product documents.
- Separate the closed judged ranking pool from the fixed retrieval catalog.
- Parse real query text using deterministic rules and dictionaries.
- Build, persist, reload, and evaluate BM25 and FAISS CPU indexes.
- Fuse BM25 and dense candidates with RRF and deterministic deduplication.
- Materialize real/derived features with offline/online parity.
- Train pointwise and LightGBM LambdaMART models on authoritative judged pairs only.
- Evaluate retrieval, fixed-candidate ranking, and end-to-end diagnostics under named protocols.
- Track experiments and artifact lineage locally without a running server.
- Serve persisted artifacts through FastAPI and compare modes through Streamlit.
- Provide unit, integration, smoke, regression, data, API, determinism, latency, and memory tests.
- Run the development and required portfolio workflows locally on the M3/8 GB machine.

### 3.2 Optional but designed

- Compact top-10–30 cross-encoder reranking.
- Relevance-preserving catalog-diversity reranking.
- MPS acceleration after parity and memory validation.
- Free Colab acceleration for isolated offline batches.
- Full reduced-US benchmark run, all-US-products stress catalog, cold-start stress test, and counterfactual-learning study.
- Docker for portability, not for required local benchmarking.

### 3.3 Implementation boundary

Goldfish tasks may implement this design after approval. This Elephant task creates or modifies only this Markdown document. Notebooks may explore results later but cannot own core functionality.

## 4. Non-Goals

The required project does not provide:

- real-time distributed search, full Amazon-scale indexing, or a representation of Amazon’s live system;
- transformer fine-tuning, GPU-required training, distributed Spark, Kubernetes, or managed infrastructure;
- paid LLM, embedding, vector-database, tracking, or hosted-model services;
- user personalization, real online learning, reinforcement learning, or genuine A/B testing;
- real behavioral, revenue, seller, inventory, price, shipping, fulfillment, margin, conversion, review, sponsorship, cancellation, or return-probability modeling;
- synthetic marketplace metadata generation or interfaces retained for such generation;
- a claim that brand diversity is seller diversity, marketplace fairness, customer welfare, or causal improvement;
- a claim that unjudged retrieved products are irrelevant;
- a canonical category taxonomy not present in ESCI;
- exhaustive hyperparameter optimization or production cloud deployment.

## 5. Functional Requirements

### 5.1 Requirements table

| ID | Requirement | Acceptance evidence |
|---|---|---|
| FR-001 | Ingest official ESCI examples, products, and sources files with checksums, source, version, and license. | Raw manifest and validation report. |
| FR-002 | Normalize queries, products, judgments, and sources without changing authoritative labels. | Key and row reconciliation report. |
| FR-003 | Filter required rows to `us` and `small_version == 1`. | Dataset predicate manifest and invariant test. |
| FR-004 | Split and sample complete normalized-query groups deterministically; development is nested within portfolio. | Repeated query-ID checksum parity and no-leakage report. |
| FR-005 | Build the fixed Task-1 US retrieval catalog independently of query-profile size. | Catalog selection predicate, membership checksum, and count. |
| FR-006 | Build exact closed judged pools for supervised fitting and primary ranking evaluation. | One complete pool per included query; no unjudged rows. |
| FR-007 | Construct deterministic product documents with field markers and null handling. | Golden document tests and template version. |
| FR-008 | Parse normalized text, tokens, numbers, model identifiers, units, brands, colors, and compatibility terms. | Curated query fixtures. |
| FR-009 | Build, persist, load, and query a sparse index. | Reload parity, candidate invariants, and retrieval metrics. |
| FR-010 | Encode products offline and persist normalized vectors plus a FAISS CPU index. | Dimension, normalization, checksum, and reload tests. |
| FR-011 | Encode queries online without re-encoding the product catalog. | Startup and request profiling. |
| FR-012 | Retrieve BM25, dense, and hybrid candidates with source scores, ranks, and provenance. | Candidate schema and fixed-catalog tests. |
| FR-013 | Deduplicate products, validate catalog membership, truncate deterministically, and break ties by stable keys. | Toy fusion and duplicate tests. |
| FR-014 | Directly rescore every candidate with shared sparse/dense pair scorers for model features. | Complete pair-score coverage and parity tests. |
| FR-015 | Materialize a versioned, leakage-reviewed ranking matrix using only source and derived features. | Feature registry and offline/online parity report. |
| FR-016 | Train and serialize a pointwise baseline and LambdaMART with valid contiguous query groups. | Population manifest, model manifest, and held-out metrics. |
| FR-017 | Persist component predictions and exactly one active-relevance stage/rank/score-comparability contract. | Promotion and fallback tests. |
| FR-018 | Optionally cross-encode only the configured top 10–30 candidates. | Bounded-count, cache, latency, and disabled-path tests. |
| FR-019 | Optionally rerank a bounded candidate head for brand/semantic diversity while preserving relevance guardrails. | Deterministic output, rank audit, and validation Pareto report. |
| FR-020 | Compute protocol-valid retrieval, ranking, diversity, latency, memory, slice, and bootstrap outputs. | Long-form metric records with protocol IDs. |
| FR-021 | Track dataset, code, config, features, models, metrics, hardware, and artifacts locally. | Reconstructable run directory. |
| FR-022 | Serve health, search, model, artifact, and bounded debug-explanation endpoints. | FastAPI contract tests. |
| FR-023 | Provide a Streamlit client comparing ranking stages and diversity effects. | UI smoke test and screenshots. |
| FR-024 | Degrade to the highest valid relevance stage when optional components fail. | Failure-injection tests. |
| FR-025 | Reject incompatible bundles before readiness and never rebuild artifacts at startup. | Compatibility and startup-operation tests. |

## 6. Non-Functional Requirements

| ID | Requirement | Target or verification |
|---|---|---|
| NFR-001 | Zero monetary cost | No required credential or paid service. |
| NFR-002 | CPU-first macOS support | Required workflow passes on the Apple M3/8 GB reference machine. |
| NFR-003 | Memory envelope | Target process RSS ≤5.5 GB, leaving headroom for macOS; report actual peaks. |
| NFR-004 | Reproducibility | Input checksums, seeds, resolved configs, lockfile, code revision, and artifact hashes. |
| NFR-005 | Determinism | Same-environment sampling, fusion, rankings, and metrics match within declared floating tolerance. |
| NFR-006 | Modularity | Retrieval, features, ranker, neural, diversity, evaluation, and serving depend on explicit contracts. |
| NFR-007 | Testability | Pure toy fixtures cover formulas; slow/download tests are separately marked. |
| NFR-008 | Observability | Structured stage timings, counts, versions, fallbacks, cache status, and errors. |
| NFR-009 | Fast startup | Load a validated serving bundle; perform no expensive offline operation. |
| NFR-010 | Caching | Bounded version-aware caches for query parses, embeddings, and optional cross scores. |
| NFR-011 | Persistence | Atomic stage writes, manifests, and success markers. |
| NFR-012 | Local latency | Compare measured p50/p95 with Section 41 targets on the reference machine. |
| NFR-013 | Maintainability | `pyproject.toml`, Ruff, pytest, practical typing, pre-commit, and module ownership. |
| NFR-014 | Graceful degradation | Optional-stage failure preserves the previous valid relevance order. |
| NFR-015 | Auditability | Debug output traces source rank through final rank without causal claims. |
| NFR-016 | Safe artifacts | Allowlisted roots, checksums, schema versions, and safer native serialization formats. |
| NFR-017 | Offline core | After initial downloads, required training/evaluation/search/demo need no internet. |
| NFR-018 | Configuration discipline | No experiment-defining value exists only as a source-code constant. |

## 7. Success Criteria

No absolute relevance result is guaranteed. The project succeeds through correct comparisons and reproducible evidence.

| Dimension | Primary measure | Success gate |
|---|---|---|
| Candidate coverage | judged-relevant Recall@100 | Validation selects hybrid; frozen test reports whether it exceeds or ties the better single retriever without retuning. |
| Early retrieval | judged MRR@10, Exact Hit@10, known-judgment coverage@10 | BM25, dense, and RRF use the same catalog/query cohort. |
| Learned ranking | official-gain NDCG@10 and NDCG@20 | LambdaMART improves at least one primary closed-pool metric over raw direct-score/RRF order without an unexplained material regression on the other. |
| Objective comparison | paired pointwise-versus-LambdaMART delta | Same rows, features, groups, and evaluation protocol. |
| Neural option | NDCG delta and added p95 latency | Reported before any diversity stage; adoption is evidence-driven. |
| Brand diversity | unique brands@10, brand HHI/entropy@10 | Optional stage improves at least one measure on queries with sufficient known-brand alternatives. |
| Semantic diversity | intra-list embedding distance@10 | Optional semantic penalty improves mean diversity on the validation-selected configuration. |
| Relevance preservation | NDCG@10/@20 before versus after diversity | Mean loss stays within the configured validation-selected budget; Exact displacement and tail losses are reported. |
| Integrity | duplicates and rank lineage | Zero duplicate products; every optional rank change is auditable. |
| Performance | stage and total latency | Actual p50/p95 reported against targets; no hidden rebuild. |
| Memory | peak RSS | Development and required portfolio workflows target ≤5.5 GB on the M3/8 GB machine. |
| Reproducibility | hashes and metric parity | Two clean development runs match deterministic outputs/tolerances. |
| Reliability | tests and artifact reload | Required test suites pass; indexes/models reload with parity. |

If hybrid, LambdaMART, neural reranking, or diversity does not improve its target metric, the project can still be complete when the evaluation is sound, the result is disclosed, and the simpler stage remains champion.

## 8. Data Sources

### 8.1 Authoritative source and benchmark

Use the [Amazon Science ESCI repository](https://github.com/amazon-science/esci-data) and cite the [Shopping Queries Dataset paper](https://arxiv.org/abs/2206.06588). Pin the release/retrieval date and SHA-256 checksums. Raw files remain immutable and Git-ignored.

Required primary population:

- `product_locale == "us"`;
- `small_version == 1` (official reduced Task 1 subset);
- official train rows for project train/validation construction;
- official test rows for the frozen final test;
- every row belonging to each selected query group.

The published reduced US subset contains 29,844 queries and 601,354 judgments before project collision quarantine/profile sampling. Release-level counts are checksum-versioned expectations, not constants silently accepted for a different release.

### 8.2 Dataset schema table

| File | Field | Normalized type | Role |
|---|---|---:|---|
| examples | `example_id` | `int64` | Source-row traceability. |
| examples | `query` | `utf8` | Raw query; consistency checked per `query_id`. |
| examples | `query_id` | `int64` | Ranking group identifier. |
| examples | `product_id` | `utf8` | Product key joined with locale. |
| examples | `product_locale` | categorical | Required value `us`. |
| examples | `esci_label` | categorical | Authoritative `E/S/C/I` target. |
| examples | `small_version` | integer/bool | Required Task 1 membership flag. |
| examples | `large_version` | integer/bool | Preserved source flag; not used to expand primary benchmark. |
| examples | `split` | categorical | Official train/test provenance. |
| products | `product_id` | `utf8` | Product key component. |
| products | `product_locale` | categorical | Product key component. |
| products | `product_title` | nullable `utf8` | Source title. |
| products | `product_description` | nullable `utf8` | Source description. |
| products | `product_bullet_point` | nullable `utf8` | Source bullets/attributes. |
| products | `product_brand` | nullable `utf8` | Source brand; normalized copy is derived. |
| products | `product_color` | nullable `utf8` | Source color; normalized copy is derived. |
| sources | `query_id` | `int64` | Query key. |
| sources | `source` | categorical | Retained for analysis; prohibited as an unreviewed shortcut. |

### 8.3 Labels and gains

| ESCI label | Label ID | Official Task 1 gain |
|---|---:|---:|
| Exact (`E`) | 3 | 1.0 |
| Substitute (`S`) | 2 | 0.1 |
| Complement (`C`) | 1 | 0.01 |
| Irrelevant (`I`) | 0 | 0.0 |

LightGBM receives integer label IDs and the explicit `label_gain=[0.0,0.01,0.1,1.0]`. Primary DCG uses the published gains directly: `gain/log2(rank+1)`. It must not apply another exponential transform. Metric records include `label_mapping_id` and `gain_mapping_id`. Retrieval Recall defaults to `E+S`, with `E`-only and optional `E+S+C` views named explicitly.

### 8.4 Missing values and deduplication

- Product PK: `(product_locale, product_id)`.
- Judgment PK: `(query_id, product_locale, product_id)`.
- Identical duplicates collapse with counts; conflicting products or labels fail strict validation.
- Source nulls remain null in normalized tables. Derived documents use empty segments plus missingness flags.
- Products with no usable source text are excluded from the retrieval catalog with an audit.
- Raw display text remains separate from normalized retrieval text.

### 8.5 Product document

```text
[TITLE] title [BRAND] brand [COLOR] color [BULLETS] bullet text
[DESCRIPTION] truncated description
```

The versioned builder strips HTML, applies Unicode NFKC, removes control characters, collapses whitespace, and caps long fields. Retriever tokenization may lowercase; stored display text does not.

### 8.6 Dataset limitations

ESCI supplies bounded judged lists, not exhaustive judgments over a live catalog. An absent judgment means unknown, not irrelevant. ESCI also lacks authoritative seller, inventory, price, shipping, fulfillment, margin, conversion, review, product-age, sponsorship, cancellation, return-probability, and canonical-category fields. Those limitations constrain claims and prevent business-rule evaluation in the required system.

### 8.7 Candidate and evaluation populations

1. **`esci_task1_us_judged_pool_v1`:** the complete official judged list for each included query. This is the only required population for supervised fitting, early stopping, pointwise/LambdaMART comparison, and official-gain NDCG.
2. **`esci_task1_us_catalog_v1`:** full-catalog mode: distinct products referenced by any `us`, `small_version == 1` example across both official splits, joined to official products. Membership reads participation fields but not label values.
3. **`esci_task1_us_compact_catalog_v1`:** required local benchmark mode: every distinct product judged for a selected portfolio query plus a configured seeded SHA-256 sample of additional Task-1 US products. Selection reads portfolio membership and stable product keys but never label values, gains, model output, or evaluation results. Development and portfolio share the same resolved compact membership.
4. **`end_to_end_diagnostic_v1`:** hybrid retrieval from the resolved named fixed catalog followed by ranker inference. Products outside the query’s judged pool remain unjudged; only qrels-aware retrieval metrics and explicitly conditional ordering diagnostics are valid.

An optional all-US-products catalog is a separately named stress test. Catalog manifests record mode, selection method, full source count, required judged-product count, distractor target/count, checksums, membership count/hash, missing documents, text template, and estimated/resident index sizes. Catalog mode, profile selection, seed, and distractor target are configuration-hashed. Compact metrics must never be described as full-catalog metrics.

## 9. Data Model

### 9.1 Provenance classes

| Class | Examples | Training rule |
|---|---|---|
| Source (`provenance=esci`) | query, title, description, bullets, brand, color | Source text/features allowed; label is target only. |
| Derived (`provenance=derived:<version>`) | normalized brand, token overlap, BM25 score, embeddings | Allowed when available online and leakage-reviewed. |
| Prediction (`provenance=model:<id>`) | LambdaMART/cross-encoder scores | Downstream use only; never feed a model its own same-fold prediction. |
| Diversity output (`provenance=diversity:<id>`) | penalties, rank change, cap relaxation | Audit/presentation only; not a relevance target. |

### 9.2 Logical schemas

| Entity/artifact | Purpose and primary key | Important fields | Format / expected size | Production analogue |
|---|---|---|---|---|
| `queries` | One query; `query_id` | raw/normalized text, locale, official/project split, source | Parquet; ~5k dev/~20k portfolio | Query warehouse |
| `products` | Canonical item; `(locale, product_id)` | source text, normalized brand/color, document, missing flags | Parquet; fixed catalog scale | Catalog service |
| `judgments` | Ground truth; `(query_id, locale, product_id)` | example ID, ESCI label, label ID, gain, official flags | Parquet; ~100k dev/~400k portfolio | Label store |
| `catalog_membership` | Named catalog; `(catalog_id, locale, product_id)` | selection version and document availability | Parquet + manifest | Catalog snapshot registry |
| `benchmark_candidate_pool` | Exact judged group; `(profile, split, query, product)` | label/gain and stable ordinal | Parquet | Curated evaluation set |
| `query_parse` | Parsed request; `(query/parser version)` | tokens, numbers, model IDs, units, brand/color/compatibility entities | Parquet offline, LRU online | Query-understanding service |
| `retrieval_candidates` | Source union; `(run, query, product)` | source scores/ranks, RRF, indicators, catalog ID | Partitioned Parquet | Candidate logs |
| `ranking_features` | Denormalized matrix; same candidate key | compact features, feature-set ID, protocol; labels only in offline judged matrices | Parquet + training matrix | Feature platform |
| `model_predictions` | Per-stage relevance; `(model, query, product)` | component score/rank, active stage, nullable active score, active rank, comparability | Parquet | Prediction log |
| `promoted_relevance_rankings` | One downstream relevance order; `(run, query, rank)` | active stage, component lineage, tie rule, fallback | Parquet/JSON | Ranking service output |
| `diversity_rankings` | Optional changed order; `(diversity run, query, product)` | input/output ranks, relevance score/rank, redundancy/brand penalties, constraint/relaxation actions, config ID, reason | Parquet/JSON | Diversification service log |
| `evaluation_outputs` | Metric facts; `(run, protocol, stage, slice, metric, cutoff)` | value, CI, query/known/unjudged counts, mapping IDs | Parquet/JSON/Markdown | Metrics warehouse |
| `experiments` | Reproduction lineage; `run_id` | config/data/code hashes, seeds, hardware, params, paths, status | JSON/Parquet + optional MLflow | Experiment platform |
| `artifact_manifest` | Compatibility DAG; `artifact_id` | type/schema/version/hashes/dependencies/created UTC/code revision | JSON sidecar | Artifact registry |
| `reranker_cache` | Optional cross-score cache; composite versioned key | score and created time | SQLite | Distributed cache |

Normalized source tables preserve truth; candidate, feature, prediction, and diversity artifacts are intentionally denormalized. Online feature payloads contain no labels.

## 10. Data Provenance and Derived Metadata

Only deterministic transformations of official fields and model/index outputs are required. Each derived column records definition version, source columns/artifacts, fit population, dtype, missing rule, and online availability.

Allowed product derivations include normalized brand/color strings, source-field lengths and missing flags, text completeness, versioned product documents, tokens, and normalized embeddings. Query derivations include normalized text, tokens, numbers, model identifiers, units, entity matches, and compatibility terms. Pair derivations include overlaps, conflicts, phrase matches, sparse/dense scores, and bounded ranks. Prediction and diversity values remain in separate downstream namespaces.

Rules:

- Derived values never overwrite source columns.
- Dictionaries/statistics are fitted on training or catalog text without test labels and persisted.
- Product identity may key caches but is not a direct high-cardinality relevance feature.
- No target-derived popularity or historical-relevance aggregate is required. Any future proposal needs out-of-fold computation and a leakage ADR.
- Brand/color normalization retains raw values, a normalized value, and missing/unknown state.
- Small hand-authored product fixtures are allowed for tests; they are not a production data generator.

## 11. Data Splitting Strategy

Random row splitting is invalid because it fragments ranked lists, corrupts group arrays, and leaks query context.

Default strategy:

1. Filter to `us` and `small_version == 1`.
2. Preserve official test groups for final evaluation. If the same normalized query text occurs in official train and test, keep test and quarantine the colliding train group from fitting.
3. Group remaining official-train rows by `(locale, normalized_query_text)` and assign approximately 85%/15% to project train/validation using a stable label-blind hash.
4. Move every row for a `query_id` together; assert split disjointness.
5. Create nested development and portfolio group sets using the same stable hash; never sample rows or select test queries from label composition.
6. Freeze test before feature/model/diversity selection.

Product overlap across query splits is permitted and reported because the default task tests new-query ranking over a shared catalog. An optional product-held-out stress test is separately named. There is no valid temporal claim because ESCI supplies no required time field.

## 12. Data Processing Pipeline

### 12.1 Restartable offline pipeline

```text
 [00 download + checksums]
             |
 [01 schema/data validation] -> failure report/quarantine
             |
 [02 normalized Task-1 US tables]
             |
 [03 grouped splits + nested profiles]
             |
 [04 fixed retrieval catalog + closed judged pools]
             |
 [05 product documents]
        /                 \
 [06 BM25 index]    [07 embeddings + FAISS]
        \                 /
 [08 retrieval candidates + closed-pool pair scores]
             |
 [09 query parsing + feature materialization/parity fixtures]
             |
 [10 pointwise + LambdaMART training]
             |
 [11 optional neural scoring + active-relevance promotion]
             |
 [12 optional diversity tuning/evaluation]
             |
 [13 frozen test reports + serving bundle promotion]

 Every stage: resolved config/dependency hashes -> temporary output -> validate
 -> atomic rename -> manifest + success marker. Matching outputs are reusable.
```

### 12.2 Stage rules

- Download is explicit and never occurs at API startup.
- Validation checks exact schema, keys, labels, locale, flags, counts, and joins.
- Polars lazy scans and DuckDB project only needed columns.
- Split/profile selection uses complete groups and label-blind hashes.
- Catalog membership is independent of query-profile sampling.
- Label IDs and official gains are persisted as different columns/mapping IDs.
- Indexes share one catalog and document version.
- Feature computation is candidate-aligned, never query×catalog materialization.
- Optional stages consume a persisted promoted relevance order.
- Interrupted output cannot be loaded; rebuilds create a new immutable version.

## 13. Query Understanding

The deterministic parser:

1. validates bounded UTF-8 input and preserves raw text;
2. applies NFKC normalization, lowercase retrieval view, whitespace collapse, and versioned regex tokenization;
3. extracts numbers, units, model identifiers, sizes/capacities, and compatibility phrases;
4. matches longest-boundary brands against a dictionary derived from official catalog brands;
5. matches colors against normalized official values plus a small versioned alias lexicon;
6. applies conservative spelling aliases only when unambiguous;
7. emits stopword-preserving and reduced token views while retaining negation, compatibility, unit, and model tokens.

Output includes normalized text, tokens, extracted entities/confidences, parser version, warnings, and a deterministic hash. Low-confidence entities become features, not hard filters. No LLM or heavy NLP package is required.

## 14. Candidate Retrieval

### 14.1 Contract

Each retriever receives a parsed query, `top_k`, locale, and `catalog_id`, and returns keyed `(product_id, raw_score, one_based_rank, retriever_id, index_id, latency_ms)`. Scores are finite; products belong to the catalog; ties use product ID.

Both retrievers also expose batch pair scoring over explicit product IDs. Closed-pool feature generation directly scores every judged pair even when it was absent from a top-K search result. Search and pair scoring use the same tokenization/document/model versions.

### 14.2 Sparse retrieval

Default local implementation: `bm25s` or an audited equivalent persisted sparse representation. `rank-bm25` is the transparent smoke reference. Pyserini is an optional comparison, not a required JDK dependency. Persist vocabulary, tokenizer metadata, BM25 parameters, postings/statistics, document map, and catalog/document hashes. Candidate default is `K_sparse=150`, validation-configured.

Failure modes include verbosity bias, keyword repetition, misspellings, and synonym gaps. Field markers and downstream interaction features mitigate rather than conceal them.

### 14.3 Dense retrieval

Default model: pinned `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional vectors. Product vectors are L2-normalized offline; query vectors are normalized online. FAISS inner product therefore equals cosine similarity.

Default index: CPU `IndexFlatIP` over the fixed catalog. It is exact, untrained, simple, and the quality reference. HNSW is a validated latency alternative but generally uses more memory. IVF/PQ is considered only for measured memory pressure and must report recall against FlatIP on identical catalog membership. Candidate default is `K_dense=150`.

### 14.4 Candidate validity

Only locale/catalog membership, valid keys/documents, and duplicate removal can exclude candidates. Parsed brand/color/compatibility signals are ranking evidence, not hard filters. If one retriever fails, hybrid can use the other and records degraded provenance.

### 14.5 Retrieval technology comparison

| Option | Strength | Local cost | Caveat | Role |
|---|---|---|---|---|
| `rank-bm25` | Simple and transparent | Python token memory | Weak persistence/scale | Smoke reference |
| `bm25s`/equivalent | Fast sparse CPU and persistable | Moderate RAM | Smaller ecosystem | **Local default** |
| Pyserini/Lucene | Mature IR behavior | JDK/JVM setup | macOS friction | Optional comparison |
| FAISS FlatIP | Exact dense search | ~146 MiB/100k 384-D float32 vectors plus IDs | Linear scan | **Dense default** |
| FAISS HNSW | Lower latency at scale | Additional memory | Approximate/tunable | Latency fallback |
| OpenSearch | Fielded retrieval and operations | Service overhead | Unnecessary locally | Production analogue |

## 15. Hybrid Retrieval

Weighted normalized fusion is sensitive to incomparable query-specific score scales. RRF uses only ranks and is robust with little tuning:

```text
rrf(product) = Σ_source 1 / (rrf_constant + source_rank)
```

Default: union BM25 and dense results, merge duplicates, retain both scores/ranks and source indicators, sort by RRF then best source rank then product ID, and truncate to `K_union` (candidate default 200). A product returned by one source receives only that source’s RRF contribution.

Evaluation populations remain distinct:

- **Closed pool:** complete official judged products; direct pair scores determine BM25/dense/RRF baselines within that pool.
- **Catalog retrieval:** source top-K and hybrid union from the fixed catalog; use qrels-aware retrieval metrics.
- **End-to-end diagnostic:** LambdaMART scores the retrieved union; unjudged products never become training rows.

## 16. Embedding Pipeline

1. Resolve a pinned model revision from the local cache; initial download is explicit.
2. Sort product IDs deterministically and stream product documents.
3. Start batches at 16–32 on the M3/8 GB machine; increase only after RSS measurement.
4. Encode `float32`, normalize, and validate dimensions, finite values, and norms.
5. Write a temporary `.npy` memory map and aligned product-ID array.
6. Checkpoint verified contiguous row ranges for restart.
7. Build/persist FAISS, write manifest, and atomically promote.

Artifact identity includes catalog, document template, model slug/revision, dimension, dtype, normalization, and config hashes. API startup loads the cached query encoder and FAISS; candidate vector lookup uses FAISS reconstruction or bounded rows from a read-only memory map. Product embeddings are never recomputed at startup.

Optional Colab output is accepted only when it uses the same resolved config/model revision and returns platform-neutral vectors, ID map, checksums, and manifest. The FAISS reference index is rebuilt or parity-checked locally.

## 17. Feature Engineering

### 17.1 Feature policy

One registry defines batch and serving formulas, dtypes, provenance, defaults, fit state, and missing behavior. Statistical/dictionary state is fitted without test labels and persisted. Compact `float32` and small integer types are preferred.

### 17.2 Feature catalog

| Feature | Class | Definition / rationale | Availability | Leakage/storage |
|---|---|---|---|---|
| Query char/token counts | Query | Text lengths indicate specificity | Train/online | Computed; low risk |
| Unique-token ratio | Query | Unique tokens divided by token count | Both | Computed |
| Digit/model-token count | Query | Regex token classes; helps exact identifiers | Both | Computed |
| Brand/color detected + confidence | Query | Parser match against real catalog values | Both | Dictionary versioned |
| Lexical specificity | Query | IDF summary over query tokens from catalog corpus | Both | Corpus/index versioned, no labels |
| Locale | Query/source | Required locale categorical | Both | Source |
| Title/description/bullet lengths | Product | Character/token counts by source field | Both | Product Parquet |
| Source-field missing flags | Product | One flag per official field | Both | Product Parquet |
| Normalized brand/color | Product | Versioned normalization of official strings | Both | Raw retained; unknown code |
| Product-text completeness | Product | Fraction/weighted count of nonempty source text fields | Both | No labels |
| Direct BM25 score | Retrieval-derived | Pair score for every candidate | Both | Index/version coupled |
| BM25 bounded rank fraction | Retrieval-derived | `(rank-1)/max(n-1,1)` within current candidate set | Both | `[0,1]`; absolute rank diagnostic only |
| Direct dense cosine | Retrieval-derived | Normalized query-product dot product | Both | Model/version coupled |
| Dense bounded rank fraction | Retrieval-derived | Same bounded formula | Both | Candidate-population shift reported |
| Closed-set RRF score/rank fraction | Retrieval-derived | RRF recomputed from direct pair-score ranks | Both | Candidate-set semantics versioned |
| Source top-K scores/ranks/RRF | Retrieval provenance | Actual retrieval-generator outputs | End-to-end only | Stored for diagnostics; excluded from `ltr_core_v1` |
| Sparse-only/dense-only/both indicators | Retrieval provenance | Which generators returned product | End-to-end only | Optional candidate-conditioned model only |
| Title token Jaccard/coverage | Interaction | Intersection/union and query-token coverage | Both | Tokenizer versioned |
| Description overlap | Interaction | Query coverage in description tokens | Both | Computed |
| Bullet overlap | Interaction | Query coverage in bullet tokens | Both | Computed |
| Exact phrase match | Interaction | Boundary-aware normalized substring | Both | Empty-query guarded |
| Brand match/conflict | Interaction | Parsed brand vs normalized official brand | Both | Confidence and missing states |
| Color match/conflict | Interaction | Parsed color vs normalized official color | Both | Confidence and missing states |
| Model-number match/conflict | Interaction | Exact normalized identifier comparison | Both | Conservative parser |
| Compatibility-term match | Interaction | Query compatibility token vs product text | Both | Rule versioned |
| Semantic similarity | Interaction | Direct dense cosine | Both when dense available | Approved missing fallback |
| Query-title length ratio | Interaction | Bounded log ratio | Both | Computed |
| LambdaMART score/rank | Prediction | Base relevance output | Downstream only | Separate prediction artifact |
| Cross score/rank | Prediction/optional | Bounded-head neural output | Optional | Model/cache coupled |
| Active stage/score/rank/comparability | Prediction | Promoted relevance contract | Downstream only | Bundle semantics validated |

### 17.3 Primary feature contract

`ltr_core_v1` contains query/product/interaction features, direct BM25 and dense scores, bounded within-set rank fractions, and closed-set RRF derived from direct pair ranks. It excludes original top-K provenance indicators because all closed-pool products are directly rescored and source membership would be population-dependent. Those fields remain available to an explicitly named `ltr_candidate_conditioned_v1` ablation trained only under a separately approved labeling protocol.

Absolute candidate rank/count, raw product ID, test-fitted encodings, labels/gains, and target-history aggregates are absent from the model matrix. Offline/online parity means identical formulas; distribution reports still compare the roughly 40-item closed pool with unions up to 200.

## 18. Baseline Ranking Methods

### 18.1 Model comparison table

| Method | Inputs | Why included | Limitation |
|---|---|---|---|
| Seeded random | Stable candidate keys | Metric sanity floor | No relevance evidence |
| BM25 | Direct sparse score | Lexical baseline | Semantic gaps |
| Dense | Cosine score | Semantic baseline | Identifier/constraint errors |
| RRF | Direct or retrieved ranks per protocol | Robust fusion baseline | Fixed rule |
| Weighted heuristic | Normalized scores + selected matches | Interpretable supervised-free fallback | Manual weights |
| Pointwise LightGBM | `ltr_core_v1` | Isolates benefit of supervision without rank objective | Ignores group ordering objective |
| LambdaMART | `ltr_core_v1` + groups | Required primary ranker | Cannot recover missed candidates |
| Cross-encoder fusion | Top LambdaMART candidates | Optional fine interaction | CPU latency |
| Diversity reranker | Active relevance + brand/embedding signals | Optional redundancy trade-off | May reduce relevance |

Primary ranking baselines use identical closed judged pools. Retrieval baselines use the fixed catalog. Random order is keyed by `(seed, query_id, product_id)`.

## 19. Learning-to-Rank Design

Pointwise models predict rows independently. Pairwise methods learn preferences but can overweight large groups. Listwise methods target ordered lists. LightGBM LambdaMART is primary because it is ranking-aware, CPU-efficient, strong on mixed tabular features, fast at inference, handles missing values, and supports importance/TreeSHAP.

Exact training construction:

1. Select official `us`, `small_version == 1`, train rows assigned to project train.
2. Retain every judgment in each selected query group; never add an unjudged row.
3. Join exactly one product; missing/conflicting joins fail rather than silently drop.
4. Assign label ID separately from official gain.
5. Build `ltr_core_v1`; directly score every judged pair with sparse/dense scorers.
6. Recompute bounded ranks and closed-set RRF within the complete judged pool.
7. Sort by stable query ordinal/product ID and construct contiguous group sizes.
8. Exclude groups with fewer than two rows or one distinct label from fitting with audited counts; retain them for defined evaluation.
9. Persist a training-population manifest with predicates, query IDs, exclusions, distributions, features, scorers, split hash, and checksums.

Use `objective=lambdarank` and `label_gain=[0.0,0.01,0.1,1.0]`. Fit categorical mappings on training only; reserve missing/unknown codes. Product IDs are not categorical features. No monotonic constraints are required. Early stopping uses validation NDCG. Test is evaluated only after selection.

Model serialization uses LightGBM text plus a manifest containing feature names/order/types, gains, params, dependencies, data/config/code hashes, validation history, and fallback compatibility. Prediction sorts descending with product-ID tie breaks. Gain/split importance and bounded TreeSHAP samples support developer explanations without causal interpretation.

XGBoost Ranker and CatBoost ranking are credible alternatives but not required duplicate stacks. Neural rankers remain optional because of CPU cost and less transparent integration.

## 20. Training Strategy

| Level | Data | Controls | Purpose |
|---|---|---|---|
| Smoke | Hand-authored fixture or 100–500 groups | Fixed params, 1–2 threads, tens of trees | Validate contracts quickly |
| Development | ~5k groups/~100k observed judgments | Cached features, early stopping, ≤5 seeded trials, 2–4 threads | Feature/model iteration |
| Portfolio | ~20k groups/~400k observed judgments | Frozen features, ~5–10 sequential trials on M3/8 GB | Final local experiments |
| Full reduced US (optional) | Up to 29,844 groups/601,354 judgments before quarantine | Champion config only | Optional scale extension |

Query count controls profiles; judgment count is observed because rows cannot be sampled. Trials run sequentially with compact matrices and controlled threads. Record wall time, peak RSS, machine/OS, seeds, feature/data hashes, and early-stopping history. Optuna is optional; a seeded parameter list is sufficient.

## 21. Optional Neural Reranking

Default optional model: `cross-encoder/ms-marco-TinyBERT-L-2-v2`; `cross-encoder/ms-marco-MiniLM-L-6-v2` is the quality-oriented alternative. Score only the top 10–30 (default candidate 20) from LambdaMART using `(normalized query, versioned product document)` pairs, small batches, controlled threads, and no gradient state.

Validation selects score normalization/fusion. Every request emits one active-relevance contract:

1. **Full-list comparable:** an explicit missing-neural rule maps every candidate onto a coherent fused scale; every candidate has an active score and `active_score_comparable=true`.
2. **Rank-only promotion:** the neural head order is followed by the deterministic LambdaMART tail; active score is null for the whole list and `active_score_comparable=false`.

The system never mixes neural-head and raw LambdaMART-tail values in one comparable score field. Disabled/failure behavior promotes LambdaMART unchanged. Cache keys include model revision, query hash, product/text hash, and preprocessing version. Missing model, timeout, memory error, or non-finite score skips the stage and logs the fallback.

## 22. Catalog Diversity Reranking

This optional stage reduces repeated brands and near-duplicate semantics after the active relevance order is finalized. Inputs are only product ID, official brand/color/text fields, normalized product embeddings, and active relevance scores/ranks.

The default is deterministic greedy MMR-like selection over a bounded head/pool:

```text
utility(product | selected) =
    normalized_active_relevance(product)
  - lambda_semantic * max_cosine_similarity(product, selected)
  - lambda_brand * same_known_brand_penalty(product, selected)
```

If `active_score_comparable=true`, relevance uses a versioned query-local score normalization. Otherwise it uses a monotone reciprocal/percentile transform of active rank. The mode is recorded. Semantic similarity uses normalized product embeddings from the active catalog/model version. The brand penalty applies only when both products have the same non-missing normalized brand. Missing brands do not match one another, do not form an artificial group, and do not count toward a brand cap.

At each rank, choose the highest utility, then higher active relevance, then lower input rank, then lexical product ID. `lambda_semantic`, `lambda_brand`, pool size, output K, protected ranks, and optional brand cap are selected on validation. Disabled mode returns the active relevance order byte-for-byte/rank-for-rank.

Every output row records input/output ranks, active stage/score/rank/comparability, redundancy penalty, brand penalty, selected-against product where relevant, cap/relaxation actions, config ID, and reason code. This is a descriptive catalog-list operation, not a causal or fairness intervention.

## 23. Diversity Constraints and Relevance Guardrails

Candidate validity (catalog membership, valid product key/document, no duplicates) is enforced before relevance ranking. The diversity stage cannot remove candidates for unavailable external attributes.

Required guardrails for any promoted diversity configuration:

- configurable `top_k` and candidate pool with hard API maxima;
- protect the first `P` active-relevance positions by default;
- permit movement only when normalized relevance gap is within a configured bound, or use protected-rank rules when scores are not comparable;
- optional hard same-known-brand cap in top K;
- never treat missing brand values as one identity;
- deterministic tie-breaking and complete pre/post lineage;
- validation-selected maximum allowable mean NDCG@10 degradation, with NDCG@20 and tail/query losses reported;
- Exact-product displacement counts and large-drop examples;
- relevance-first fallback on missing embeddings, invalid config, infeasibility, or guardrail failure.

For an infeasible brand cap, first fill from uncapped missing-brand items and other known brands; if fewer than K remain, relax the cap minimally one slot at a time and record each action. It never returns fewer results solely to satisfy diversity. The catalog-validity duplicate rule never relaxes.

Compared approaches:

| Method | Strength | Weakness | Decision |
|---|---|---|---|
| Brand penalty only | Very interpretable | Misses semantic duplicates | Ablation |
| Semantic MMR only | Captures near duplicates | Embedding errors/latency | Ablation |
| Combined greedy utility | Fast, deterministic, auditable | Locally rather than globally optimal | **Optional default** |
| Hard brand cap | Enforceable list constraint | Can be infeasible or harm relevance | Optional constraint |
| Integer optimization | Globally expressive | Solver/dependency complexity | Deferred |

## 24. Personalization

Personalization is future work because ESCI has no user or session history. A future system could use consented session context, recent queries, category/brand preferences, and user/session embeddings with retention, freshness, cold-start, and privacy controls. The required system always uses contextual query and catalog evidence only.

## 25. Offline Evaluation

### 25.1 Metric table

| Metric | Definition | Valid stage/use |
|---|---|---|
| Recall@K | Retrieved judged-relevant products / all judged-relevant products | Catalog retrieval; threshold named (`E`, `E+S`, optional `E+S+C`) |
| Exact Hit@K | Query has an Exact result in top K | Retrieval/ranking |
| Known-judgment coverage@K | Returned products with judgments / returned products | Catalog retrieval diagnostic |
| Unjudged rate@K | One minus known coverage | Catalog retrieval diagnostic |
| Precision@K | Relevant / K | Closed judged pool only |
| MRR@K | Reciprocal rank of first threshold-relevant product | Retrieval or closed pool with protocol named |
| MAP@K | Mean binary average precision | Closed pool only |
| NDCG@K | Official direct gains normalized by judged-pool ideal | Primary closed-pool ranking/diversity relevance |
| Unique brands@K | Distinct non-missing normalized brands | Diversity; report known-brand denominator |
| Brand HHI@K | Sum squared exposure shares among known brands | Diversity concentration |
| Brand entropy@K | Entropy of known-brand exposure | Diversity concentration |
| Intra-list diversity@K | Mean pairwise `1-cosine` among available product embeddings | Semantic diversity |
| Product coverage@K | Unique product IDs exposed across queries / catalog | Aggregate diversity |
| Duplicate violations | Repeated product IDs in a list | Must be zero |
| Relevance delta | NDCG after diversity − before diversity | Signed diversity effect |
| Relevance loss | `max(0, -relevance_delta)` | Guardrail harm measure |
| Exact displacement | Exact items moved beyond configured positions/cutoff | Diversity audit |
| Missing-brand rate | Missing brand values / evaluated results | Metric interpretability |
| Stage latency | p50/p95/p99 wall time after warmup | System evaluation |
| Peak RSS/artifact bytes | Process peak and persisted sizes | Resource evaluation |

Brand metrics describe catalog brand variety only and carry no fairness interpretation.

### 25.2 Evaluation protocol matrix

| Protocol ID | Candidate population | Allowed metrics | Prohibited interpretation |
|---|---|---|---|
| `closed_pool_task1_v1` | Complete official judged list/query | Official-gain NDCG, thresholded MRR/MAP/Precision, slices | Not retrieval from a catalog |
| `retrieval_catalog_task1_us_v1` | BM25/dense/RRF results from fixed catalog | Judged Recall, Exact Hit, judged MRR, known coverage, unjudged rate, latency | No naive Precision/MAP/NDCG over unjudged items |
| `end_to_end_diagnostic_v1` | Retrieved union scored by ranker | Retrieval metrics plus explicitly conditional ordering diagnostic | Not official Task 1 NDCG |
| `closed_pool_diversity_v1` | Same closed judged list before/after optional diversity | Before/after NDCG@10/@20, signed delta/loss, Exact displacement, brand/semantic metrics, latency | No business or fairness claim |

Every metric row includes protocol, candidate-population/catalog/profile IDs, stage/config IDs, threshold/gain mapping, query count, empty count, and known/unjudged counts where applicable.

## 26. Evaluation Methodology

- Freeze test queries, candidate populations, configs, and artifact versions before final evaluation.
- Compare rankers on identical complete judged pools; compare retrievers on the identical fixed catalog.
- Empty retrieval lists remain in the cohort and receive zero for applicable metrics.
- Compute per-query metrics and paired deltas. Bootstrap normalized-query leakage groups with replacement using a fixed seed for 95% confidence intervals, preserving every query/product row in each sampled group.
- Report mean, median, confidence interval, and tail examples where useful.
- Slice by query length, lexical specificity, brand/color/model/compatibility presence, source, judgment composition, and head/tail based only on non-label statistics.
- Report Exact-first and `E+S` retrieval behavior plus complement-heavy queries.
- Benchmark cold and warm startup, warmed repeated request latency, and modest concurrency.
- Record exact hardware, OS, power state, threads, and background conditions for RSS/latency.
- Inspect a seeded sample of wins, losses, parser errors, and large diversity moves.

Test results never select features, K values, neural fusion, diversity strength, or guardrails. A post-test correction creates a new named evaluation generation.

## 27. Required Ablation Studies

### 27.1 Ablation plan

| ID | Comparison | Fixed controls | Purpose |
|---|---|---|---|
| ABL-01 | BM25 retrieval | Catalog, query cohort, K, protocol | Lexical baseline |
| ABL-02 | Dense retrieval | Same as ABL-01 | Semantic baseline |
| ABL-03 | Hybrid RRF vs best single | Same catalog/cohort, fixed union cap | Complementary retrieval value |
| ABL-04 | Closed-pool RRF vs LambdaMART | Identical judged pools/gains | Learned ordering value |
| ABL-05 | Pointwise vs LambdaMART | Same rows/features/groups | Ranking-objective value |
| ABL-06 | LambdaMART before vs after diversity | Same active relevance order and pool | Overall relevance/diversity trade-off |
| ABL-07 | Diversity with vs without semantic penalty | Same brand term/guardrails | Semantic redundancy contribution |
| ABL-08 | Diversity with vs without brand penalty/cap | Same semantic term/guardrails | Brand concentration contribution |
| ABL-09 | LambdaMART vs LambdaMART + cross-encoder | Same closed pool/top-neural K; diversity off | Neural quality/latency |
| ABL-10 | Cross-encoder order before vs after diversity | Same promoted relevance and diversity config | Keep neural and diversity effects attributable |

Each row records protocol/population IDs, paired metric deltas/CIs, latency, RSS/artifact cost, config hashes, and query counts. Retrieval and closed-pool values never share an unlabeled metric column.

## 28. Position Bias and Counterfactual Evaluation

This optional research module requires impression logs with displayed positions, actions, and logging-policy propensities. ESCI has no such logs. A clearly isolated educational simulator may illustrate examination probability, inverse propensity scoring, self-normalized IPS, and doubly robust estimation, but no result from it enters the required benchmark or completion criteria. Claims require positivity, correct propensity handling, consistency, and no unmeasured confounding.

## 29. Explainability

User-facing reason templates are factual and predicate-backed:

- “Strong title-term match”
- “Brand matches your query”
- “Color matches your query”
- “Model identifier matches”
- “Strong semantic text match”
- “Selected to reduce near-duplicate results”

Developer debug output includes retrieval provenance, direct scores/ranks, feature values, LambdaMART score/rank, optional cross score, active stage/score/rank/comparability, diversity penalties/actions, stage-by-stage ranks, versions, and optional bounded TreeSHAP contributions. Explanations are associations and rule traces, never causal statements.

## 30. Local Storage and Artifact Management

### 30.1 Artifact dependency flow

```text
 raw checksums -> normalized tables -> split/profile manifests
                         |                    |
                         +-> catalog + judged pools
                                     |
                              product documents
                                /          \
                         BM25 index      embeddings -> FAISS
                                \          /
                           candidates + pair scores
                                     |
                            feature registry/matrices
                                     |
                          pointwise/LambdaMART model
                                     |
                     optional cross scores -> relevance promotion
                                     |
                         optional diversity configuration
                                     |
                     serving bundle + evaluations + reports

 A consumer loads only when every parent hash matches its manifest.
```

### 30.2 Artifact catalog

| Artifact | Format | Required metadata | Loaded when |
|---|---|---|---|
| Raw manifest | JSON | URL, license, bytes, checksum, retrieved UTC | Ingestion |
| Normalized tables | Parquet | Schema, counts, source hash | Offline stages |
| Split/profile manifest | JSON + query IDs | Hash rule, seed, nested counts, distributions | All experiments |
| Catalog membership | Parquet + JSON | Predicate, source/product hashes, count, missing docs | Index/retrieval |
| Closed judged pools | Parquet + JSON | Profile/split, group counts, label/gain mappings | Training/evaluation |
| Resource reports | JSON/Markdown | Projected/measured bytes and RSS, catalog hash | M2/M3/M5/M6/promotion |
| Product documents | Parquet | Template/tokenizer/truncation/catalog versions | Index/features |
| Sparse index | Native + NumPy/JSON | BM25/tokenizer/document-map hashes | Retrieval/API |
| Embeddings | `.npy` memmap + ID array | Model revision, dimension, dtype, norms | FAISS/features/diversity |
| FAISS index | `.faiss` + manifest | Type/params/vector and ID hashes | Retrieval/API |
| Candidates/pair scores | Partitioned Parquet | Index IDs, K, fusion, protocol/population | Features/evaluation |
| Feature registry/state | JSON/Parquet | Names, dtypes, provenance, defaults, fit state | Training/API |
| Feature matrix | Parquet/optional binary | Candidate/data/feature hashes, label inclusion | Training/evaluation |
| Training population | JSON + query IDs | Predicates, exclusions, distributions, feature/scorer IDs | Training/promotion |
| Models | LightGBM text + JSON | Features, gains, params, dependencies, metrics | Evaluation/API |
| Cross cache | SQLite | Model/input versions and size policy | Optional stage |
| Promoted relevance | Parquet/JSON | Active stage, components, fusion mode, nullable score, rank, comparability, fallback | Diversity/API |
| Diversity output/config | Parquet/JSON + YAML | Input/output ranks, penalties, actions, lambdas, cap, guardrails, validation run | Optional evaluation/API |
| Experiment run | JSON/Parquet + optional MLflow | Full lineage, hardware, metrics, status | Reporting |
| Evaluation report | Parquet/JSON/Markdown/plots | Every compared artifact and protocol hash | Portfolio |
| Serving bundle | Immutable directory manifest | Compatibility matrix and readiness mode | API startup |

Paths follow `artifact_type/dataset_version/profile/component_version/config_hash/`. Writes use temporary locations and atomic promotion. “Latest” is not a serving dependency; the API loads an explicit bundle ID. Large artifacts are Git-ignored.

## 31. Experiment Tracking

Canonical `run.json` and long-form `metrics.parquet` are always written. MLflow may mirror them into a local file/SQLite backend; no server is required.

Track dataset/checksums/profile/split, benchmark predicate, catalog/population/protocol IDs, label/gain mappings, training population, feature set, code revision/dirty hash, retriever/index/model IDs, active relevance contract, optional diversity config, hyperparameters, seeds, threads, candidate counts, metrics/CIs, latency, peak RSS, artifact paths/hashes, environment versions, hardware, and status. Failed runs retain their config/error but cannot be promoted. Final reporting names one validation-selected relevance bundle and, separately, an optional diversity configuration.

## 32. Configuration Management

Use strictly validated YAML with unknown-key rejection:

- `base.yaml`: paths, seed, locale, logging, threads;
- `profiles/*.yaml`: grouped query targets;
- `catalogs/*.yaml`: immutable membership predicates;
- `retrieval/*.yaml`: tokenization, BM25, dense model, K, FAISS, RRF;
- `features/*.yaml`: feature groups/registry versions;
- `ranking/*.yaml`: gains, LightGBM params/search/early stopping;
- `reranker/*.yaml`: model, top K, batch, cache, fusion/comparability mode;
- `diversity/*.yaml`: enabled, pool/K, lambdas, cap, protected ranks, guardrails;
- `evaluation/*.yaml`: protocol, population, gains, cutoffs, slices, bootstrap;
- `serving/*.yaml`: explicit bundle, caches, deadlines, strict/fallback mode.

Resolution is base → component/profile → explicit CLI override. Canonical sorted resolved config is hashed and copied into the run. Environment variables may change paths/ports but not silently alter model semantics.

## 33. API Design

### 33.1 Online inference pipeline

```text
 STARTUP: bundle ID -> manifest DAG/checksums/schema -> load display projection,
          BM25, FAISS, query encoder, feature state, LambdaMART
          -> optionally load cross-encoder -> bounded warm probes -> ready

 REQUEST: validate -> parse -> BM25 || dense -> RRF/dedupe
          -> direct pair scores/features -> LambdaMART
          -> [optional neural] -> active relevance promotion
          -> [optional diversity] -> explanations -> JSON + timings
```

### 33.2 Endpoints

| Endpoint | Purpose | Request | Response / errors |
|---|---|---|---|
| `GET /health/live` | Process liveness | None | Cheap status/timestamp |
| `GET /health/ready` | Artifact readiness | None | Components/degraded flags; `503` without valid relevance path |
| `POST /v1/search` | Ranked search | Bounded query, top K, `bm25|dense|hybrid|lambdamart`, optional neural/diversity flags, debug | Results, timings, active stage, fallbacks, versions; `422` invalid, `503` unavailable |
| `GET /v1/model-info` | Safe model summary | None | Model/feature/diversity versions and availability |
| `GET /v1/artifact-info` | Reproduction summary | None | Bundle/data/index hashes without unsafe local paths |
| `POST /v1/debug/explain` | Bounded rank trace | Search request + bounded candidates/IDs | Provenance, features, component/active ranks, diversity actions; local-only default |

Results may contain product ID, title, description snippet, brand, color, retrieval provenance, component and active relevance scores/ranks, comparability flag, optional diversity penalties/pre-post ranks, reason codes, timing, and artifact/model IDs. ESCI labels appear only for known evaluation queries and are explicitly ground truth. Caches include bundle/mode/options in keys and are bounded.

FastAPI depends on the relevance bundle only. Missing diversity artifacts disable that option without affecting readiness.

## 34. Streamlit Demo

Streamlit is a thin API client and provides:

- search box and ESCI-compatible examples;
- bounded top-K selector;
- BM25/dense/hybrid/LambdaMART comparison;
- optional neural and diversity toggles with unavailable-state explanations;
- cards using official product title, description snippet, brand, and color;
- retrieval/score/provenance and stage latency breakdowns;
- unique-brand, brand concentration/entropy, and intra-list-diversity metrics;
- pre/post diversity rank-change visualization and reasons;
- relevance effects only for known evaluation queries;
- artifact/model/config identifiers and degraded-mode banner;
- an always-available dataset-limitations note pointing to Section 8.6.

The UI never calls models directly, rebuilds artifacts, or claims live Amazon behavior.

## 35. Repository Structure

```text
ecommerce_market_ranker/
├── ELEPHANT.md
├── README.md
├── pyproject.toml
├── lockfile
├── configs/
│   ├── profiles/  catalogs/  retrieval/  features/
│   ├── ranking/   reranker/   diversity/
│   └── evaluation/  serving/
├── data/
│   ├── raw/  interim/  processed/  samples/
├── artifacts/
│   ├── embeddings/  indexes/  features/  models/
│   └── rankings/  evaluations/
├── src/market_rank/
│   ├── data/  query/  retrieval/  features/  ranking/
│   ├── reranking/  diversity/  evaluation/  serving/  utils/
├── scripts/
├── app/
├── tests/
│   ├── unit/  integration/  smoke/  regression/  fixtures/
├── notebooks/
├── experiments/
├── docs/
└── reports/
```

`src/market_rank` owns reusable logic. Scripts are thin CLI adapters. Notebooks only explore/import package code. `data` holds lifecycle tables, `artifacts` holds expensive derived state, `experiments` holds run lineage, and `reports` holds curated outputs.

## 36. Module and Interface Design

| Module | Responsibility | Inputs → outputs | Persistence / tests |
|---|---|---|---|
| `DatasetLoader` | Project official files under schema | paths → lazy frames | Raw manifest; missing/schema fixtures |
| `DatasetSampler` | Filter benchmark, quarantine collisions, create nested groups | judgments/seed/targets → IDs/manifest | Determinism/leakage tests |
| `CatalogBuilder` | Build label-value-independent catalog membership and judged pools | examples/products → manifests/tables | Predicate/checksum tests |
| `ProductTextBuilder` | Create versioned documents/display text | products/template → documents | Golden Unicode/null/truncation tests |
| `QueryParser` | Normalize and extract entities | raw query → parsed query | Curated fixtures; total-function behavior |
| `SparseRetriever` | Build/load/search/pair-score BM25 | documents/query/product IDs → index/candidates/scores | Reload/search/pair parity |
| `DenseRetriever` | Encode/build/load/search/pair-score | documents/query/product IDs → vectors/index/candidates/scores | Dimension/norm/exact fixtures |
| `HybridRetriever` | Union/dedupe/RRF/truncate | source candidates/config → union | RRF/tie/duplicate tests |
| `FeatureBuilder` | Shared pair feature contract | parsed query/products/candidates → frame | Registry/state/matrices; parity |
| `Ranker` | Build population, train/load/predict | closed features/labels/groups → model/predictions | Group/gain/serialization tests |
| `NeuralReranker` | Optional bounded scoring and promotion | query/head/base predictions → promoted relevance | Cache/fallback/comparability tests |
| `CatalogDiversityReranker` | Optional deterministic brand/semantic reranking | promoted order/products/vectors/config → changed order/audit | Guardrail, missing-brand, cap, lineage tests |
| `RankingEvaluator` | Enforce protocol-valid metrics/CIs/slices | rankings/judgments/protocol → metric facts/reports | Toy metrics/prohibited combinations |
| `ArtifactRegistry` | Manifest DAG and bundle promotion | artifacts/IDs → verified bundle | Corruption/incompatibility tests |
| `ExperimentTracker` | Canonical local runs | config/metrics/artifacts → run directory | Round-trip/failure tests |
| `SearchOrchestrator` | Compose online stages/fallbacks | request/bundle → response | End-to-end/API tests |

Public boundaries use immutable dataclasses or validated models. Domain exceptions distinguish invalid data, missing/incompatible artifacts, unavailable optional stages, and invalid requests.

## 37. Testing Strategy

- **Unit:** parser rules, document building, RRF, features, gains/NDCG, diversity utility, constraints, manifests.
- **Integration:** normalize→sample; build/load/query indexes; candidate→feature→ranker; promoted relevance→diversity; API with tiny persisted bundle.
- **Smoke:** one command builds a small hand-authored fixture bundle and searches it; a gated ESCI smoke run validates the official schema.
- **Regression:** golden documents, candidates, feature rows, metric values, rank lineage, and API schema.
- **Determinism:** shuffled input/chunking produces stable groups, fusion, diversity results, and hashes within native-library tolerance.
- **Artifact:** cold reload, checksum corruption, dimension mismatch, feature mismatch, and old schema.
- **API:** limits, errors, cache keys, degraded modes, concurrency, and debug redaction.

Critical invariants:

1. Every candidate belongs to its request/query and active catalog.
2. No normalized-query group crosses project splits.
3. Group sizes are positive/contiguous and sum to matrix rows.
4. No output contains duplicate product IDs.
5. Every `ltr_core_v1` row has an official ESCI judgment.
6. Direct sparse/dense scores exist for every closed-pool row.
7. Label IDs and official gains remain distinct and reproduce hand-calculated NDCG.
8. Development query IDs are nested in portfolio; catalog hash is identical.
9. Empty retrieval lists stay in evaluation with zero applicable metrics.
10. Bounded rank fractions are finite in `[0,1]`; singleton is `0.0`.
11. Offline/online feature formulas match.
12. Index/model reload preserves outputs.
13. API startup performs no download, encoding, index build, or training.
14. Neural disabled/failure promotes LambdaMART exactly.
15. Active score is populated for all candidates only when comparable; otherwise null for all.
16. Diversity disabled returns active relevance order exactly.
17. Diversity reranking is deterministic and respects maximum pool/K.
18. Missing brands never match one another or consume a shared cap bucket.
19. Same-brand cap and minimal relaxation audit agree.
20. Semantic penalty matches toy cosine examples.
21. Relevance-first fallback activates on missing vectors/config/guardrail failure.
22. Every diversity row has valid pre/post lineage and reason.
23. NDCG loss and Exact displacement reports match toy lists.
24. Diversity metrics match hand-computed examples.

Network/download and long performance tests are opt-in. Ordinary CI avoids brittle one-shot timing assertions.

## 38. Data Validation

Required checks:

- exact source schema/dtypes/checksums and nonzero rows;
- primary-key uniqueness and consistent query text per ID;
- supported locale, Task flag, split, and label domains;
- no normalized-query split overlap;
- complete group/profile nesting;
- duplicate/conflicting product/judgment handling;
- nonempty queries/documents and one product join per judgment/candidate;
- catalog membership matches the named full or required-judged-plus-label-blind-distractor
  predicate and its persisted selection counts;
- training population contains only included judged groups and audited exclusions;
- label/gain mappings and protocol IDs are valid;
- candidate keys, finite scores, ranks, deduplication, and bounds;
- embedding dimensions, alignment, norms, and finite values;
- feature schema/order/dtypes and no labels in online payloads;
- active-relevance stage/rank/score/comparability completeness;
- diversity input/output key equality, rank permutation, penalties, cap actions, and config lineage.

Reports include severity, pass/fail, counts, sample offending keys, and artifact version. Hard failures block promotion.

## 39. Observability

Structured JSON events contain UTC timestamp, level, request/run ID, bundle/catalog/index/model/config IDs, stage, duration, input/output counts, cache hit, fallback, warning/error class, and memory checkpoints. Raw queries are not logged by default; store a salted hash plus length/parser flags.

Online timers: parse, sparse, query embedding, dense, fusion, pair scoring/features, LambdaMART, neural, relevance promotion, diversity, serialization, and total. Diversity logs lambda/config, input/output K, cap relaxation, missing-brand count, and rank-change count. Offline logs rows/sec, batches, RSS, bytes, cache/reuse, and stage status. Logs remain analyzable with DuckDB/Polars.

## 40. Memory Management

| Workflow | Loaded/streamed | Controls |
|---|---|---|
| Ingestion | Projected Parquet scans and small aggregates | Polars lazy/predicate pushdown, DuckDB, no full pandas copies |
| Index build | Product IDs/documents or embedding batches | Truncation, batch 16–32 initially, memmap, controlled threads |
| Training | Candidate-aligned compact features/groups | Column projection, float32/small ints, one trial/process at a time |
| Evaluation | One stage/partition plus per-query metrics | Stream partitions and bounded bootstrap arrays |
| API startup | Compact display lookup, sparse index, FAISS, query encoder, ranker, dictionaries | Explicit bundle, no duplicate full frames, lazy neural load |
| Request | At most configured source/union/neural/diversity pools | Candidate-only vectorized work |

Do not simultaneously retain offline training frames and serving artifacts. Separate stages/processes allow macOS to reclaim memory. Use FAISS reconstruction or bounded memory-map rows rather than copying the complete vector matrix into another structure.

## 41. Latency Targets

### 41.1 Latency budget

| Stage | Warm target | Notes |
|---|---:|---|
| Query preprocessing | <50 ms | Parser/dictionaries |
| BM25 retrieval | <200 ms | Loaded index |
| Query embedding + dense search | <200 ms | Cache miss reported separately |
| Fusion/deduplication | <50 ms | Union bounded |
| Pair features | <250 ms | Union only |
| LambdaMART | <50 ms | Batch score |
| Diversity reranking | <100 ms | Optional bounded greedy stage |
| Serialization/overhead | <100 ms | Debug off |
| Total without neural | target <1 s | p50/p95 reported |
| Optional cross-encoder | <2 s | Top 10–30 CPU |
| Total with neural | target <3 s | Hit/miss separately |
| Cold startup | goal <30 s | No build/download |

Targets are design goals, not promises. For latency, reduce approved K values or validate HNSW. For memory, use compact representations only after quality comparison. Catalog membership never changes under the same catalog/protocol ID.

### 41.2 Ranking-stage funnel

```text
 Fixed Task-1 US catalog:       tens/hundreds of thousands
        BM25 top 150 ---\
                         +--> RRF union/dedupe: <=200
        dense top 150 --/              |
                                  LambdaMART: <=200
                                         |
                              optional neural: top 20
                                         |
                         active relevance order: <=200
                                         |
                     optional diversity pool: configured <=200
                                         |
                                  response: top 10–50
```

## 42. macOS and Optional Colab Compatibility

- Required machine: Apple M3 with 8 GB unified memory; record exact Mac model, macOS, power state, and free memory.
- Target Python 3.11 initially, subject to a locked compatible dependency set.
- Prefer tested ARM64 wheels; document Intel fallbacks and Homebrew `libomp` only if LightGBM needs it.
- FAISS CPU packaging must have a pinned macOS route plus a documented Conda fallback.
- CPU is authoritative. MPS is optional and must pass numerical parity, latency, and unified-memory profiling.
- Start with 2–4 compute threads and 16–32 embedding batch; avoid BLAS/PyTorch/LightGBM oversubscription.
- Target ≤5.5 GB process RSS and run embedding, training, evaluation, and serving as separate stages.
- Use `pathlib`; account for macOS `spawn`; avoid Linux-only assumptions.
- Docker is optional and not the required benchmark path because VM overhead reduces available memory.

Free Colab is an optional batch accelerator only. Suitable jobs are product embedding generation, optional cross scoring, or a separately named larger run. It must execute reusable package/CLI logic rather than notebook-only implementations, pin dependencies/model revisions, write manifests/checksums, and export platform-neutral artifacts. Colab’s variable hardware and ephemeral runtime cannot define final latency/RSS results, API serving, or the required local completion gate.

## 43. Failure Modes and Fallbacks

| Failure | Detection | Behavior |
|---|---|---|
| Dense model/index unavailable | Startup probe/checksum | Mark unavailable; hybrid→BM25 when allowed |
| Sparse index unavailable | Startup probe | Dense-only if allowed; otherwise not ready |
| One retriever returns nothing | Per-source count | RRF uses available list and marks degraded |
| No candidates | Empty valid union | Return structured empty result; never fabricate |
| Parser rule fails | Domain exception | Minimal normalization with empty entities |
| Ranker incompatible | Feature/model manifest mismatch | Fall back to configured hybrid/heuristic or fail strict readiness |
| Neural disabled/fails/times out | Config/deadline/score checks | Promote LambdaMART unchanged |
| Diversity disabled/artifact missing | Bundle/options check | Return active relevance order; service remains ready |
| Embeddings missing for diversity | Candidate/vector audit | Skip semantic term if config permits or relevance-first fallback |
| Diversity config infeasible | Cap/guardrail audit | Minimal documented cap relaxation or relevance-first fallback |
| Diversity guardrail fails offline | Validation report | Do not promote configuration |
| Cache corrupt | Read/checksum error | Evict and recompute without cache |
| Memory pressure | RSS/startup/allocation checks | Release stages, reduce approved batches/K, disable optional models; preserve catalog ID |
| Unexpected score | Finite/comparability validation | Drop optional stage or fail affected relevance path |

## 44. Security and Privacy

Validate query length/Unicode, top K, mode, body size, deadlines, and concurrency. Resolve artifacts only below allowlisted roots. Verify checksums and prefer native safe formats over untrusted pickle. Pin dependencies and review vulnerabilities/licenses. Bind locally by default; public exposure would require authentication, TLS, and rate limits. Redact traces/paths from responses, hash queries by default, and document log retention. ESCI supplies no real user profiles to this project.

## 45. Production-Scale Evolution

### 45.1 Local versus hypothetical production diagram

```text
 LOCAL PORTFOLIO                      HYPOTHETICAL PRODUCTION
 ESCI + Parquet/DuckDB        ---->   governed catalog/label lake
 Polars single-node stages    ---->   distributed batch/stream compute
 bm25s local index            ---->   OpenSearch/Lucene retrieval tier
 FAISS CPU                    ---->   sharded ANN service
 Parquet feature state        ---->   offline/online feature platform
 LightGBM on one Mac          ---->   scheduled training platform
 files + local MLflow         ---->   experiment/model registry
 FastAPI one process          ---->   autoscaled serving tier
 greedy diversity module      ---->   dedicated diversification service
 JSON logs + DuckDB           ---->   centralized metrics/traces/logs
 Streamlit                    ---->   product search clients
```

### 45.2 Component mapping

| Capability | Local | Production analogue | Preserved boundary |
|---|---|---|---|
| Data | Parquet, DuckDB, Polars | Lake/warehouse/distributed compute | Normalized schemas/manifests |
| Sparse retrieval | `bm25s` | OpenSearch/Lucene | Candidate contract |
| Dense retrieval | FAISS CPU | Distributed ANN | Vector/query contract |
| Features | Versioned Parquet/shared functions | Feature platform | Registry/parity tests |
| Training | LightGBM local | Managed jobs | Matrix/group/model manifest |
| Artifacts | Content-addressed directories | Object store/registry | DAG/promotion rules |
| Tracking | JSON/Parquet + local MLflow | Hosted platform | Canonical run schema |
| Serving | FastAPI | Autoscaled service | HTTP/stage contracts |
| Cache | LRU/SQLite | Distributed cache | Versioned keys |
| Diversity | In-process greedy stage | Dedicated reranking tier | Input/output/audit contract |
| Observability | JSON + local analysis | Central telemetry | Event/timer schema |

Production adds catalog updates, shadow indexes, freshness, canaries, SLOs, rollback, privacy governance, and online experimentation. None is required locally.

## 46. Alternatives Considered

| Alternative | Why not default | Reconsider when |
|---|---|---|
| Keep full synthetic marketplace simulation | Adds seller, inventory, price, shipping, fulfillment, margin, conversion, review, sponsorship, cancellation, and return-probability machinery without real observations | A legally usable dataset supplies those fields |
| Retain simulation interfaces but disable them | Preserves complexity and weakens the focused scientific story | Never without a new approved use case |
| Remove all post-ranking work | Loses a useful real/derived list-diversity demonstration | If diversity has no measurable value |
| OpenSearch locally | JVM/service operations distract from ML workflow | Fielded/operational parity dominates |
| Pyserini required | JDK and setup cost | Packaging/performance evidence justifies it |
| Large neural retriever/ranker | CPU/memory cost | Hardware and measured value change |
| Transformer fine-tuning | Outside required compute budget | Separate optional research environment |
| Hosted vector/model service | Cost, credentials, network dependence | Production organization chooses it |
| GPU-required training | Not necessary for tree ranker | Separate future neural objective |
| LLM query rewriting | Complexity/latency and weak evaluation story | Small local model proves value |
| Collaborative filtering | No user histories | Governed interaction data exists |
| Treat unjudged as negative | Scientifically invalid | Only with new judgments |
| ANN from start | FlatIP is exact/simple at benchmark scale | Measured latency requires it |
| Integer diversity optimizer | Solver/latency complexity | Greedy method demonstrably insufficient |

## 47. Risks and Mitigations

### 47.1 Risk register

| Risk | Probability | Impact | Mitigation | Contingency |
|---|---|---|---|---|
| Incomplete judgments outside supplied lists | High | High | Protocol-separated qrels-aware retrieval and closed-pool ranking | Publish unjudged rate and both tracks |
| Query-group leakage | Medium | High | Normalized-query grouping and invariant tests | Invalidate/regenerate descendants |
| Target leakage | Medium | High | Feature registry, prohibited aggregates, training-population manifest | Remove feature and rerun |
| Closed-pool/online distribution shift | High | Medium | Direct pair features, bounded ranks, distribution report | Candidate-conditioned model only with valid labels |
| Hybrid or LambdaMART weak gains | Medium | Medium | Fair pools, error slices, honest champion selection | Retain simpler method |
| CPU neural latency | High | Medium | Optional top-20, cache, batches | Disable neural stage |
| Fixed catalog exceeds memory target | High | High | Early sizing, component/combined load gates, memmaps | Validated compression; explicit catalog ADR if unresolved |
| macOS dependency friction | Medium | High | Locked wheels and documented fallbacks | Disable only optional component or use alternate package |
| Diversity harms relevance | Medium | High | Protected ranks, validation loss budget, Exact audit | Do not promote/disable diversity |
| Brand missingness distorts metrics | High | Medium | Known-brand denominator/rate; missing values never grouped | Prefer semantic-only mode |
| Embedding similarity creates false redundancy | Medium | Medium | Ablation and manual error sample | Disable semantic penalty |
| Optional scope expands | High | Medium | Milestone gates and optional flags | Ship relevance core first |
| Synthetic marketplace subsystem is reintroduced informally | Low | High | Explicit removal decision, no schemas/configs/modules/artifacts | Require new Elephant/ADR and real data source |
| Colab artifacts drift from local environment | Medium | Medium | Pinned configs/revisions, neutral formats, local parity | Recompute locally |
| Native-library nondeterminism | Medium | Medium | Seeds, threads, version pinning, tolerance | Record variance/deterministic mode |
| Artifact incompatibility | Medium | High | Manifest DAG and startup probes | Fall back to prior bundle |

## 48. Implementation Milestones

### 48.1 Milestone plan

| Milestone | Objective | Inputs → outputs | Acceptance criteria | Dependencies / complexity |
|---|---|---|---|---|
| M0 Environment | Quality-gated package skeleton | Elephant → installable repo | macOS setup, lock, lint/type/test commands | Approved Elephant / M |
| M1 Ingestion/profiles | Task-1 US load and grouped profiles | Raw files → dataset/split/profile manifests | Checksums/schema/predicate/nesting/leakage audit | M0 / M |
| M2 Normalized data | Canonical tables, documents, catalog/pools, sizing | M1 → Parquet/manifests/resource estimate | Keys/joins/gains/catalog exact; 5.5 GB proceed/block gate | M1 / M |
| M3 BM25 | Persisted sparse retrieval and pair scoring | Documents/catalog/pools → index/candidates/scores | Reload parity, complete pair scores, metrics/RSS | M2 / M |
| M4 Evaluation framework | Protocol-valid metrics/bootstrap | Candidates/judgments → reports | Toy gains, prohibited metric checks, fixed cohorts | M3 / M |
| M5 Dense | Embeddings and FAISS CPU | Documents/model → vectors/index | Restart, norm/alignment/reload/latency/RSS | M2 + model cache / L |
| M6 Hybrid | RRF and combined load gate | Sparse+dense → union/retrieval report | Determinism, fair comparison, combined RSS/catalog hash | M3–M5 / M |
| M7 Query/features | Parser and `ltr_core_v1` | Pools/candidates/tables → matrices/state | Pair coverage, parity, leakage/distribution report | M6 / L |
| M8 Rankers | Pointwise and LambdaMART | Closed matrices → population/models | Valid rows/groups/gains, early stop, reload | M7 / L |
| M9 Ranking evaluation | Closed/end-to-end ablations | Models/predictions → reports | ABL-01–05, CIs, slices, protocol IDs | M8 / M |
| M10 Neural optional | Cross reranker and relevance promotion | Model cache + ranks → active relevance | ABL-09, bounds/cache/fallback/comparability | M9 / M |
| M11 Diversity optional | Greedy brand/semantic stage | Active relevance + products/vectors → ranks/audits | ABL-06–08/10, guardrails, metrics, deterministic tests | M9; M10 optional / M |
| M12 FastAPI | Bundle-backed search | Relevance bundle; optional stages → HTTP | No rebuild, contracts, fallbacks, readiness | M9; M10/M11 optional / L |
| M13 Streamlit | Interactive API client | API → demo | Modes/cards/metrics/rank changes/limitations | M12 / M |
| M14 Hardening | Tests, profiling, documentation | Complete core → release candidate | Required suites, M3 latency/RSS, runbooks | M0–M13 / L |
| M15 Portfolio | Frozen experiments/presentation | Promoted bundle/test → final report | Tables, ablations, screenshots, reproduction, limitations | M14 / M |

Every milestone leaves a working repository. M10/M11 may be skipped without blocking M12. A change to promoted relevance invalidates diversity descendants by manifest hash. M15 may report an honest non-winning model/stage.

## 49. Goldfish Decomposition Strategy

Each Goldfish implements one short, testable unit; cites relevant Elephant decisions; names exact files; defines inputs, outputs, configuration, acceptance commands, tests, failures, and rollback; and leaves the repository working. It cannot redesign settled architecture implicitly.

Proposed first sequence:

1. Repository quality skeleton and M3/8 GB environment contract.
2. Strict configuration loader and canonical hashing.
3. Artifact manifest and atomic stage-output protocol.
4. ESCI raw manifest and schema validator.
5. Task-1 US filter and normalized-query split logic.
6. Deterministic nested profile sampler.
7. Canonical normalized query/product/judgment/source tables.
8. Fixed catalog and closed-pool builder.
9. Versioned product-document builder.
10. Official-gain ranking metric library.

Later Goldfish follow milestone order. Never combine data download, both indexes, ranker training, and serving in one task.

## 50. Definition of Done

### 50.1 Checklist

- [ ] ESCI source/license/version/checksums are recorded and ingestion is reproducible.
- [ ] Primary benchmark is `us`, `small_version == 1`, with official split provenance and gains.
- [ ] Development/portfolio profiles preserve complete nested query groups without leakage.
- [ ] Fixed retrieval catalog and closed judged pools have reproducible membership manifests.
- [ ] Resource gates measure sparse, dense, and combined serving RSS without changing catalog identity.
- [ ] Normalized source tables and product documents pass validation.
- [ ] BM25 builds, persists, reloads, pair-scores, retrieves, and is evaluated.
- [ ] Product vectors and FAISS CPU build, persist, reload, pair-score, and retrieve.
- [ ] Hybrid RRF deduplicates, records provenance, and is fairly evaluated.
- [ ] Retrieval, closed-pool, and end-to-end diagnostic metrics remain protocol-separated.
- [ ] Query parser and `ltr_core_v1` features are versioned and offline/online parity-tested.
- [ ] Every primary ranker row is officially judged; no unjudged negative enters training.
- [ ] Pointwise and LambdaMART train with correct groups/gains and serialize with parity.
- [ ] Required ranking/retrieval ablations and query-level confidence intervals are reported.
- [ ] Optional neural stage, if included, is bounded and failure-skippable.
- [ ] Exactly one valid active-relevance contract exists per ranked request.
- [ ] Optional diversity, if included, is deterministic, uses only source/derived catalog signals, and can be disabled with exact relevance-order parity.
- [ ] Diversity reports brand/semantic metrics, NDCG deltas, Exact displacement, missing-brand rate, latency, and rank lineage.
- [ ] No output contains duplicate products; cap relaxations and guardrail fallbacks are audited.
- [ ] Expensive artifacts persist with compatible manifests and atomic success state.
- [ ] Experiment records reconstruct data, code, config, models, hardware, metrics, and artifacts.
- [ ] FastAPI loads an explicit relevance bundle without rebuilding or downloading.
- [ ] Streamlit compares ranking modes and optional stages using official product fields.
- [ ] Required test suites pass, including diversity-disabled parity and metric toy cases.
- [ ] Latency and peak RSS are published against targets on the Apple M3/8 GB machine.
- [ ] Required development and portfolio workflows run locally without CUDA, MPS dependence, or paid APIs.
- [ ] README documents setup, architecture, protocols, results, reproduction, and limitations.
- [ ] Final claims distinguish source, derived, predicted, and optional diversity outputs.

Optional neural/diversity/full/counterfactual work is not required for core completion. Correct evidence and an operational relevance pipeline are required.

## 51. Portfolio Presentation

README/report structure:

1. problem, scientific story, and measured headline;
2. architecture and 60-second local demo;
3. exact environment/data/model/artifact commands;
4. separate retrieval-catalog and closed-pool ranking tables with protocol IDs;
5. pointwise/LambdaMART/neural ablations with paired confidence intervals;
6. optional diversity Pareto table/plot: relevance delta versus brand and semantic diversity;
7. Recall@K curves, NDCG slices, stage-latency waterfall, RSS/artifact table;
8. before/after rank examples with provenance and reason codes;
9. demo screenshots, reproducibility IDs, limitations, and future work.

Suggested resume bullets, filled only with measured values:

- “Built a CPU-first multi-stage product search system combining BM25, MiniLM/FAISS, RRF, and LambdaMART; evaluated on official Amazon ESCI Task 1 judgments with query-level bootstrap intervals.”
- “Designed protocol-separated retrieval/ranking evaluation, versioned offline/online features, immutable artifacts, FastAPI serving, and an optional relevance-guarded diversity stage on an 8 GB M3 Mac.”

Do not cherry-pick examples or imply live Amazon performance.

## 52. Future Work

Clearly separate future extensions include:

- a legally usable real source for seller, inventory, price, shipping, fulfillment, margin, conversion, review, product-age, sponsorship, cancellation, and return-probability fields;
- synthetic marketplace simulation only as a separately approved research sandbox, never retroactively part of this benchmark;
- consented session/user histories and personalization;
- real impression/action logs, position-bias correction, and counterfactual evaluation;
- online experiments and contextual bandits;
- local small-model query rewriting;
- price-aware query understanding only after an authoritative catalog source exists;
- multimodal product embeddings and graph-based retrieval;
- temporal catalog/index updates and distributed serving.

Every extension needs new data contracts, evaluation protocols, privacy/ethical review, and an explicit decision record.

## 53. Open Questions

| Question | Default assumption | Resolve by |
|---|---|---|
| Which official release/checksums are current? | Pin Amazon Science files retrieved for M1. | M1 |
| How many groups remain after normalized-query collision quarantine? | Target ~5k development and ~20k portfolio; report observed rows. | M1 |
| Does `bm25s` install/reload efficiently on supported Macs? | Keep it default with `rank-bm25` smoke reference. | M3 |
| Does the fixed combined bundle fit 5.5 GB and FlatIP meet latency? | Estimate M2, measure components M3/M5, combined M6. | M2–M6 |
| Which optional cross-encoder meets p95 budget? | TinyBERT L-2 first. | M10 |
| What diversity lambdas and NDCG loss budget are acceptable? | Select validation Pareto point; do not assume benefit. | M11 |
| Is known-brand coverage sufficient for stable brand metrics? | Always report missing rate and query feasibility. | M11 |
| Which exact M3 Mac/macOS defines final benchmark? | Chip/memory fixed; record model, OS, power, disk. | M14 |

## 54. Decision Log

Each decision uses a consistent format and can be superseded only by an explicit ADR/Elephant revision.

### D-001 Dataset selection

**Decision:** Use English (`us`) reduced ESCI Task 1 (`small_version == 1`) with official gains.

**Context:** The project needs public grouped product relevance judgments.

**Options considered:** ESCI; generic IR corpora; manufactured relevance; proprietary logs.

**Chosen approach:** Preserve official train/test provenance, grouped project validation, E/S/C/I labels, and gains `[0.0,0.01,0.1,1.0]`.

**Rationale:** Domain-relevant, graded, public, and documented.

**Trade-offs:** Bounded judgments and missing business/behavior fields.

**Future reconsideration trigger:** A legally usable stronger product-search benchmark appears.

### D-002 Candidate populations

**Decision:** Separate fixed closed judged pools, a fixed Task-1 retrieval catalog, and end-to-end diagnostics.

**Context:** Products outside a supplied query list are unjudged.

**Options considered:** Treat unjudged as negative; profile-specific catalog; fixed benchmark catalog plus named protocols.

**Chosen approach:** Use the last option; every metric carries protocol/population IDs.

**Rationale:** Prevents invalid labels and incomparable metrics.

**Trade-offs:** More evaluation/reporting complexity.

**Future reconsideration trigger:** Exhaustive catalog qrels become available.

### D-003 Sparse retrieval

**Decision:** `bm25s`/equivalent persisted CPU index; `rank-bm25` smoke reference.

**Context:** Need macOS-friendly BM25 persistence under tight memory.

**Options considered:** `rank-bm25`, `bm25s`, Pyserini, OpenSearch.

**Chosen approach:** Lightweight persisted default without required JVM/service.

**Rationale:** Practical local performance and portability.

**Trade-offs:** Less production parity than Lucene.

**Future reconsideration trigger:** Installation, correctness, or performance gates fail.

### D-004 Dense model and index

**Decision:** Pinned `all-MiniLM-L6-v2` plus CPU `IndexFlatIP` over normalized vectors.

**Context:** Exact reference behavior must fit an M3/8 GB machine.

**Options considered:** MiniLM variants, BGE-small; FlatIP, HNSW, IVF/PQ.

**Chosen approach:** Compact 384-D model and exact FlatIP; HNSW for measured latency, compression for measured memory only after comparison.

**Rationale:** Simple, reproducible, and portfolio-relevant.

**Trade-offs:** Domain mismatch and linear scan.

**Future reconsideration trigger:** A compact alternative wins within quality/resource gates.

### D-005 Hybrid fusion

**Decision:** RRF pre-order plus candidate union.

**Context:** Sparse/dense score scales differ by query.

**Options considered:** Normalized weighted sums, Borda, RRF, learned fusion.

**Chosen approach:** Validation-configured RRF with source evidence retained.

**Rationale:** Robust and transparent.

**Trade-offs:** Pre-order ignores raw score magnitude.

**Future reconsideration trigger:** Calibrated fusion wins reliably.

### D-006 Primary feature contract

**Decision:** Use only source/derived query, product, pair, and direct-scoring features in `ltr_core_v1`.

**Context:** Original top-K provenance has incompatible semantics between closed pools and retrieved unions.

**Options considered:** Include all source ranks; train only retrieved intersections; direct-score all judged pairs.

**Chosen approach:** Direct-score every pair, use bounded ranks/closed-set RRF, and isolate source-provenance features to a named optional model.

**Rationale:** Authoritative labels and online-available formulas for every row.

**Trade-offs:** Closed-pool/online distributions still differ and require reporting.

**Future reconsideration trigger:** A properly labeled candidate-conditioned dataset exists.

### D-007 Primary ranker

**Decision:** LightGBM LambdaMART.

**Context:** Graded grouped labels, mixed features, and CPU constraints.

**Options considered:** Heuristic, pointwise, XGBoost/CatBoost ranking, neural rankers.

**Chosen approach:** LambdaMART with explicit gains and group arrays; pointwise is baseline.

**Rationale:** Ranking-aware, fast, mature, and explainable.

**Trade-offs:** Cannot recover missed candidates.

**Future reconsideration trigger:** Another ranker wins fairly within resource budgets.

### D-008 Feature storage

**Decision:** Parquet is canonical; compact binary matrices are disposable accelerators.

**Context:** Features need inspection, versioning, projection, and reuse.

**Options considered:** CSV, database-only, Parquet, service-backed store.

**Chosen approach:** Partitioned Parquet plus registry/state manifests.

**Rationale:** Typed, compressed, local, and interoperable.

**Trade-offs:** Online point lookup needs an adapter.

**Future reconsideration trigger:** Real-time freshness/multi-service requirements emerge.

### D-009 Optional neural reranker

**Decision:** TinyBERT L-2 over top 10–30 with explicit active-score comparability.

**Context:** Neural text interaction may improve relevance but CPU latency is constrained.

**Options considered:** No neural stage, TinyBERT, MiniLM L-6, larger/fine-tuned models.

**Chosen approach:** Optional bounded stage that emits either a full-list comparable score or rank-only promotion and falls back exactly.

**Rationale:** Demonstrates the concept without becoming a dependency.

**Trade-offs:** Added latency and score-contract complexity.

**Future reconsideration trigger:** No quality gain, excessive p95, or a better compact model.

### D-010 Catalog diversity

**Decision:** Optional deterministic greedy reranking using active relevance, official brand data, and product embeddings.

**Context:** Relevant lists can contain repeated brands and semantically near-duplicate products.

**Options considered:** No post-rank stage; brand penalty; semantic MMR; combined greedy; integer optimization.

**Chosen approach:** Combined greedy utility with ablations, protected relevance ranks, optional cap, and validation loss budget.

**Rationale:** Meaningful multi-objective demonstration from real/derived evidence with low CPU cost.

**Trade-offs:** It may reduce relevance and cannot imply fairness or business benefit.

**Future reconsideration trigger:** Validation shows no useful Pareto point or a better method is justified.

### D-011 Experiment tracking

**Decision:** Canonical JSON/Parquet runs plus optional file-backed MLflow.

**Context:** Reproduction must work offline without a service.

**Options considered:** MLflow-only, flat files, hosted tracking, custom database.

**Chosen approach:** Portable files are authoritative; MLflow mirrors for UI.

**Rationale:** Free, inspectable, and robust.

**Trade-offs:** Less collaboration functionality.

**Future reconsideration trigger:** Multi-user governance is required.

### D-012 API architecture

**Decision:** One FastAPI process loads an immutable relevance bundle; Streamlit is a client.

**Context:** Need realistic boundaries without local microservice overhead.

**Options considered:** Notebook-only, direct Streamlit model calls, one API, separate services.

**Chosen approach:** Modular in-process stages behind versioned schemas; optional stages do not gate readiness.

**Rationale:** Testable and simple locally.

**Trade-offs:** Components cannot scale independently.

**Future reconsideration trigger:** Independent scaling/fault isolation is required.

### D-013 Dataset profiles

**Decision:** Nested complete-query profiles controlled by query count.

**Context:** Row sampling breaks ranking groups; reduced US has fewer than 50k queries.

**Options considered:** Full only, first-N, row sample, stable group sample.

**Chosen approach:** ~5k development and ~20k portfolio groups; full reduced US optional.

**Rationale:** Fast iteration and honest scale within 8 GB.

**Trade-offs:** Profile results are not full-dataset results.

**Future reconsideration trigger:** Resource profiling supports a larger separately named population.

### D-014 Local-first with optional Colab

**Decision:** Local execution is authoritative; Colab may accelerate isolated offline batches.

**Context:** The M3 has limited memory, while hosted free resources are variable and ephemeral.

**Options considered:** Local only; Colab primary; local core plus manifest-compatible remote batches.

**Chosen approach:** Third option, with local artifact parity and final local serving/benchmarks.

**Rationale:** Preserves reproducibility and macOS proof while providing a practical escape valve.

**Trade-offs:** Additional artifact-transfer/version checks.

**Future reconsideration trigger:** Required local workflow cannot meet resource gates after approved optimizations.

### D-015 Remove synthetic marketplace metadata

**Decision:** Remove synthetic marketplace metadata from the required system.

**Context:** ESCI provides real relevance judgments and product text but no real seller, price, inventory, conversion, shipping, fulfillment, margin, review, product-age, sponsorship, cancellation, or return-probability data. The synthetic subsystem added substantial implementation and evaluation complexity while weakening the clarity of the project’s scientific claims.

**Options considered:** (1) keep the full simulation; (2) make it optional but retain all interfaces; (3) remove it from the architecture and replace it with real/derived catalog-diversity reranking; (4) remove all post-ranking optimization.

**Chosen approach:** Option 3.

**Rationale:** It preserves a meaningful multi-objective demonstration using official brands, colors, product IDs, text, and derived embeddings while focusing the project on retrieval and learning-to-rank.

**Trade-offs:** The project no longer demonstrates business/offer policies. It gains credibility, tractability, and a clearer portfolio narrative.

**Future reconsideration trigger:** A legally usable dataset supplies real fields needed for those policies.

---

## Recommended First Goldfish

**Title:** Goldfish 001 — Repository Quality Skeleton and macOS Environment Contract

**Scope:** Create only the minimal installable Python project structure, `pyproject.toml`, deterministic dependency-lock approach, Ruff/pytest/type-check/pre-commit configuration, Git-ignore rules for data/artifacts/model caches, a tiny version module, and one smoke test. Document Apple M3/8 GB setup, the 5.5 GB process-RSS target, optional manifest-compatible Colab batch execution, and the initial-download/offline boundary. Do not ingest ESCI, implement retrieval/model logic, or create optional diversity functionality. Acceptance is a clean environment in which formatting, linting, typing, and tests run without paid credentials.
