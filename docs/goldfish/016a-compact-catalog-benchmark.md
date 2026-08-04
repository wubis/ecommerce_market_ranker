# Goldfish 016A — Compact Catalog Benchmark

| Field | Value |
|---|---|
| Status | Implemented |
| Parent | Goldfish 016 frozen portfolio release |
| Motivation | Make the real pipeline viable on an Apple M3 with 8 GB RAM and bounded storage |
| Catalog ID | `esci_task1_us_compact_catalog_v1` |

## Objective

Retain the ESCI Task-1 US benchmark, complete judged pools, official gains, validation-only model
selection, and frozen project-test evaluation while replacing the full retrieval catalog with one
smaller, reproducible research catalog. Compact results are never presented as full-catalog or live
Amazon results.

## Selection Contract

The source population remains every distinct product referenced by the release-wide Task-1 US
predicate. Selection does not inspect `esci_label`, gains, model scores, or project-test outcomes.

The compact catalog is the union of:

1. every distinct product judged for a selected portfolio query, across project train, validation,
   and test; and
2. up to `dataset.compact_catalog_distractor_products` additional source-catalog products selected
   by seeded SHA-256 priority over selection version, runtime seed, locale, and product ID.

All products for a normalized query remain in its complete judged pool. Required judged products
with no usable source text remain audited catalog exclusions, matching the existing retrieval
contract. Development and portfolio experiments use the same compact membership.

The default distractor target is 100,000. The final observed catalog can be larger because required
judged products are never discarded. If fewer distractors exist, all available distractors are
retained.

## Persisted Evidence

`foundation-manifest.json` records:

- full source product count;
- required judged-product count;
- configured and observed distractor counts;
- selected candidate count;
- compact/full mode and selection method;
- compact catalog ID and membership hash;
- the existing no-text exclusions and resource estimate.

This extension publishes schema version 2 under `data-foundation-v2`; earlier full-catalog
foundations remain immutable and cannot be mistaken for compact inputs.

The manifest validates that candidates equal required plus distractors, that the observed sample
matches the configured bound, and that every portfolio-judged product is present before text-based
retrieval exclusions. Raw row order and label changes cannot alter selection.

## Configuration

```yaml
dataset:
  catalog_mode: compact
  catalog_selection_version: portfolio-judged-plus-sha256-v1
  compact_catalog_distractor_products: 100000
```

`catalog_mode: full` preserves the prior `esci_task1_us_catalog_v1` behavior. Catalog semantics are
part of the resolved configuration hash, so switching modes or sample size requires rebuilding the
complete downstream artifact DAG.

## Claim Boundary

Allowed wording:

- deterministic compact ESCI Task-1 US research catalog;
- complete judged products plus a label-blind distractor sample;
- closed-pool ranking and compact-catalog retrieval results;
- measured local latency/RSS for the exact compact serving bundle.

Prohibited wording:

- full-catalog retrieval quality;
- live Amazon search performance;
- exhaustive relevance, production scale, business impact, or online behavior.

The generated Goldfish 016 report and limitations file include the compact catalog ID and boundary
automatically.

## Acceptance Criteria

1. Every portfolio-judged product is selected before document exclusions.
2. Distractor selection is seeded, SHA-256 based, label blind, and row-order invariant.
3. Compact and full catalogs have different persisted IDs.
4. Sparse, dense, evaluation, serving, and portfolio schemas accept only named catalog IDs.
5. Complete downstream fixture lineage passes with the compact catalog.
6. Final narrative explicitly rejects full-catalog interpretation.
7. Resource estimates use the observed compact membership.
8. Lock, Ruff, strict mypy, pytest, and pre-commit pass.

## Operational Note

The pinned raw ESCI release is still required and is about 1.16 GB. Compact mode reduces derived
products, documents, BM25 state, embeddings, FAISS, serving projection, and downstream artifacts;
it does not make a nearly full system disk safe. Maintain adequate free disk space before the real
build.

## Rollback

Set `dataset.catalog_mode: full`, rebuild from the configuration-hashed profile/foundation stage,
and use only the resulting full-catalog artifact lineage. Do not relabel compact artifacts as full.
