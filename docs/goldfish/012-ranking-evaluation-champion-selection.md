# Goldfish 012 — Ranking Evaluation and Champion Selection

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 012 |
| Parent design | `ELEPHANT.md`, Sections 7, 8.7, 25–27, 31–32, 36–37, 48 |
| Milestone | M9 — Protocol-safe ranking evidence and active relevance promotion |

## Objective

Consume one compatible Goldfish 011 model artifact and its recursively verified Goldfish
009–010 ancestors, evaluate all three relevance orders on a fixed project-validation cohort,
and promote exactly one deterministic active-relevance contract. The three orders are direct
RRF, pointwise LightGBM, and LambdaMART.

```bash
uv run market-rank ranking evaluate
uv run market-rank ranking evaluate --profile portfolio
```

This command performs no training, retrieval, feature generation, downloading, project-test
evaluation, serving-bundle publication, or test-informed selection.

## Two Protocols, No Metric Leakage

Goldfish 012 keeps two candidate populations physically and semantically distinct.

### `closed_pool_task1_v1`

Every catalog-eligible official judged product for each project-validation query is present.
The three stages rank identical product sets:

- `rrf` uses `closed_rrf_score`, derived from direct BM25 and dense scores within that complete
  judged pool;
- `pointwise` uses the persisted pointwise model;
- `lambdamart` uses the persisted LambdaMART model.

This protocol permits official direct-gain NDCG, thresholded Precision, MAP, MRR, and Exact Hit.
NDCG uses gains `I=0`, `C=0.01`, `S=0.1`, and `E=1` directly, with no exponential transform.

### `end_to_end_diagnostic_v1`

The fixed hybrid union produced by Goldfish 009 is joined exactly to Goldfish 010's label-free
retrieved-union features. `rrf` preserves the original hybrid rank; the two models reorder that
same union. Empty candidate lists remain in the query cohort.

Products outside a query's supplied judgment list remain unknown. This protocol emits only
judged Recall, Exact Hit, judged MRR, known-judgment coverage, and unjudged rate. It never emits
naive Precision, MAP, or NDCG, and its values are explicitly conditional diagnostics rather than
official Task 1 ranking claims.

## Validation and Test Quarantine

The fixed selection split is `validation`. Lazy scans predicate-filter project split before
collecting closed matrices, retrieved-union matrices, original hybrid candidates, and query
context. The command never materializes project-test rows and records `test_evaluated=false` in
the evaluation manifest, experiment record, and active-relevance contract.

Goldfish 009's ABL-01–03 artifact remains an immutable ancestor and supplies inherited retrieval
evidence, but its metrics are not consulted by the Goldfish 012 champion policy. Goldfish 016
owns the frozen portfolio/test report.

## Query Metrics, Intervals, and Slices

Every query/stage/protocol/threshold/metric/cutoff fact is persisted in
`query-metrics.parquet`. All stages retain the identical query cohort, including empty
end-to-end results.

`metrics.parquet` reports mean, median, and fixed-seed 95% confidence intervals. Bootstrap units
are normalized-query leakage groups sampled with replacement, preserving all queries within a
sampled group. Replicates are computed in bounded batches from the existing evaluation config.

Overall results and the following named slices are reported independently:

- query length: `1-2`, `3-5`, or `6+` tokens;
- low/high catalog-IDF lexical specificity;
- query brand, color, model-token, and compatibility presence;
- official source and project split;
- judgment composition: Exact present, Substitute without Exact, complement-heavy, or
  irrelevant-heavy.

Entity and specificity slices use online-available, non-target query signals. Judgment
composition is explicitly named as a label-composition diagnostic and never becomes a feature.

## Required Ablations

`run.json` accounts for every required pre-neural ablation:

| ID | Status in Goldfish 012 | Evidence |
|---|---|---|
| ABL-01 | inherited | Goldfish 009 BM25 catalog retrieval |
| ABL-02 | inherited | Goldfish 009 dense catalog retrieval |
| ABL-03 | inherited | Goldfish 009 hybrid-vs-single retrieval |
| ABL-04 | evaluated | closed-pool LambdaMART versus direct RRF |
| ABL-05 | evaluated | closed-pool LambdaMART versus pointwise |

`comparisons.parquet` stores paired query-level improvements, grouped-bootstrap intervals, and
win/tie/loss counts for ABL-04/05. It also stores separately named `E2E-01/02` conditional
pointwise/LambdaMART-versus-hybrid comparisons under the end-to-end protocol. Protocols never
share an unlabeled metric column.

`failure-analysis.parquet` retains the largest absolute per-query NDCG@10 deltas for ABL-04 and
ABL-05, bounded independently by configuration. Each row carries query slices, baseline and
treatment values, signed delta, and win/tie/loss outcome. These are deterministic associations
for investigation, not causal explanations.

## Champion Policy

Champion selection uses only mean validation `ndcg_official_gain` across configured closed-pool
cutoffs. RRF is always eligible. Each learned model becomes eligible only when it:

1. improves at least one configured NDCG cutoff by more than
   `minimum_model_improvement`; and
2. has no cutoff regression larger than `material_regression_tolerance`.

Among eligible candidates, the highest mean across configured NDCG cutoffs wins. Improvements
within `selection_tie_tolerance` retain the simpler earlier order: RRF, then pointwise, then
LambdaMART. This means a non-winning model is a valid scientific result; the implementation does
not force LambdaMART promotion.

The selected candidate becomes `active-relevance-v1`. The contract records:

- selected stage and optional model ID;
- validation protocol, cutoffs, query count, and every candidate decision;
- the complete model and feature identities;
- one score field populated for every result with `active_score_comparable=true`;
- `rrf-on-model-failure-v1`, which restores the hybrid RRF order if a selected model is missing
  or invalid;
- `test_evaluated=false`.

When RRF itself wins, no model ID is asserted. Goldfish 013 will package this contract into the
explicit serving bundle and implement the runtime fallback.

## Artifact and Experiment Record

```text
ranking-evaluation/<dataset-version>/<profile>/ranking-eval-v1/<config-sha256>/
├── predictions.parquet
├── query-metrics.parquet
├── metrics.parquet
├── comparisons.parquet
├── failure-analysis.parquet
├── active-relevance.json
├── run.json
├── ranking-evaluation.json
├── manifest.json
└── _SUCCESS
```

`predictions.parquet` records every materialized stage score/rank and marks exactly one active
stage per query/product. Closed rows retain judgments for audit; end-to-end rows remain
label-null. `run.json` is the canonical completed experiment record: config/profile, exact parent
artifact hashes, both protocols, ABL-01–05 status, active contract, output counts, and resource
measurement.

The evaluation artifact has one direct parent—the Goldfish 011 ranking-model artifact. Recursive
Goldfish 003 verification covers the feature, retrieval-evaluation, sparse, dense, and foundation
ancestors. Compatible artifact reuse occurs before matrix reads, prediction, or bootstrap work.

## M3/8 GB Resource Contract

Only validation rows and the 44 model features are collected. Both persisted models are loaded
together, score compact float32 matrices, and reuse the bounded candidate union. Bootstrap
arrays operate on query/group summaries rather than repeated candidate rows.

Peak process RSS is observed after dependency/input load, evaluation, and report/contract
staging. It must remain at or below `runtime.rss_limit_mb` (5,632 MiB). Initial overage blocks
staging; final overage removes staged reports and cannot publish `_SUCCESS`.

## Configuration

```yaml
ranking_evaluation:
  component_version: ranking-eval-v1
  selection_split: validation
  closed_cutoffs: [10, 20]
  diagnostic_cutoffs: [10, 100]
  minimum_model_improvement: 0.0
  material_regression_tolerance: 0.005
  selection_tie_tolerance: 1.0e-12
  failure_analysis_queries: 20
  max_closed_rows: 200000
  max_candidate_rows: 200000
```

Cutoffs must be nonempty, positive, unique, and sorted. Population limits are hard failure gates;
rows and query groups are never sampled to make them fit.

## Out of Scope

- project-test or final frozen portfolio evaluation;
- training, hyperparameter search, or changing feature definitions;
- optional neural or diversity reranking and their ablations;
- serving-bundle promotion or runtime orchestration;
- API/Streamlit behavior, latency qualification, screenshots, or final narrative claims.

## Acceptance Criteria

1. One recursively verified Goldfish 011 artifact anchors all evaluation lineage.
2. Only project-validation rows are materialized; project test remains unobserved by selection.
3. Closed-pool stages contain exactly the same complete judged products per query.
4. End-to-end feature rows match original hybrid query/product keys and contiguous RRF ranks.
5. RRF, pointwise, and LambdaMART scores/ranks are deterministic with product-ID tie breaks.
6. Official-gain NDCG is available only under `closed_pool_task1_v1`.
7. End-to-end diagnostics contain no Precision, MAP, or NDCG.
8. Empty end-to-end lists remain in every stage's query cohort.
9. Query metrics, overall summaries, named slices, and grouped intervals persist exact protocol
   and population IDs.
10. ABL-01–03 lineage is inherited and ABL-04/05 paired evidence is evaluated.
11. Conditional end-to-end comparisons remain separately named and interpreted.
12. Failure rows are bounded, deterministic, signed, and carry slice context.
13. Champion eligibility enforces improvement and material-regression guardrails.
14. Exact ties retain the simpler stage; a weaker LambdaMART cannot be forced active.
15. Exactly one complete active-relevance contract is persisted and mirrored in `run.json`.
16. Each query/product's stage rows mark exactly one active relevance stage.
17. Compatible artifacts reuse before matrix reads or inference.
18. Load/evaluation/promotion RSS passes before immutable promotion; failure rolls back.
19. CLI failures are bounded and ordinary imports perform no evaluation or download.
20. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove `ranking_evaluation` configuration, the ranking evaluation module/CLI/tests, this
document, and local `ranking-evaluation` artifacts. Goldfish 006–011 logic remains intact,
although configuration-hashed ancestor artifacts must be rebuilt after removing the Goldfish
012 fields.
