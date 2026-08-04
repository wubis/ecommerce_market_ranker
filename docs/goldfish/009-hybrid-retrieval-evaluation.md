# Goldfish 009 — Hybrid Retrieval and Retrieval Evaluation

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 009 |
| Parent design | `ELEPHANT.md`, Sections 14–16, 25–27, 30–32, 36–43, 48–50; D-005/D-006 |
| Milestone | M6 — Hybrid retrieval and the retrieval portion of M4 evaluation |

## Objective

Fuse Goldfish 007 BM25 and Goldfish 008 dense candidates with deterministic reciprocal-rank
fusion (RRF), retain complete source evidence, generate one fixed-catalog candidate artifact,
and compare sparse, dense, and hybrid retrieval on exactly the same query cohort. Persist
protocol-safe query metrics, grouped confidence intervals, named slices, paired deltas, and a
simultaneous sparse+dense RSS gate suitable for the Apple M3 with 8 GB unified memory.

```bash
uv run market-rank retrieval evaluate-hybrid
uv run market-rank retrieval evaluate-hybrid --profile portfolio
```

The default profile is `development`. The command requires compatible promoted Goldfish 006,
007, and 008 artifacts and the exact cached MiniLM query encoder. It performs no download,
embedding build, index build, or implicit parent repair.

## Deterministic RRF Contract

For each product returned by sparse and/or dense retrieval:

```text
rrf(product) = sum_source 1 / (rrf_constant + source_rank)
```

Defaults are `rrf_constant=60`, sparse K=150, dense K=150, and union K=200. RRF deliberately
uses ranks rather than incomparable BM25 and cosine score magnitudes. Products returned by only
one source receive only that source's contribution.

`fuse_rrf` validates each source independently:

- product IDs are nonempty and unique;
- ranks are contiguous and one-based;
- raw scores are finite;
- locale, retriever ID, and index ID are present;
- a product shared by sources has one locale.

The union is deduplicated and sorted by descending RRF, then best source rank, then product ID.
It is truncated only after that complete deterministic ordering. One empty source produces a
valid degraded union; two empty sources produce a structured empty result without fabricating
candidates.

Every hybrid candidate retains nullable sparse score/rank/retriever/index fields, nullable dense
score/rank/retriever/index fields, source count, best source rank, RRF score/rank, and fusion
retriever ID.

## Evaluation Artifact

```text
retrieval-evaluation/<dataset-version>/<profile>/retrieval-eval-v1/<config-sha256>/
├── candidates/
│   ├── part-00000.parquet
│   └── ...
├── query-metrics/
│   ├── part-00000.parquet
│   └── ...
├── aggregate-metrics.parquet
├── comparison-metrics.parquet
├── retrieval-evaluation.json
├── manifest.json
└── _SUCCESS
```

The Goldfish 003 manifest pins three exact parent manifest hashes: data foundation, sparse index,
and dense index. Before any query runs, their config, foundation, catalog ID/hash, document
count, and immutable lineage must agree.

Candidate and query-metric rows are written in bounded deterministic Parquet partitions. The
builder never retains the complete candidate corpus as Python objects. Partition row limits are
strict configuration, making the same package/CLI path usable for development and portfolio
profiles.

## Fixed Cohort and Protocol

All three stages use:

- the identical profile query IDs and normalized query texts;
- the identical fixed `esci_task1_us_catalog_v1` membership;
- the identical Goldfish 006 judgment set per query;
- the same cutoffs, relevance thresholds, and protocol;
- explicit rows for empty retrieval results.

The only protocol is `retrieval_catalog_task1_us_v1`. Query metrics are:

- judged Recall;
- Exact Hit;
- judged MRR;
- known-judgment coverage;
- unjudged rate.

Naive catalog precision, MAP, and NDCG are unavailable because unjudged catalog products are
not irrelevant labels. The fixed thresholds are `exact` (`E`) and `exact_substitute` (`E+S`),
with default cutoffs 10 and 100. Official persisted float32 gains are checked against their
label mapping, then canonical exact gains are used by the metric contract.

Per-query records carry protocol, population/catalog/profile IDs, query and normalized-group
IDs, source/project split, stage, threshold, metric, cutoff, counts, and empty-result status.

## Grouped Confidence Intervals

Point estimates are the mean and median over query IDs. Confidence intervals resample
`normalized_query_sha256` groups—not individual rows—with replacement, preserving every query
inside a sampled leakage group. This prevents duplicated normalized query text from becoming
independent bootstrap evidence.

Each metric/slice/comparison receives a deterministic seed derived from the configured runtime
seed and its canonical identity. The default is 1,000 replicates, generated in batches of 100;
the implementation never creates the full replicates-by-groups matrix at once. The artifact
records 2.5th/97.5th percentiles, group/query counts, method ID, seed, and replicate count.

## Retrieval Slices

`aggregate-metrics.parquet` contains an overall row plus independently named slices for:

- query length: 1–2, 3–5, or 6+ normalized whitespace tokens;
- official query source;
- project split;
- presence/absence of an Exact judgment.

Every slice retains protocol, stage, threshold, cutoff, mean/median/CI, query/group/empty counts,
returned/judged/unjudged/relevant totals, and bootstrap identity. Query-length parsing here is a
minimal evaluation descriptor, not the query-understanding parser planned for Goldfish 010.

## Fair Stage Comparisons

`comparison-metrics.parquet` joins stages one-to-one on query ID and normalized group before
computing a delta. It reports:

- hybrid minus sparse;
- hybrid minus dense;
- hybrid minus the better aggregate single retriever for the same metric/threshold/cutoff.

For Recall, Exact Hit, judged MRR, and known coverage, positive improvement means hybrid is
higher. For unjudged rate, positive improvement means hybrid is lower. A sparse/dense aggregate
tie selects sparse by a fixed rule. Each row records mean/median improvement, grouped 95% CI,
win/tie/loss counts, selected baseline, direction, query/group counts, and bootstrap contract.

These are profile-scoped comparisons, not final frozen test claims. Goldfish 016 owns the final
validation-selected and frozen portfolio narrative.

## Combined Sparse+Dense Resource Gate

The command loads the recursively verified sparse index, dense FAISS index/vector memmap, and
cached query encoder in one fresh process. It records process peak RSS immediately after the
combined load and again after candidate/report construction. The maximum of both phases must be
at most `runtime.rss_limit_mb` (5,632 MiB by default).

The resource record also contains sparse, dense, and evaluation artifact bytes plus evaluated
query count. An over-limit load fails before report staging. An evaluation-phase overage rolls
back the completed staged report. Neither path can publish `_SUCCESS`.

Stage summaries separately persist candidate rows, empty queries, and warm p50/p95/maximum
latency for sparse search, dense query+search, and RRF fusion. These are observations, not final
reference-machine claims; Goldfish 015 performs qualification under frozen hardware conditions.

## Configuration

```yaml
retrieval:
  hybrid:
    component_version: rrf-v1
    rrf_constant: 60
    sparse_top_k: 150
    dense_top_k: 150
    union_top_k: 200
    max_union_top_k: 1000
evaluation:
  component_version: retrieval-eval-v1
  default_profile: development
  cutoffs: [10, 100]
  bootstrap_replicates: 1000
  bootstrap_batch_replicates: 100
  candidate_partition_rows: 100000
  metric_partition_rows: 100000
```

Configuration validation ensures source K values do not exceed their index hard maxima, union K
does not exceed its hard maximum, cutoffs are positive/unique/sorted, every cutoff fits every
compared stage depth, and bootstrap batches do not exceed total replicates.

## Out of Scope

- query parsing and ranking feature materialization (Goldfish 010);
- closed-pool RRF ranking, LambdaMART, or ranking-model evaluation;
- tuning K/RRF on frozen test results or making portfolio quality claims;
- final production ESCI candidate/report generation in automated tests;
- API fallback orchestration when one live retriever raises an exception;
- final latency/RSS qualification (Goldfish 015).

## Acceptance Criteria

1. Source inputs reject duplicates, noncontiguous ranks, nonfinite scores, missing provenance, and
   locale conflicts.
2. RRF union/deduplication, truncation, degraded behavior, and tie order are deterministic.
3. Every hybrid row preserves nullable score/rank/retriever/index evidence from both sources.
4. Foundation, sparse, and dense artifacts share exact catalog/document/config lineage.
5. Sparse, dense, and hybrid use one fixed query/catalog cohort, including empty results.
6. Catalog evaluation exposes only protocol-safe retrieval metrics at fixed thresholds/cutoffs.
7. Candidate and query-metric outputs use bounded deterministic partitions.
8. Overall and named-slice summaries persist query/group/count context.
9. Confidence intervals resample normalized-query groups with fixed, bounded seeds/batches.
10. Paired hybrid-vs-source/best-single comparisons use identical query keys and direction-aware
    improvement deltas.
11. Combined load and evaluation peak RSS must pass before promotion.
12. Imports/tests perform no model/data download or production evaluation run.
13. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove hybrid/evaluation configuration, `retrieval.hybrid`, `evaluation.retrieval`, the CLI
command, tests, documentation, and local `retrieval-evaluation` artifacts. Goldfish 007/008
index logic remains intact, although configuration-hashed local artifacts must be rebuilt after
removing the Goldfish 009 config fields.
