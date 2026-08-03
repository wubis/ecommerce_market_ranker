# Goldfish 007 — Sparse Retrieval and Evaluation Foundation

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 007 |
| Parent design | `ELEPHANT.md`, Sections 14, 25–27, 30, 36–38, 40, 48–50 |
| Milestones | M3 — BM25 and the metric primitives required from M4 |

## Objective

Build, persist, reload, search, and explicitly pair-score one deterministic BM25 index aligned
to the Goldfish 006 fixed catalog. Establish the protocol-safe query-level metric primitives
needed to evaluate this and later retrievers without mixing closed-pool ranking claims with
fixed-catalog retrieval diagnostics.

```bash
uv run market-rank retrieval build-bm25
```

No network access occurs. The command requires and recursively verifies the compatible
Goldfish 006 foundation artifact.

## Sparse Artifact

```text
sparse-index/<dataset-version>/portfolio/bm25-v1/<config-sha256>/
├── document-map.parquet
├── vocabulary.txt
├── postings-offsets.u64
├── posting-doc-ids.u32
├── posting-term-frequencies.u32
├── document-lengths.u32
├── document-frequencies.u32
├── inverse-document-frequencies.f32
├── sparse-index.json
├── manifest.json
└── _SUCCESS
```

The Goldfish 003 manifest records the exact Goldfish 006 parent manifest hash and independently
checksums every payload. `sparse-index.json` additionally records the catalog-membership hash,
product-document version, tokenizer/BM25 parameters, dimensions/counts, query-term policy,
artifact size, wall time, peak RSS, and validation results.

## Tokenization and BM25 Contract

`unicode-word-v1` applies Unicode NFKC and case folding, then extracts Unicode alphanumeric
words with optional internal ASCII hyphen/apostrophe. It preserves numbers and model-like
hyphenated tokens, removes marker punctuation/underscores, has no fitted vocabulary rules or
stopword list, and runs identically for product documents and online queries.

BM25 uses:

```text
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

score(t, d) = IDF(t) * tf(t,d) * (k1 + 1)
              / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))
```

Defaults are `k1=1.2`, `b=0.75`, `top_k=150`, and maximum `top_k=1000`. Repeated query terms
are intentionally treated once (`query_term_policy=unique`). Results sort by descending raw
score, then lexical product ID. Only fixed-catalog products can be returned.

## Memory-Bounded Build

The builder avoids a Python object graph containing the complete inverted index:

1. Goldfish 006 catalog ordinals, keys, document hashes, and document rows are checked for exact
   one-to-one alignment.
2. Documents are tokenized sequentially.
3. Unique document-term frequencies are inserted in bounded batches into a temporary SQLite
   `WITHOUT ROWID` table located inside the artifact staging filesystem.
4. Vocabulary/statistics and postings are streamed in deterministic token/document order into
   compact typed arrays.
5. The temporary database is removed before promotion.

Runtime loads postings, offsets, lengths, frequencies, and IDF read-only through `mmap`; only
the vocabulary dictionary and compact document identity map are ordinary in-memory structures.
The implementation uses only Python's standard library plus the existing Polars dependency.

## Search and Pair Scoring

`SparseIndex.search(query, top_k)` returns immutable candidates containing product/locale,
finite raw BM25 score, one-based rank, retriever ID, and immutable index artifact ID. Empty or
unknown-token queries return an empty tuple. Duplicate products are impossible by construction.

`SparseIndex.score_pairs(query, product_ids)` is separate from top-K retrieval. It:

- requires unique fixed-catalog product IDs;
- preserves caller order;
- returns every requested product;
- assigns a finite zero score when no query token matches;
- uses exactly the same persisted tokenizer/statistics/formula as search.

This completeness is required for later closed judged-pool features and prevents absent top-K
membership from being confused with a negative or missing sparse score.

## Reload and Resource Contract

Artifact loading recursively verifies lineage and payload checksums, validates typed-array byte
lengths/counts, checks contiguous catalog ordinals and final posting offsets, then memory-maps
the arrays. Cold reload must reproduce candidates, scores, ranks, and pair scores exactly.

The build records wall time, payload bytes, and process peak RSS. Peak RSS must not exceed
`runtime.rss_limit_mb` (5,632 MiB by default); `SparseResourceError` retains the full measurement
and blocks promotion. Final reference-machine timing/RSS qualification remains Goldfish 015.

## Protocol-Safe Metric Foundation

`market_rank.evaluation.metrics` exposes two explicit protocols:

### `closed_pool_task1_v1`

Requires the ranked products to equal the complete official judged set. It permits:

- official-gain NDCG (gains are used directly, never exponentiated again);
- threshold-explicit precision, MAP, and MRR;
- Exact Hit.

### `retrieval_catalog_task1_us_v1`

Allows unjudged fixed-catalog products. It emits only:

- judged Recall;
- Exact Hit;
- judged MRR;
- known-judgment coverage;
- unjudged rate.

It does not expose naive catalog Precision, MAP, or NDCG. Empty retrieval lists remain explicit
query results with zero applicable metrics. Every metric record carries its protocol, cutoff,
returned/judged/unjudged counts, and total relevant-judgment count.

## Configuration

```yaml
retrieval:
  sparse:
    tokenizer_version: unicode-word-v1
    component_version: bm25-v1
    k1: 1.2
    b: 0.75
    default_top_k: 150
    max_top_k: 1000
    sqlite_batch_rows: 10000
```

Configuration is strict and canonically hashed. Changing tokenizer, BM25 parameters, limits, or
batch semantics creates new lineage coordinates instead of overwriting an index.

## Failures and Idempotency

Missing/incompatible foundations, catalog/document misalignment, empty vocabularies, SQLite or
filesystem errors, invalid typed array sizes, unknown explicit product IDs, duplicate pair IDs,
invalid query/top-K bounds, corruption, or excess RSS fail with domain exceptions. Staged
partials are removed by the artifact transaction. Compatible immutable indexes are verified and
reused without rebuilding.

## Out of Scope

- running the production ESCI build or claiming its final latency/quality;
- corpus-level BM25 evaluation reports/bootstrap intervals (Goldfish 009);
- dense embeddings/FAISS (Goldfish 008);
- RRF, retrieval comparison, query parsing, features, rankers, APIs, or UI.

## Acceptance Criteria

1. Catalog/document ordinals and hashes align exactly before indexing.
2. Tokenization and BM25 parameters are versioned and identical online/offline.
3. Persisted vocabulary, statistics, and compact typed postings reload without rebuilding.
4. Search returns only catalog members, finite scores, contiguous ranks, and deterministic ties.
5. Pair scoring covers every requested known product including zero matches.
6. Cold reload preserves search and pair outputs exactly.
7. Shuffled raw input produces byte-identical deterministic index payloads.
8. Artifact/resource facts and exact Goldfish 006 lineage are persisted and verified.
9. Closed-pool and catalog-retrieval metric protocols cannot be mixed.
10. Imports/tests perform no network or production index build; all locked gates pass.

## Rollback

Remove retrieval configuration, sparse/evaluation modules, CLI command, tests, documentation,
and local `sparse-index` artifacts. Goldfish 006 and all source-data artifacts remain intact.
