# Goldfish 010 — Query Understanding and Ranking Features

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 010 |
| Parent design | `ELEPHANT.md`, Sections 9–13, 17, 30, 36–40, 48–50; D-006/D-008 |
| Milestone | M7 — Deterministic query understanding and `ltr_core_v1` |

## Objective

Turn the Goldfish 006 source tables, Goldfish 007/008 direct pair scorers, and Goldfish 009
retrieved union into one versioned feature-platform artifact. The same pure query parser and
feature formulas serve offline materialization and future online requests. The artifact contains
a label-free parser state, an ordered leakage-reviewed registry, labeled closed-pool matrices,
label-free retrieved-union matrices, population distributions, parity fixtures, and an observed
RSS gate for the Apple M3 with 8 GB unified memory.

```bash
uv run market-rank features build-ranking
uv run market-rank features build-ranking --profile portfolio
```

The default profile is `development`. The command requires compatible promoted Goldfish
006–009 artifacts and the exact cached MiniLM query encoder. It performs no download, index
build, embedding build, training, or implicit parent repair.

## Deterministic Query Parser

`query-parser-v1`:

1. validates string type and configured UTF-8 byte, character, and token bounds;
2. preserves raw text and applies NFKC, case folding, and whitespace collapse;
3. tokenizes with the same versioned Unicode word rule as BM25;
4. preserves full and stopword-reduced views while retaining model/unit/compatibility evidence;
5. extracts numbers, normalized number-unit measurements, conservative letter+digit model
   identifiers, and compatibility tokens/phrases;
6. matches the longest token-boundary brand from official catalog values;
7. matches official colors and a small versioned alias lexicon;
8. applies only an explicit conservative spelling-alias table and records a warning;
9. emits parser/state/query hashes and non-filtering entity confidences.

Brand and color dictionaries come from label-free official fields for products with usable text.
Their exact normalized values, aliases, parser version, and hash are persisted in feature state.
Detected entities are evidence only; they never remove a retrieval candidate.

## Primary Feature Registry

`feature-registry-v1` defines an ordered 44-column `ltr_core_v1`. Every entry records name,
compact dtype, class, provenance, default, missing behavior, formula, online availability, and
explicit false flags for label, product-identity, and source-top-K use.

The registry covers:

- query length, uniqueness, digit/model/entity/confidence, locale, and mean catalog-IDF signals;
- official product field lengths, missingness, completeness, and train-fitted brand/color codes;
- direct BM25 and dense pair scores for every row;
- bounded within-set BM25, dense, and direct-score RRF rank fractions;
- title Jaccard/coverage, description/bullet coverage, exact phrase, brand/color/model conflicts,
  compatibility, and bounded query-title log-length ratio.

`rank_fraction = (rank - 1) / max(n - 1, 1)` is finite in `[0,1]`; a singleton is `0`. Direct
score ranks sort descending by score and then product ID. Direct-score RRF uses the configured
Goldfish 009 constant and is ranked by the same stable rule.

Original generator scores/ranks/membership, absolute candidate rank/count, raw product identity,
labels/gains, test-fitted encodings, and target-history aggregates are not model features.
Product ID remains a row key. Label ID and gain appear only as metadata in the closed training
matrix.

## Fit State and Leakage Boundary

Parser dictionaries use the fixed label-free catalog. Brand/color categorical codes use only
official source fields for products appearing in the complete portfolio project-train pool; no
validation/test rows or relevance values participate. Code `0` means missing, code `1` means
unknown, and sorted known values receive contiguous codes beginning at `2`.

The persisted state pins the registry hash, parser state, scorer/index identities, RRF constant,
category mappings, fit predicate, missing/unknown codes, and state hash. `leakage-report.json`
records successful checks for target history, product identity, generator provenance,
train-only fitting, candidate label absence, and online formula availability.

## Candidate-Aligned Matrices

```text
ranking-features/<dataset-version>/<profile>/ranking-features-v1/<config-sha256>/
├── feature-registry.json
├── feature-state.json
├── parsed-queries.parquet
├── closed-matrix/part-*.parquet
├── candidate-matrix/part-*.parquet
├── distribution-report.parquet
├── parity-fixtures.parquet
├── leakage-report.json
├── ranking-features.json
├── manifest.json
└── _SUCCESS
```

The closed matrix includes every official judged pair in the selected profile that belongs to
the fixed retrieval catalog, including pairs that neither retriever returned. Judged products
without usable source text are outside that catalog, cannot receive legitimate sparse/dense
scores, and are counted as explicit manifest exclusions rather than assigned fabricated zeros.
Both indexes directly score every eligible query-product pair before within-pool BM25/dense/RRF
ranks are derived. Closed rows contain `label_id` and `gain` metadata.

The candidate matrix contains the Goldfish 009 hybrid union for the identical query cohort. It
is physically label-free. Every union product is directly rescored by both retrievers so missing
generator membership never creates a missing primary feature. Within-union ranks therefore have
the same formula as within-closed-pool ranks while retaining explicitly different population
semantics.

Rows are ordered by query ID and product ID. Query groups remain intact within Parquet
partitions. Work scans bounded query batches, loads only the current batch's official product
fields, and rejects any group above the configured 200-row maximum. It never materializes a
query×catalog cross product.

## Distribution and Formula-Parity Evidence

`distribution-report.parquet` stores row/null counts plus mean, population standard deviation,
minimum, and maximum for every registered feature in both `closed_judged` and
`retrieved_union`. This exposes the known distribution shift between approximately 40 judged
products and unions up to 200 without pretending their ranks are globally comparable.

`parity-fixtures.parquet` stores bounded real fixture vectors from both populations with a
SHA-256 over the ordered feature values. Offline materialization calls the public
`compute_core_features` formula entry point that serving will call; unit and integration tests
lock parser behavior, registry order, formula values, dtypes, feature hashes, and cold reload.

## Lineage and Resource Gate

The artifact has four exact immutable parents: data foundation, sparse index, dense index, and
profile-scoped retrieval evaluation. Their config, foundation, catalog membership, parent
manifest hashes, and profile must agree before the query encoder loads.

The process loads the sparse index, dense FAISS/vector state, and query encoder together. It
records peak RSS after load and after materialization. The larger value must remain at or below
`runtime.rss_limit_mb` (5,632 MiB by default). Initial overage blocks staging; final overage
rolls back all staged feature files. Production-scale data is not generated by ordinary tests.

## Configuration

```yaml
query_understanding:
  parser_version: query-parser-v1
  max_query_chars: 512
  max_query_bytes: 2048
  max_query_tokens: 64
  brand_min_chars: 2
ranking_features:
  component_version: ranking-features-v1
  feature_set_id: ltr_core_v1
  registry_version: feature-registry-v1
  state_version: feature-state-v1
  matrix_partition_rows: 100000
  query_batch_size: 64
  parity_fixture_rows: 128
  max_rows_per_query: 200
```

Configuration rejects a complete-query row bound larger than one partition and a row bound
smaller than the configured hybrid union.

## Out of Scope

- pointwise or LambdaMART training, tuning, serialization, or champion selection;
- candidate-conditioned features based on source membership/top-K provenance;
- target encoding, product-ID features, popularity, price, inventory, clicks, or business data;
- heavy NLP, LLM parsing, learned spelling correction, or entity-based hard filtering;
- final distribution/quality claims from the frozen portfolio test;
- API loading and serving-bundle promotion.

## Acceptance Criteria

1. Query parsing is bounded, Unicode-normalized, versioned, deterministic, and raw-text
   preserving.
2. Numbers, units, model tokens, compatibility evidence, longest-boundary brands, colors,
   aliases, warnings, confidences, and hashes have fixture coverage.
3. Parser dictionaries use label-free catalog fields and persist an exact state hash.
4. Registry names/order/dtypes/formulas/defaults/provenance and online availability are strict.
5. `ltr_core_v1` excludes labels, target history, product identity, absolute ranks/counts, and
   source-top-K provenance.
6. Categorical mappings fit only portfolio project-train source fields with reserved missing and
   unknown codes.
7. Every catalog-eligible selected judgment has complete direct BM25 and dense scores; excluded
   no-document judgments are counted.
8. Closed and candidate ranks use the same stable bounded formulas within their named
   populations.
9. Candidate payloads contain no labels or gains.
10. Matrices are candidate-aligned, query-contiguous, uniquely keyed, bounded, and partitioned.
11. Distribution reports cover every registry feature in both populations.
12. Bounded parity fixtures preserve ordered formula-vector hashes.
13. Four-parent hashes, catalog membership, config, and profile are exact and recursively
    verified.
14. Compatible artifacts reuse before query-encoder loading.
15. Load and materialization RSS pass before immutable promotion; failures leave no success
    marker.
16. Imports/tests perform no download or production feature run.
17. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove the `query_understanding` and `ranking_features` configuration blocks, query/features
packages, feature CLI command, tests, this document, and local `ranking-features` artifacts.
Goldfish 006–009 logic remains usable, although configuration-hashed local parent artifacts must
be rebuilt after removing the Goldfish 010 fields.
