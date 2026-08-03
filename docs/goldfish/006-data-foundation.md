# Goldfish 006 — Canonical Data Foundation, Fixed Catalog, Judged Pools, and Documents

| Field | Value |
|---|---|
| Status | Implemented |
| Parent design | `ELEPHANT.md`, Sections 8–12, 30, 36–40, 48–50 |
| Milestone | M2 — Normalized data |
| Consolidated scope | Original proposed Goldfish 007–009 plus the M2 resource gate |

## Objective

Promote one complete, validated M2 foundation for retrieval and ranking work: canonical source
tables, fixed label-blind catalog membership, complete nested judged pools, versioned product
documents, and a preliminary Apple M3/8 GB serving-memory proceed/block estimate.

```bash
uv run market-rank data build-esci-foundation
```

The stage is local and offline. It requires the compatible Goldfish 005 artifact, recursively
verifies its Goldfish 004 parent through the artifact protocol, and rehashes all three pinned
raw files before transformation.

## Artifact Contract

```text
data-foundation/<dataset-version>/portfolio/data-foundation-v1/<config-sha256>/
├── queries.parquet
├── sources.parquet
├── judgments.parquet
├── products.parquet
├── product-documents.parquet
├── catalog-membership.parquet
├── catalog-exclusions.parquet
├── judged-pools.parquet
├── foundation-manifest.json
├── manifest.json
└── _SUCCESS
```

Every table has a declared primary key, deterministic ordering, observed row count, columns,
byte size, and SHA-256 in `foundation-manifest.json`. The Goldfish 003 manifest independently
hashes every payload and records the exact Goldfish 005 parent manifest hash.

## Canonical Populations

### Queries and sources

`queries.parquet` contains one non-quarantined portfolio query ID with raw and normalized text,
locale, official/project split, source, and nested-profile flags. `sources.parquet` preserves
the official source field separately. Missing joins are hard failures.

### Judgments

`judgments.parquet` contains the complete judged list for each portfolio query. Its key is
`(query_id, locale, product_id)`. Identical judgment duplicates collapse with a count and
minimum source example ID; conflicting labels or official splits fail. It persists both:

| ESCI | Label ID | Official gain |
|---|---:|---:|
| I | 0 | 0.0 |
| C | 1 | 0.01 |
| S | 2 | 0.1 |
| E | 3 | 1.0 |

IDs and gains remain different columns with mapping IDs. A stable product-ID ordinal gives
each complete query group deterministic contiguous ranking order.

### Fixed catalog and products

Catalog candidates are distinct product keys referenced by the full release-wide Task-1 US
predicate across official train and test. Profile membership and label values are never read
for catalog selection. This keeps `esci_task1_us_catalog_v1` identical across development and
portfolio experiments.

Candidate keys join exactly one official product. `products.parquet` retains official nullable
fields and separate derived clean text, normalized brand/color, missingness flags, and document
availability. Products without any usable official text are excluded from retrieval membership
and written to `catalog-exclusions.parquet` with reason `no_usable_source_text`.

### Product documents

`product-document-v1` strips HTML, applies Unicode NFKC, removes control characters, collapses
whitespace, and applies per-field character caps. It emits:

```text
[TITLE] title [BRAND] brand [COLOR] color [BULLETS] bullets [DESCRIPTION] description
```

The raw display fields remain unchanged. Documents record character/UTF-8 byte counts, version,
and content SHA-256. `catalog-membership.parquet` provides deterministic zero-based catalog
ordinals and document hashes for later index alignment.

### Closed judged pools

`judged-pools.parquet` materializes `esci_task1_us_judged_pool_v1` for both development and
portfolio. Development rows are an exact subset of portfolio rows; every selected query keeps
all judgments. This is the required supervised fitting and official-gain evaluation population.
It is distinct from the fixed retrieval catalog and must not interpret unjudged catalog items as
irrelevant.

## Resource Gate

`m2-catalog-linear-v1` is explicitly preliminary. It estimates a later serving working set as:

- 1.5 × document UTF-8 bytes for a sparse-index placeholder;
- `catalog products × 384 × 4` bytes for float32 MiniLM vectors;
- 256 bytes per catalog ID/alignment record;
- compact title/brand/color display bytes;
- configured fixed runtime/model reserve (512 MiB by default).

The sum must not exceed `runtime.rss_limit_mb` (5,632 MiB by default). Failure raises
`ResourceGateError` carrying the complete estimate and prevents artifact promotion. M3, M5, and
M6 replace projected components with measured sparse, dense, and combined RSS; this estimate is
not a performance claim.

## Memory and Determinism

- Polars lazy scans apply predicate/projection pushdown and stream Parquet outputs.
- Raw rows are never converted into full Python object collections.
- Large canonical tables are written sequentially; product frames are released before document
  scans.
- Tables are deterministically sorted before persistence.
- Catalog construction is label-blind and profile-independent.
- Fixture tests prove parity under raw-row reordering and label replacement.
- No network access, index building, embeddings, or training occurs.

## Configuration

Goldfish 006 adds strict document versions/caps and the M2 runtime reserve to `DatasetConfig`.
All values participate in the canonical Goldfish 002 hash, so semantic changes create new raw
validation, profile, and foundation coordinates rather than overwriting prior artifacts.

## Failures and Reuse

Missing/incompatible parents, changed raw checksums, invalid assignment schema, missing joins,
conflicting judgments, duplicate output keys, invalid catalog partitioning, incomplete pools,
or a failed resource gate block promotion. CLI errors remain one line. A compatible immutable
foundation is recursively verified and reused without rebuilding tables.

## Out of Scope

- BM25 tokenization, indexing, pair scoring, retrieval, or evaluation;
- ranking metric/bootstrap implementation;
- embeddings and FAISS;
- RRF, query parsing, features, ranker training, APIs, or UI.

The intended next consolidated stage is Goldfish 007: a persisted BM25 baseline plus the
protocol-safe ranking/retrieval metric framework needed to evaluate it.

## Acceptance Criteria

1. Eight canonical tables have exact schemas, keys, deterministic order, and integrity hashes.
2. Queries/judgments contain only complete non-quarantined portfolio groups and preserve nested
   development membership.
3. Label IDs and official gains are mapped separately and exactly.
4. Fixed catalog membership uses all Task-1 US product participation and no label/profile value.
5. Product joins are exact; no-text candidates are excluded with an audit.
6. Documents satisfy normalization, HTML/control removal, caps, template, and hash contracts.
7. Development judged rows are an exact subset of complete portfolio judged rows.
8. The preliminary resource estimate is transparent and blocks over-limit promotion.
9. The artifact declares the exact Goldfish 005 parent and compatible reruns reuse safely.
10. No production dataset is used by tests and all locked quality gates pass.

## Rollback

Remove the M2 configuration additions, foundation module/exports, CLI command, integration tests,
documentation, and local `data-foundation` artifacts. Goldfish 005 and its parents remain intact.
