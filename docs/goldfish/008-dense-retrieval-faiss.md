# Goldfish 008 — Dense Retrieval and FAISS

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 008 |
| Parent design | `ELEPHANT.md`, Sections 14, 16, 30, 36–43, 48–50; D-004 |
| Milestone | M5 — Dense retrieval |

## Objective

Generate and persist one catalog-aligned matrix of normalized 384-dimensional MiniLM product
vectors, build an exact CPU FAISS `IndexFlatIP`, and expose deterministic catalog search plus
complete explicit-pair scoring. The workflow must be resumable, offline after an explicit model
cache operation, and viable on the Apple M3 with 8 GB unified memory.

```bash
uv run market-rank retrieval cache-minilm --allow-network
uv run market-rank retrieval build-dense
```

The first command is the only model-network boundary. `build-dense`, artifact loading, imports,
tests, and later application startup are cache-only and never download or rebuild model state.

## Pinned Model and Index Contract

| Property | Contract |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Revision | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| Backend | Sentence Transformers, CPU only |
| Dimension/dtype | 384 / `float32` |
| Normalization | L2 unit norm, required range `[0.999, 1.001]` |
| FAISS | CPU `IndexFlatIP`, exact and untrained |
| Similarity | Inner product, equal to cosine for normalized vectors |
| Default K / maximum K | 150 / 1,000 |
| Initial M3 batch | 16 documents |

Model ID, full commit revision, document template version, catalog hash, dimension, dtype,
normalization, index type, and resolved configuration hash are immutable artifact metadata.
Loading a different encoder ID, revision, or dimension fails before serving a query.

The checked-in dependency route uses `sentence-transformers>=5.6,<6`, `faiss-cpu>=1.14,<2`, and
`numpy>=2,<3`. The lockfile selects an ARM64 macOS FAISS CPU wheel on the reference machine.

## Explicit Offline Model Cache

`cache-minilm` resolves the exact commit with Hugging Face `snapshot_download`. Without
`--allow-network`, it is a cache probe and gives a bounded instruction on a miss. With the flag,
network permission is explicit and limited to the pinned repository/revision. The resulting
cache lives under `models/huggingface` by default and is excluded from Git.

`SentenceTransformerEncoder` resolves the same snapshot with `local_files_only=True`, forces
CPU execution, disables remote code trust, caps Torch threads from `runtime.max_threads`, and
requests normalized float32 NumPy output. It never uses CUDA or requires MPS.

## Artifact and Lineage

```text
dense-index/<dataset-version>/portfolio/minilm-l6-v2-flatip-v1/<config-sha256>/
├── product-embeddings.npy
├── document-map.parquet
├── flatip.faiss
├── dense-index.json
├── manifest.json
└── _SUCCESS
```

The Goldfish 003 manifest recursively pins the exact Goldfish 006 foundation manifest and
checksums every payload. The document map retains contiguous catalog ordinals, locale, product
ID, document hash, and document-template version. Vector row `i`, FAISS ID `i`, and document-map
ordinal `i` therefore refer to the same product.

The `.npy` matrix is platform-neutral and read through NumPy memory mapping. The FAISS artifact
is the local CPU index; optional remote workflows must return the neutral vectors and ordered ID
map for local FAISS rebuild/parity because native FAISS files are not assumed portable across
architectures.

## Memory-Bounded and Restartable Build

Products are sorted by the foundation catalog ordinal. A stable, artifact-specific workspace
under `artifacts/.dense-build` contains a partial `.npy` memmap and strict checkpoint JSON. Each
batch follows this sequence:

1. encode at most `embedding_batch_size` documents;
2. validate exact shape, float32 dtype, finite values, and unit norms;
3. write and flush the contiguous memmap slice;
4. atomically advance `completed_documents` in the checkpoint;
5. measure process peak RSS and stop before promotion if it exceeds the configured limit.

An interrupted run verifies checkpoint lineage, shape, and all completed-row norms before
continuing at the first unfinished ordinal. It never trusts a disjoint or unverified range.
Successful promotion removes only that artifact's scoped workspace. Failed builds retain the
checkpoint for diagnosis/resume and cannot create `_SUCCESS`.

After embeddings complete, FAISS copies the matrix into exact `IndexFlatIP` state. This
temporary overlap is measured against `runtime.rss_limit_mb` (5,632 MiB by default). Offline
embedding, index construction, evaluation, and serving remain separate processes so macOS can
reclaim memory between stages.

## Search and Explicit-Pair Scoring

`DenseIndex.search(query, top_k)`:

- embeds a nonblank query with the exact compatible encoder;
- validates finite unit-normalized query output;
- performs exact FlatIP search over the resolved named full or compact fixed catalog;
- sorts by descending score and then lexical product ID for deterministic boundary ties;
- returns finite scores, contiguous one-based ranks, immutable retriever/index IDs, and measured
  end-to-end dense latency for the request.

A blank query returns no candidates. Query text is bounded to 4,096 characters and K is bounded
by the persisted hard maximum.

`DenseIndex.score_pairs(query, product_ids)` bypasses top-K membership and multiplies the query
vector against the requested rows of the read-only product memmap. It preserves input order and
returns every unique known product. Blank-query pair scores are explicit zeros. Unknown or
duplicate product IDs fail instead of becoming silent missing values.

## Validation, Measurements, and Reload Parity

Build metadata records:

- embedding, FAISS, and total build time;
- matrix, FAISS, and total payload bytes;
- process peak RSS and configured gate;
- completed and resumed document counts;
- min/max product-vector norms;
- warm query-encode plus exact-search p50, p95, and maximum latency over a deterministic query
  sample.

Latency observations are engineering measurements, not benchmark claims. Goldfish 015 performs
the final M3/8 GB qualification with frozen production artifacts.

Cold loading recursively verifies payload checksums and parent lineage, then checks document
ordinals, matrix shape/dtype/norm facts, FAISS type/dimension/count, and encoder identity.
Fixture tests prove exact score/rank/pair parity across reload and stable results under raw-row
reordering.

## Configuration

```yaml
retrieval:
  dense:
    model_id: sentence-transformers/all-MiniLM-L6-v2
    model_revision: c9745ed1d9f207416be6d2e6f8de32d1f16199bf
    component_version: minilm-l6-v2-flatip-v1
    embedding_dimension: 384
    embedding_batch_size: 16
    default_top_k: 150
    max_top_k: 1000
    latency_sample_queries: 20
    model_cache_dir: models/huggingface
```

All values are strictly typed and canonically hashed. Batch, K, cache-path, model, or component
changes create different configuration lineage rather than overwriting an artifact.

## Out of Scope

- downloading the model implicitly or generating the production ESCI embeddings in tests;
- hybrid RRF, sparse+dense comparison reports, confidence intervals, retrieval slices, or the
  combined serving-memory gate (Goldfish 009);
- HNSW, IVF, PQ, CUDA, or MPS optimization without measured need and reference parity;
- query parsing, ranking features/models, API serving, or UI;
- final latency/RSS claims on the reference hardware (Goldfish 015).

## Acceptance Criteria

1. Model ID and full immutable revision are pinned; builds are cache-only.
2. Vector rows align one-to-one with foundation catalog/document ordinals and hashes.
3. Embeddings are finite, normalized, float32, 384-wide, and written in bounded batches.
4. Interrupted builds resume only a verified contiguous prefix.
5. Exact CPU FlatIP persists and reloads with compatible type, count, and dimension.
6. Search uses finite cosine-equivalent scores, contiguous ranks, and deterministic ties.
7. Pair scoring covers every requested known product independent of top-K membership.
8. Cold reload preserves search and pair results exactly.
9. Peak RSS, payload bytes, phase timing, and warm latency distributions are persisted.
10. Excess RSS, bad norms/shapes, incompatible encoders, corruption, or lineage drift block
    promotion with domain errors.
11. Imports and tests perform no model download or production embedding build.
12. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove dense configuration/module/CLI/tests/documentation, the three direct dependencies and
their lock entries, local `models/` cache, scoped `.dense-build` checkpoints, and promoted
`dense-index` artifacts. Goldfish 007 sparse retrieval and Goldfish 006 source-data artifacts
remain conceptually unchanged, although their configuration-hashed fixtures must be rebuilt
after a configuration rollback.
