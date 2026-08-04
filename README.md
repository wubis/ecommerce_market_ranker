# MarketRank

MarketRank is a planned CPU-first, multi-stage e-commerce search and ranking system built
around the public Amazon ESCI Task 1 relevance judgments. The authoritative architecture is
defined in [ELEPHANT.md](ELEPHANT.md).

This repository has completed **Goldfish 011**: environment/tooling, reproducible ESCI data
foundations, persisted fixed-catalog BM25 retrieval, protocol-safe metric primitives, and a
checkpointed MiniLM/FAISS dense retriever. Deterministic RRF fusion now produces partitioned
fixed-cohort retrieval reports with grouped confidence intervals, slices, paired comparisons,
and a combined sparse+dense memory gate. A deterministic query parser and ordered 44-column
`ltr_core_v1` now produce bounded closed-pool and retrieved-union feature artifacts with
leakage, distribution, parity, and resource evidence. Exact grouped populations now train and
persist directly comparable pointwise LightGBM and LambdaMART models with validation-only early
stopping, reload parity, explanation evidence, and a resource gate. It intentionally contains
no champion selection, serving, or demo logic.

## Reference environment

The required local reference environment is:

- Apple M3 Mac;
- 8 GB unified memory;
- macOS;
- Python 3.11;
- CPU-first execution, with no CUDA or required MPS support;
- target peak process RSS of 5.5 GB for required workflows.

Later data processing, embedding, training, evaluation, and serving workflows will run as
separate stages so macOS can reclaim memory between them. Performance claims must be measured
on this local reference environment.

## Prerequisites

Install Python 3.11, [`uv`](https://docs.astral.sh/uv/), and the OpenMP runtime required by
LightGBM on macOS. With Homebrew:

```bash
brew install python@3.11 uv libomp
```

The project constrains Python to the 3.11 series so local and optional remote batch
environments use the same interpreter contract.

Pre-commit also requires a modern Git. Verify that the first `git` on `PATH` supports the
command it uses:

```bash
git --version
git ls-files --deduplicate >/dev/null
```

If an obsolete `/usr/local/bin/git` shadows macOS Git, remove that stale installation or put
`/usr/bin` ahead of it before running pre-commit. Installing a current Git with Homebrew is
another valid fix.

## Setup

Create the locked development environment:

```bash
uv sync --group dev --python 3.11
```

Run the quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Install the local pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

The hooks call tools from `uv.lock`; they do not create separate unpinned hook environments.

## Dependency locking

`pyproject.toml` declares direct dependencies and tool configuration. `uv.lock` records the
complete resolved environment. Dependency changes must update both files with `uv lock` or
`uv sync`, and CI/local verification must use `uv run` rather than globally installed tools.

Do not hand-edit `uv.lock`.

## Configuration contract

[`configs/base.yaml`](configs/base.yaml) is the first versioned runtime configuration. Load it
through `market_rank.config.load_config`; do not parse project YAML independently in future
modules.

The loader:

- merges YAML files from left to right, recursively for mappings;
- applies explicit dotted-key overrides last;
- rejects duplicate YAML keys, non-string keys, unknown fields, and invalid types;
- returns immutable Pydantic models;
- serializes resolved semantics to canonical sorted JSON;
- computes a full SHA-256 configuration hash;
- records source paths for lineage without including machine-specific paths in that hash.

Example:

```python
from pathlib import Path

from market_rank.config import load_config

resolved = load_config(
    [Path("configs/base.yaml")],
    overrides={"runtime.max_threads": 2},
)
print(resolved.sha256)
```

Later Goldfish tasks will add component schemas. They must extend the typed root model rather
than accepting arbitrary keys.

## Artifact lifecycle

[`market_rank.artifacts`](src/market_rank/artifacts.py) is the only stage-output promotion
protocol. An `ArtifactStore` is rooted at one explicit allowlisted directory. Artifact IDs
and paths follow:

```text
artifact_type/dataset_version/profile/component_version/config_sha256/
```

Writers use a temporary sibling directory and must explicitly commit inside the context. A
commit hashes every regular payload file, writes a strict canonical `manifest.json`, writes a
matching `_SUCCESS` marker, and atomically renames the directory into place. Exceptions and
uncommitted contexts discard temporary output. Existing targets are immutable.

```python
from pathlib import Path

from market_rank.artifacts import ArtifactStore
from market_rank.config import load_config

config = load_config([Path("configs/base.yaml")])
store = ArtifactStore(config.config.paths.artifacts_dir)

with store.stage(
    artifact_type="example",
    dataset_version="dataset-v1",
    profile="development",
    component_version="v1",
    config_sha256=config.sha256,
    code_revision="local-dev",
) as stage:
    stage.path("payload.txt").write_text("complete\n", encoding="utf-8")
    artifact = stage.commit()

verified = store.load(artifact.manifest.artifact_id)
```

Consumers load only explicit artifact IDs. Loading recursively verifies every declared parent
artifact and fails closed on path escape, symbolic links, missing success state, undeclared
files, byte-size or checksum changes, strict schema violations, or parent-manifest hash
mismatches. “Latest” aliases are not dependencies.

## Download and validate ESCI raw data

[`configs/data/esci-release-7916cdf6ab75.json`](configs/data/esci-release-7916cdf6ab75.json)
pins the official Amazon Science repository revision, Apache-2.0 license, paper, exact source
filenames, byte sizes, and SHA-256 checksums. The three source files total about 1.16 GB. They
are deliberately not downloaded by setup, tests, imports, validation-only calls, or
application startup.

From the repository root, explicitly run:

```bash
uv run market-rank data download-esci
```

The equivalent fully explicit console command is:

```bash
market-rank data download-esci \
  --manifest configs/data/esci-release-7916cdf6ab75.json \
  --config configs/base.yaml
```

The workflow distinguishes four states:

1. **Pinned release metadata** is the tracked JSON contract.
2. **Downloaded raw files** are verified bytes in `data/raw/esci/`.
3. **Validated raw dataset** is the complete structural validation report.
4. **Promoted validation artifact** is immutable evidence under
   `artifacts/raw-validation/...`.

Downloads use bounded 1 MiB chunks, separate connect/read timeouts, bounded transient retries,
same-directory hidden partials, incremental size/SHA-256 verification, and atomic promotion.
Validation then lazily checks exact columns, semantic types, required values, domains, primary
keys, query consistency, and cross-file joins. Invalid reports retain full programmatic detail
and cannot be promoted.

Reruns are safe and idempotent: matching files are rehashed and reused without network access,
and a compatible immutable validation artifact is reused. A mismatched existing final file is
never overwritten; the command exits with a concise error and leaves it untouched for manual
inspection. After confirming no process is using a hidden `.partial-*.tmp` file left by an
abrupt termination, that partial may be removed and the command rerun. Permanent HTTP errors
are not retried; transient connection and selected HTTP failures retry at most three times.

After successful acquisition, core validation and downstream work can run offline.

## Build Task-1 US splits and nested profiles

Once the current configuration has a successful raw-validation artifact, run:

```bash
uv run market-rank data build-esci-profiles
```

The fully explicit form is:

```bash
market-rank data build-esci-profiles \
  --manifest configs/data/esci-release-7916cdf6ab75.json \
  --config configs/base.yaml
```

If configuration semantics changed since the last acquisition, rerun `download-esci` first.
Matching raw files are rehashed without network access and a compatible validation artifact is
published for the new configuration hash.

Goldfish 005 applies only `product_locale == "us"` and `small_version == 1`. It groups queries
using versioned Unicode NFKC, case folding, and whitespace collapse. Official test queries are
frozen as project test. Any official-train query whose normalized text also occurs in official
test is quarantined; remaining train groups receive deterministic approximately 85%/15%
project train/validation assignments.

Development and portfolio use stable label-blind priorities over complete normalized-query
groups. Defaults target 5,000 and 20,000 groups, with development guaranteed to be a subset of
portfolio. The promoted artifact contains a compact assignment Parquet and strict audit
manifest under `artifacts/dataset-profiles/...`. It never copies query text to terminal output,
never samples judgment rows, and never uses ESCI labels to choose splits or profiles. A rerun
verifies and reuses a compatible immutable artifact.

## Build the canonical M2 data foundation

After Goldfish 005 succeeds, run:

```bash
uv run market-rank data build-esci-foundation
```

The stage publishes canonical queries, sources, judgments, catalog products, versioned product
documents, fixed retrieval membership, no-text exclusions, and complete development/portfolio
judged pools. It keeps official label IDs and gains separate, constructs catalog membership from
all Task-1 US participation without reading labels or profile selection, and records exact table
keys/counts/checksums.

Product documents use the versioned marker template described in
[`docs/goldfish/006-data-foundation.md`](docs/goldfish/006-data-foundation.md), while official
display fields remain unchanged. A preliminary resource gate estimates sparse index, 384-wide
float32 vectors, ID/display state, and runtime reserve against the 5.5 GB process-RSS limit. An
over-limit estimate prevents promotion; later retrieval Goldfishes replace estimates with
measured component and combined RSS.

The complete local data sequence is:

```bash
uv run market-rank data download-esci
uv run market-rank data build-esci-profiles
uv run market-rank data build-esci-foundation
```

All three commands are idempotent. After the explicit initial download they operate offline,
verify their exact immutable parents, and reuse compatible outputs.

## Build and load the BM25 baseline

After the current data foundation exists, build the persisted fixed-catalog index:

```bash
uv run market-rank retrieval build-bm25
```

Goldfish 007 uses deterministic Unicode tokenization and an audited BM25 implementation. The
build writes document-term frequencies to a temporary disk-backed SQLite table in bounded
batches, then streams a sorted vocabulary, statistics, and compact typed posting arrays into an
immutable `sparse-index` artifact. Runtime memory-maps those arrays; it never rebuilds or
downloads at load time.

Reusable package APIs support both catalog top-K search and explicit product-pair scoring. Pair
scoring returns a finite value for every requested catalog product, including zero when no term
matches, so later closed-pool features do not confuse top-K absence with missing evidence. Cold
reload is exact and compatible command reruns reuse the existing artifact.

The accompanying metric layer keeps candidate populations explicit:

- `closed_pool_task1_v1` permits official-gain NDCG, thresholded precision/MAP/MRR, and Exact
  Hit only when the complete judged product set is ranked;
- `retrieval_catalog_task1_us_v1` permits judged Recall, Exact Hit, judged MRR, known-judgment
  coverage, and unjudged rate, but deliberately exposes no naive catalog NDCG/MAP/precision.

See [`docs/goldfish/007-sparse-retrieval-evaluation.md`](docs/goldfish/007-sparse-retrieval-evaluation.md)
for persistence, formula, resource, and protocol details.

## Build and load the dense baseline

Cache the exact pinned model only through the explicit network command, then run the dense build
offline:

```bash
uv run market-rank retrieval cache-minilm --allow-network
uv run market-rank retrieval build-dense
```

Goldfish 008 pins `sentence-transformers/all-MiniLM-L6-v2` to a full commit revision. Product
documents are encoded on CPU in batches of 16 into a normalized 384-wide float32 `.npy` memmap.
Each completed contiguous range is durably checkpointed, so an interrupted local job resumes at
the first unfinished product instead of starting over.

The builder creates an exact FAISS CPU `IndexFlatIP`, persists the ordered product/document map,
and records build phases, artifact bytes, peak RSS, and deterministic warm query-latency samples.
Search uses cosine-equivalent inner products with product-ID tie breaking. Explicit-pair scoring
reads bounded rows from the vector memmap and covers every requested known catalog product even
when it was absent from dense top-K.

Runtime loading verifies immutable Goldfish 006 lineage, vector dtype/shape/norms, catalog
ordinals, FAISS type/count/dimension, and query-encoder model identity. It never downloads model
weights or rebuilds embeddings/indexes. See
[`docs/goldfish/008-dense-retrieval-faiss.md`](docs/goldfish/008-dense-retrieval-faiss.md).

## Evaluate sparse, dense, and hybrid retrieval

After both compatible indexes exist, build the development report or the larger portfolio
report:

```bash
uv run market-rank retrieval evaluate-hybrid
uv run market-rank retrieval evaluate-hybrid --profile portfolio
```

Goldfish 009 unions BM25 and dense top-150 results with RRF (`k=60`), deduplicates them, retains
both sources' scores/ranks/retriever/index provenance, applies deterministic tie breaking, and
caps the hybrid union at 200 products. One empty source is represented as degraded provenance;
no-candidate queries remain explicit evaluation rows.

All three stages are compared on the identical fixed catalog and profile query cohort under
`retrieval_catalog_task1_us_v1`. Reports include only judged Recall, Exact Hit, judged MRR,
known-judgment coverage, and unjudged rate for `E` and `E+S` at cutoffs 10 and 100—never naive
catalog precision, MAP, or NDCG.

Candidate and query-metric rows are partitioned. Aggregate reports include fixed-seed 95%
confidence intervals that resample normalized-query groups, named query-length/source/split/
Exact-presence slices, and paired hybrid-vs-sparse/dense/best-single improvements. Promotion
also requires the simultaneously loaded sparse+dense process and completed evaluation phase to
remain below the configured 5.5GB RSS limit. See
[`docs/goldfish/009-hybrid-retrieval-evaluation.md`](docs/goldfish/009-hybrid-retrieval-evaluation.md).

## Build query understanding and ranking features

After the compatible Goldfish 006–009 artifacts exist, materialize the development or portfolio
feature artifact:

```bash
uv run market-rank features build-ranking
uv run market-rank features build-ranking --profile portfolio
```

Goldfish 010 persists a bounded `query-parser-v1` state from official label-free catalog
brand/color values. It performs NFKC/casefold/whitespace normalization, versioned tokenization,
conservative number/unit/model/compatibility extraction, longest-boundary brand matching, color
aliases, explicit spelling aliases, entity confidences, warnings, and deterministic hashes.
Parser signals are ranking evidence and never hard filters.

The ordered 44-feature `ltr_core_v1` registry combines query/product interactions with direct
BM25 and dense scores, bounded within-set rank fractions, and direct-score RRF. Category codes
fit only portfolio project-train source fields with reserved missing/unknown codes. Labels,
target history, product identity, absolute ranks/counts, and original top-K provenance are not
model features.

The closed matrix contains every catalog-eligible judged pair with label/gain metadata and
counts judgments excluded for lacking a retrievable document; the retrieved-union matrix is
physically label-free. Both populations directly score every eligible pair with both indexes.
Query groups stay intact in bounded Parquet partitions, and reports persist population
distributions, shared-formula parity vectors, exact four-parent lineage, leakage checks, and a
5.5GB RSS promotion gate. See
[`docs/goldfish/010-query-understanding-ranking-features.md`](docs/goldfish/010-query-understanding-ranking-features.md).

## Train pointwise and LambdaMART rankers

After building a compatible feature artifact, train both supervised objectives on the exact same
development or portfolio population:

```bash
uv run market-rank ranking train
uv run market-rank ranking train --profile portfolio
```

Goldfish 011 materializes only catalog-eligible judged project-train and validation rows. Project
test is excluded before materialization. Complete query groups are sorted and converted to exact
group arrays; groups with fewer than two rows or one distinct label are audited and excluded
without sampling. Label IDs remain separate from verified official gains.

Pointwise LightGBM uses `regression_l2`; LambdaMART uses `lambdarank`. Both consume identical
float32 `ltr_core_v1` matrices, labels, group checksums, categorical fields, seeds, threads, and
official gains. Validation NDCG@10/@20 alone selects the best iteration. The command does not
evaluate project test or select a champion.

The immutable artifact includes LightGBM text models, population predicates/query IDs/groups,
per-iteration validation history, feature importance, bounded contribution samples, and cold
reload parity. Both models are loaded together before promotion, and matrix/training/reload RSS
must remain below 5.5GB. See
[`docs/goldfish/011-pointwise-lambdamart-rankers.md`](docs/goldfish/011-pointwise-lambdamart-rankers.md).

## Download and offline boundary

Internet access is allowed only during explicit setup operations:

1. installing or updating locked Python packages;
2. downloading the official ESCI dataset;
3. downloading approved open-source pretrained model revisions.

Downloaded data, model caches, generated indexes, features, experiments, and evaluation
artifacts are local state and are excluded from Git. Once those explicit downloads are
complete, the required core workflow must run without internet access. Application startup
must never download dependencies, data, or models.

## Optional Colab batch acceleration

Free Google Colab may later accelerate an isolated offline batch such as product embedding
generation or optional cross-encoder scoring. It is not the canonical runtime and cannot be
used for final macOS latency or memory claims.

A Colab-produced artifact is acceptable only when the batch:

- invokes reusable package or CLI logic rather than notebook-only code;
- uses the same resolved configuration and pinned model revision;
- exports platform-neutral data plus ordered ID mappings;
- records input/output checksums, dependency versions, and an artifact manifest;
- passes local compatibility and parity validation before promotion.

FastAPI, Streamlit, smoke tests, artifact loading, and the required portfolio workflow remain
local-first.

## Current layout

```text
ecommerce_market_ranker/
├── ELEPHANT.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .pre-commit-config.yaml
├── configs/
│   ├── base.yaml
│   └── data/esci-release-7916cdf6ab75.json
├── docs/goldfish/002-strict-configuration.md
├── docs/goldfish/003-artifact-protocol.md
├── docs/goldfish/004-esci-raw-validation.md
├── docs/goldfish/004a-esci-download-command.md
├── docs/goldfish/005-task1-us-splits-profiles.md
├── docs/goldfish/006-data-foundation.md
├── docs/goldfish/007-sparse-retrieval-evaluation.md
├── docs/goldfish/008-dense-retrieval-faiss.md
├── docs/goldfish/009-hybrid-retrieval-evaluation.md
├── docs/goldfish/010-query-understanding-ranking-features.md
├── docs/goldfish/011-pointwise-lambdamart-rankers.md
├── docs/goldfish/consolidated-roadmap.md
├── src/market_rank/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── esci_raw.py
│   │   ├── foundation.py
│   │   └── profiles.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── retrieval.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── artifact.py
│   │   ├── core.py
│   │   └── registry.py
│   ├── query/
│   │   ├── __init__.py
│   │   └── parser.py
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── population.py
│   │   └── training.py
│   └── retrieval/
│       ├── __init__.py
│       ├── dense.py
│       ├── hybrid.py
│       └── sparse.py
└── tests/
    ├── integration/test_dense_retrieval.py
    ├── integration/test_esci_foundation.py
    ├── integration/test_ranker_training.py
    ├── integration/test_ranking_features.py
    ├── integration/test_retrieval_evaluation.py
    ├── integration/test_sparse_retrieval.py
    ├── smoke/test_import.py
    └── unit/
        ├── test_artifacts.py
        ├── test_config.py
        ├── test_esci_download.py
        ├── test_esci_profiles.py
        ├── test_features.py
        ├── test_hybrid.py
        ├── test_metrics.py
        ├── test_query_parser.py
        ├── test_training_population.py
        └── test_esci_raw.py
```

Later directories and modules will be introduced only by approved Goldfish tasks.
