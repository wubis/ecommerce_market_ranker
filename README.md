# MarketRank

MarketRank is a planned CPU-first, multi-stage e-commerce search and ranking system built
around the public Amazon ESCI Task 1 relevance judgments. The authoritative architecture is
defined in [ELEPHANT.md](ELEPHANT.md).

This repository has completed **Goldfish 006**: environment/tooling, strict layered
configuration, atomic artifacts, pinned ESCI validation/download, deterministic Task-1 US
profiles, and the canonical M2 data foundation. It intentionally contains no retrieval index,
ranking metrics, embeddings, features, training, serving, or demo logic.

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

Install Python 3.11 and [`uv`](https://docs.astral.sh/uv/). With Homebrew:

```bash
brew install python@3.11 uv
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
├── src/market_rank/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── cli.py
│   ├── config.py
│   └── data/
│       ├── __init__.py
│       ├── download.py
│       ├── esci_raw.py
│       ├── foundation.py
│       └── profiles.py
└── tests/
    ├── integration/test_esci_foundation.py
    ├── smoke/test_import.py
    └── unit/
        ├── test_artifacts.py
        ├── test_config.py
        ├── test_esci_download.py
        ├── test_esci_profiles.py
        └── test_esci_raw.py
```

Later directories and modules will be introduced only by approved Goldfish tasks.
