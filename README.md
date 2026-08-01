# MarketRank

MarketRank is a planned CPU-first, multi-stage e-commerce search and ranking system built
around the public Amazon ESCI Task 1 relevance judgments. The authoritative architecture is
defined in [ELEPHANT.md](ELEPHANT.md).

This repository is currently at **Goldfish 001**: environment and quality-tooling setup only.
It intentionally contains no ingestion, retrieval, feature, training, serving, or demo logic.

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
├── src/market_rank/__init__.py
└── tests/smoke/test_import.py
```

Later directories and modules will be introduced only by approved Goldfish tasks.
