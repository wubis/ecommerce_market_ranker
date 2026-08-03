# Goldfish 004 — Pinned ESCI Raw Manifest and Schema Validator

| Field | Value |
|---|---|
| Status | Implemented |
| Parent design | `ELEPHANT.md`, Sections 5–6, 8, 12, 30, 44, 48–50 |
| Milestone | M1 — Ingestion/profiles |
| Scope size | One short, independently testable raw-data trust boundary |

## Objective

Pin the authoritative Amazon Science ESCI release, license, paper, filenames, byte sizes, and
SHA-256 checksums; validate already-downloaded local files before any transformation; and
atomically publish a strict validation report for valid inputs.

## Pinned Source

Goldfish 004 pins official repository commit
`7916cdf6ab75a462e77f20ab40428a10923998d5`. The checked-in release manifest records the
Apache-2.0 license, Shopping Queries Dataset paper, and these source files:

| Role | File | Bytes |
|---|---|---:|
| examples | `shopping_queries_dataset_examples.parquet` | 51,286,808 |
| products | `shopping_queries_dataset_products.parquet` | 1,108,857,465 |
| sources | `shopping_queries_dataset_sources.csv` | 1,682,802 |

The Parquet checksums are the official Git LFS object SHA-256 values at the pinned revision;
the CSV checksum is calculated from the pinned Git blob. Dataset files remain Git-ignored.

## Inputs and Outputs

Inputs are the checked-in release manifest, a local raw directory containing the three exact
filenames, an aware UTC retrieval timestamp, the Goldfish 002 config hash, a code revision,
and a Goldfish 003 artifact store.

Validation returns a strict `RawValidationReport` whether input is valid or invalid. Only a
valid report may be promoted. Promotion creates a `raw-validation` artifact containing the
canonical release manifest and canonical validation report.

## In Scope

- Strict JSON release manifest with duplicate/unknown-key rejection and canonical SHA-256.
- Pinned official repository revision, license, paper, file URL, byte size, and checksum.
- Safe local paths with symbolic-link and traversal rejection.
- Streaming 1 MiB file checksums.
- Polars lazy scans with exact ordered columns and semantic physical-type checks.
- Required-null, non-empty, locale, ESCI label, split, and binary-flag checks.
- Primary-key uniqueness and query-text consistency checks.
- Example-to-product and example-to-source referential checks.
- A complete bounded diagnostic report on ordinary validation failure.
- Atomic promotion through the Goldfish 003 protocol.
- Toy Parquet/CSV fixtures; no large source files in tests or Git.

## Out of Scope

- Downloading files, Git LFS installation, retries, credentials, or network access. The
  explicit download boundary is defined separately by Goldfish 004A.
- Filtering to `us` and `small_version == 1`; that belongs to Goldfish 005.
- Deduplication, normalization, collision quarantine, grouped splitting, or profile sampling.
- Canonical normalized tables, product documents, retrieval, features, training, or serving.
- Treating absent judgments as irrelevant.

## Public Interface

```text
load_release_manifest(path) -> ResolvedReleaseManifest

validate_raw_dataset(
    release,
    raw_root,
    retrieved_utc=aware_utc_datetime,
) -> RawValidationReport

publish_raw_validation(
    release,
    report,
    artifact_store,
    config_sha256,
    code_revision,
) -> LoadedArtifact
```

`RawValidationReport.require_valid()` raises `RawDataValidationError` and attaches the full
report. Integrity failures stop schema parsing for the affected file. Structural failures
prevent cross-file joins when their inputs are unsafe.

## Files

| File | Change |
|---|---|
| `configs/data/esci-release-7916cdf6ab75.json` | Pin the official source release. |
| `src/market_rank/data/esci_raw.py` | Implement contracts, validation, and promotion. |
| `src/market_rank/data/__init__.py` | Export the data trust-boundary API. |
| `tests/unit/test_esci_raw.py` | Cover integrity, schemas, domains, keys, joins, and failure. |
| `pyproject.toml`, `uv.lock` | Add and lock Polars for lazy local validation. |
| `.gitignore` | Scope generated-data ignores to repository-root lifecycle directories. |
| `README.md` | Document explicit acquisition and validation usage. |

## Acceptance Criteria

1. The checked-in official release manifest loads canonically and pins all three file hashes.
2. Matching toy files pass integrity, schema, domain, key, consistency, and join checks.
3. Missing, changed, malformed, unsafe, or incompatible files return an invalid report.
4. Invalid reports cannot be atomically promoted and retain actionable check details.
5. Valid evidence is promoted with the resolved config hash and reloads through Goldfish 003.
6. Validation is lazy/streaming and does not materialize the full products dataset in Python.
7. Ruff format/lint, strict mypy, full pytest, and pre-commit pass.

## Failure and Rollback

Validation never modifies raw input. Failed reports are returned in memory and may be logged
by a future CLI, but are not promoted as valid stage output. Artifact transactions roll back
on publication failure. Rollback removes the release manifest, data module, tests, Polars
dependency, documentation, and regenerated lockfile; Goldfish 001–003 remain usable.
