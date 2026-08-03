# Goldfish 005 — Task-1 US Cohort, Leakage-Safe Splits, and Nested Profiles

| Field | Value |
|---|---|
| Status | Implemented |
| Parent design | `ELEPHANT.md`, Sections 8, 11–12, 20, 36–38, 48–50, D-013 |
| Milestone | M1 — Ingestion/profiles |
| Consolidated scope | Original proposed Goldfish 005 and 006 |

## Objective

Turn the validated ESCI examples source into a deterministic, auditable Task-1 US cohort
assignment artifact. This larger Goldfish owns the exact benchmark predicate, normalized-query
collision quarantine, project train/validation/test assignment, and nested development and
portfolio query profiles.

```bash
uv run market-rank data build-esci-profiles
```

This stage is offline. It requires the compatible Goldfish 004 validation artifact and exact
pinned examples file, and it declares that validation artifact as its immutable parent.

## Input and Output Contract

Inputs are the pinned release manifest, resolved Goldfish 002 configuration, verified raw
examples Parquet, compatible Goldfish 004 validation artifact, and code revision. Output is:

```text
dataset-profiles/<dataset-version>/nested/split-profile-v1/<config-sha256>/
├── query-assignments.parquet
├── profile-manifest.json
├── manifest.json
└── _SUCCESS
```

`query-assignments.parquet` contains one row per included Task-1 US `query_id`:

- official and project split;
- SHA-256 identity of its normalized-query leakage group;
- complete observed judgment count;
- development and portfolio membership;
- nullable collision-quarantine reason.

It contains no label values or label-derived statistics. `profile-manifest.json` records the
predicate, algorithms and seed, targets and observed counts, query/group membership hashes,
split counts, collision audit, config/release/parent hashes, and hard invariant results.

## Benchmark and Normalization Rules

The benchmark predicate is fixed to:

```text
product_locale == "us" and small_version == 1
```

`nfkc-casefold-ws-v1` applies Unicode NFKC, Unicode case folding, and whitespace collapse.
This representation is used only as a leakage-group identity in Goldfish 005; the canonical
query table remains Goldfish 006 scope.

## Split and Quarantine Rules

1. Every official test query remains project `test`.
2. If a normalized query occurs in both official train and official test, the official test
   queries remain and every colliding official-train query becomes `quarantine`.
3. Remaining official-train normalized-query groups are assigned together to project `train`
   or `validation` by a seeded, label-blind SHA-256 bucket. The default threshold is 8,500 of
   10,000 buckets (approximately 85%/15%).
4. Product overlap is allowed; normalized-query group overlap is a hard failure.
5. Quarantined queries cannot enter either profile.

The assignment never samples rows and never examines `esci_label`.

## Nested Profile Rules

All eligible normalized-query groups receive an independent seeded SHA-256 priority. The
portfolio profile takes the first configured target and development takes the prefix at its
smaller target. Defaults are 20,000 and 5,000 groups. If a fixture or release has fewer
eligible groups, observed selection is capped at the available count and reported.

Consequences:

- development query IDs are an exact subset of portfolio query IDs;
- every selected `query_id` retains all of its judgments;
- multiple IDs with the same normalized query stay in one project split and profile decision;
- shuffled row/file ordering cannot affect assignments;
- labels and label composition cannot affect assignments.

## Resource and Failure Contract

- The examples source is rehashed in 1 MiB chunks before use.
- Polars lazily filters and aggregates the raw rows to one in-memory record per query ID, rather
  than loading the approximately 601k judgment rows into Python objects.
- No network access occurs.
- Missing validation lineage, changed raw bytes, empty cohorts, inconsistent groups, leakage,
  non-nested membership, or invalid configuration fail before promotion.
- Terminal output contains only bounded counts and the final artifact ID.
- Existing compatible immutable artifacts are verified and reused.

## Configuration

`configs/base.yaml` now includes strict dataset controls:

```yaml
dataset:
  query_normalization_version: nfkc-casefold-ws-v1
  split_version: normalized-query-sha256-v1
  profile_version: nested-query-sha256-v1
  train_basis_points: 8500
  development_query_groups: 5000
  portfolio_query_groups: 20000
```

Development must not exceed portfolio. All fields participate in the canonical configuration
hash. Changing the seed, threshold, version, or targets creates new artifact coordinates.

## Out of Scope

- canonical query, product, judgment, or source tables;
- product deduplication and joins;
- fixed retrieval catalog or judged-pool materialization;
- product document construction;
- gains, ranking metrics, retrieval, features, training, or serving.

Those are later Goldfishes. In the consolidated plan, Goldfish 006 will own canonical tables,
fixed catalog/pools, and versioned product documents as one cohesive M2 data-foundation stage.

## Acceptance Criteria

1. Only exact Task-1 US query groups enter assignments.
2. Official test is preserved and colliding train groups are audited and excluded.
3. No eligible normalized-query group crosses project splits.
4. Every query ID has exactly one assignment and retains its full judgment count.
5. Development is nested exactly in portfolio and configured query-group targets are honored.
6. Assignment is deterministic under input reordering and independent of labels.
7. The artifact declares and verifies its raw-validation parent.
8. Mutated raw source bytes and missing validation evidence fail before scanning/promotion.
9. Compatible reruns reuse the immutable artifact.
10. Ruff, strict mypy, pytest, and pre-commit pass without production data or network access.

## Rollback

Remove the dataset configuration extension, profile module and exports, CLI subcommand, tests,
documentation, and any local `dataset-profiles` artifacts. Goldfish 004A remains usable; raw
source files and validation artifacts are not modified.
