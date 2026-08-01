# Goldfish 003 — Artifact Manifest and Atomic Stage-Output Protocol

| Field | Value |
|---|---|
| Status | Implemented |
| Parent design | `ELEPHANT.md`, Sections 6, 9, 30, 31, 44, 49, and 50 |
| Milestone | M0 — Cross-cutting persistence foundation |
| Scope size | One short, independently testable artifact lifecycle |

## Objective

Provide one authoritative protocol for writing immutable stage outputs to temporary
same-filesystem directories, recording strict integrity and compatibility manifests, and
atomically promoting only complete artifacts with a success marker.

## Inputs and Outputs

Inputs are five canonical artifact coordinates, the Goldfish 002 configuration SHA-256, a
code revision, optional exact parent-manifest hashes, and one or more payload files. The
output directory is:

```text
artifact_type/dataset_version/profile/component_version/config_sha256/
```

It contains payloads, canonical `manifest.json`, and `_SUCCESS`. The success marker stores the
SHA-256 of the exact manifest bytes.

## In Scope

- Strict, immutable Pydantic manifest models with schema version 1.
- Artifact ID derivation from validated portable path segments.
- File byte counts and streaming SHA-256 checksums.
- Exact parent artifact IDs and manifest hashes.
- Canonical, compact, sorted JSON manifests with aware UTC creation time and code revision.
- Temporary directories created beside the final target for same-filesystem atomic rename.
- Explicit commit, rollback on failure or abandonment, and immutable-target refusal.
- Explicit allowlisted store roots, traversal rejection, and symbolic-link rejection.
- Load-time manifest, success-marker, payload-set, size, checksum, and dependency validation.
- Unit tests and README usage documentation.

## Out of Scope

- ESCI URLs, licenses, release checksums, or schema validation; those belong to Goldfish 004.
- Artifact-type-specific metadata schemas.
- Experiment `run.json`, metrics, promotion aliases, garbage collection, or remote storage.
- Cross-filesystem copying, distributed writers, locking, or concurrent publication.
- Data ingestion, index building, model training, serving, or ML serialization choices.

## Public Interface

```text
ArtifactStore(root)

store.stage(
    artifact_type,
    dataset_version,
    profile,
    component_version,
    config_sha256,
    code_revision,
    dependencies=(),
) -> ArtifactTransaction

transaction.path(relative_path) -> Path
transaction.commit() -> LoadedArtifact

store.load(
    artifact_id,
) -> LoadedArtifact
```

Stages write only inside `with store.stage(...) as transaction`. A successful body still
requires explicit `commit()`; otherwise the temporary directory is discarded. Loading
recursively resolves every declared parent from the same store and requires its verified
manifest SHA-256 to match exactly.

## Files

| File | Change |
|---|---|
| `src/market_rank/artifacts.py` | Implement schemas, transaction, promotion, and verification. |
| `tests/unit/test_artifacts.py` | Test completion, integrity, dependencies, paths, and rollback. |
| `README.md` | Document the lifecycle and minimal usage. |
| `docs/goldfish/003-artifact-protocol.md` | Define this Goldfish contract. |

## Acceptance Criteria

1. Final paths follow the Elephant coordinate layout and use the resolved config hash.
2. A committed artifact has checksummed payloads, a strict manifest, and matching `_SUCCESS`.
3. The final artifact directory appears only through same-filesystem atomic rename.
4. Exceptions, abandoned transactions, validation failures, and duplicate targets leave no
   promoted partial artifact.
5. Traversal, reserved payload paths, symbolic links, missing or extra payloads, and checksum
   changes fail closed.
6. A child loads only when every declared parent manifest hash matches exactly.
7. Ruff format/lint, strict mypy, full pytest, and pre-commit pass.

## Failure and Rollback

All protocol failures use artifact-domain exceptions. Failed temporary directories are
removed when the transaction context exits; an interrupted process may leave only a hidden
temporary sibling without `_SUCCESS`, never a promoted target. Cleanup of crash leftovers is
future maintenance scope. Rollback removes this module, tests, document, and README section;
Goldfish 001 and 002 remain usable.
