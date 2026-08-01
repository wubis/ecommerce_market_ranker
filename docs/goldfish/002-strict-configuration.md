# Goldfish 002 — Strict Layered Configuration and Canonical Hashing

| Field | Value |
|---|---|
| Status | Implemented |
| Parent design | `ELEPHANT.md`, Sections 6, 10, 30–32, and 49 |
| Milestone | M0 — Environment |
| Scope size | One short, independently testable configuration foundation |

## Objective

Provide one authoritative API for loading versioned YAML configuration, applying ordered
layers and explicit overrides, validating a typed schema, and generating a deterministic
semantic hash for future artifact and experiment lineage.

## In Scope

- A strict typed base schema for project identity, paths, runtime limits, and logging.
- A checked-in `configs/base.yaml` matching the M3/8 GB runtime contract.
- Left-to-right recursive mapping merge; later scalar/list values replace earlier values.
- Explicit dotted-key overrides applied after file layers.
- Rejection of duplicate YAML keys, non-string keys, unknown fields, invalid types, and
  non-mapping documents.
- Immutable resolved models.
- Canonical JSON with sorted keys, compact separators, UTF-8 preservation, and no NaN values.
- Full SHA-256 hash of the canonical semantic document.
- Source path lineage excluded from the semantic hash.
- Unit tests and README usage documentation.

## Out of Scope

- Component-specific retrieval, feature, ranking, evaluation, diversity, or serving schemas.
- Environment-variable parsing.
- A command-line interface.
- Persisting resolved configs into experiment directories.
- Artifact manifests or atomic stage-output logic; those belong to Goldfish 003.
- Data ingestion or ML behavior.

## Public Interface

```text
load_config(
    paths: Sequence[Path],
    overrides: Mapping[str, object] | None = None,
) -> ResolvedConfig

ResolvedConfig:
    config: AppConfig
    canonical_json: str
    sha256: str
    source_paths: tuple[Path, ...]
```

Layer precedence is `paths[0] < ... < paths[n] < dotted overrides`. Source paths are resolved
for lineage. They do not affect `canonical_json` or `sha256`.

## Files

| File | Change |
|---|---|
| `pyproject.toml` | Add Pydantic, PyYAML, and YAML typing support. |
| `uv.lock` | Lock the new dependency graph. |
| `configs/base.yaml` | Define the first base runtime configuration. |
| `src/market_rank/config.py` | Implement schema, loader, merge, override, errors, and hash. |
| `tests/unit/test_config.py` | Test success, precedence, strictness, determinism, and failures. |
| `README.md` | Document the configuration contract and example. |

## Acceptance Criteria

1. The checked-in base configuration validates.
2. Later layers recursively override earlier mappings without mutating inputs.
3. Dotted overrides have highest precedence.
4. Unknown or duplicate keys and invalid types fail with a configuration-domain exception.
5. Semantically identical configurations produce identical canonical JSON and SHA-256 hashes
   regardless of YAML key order or layer decomposition.
6. A semantic value change changes the hash.
7. Models are immutable.
8. Ruff format/lint, strict mypy, full pytest, and pre-commit pass.

## Failure and Rollback

Invalid input fails before any downstream action and includes the source or validation detail
in the exception. No partial output is persisted. Rollback removes this Goldfish's config
module, base YAML, tests, dependency additions, documentation, and regenerated lockfile; the
Goldfish 001 package skeleton remains usable.

