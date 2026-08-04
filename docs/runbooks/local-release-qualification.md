# Local Release Qualification Runbook

## Purpose

Use this runbook to produce MarketRank release-candidate latency, memory, readiness, offline, and
lineage evidence on the required Apple M3/8 GB Mac. A green software test suite is necessary but
is not a substitute for a passing portfolio qualification artifact.

## Prerequisites

1. Use a clean checkout at the intended release revision.
2. Install the locked Python 3.11 environment and required Homebrew `libomp` runtime.
3. Complete explicit data/model downloads, then disconnect or otherwise avoid network reliance.
4. Build and freeze the compatible portfolio artifacts through Goldfish 013.
5. Keep the Mac on AC power, close nonessential applications, and allow memory pressure to settle.
6. Record honest background conditions in one short sentence.

Verify the repository before measuring:

```bash
uv sync --frozen --group dev
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

If `/usr/local/bin/git` is the obsolete 2.15 installation on this machine, put Apple Git first for
pre-commit:

```bash
env PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
  /opt/homebrew/bin/uv run pre-commit run --all-files
```

## Run

Start from a new terminal process; do not run training, evaluation, the API, or Streamlit in the
same process beforehand.

```bash
uv run market-rank qualification run \
  --bundle-id serving-bundle/<dataset-version>/portfolio/serving-bundle-v1/<config-sha256> \
  --background-conditions "AC power; all nonessential applications closed"
```

On success, record the printed `release-qualification/...` artifact ID. Verify its two payloads
and recursive lineage with the normal artifact loader before using it in Goldfish 016.

On failure, inspect the JSON named in the error under
`reports/generated/qualification/failed/`. Failed reports are diagnostics, not release evidence.

## Recovery Matrix

| Failure | Action |
|---|---|
| Bundle missing or invalid ID | Use the complete Goldfish 013 ID; never use `latest` or discovery. |
| Checksum/corruption failure | Quarantine the affected generated artifact and rebuild its owning stage from verified parents; never edit checksums. |
| Dense/ranker unavailable | Verify the pinned local MiniLM snapshot and exact bundle/model lineage; rerun only after clean offline load succeeds. |
| Network attempt | Identify the startup code path and remove the download/remote fallback. Do not weaken the guard. |
| Wrong chip/memory/architecture | Move the run to the defined Apple M3/8 GB reference host; do not publish substitute-host numbers as qualification. |
| Dirty revision | Commit intended code or reset unrelated local state safely, then rebuild config-hashed artifacts where required. |
| Battery power | Connect AC power and start a new run with corrected conditions. |
| Degraded component | Repair/rebuild the failed component and promote a compatible serving bundle. Degraded serving is supported operationally but cannot qualify a release. |
| RSS over 5,632 MiB | Confirm no other workflow shares the process; then optimize representations/load order without changing catalog identity. Record failed evidence. |
| Startup or latency over target | Repeat only after stabilizing background conditions. If persistent, profile the named stage and use a new reviewed config generation; never delete a slow result to claim passage. |
| Qualification payload corruption | Discard the corrupt generated qualification artifact and rerun from its verified serving dependency. |

## Reproduction and Reporting Rules

- Report the exact qualification artifact, serving bundle, config hash, clean Git revision, macOS,
  Mac model/chip/memory, power source, threads, package versions, and operator conditions.
- Use the persisted distributions; do not replace p95 with a faster single request.
- Do not compare scores across modes when the API marks them incomparable.
- Do not interpret list-composition diagnostics as relevance, fairness, or business metrics.
- Do not claim the fixed ESCI catalog represents live Amazon search.
- Any semantic config, target, artifact, or code change requires a new qualification lineage.

