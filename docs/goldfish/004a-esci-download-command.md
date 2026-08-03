# Goldfish 004A — Explicit ESCI Download and Validation Command

| Field | Value |
|---|---|
| Status | Implemented |
| Parent design | Goldfish 002–004; `ELEPHANT.md`, Sections 6, 8, 12, 30, 42, 44, 49–50 |
| Milestone | M1 — Ingestion/profiles |
| Scope size | One explicit, resumable raw acquisition command |

## Objective

Provide a usable command that explicitly downloads the three files described by the existing
pinned ESCI release manifest, verifies and atomically promotes each file, runs Goldfish 004
validation, and publishes or safely reuses the Goldfish 003 validation artifact.

```bash
market-rank data download-esci \
  --manifest configs/data/esci-release-7916cdf6ab75.json \
  --config configs/base.yaml
```

From the repository's locked environment, `uv run market-rank data download-esci` supplies
those defaults.

## Lifecycle States

1. **Pinned release metadata:** tracked JSON containing URLs, filenames, sizes, and hashes.
2. **Downloaded raw files:** immutable verified source bytes in configured `data/raw/esci/`.
3. **Validated raw dataset:** the complete Goldfish 004 `RawValidationReport`.
4. **Promoted validation artifact:** immutable manifest/report evidence under the Goldfish 003
   artifact root.

No stage implies the next. A downloaded file is not trusted until validation passes, and a
valid report is not durable lineage until publication or compatible artifact reuse succeeds.

## Network and Resource Contract

- Network access begins only inside `acquire_esci_files`, reached by this explicit command.
- Imports, validation-only calls, tests, setup, and future API startup never instantiate a
  connection.
- The standard-library transport uses distinct 15-second connect and 60-second read timeouts,
  at most five redirects, a descriptive user agent, and three attempts by default.
- HTTP 408, 425, 429, 500, 502, 503, and 504 plus connection/read failures are retryable.
  Other HTTP failures are permanent and are not retried.
- Downloads use 1 MiB chunks and never materialize a complete file in memory.
- Tests inject fake transport/response objects and do not access the internet.

## File Safety and Idempotency

Each missing file is written to a unique hidden
`.filename.partial-<random>.tmp` sibling. Size and SHA-256 are computed incrementally. Only
matching bytes are flushed and atomically renamed to the final pinned filename. Handled
failures remove their partial file; process termination may leave only the clearly marked
partial sibling.

Final files are never intentionally overwritten:

- an exact existing file is rehashed and reused without transport access;
- a mismatched existing file raises `ExistingRawFileMismatchError` and remains untouched;
- a downloaded mismatch raises `DownloadedFileMismatchError` and is discarded;
- there is no destructive `--force` option.

After every file verifies, the workflow records its actual completion time and invokes
`validate_raw_dataset`. Invalid validation raises `RawDataValidationError` with the full
report and never calls publication. A valid report is published with the resolved config hash
and detected or explicit code revision. Reruns verify the files again and reuse an existing
validation artifact only when its release and report semantics match.

## Public Interfaces

```text
HttpDownloadTransport.open(...)

acquire_esci_files(
    release,
    raw_root,
    transport=None,
    policy=None,
) -> AcquisitionResult

download_validate_esci(
    release,
    resolved_config,
    code_revision,
    ...,
) -> DownloadWorkflowResult

ensure_raw_validation_artifact(...) -> RawValidationPublication
```

`DownloadTransport`, validator, publisher, clock, sleeper, paths, and progress callback are
injectable. CLI parsing owns no download or validation logic.

## Terminal Output and Failures

Progress is bounded to filename/attempt, reused or downloaded status, verified bytes,
validation status, and final artifact ID. Domain failures produce a one-line error and exit
code 1 without a traceback or raw data. Invalid validation exceptions retain the full report
for programmatic callers.

## Files

| File | Change |
|---|---|
| `pyproject.toml` | Add the `market-rank` console-script entry point. |
| `src/market_rank/cli.py` | Parse commands, detect revision, render bounded progress/errors. |
| `src/market_rank/data/download.py` | Implement transport, retries, streaming, and workflow. |
| `src/market_rank/data/esci_raw.py` | Add compatible validation-artifact reuse. |
| `src/market_rank/data/__init__.py` | Export reusable acquisition interfaces. |
| `tests/unit/test_esci_download.py` | Exercise download, failure, reuse, orchestration, and CLI. |
| `README.md` | Document setup, states, recovery, offline behavior, and layout. |

No networking dependency is added. The standard library materially satisfies the required
timeouts, redirects, retry classification, streaming, and injection boundaries.

## Acceptance Criteria

1. The default and fully explicit CLI forms resolve the existing Goldfish 002/004 inputs.
2. All release-specific download facts come only from the strict pinned manifest.
3. All three files stream into same-directory partials and promote only after exact integrity.
4. Existing exact files and compatible validation artifacts are reused idempotently.
5. Existing mismatches remain untouched; handled failed downloads leave no partials.
6. Retryable failures are bounded and permanent HTTP failures are attempted once.
7. Validation starts only after all files verify; invalid reports cannot publish.
8. Imports and deterministic tests open no external network connection.
9. No normalization, US/Task-1 filtering, splitting, or Goldfish 005 behavior exists.
10. Lock check, Ruff, strict mypy, pytest, and pre-commit pass.

## Rollback

Remove the console entry point, CLI, downloader, 004A tests/documentation, and artifact-reuse
extension. Existing Goldfish 004 manual validation remains intact. Raw files and artifacts are
user-owned local state and are never deleted by rollback.
