# Goldfish 015 — Hardening and M3/8 GB Qualification

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 015 |
| Parent design | `ELEPHANT.md`, Sections 26, 31–32, 37, 41–44, 47–48, D-012–D-013 |
| Milestone | M14 — Fail-closed local release qualification |

## Objective

Turn the complete local system into a release-candidate boundary with repeatable evidence. One
fresh CLI process must load one explicit Goldfish 013 serving bundle without network access,
exercise the API and orchestrator under a frozen workload, record safe host/dependency/lineage
facts, enforce resource and latency targets, and promote an immutable qualification artifact only
when every gate passes.

```bash
uv run market-rank qualification run \
  --bundle-id serving-bundle/<dataset-version>/portfolio/serving-bundle-v1/<config-sha256> \
  --background-conditions "AC power; all nonessential applications closed"
```

The command requires an exact bundle ID and explicit operator conditions. A clean Git revision is
required by default. There is no `latest`, partial pass, override-to-green, or remote benchmark
path.

## Qualification Workload

The checked-in `rc1` workload uses five fixed ESCI-compatible queries, top-K 10, two warmup rounds,
five measured rounds, and all six relevance modes: active, BM25, dense, hybrid RRF, pointwise, and
LambdaMART. It then issues three rounds at concurrency two, matching the serving semaphore bound.
Queries are stored only as SHA-256 digests in the report; the resolved configuration carries the
reproducible workload definition.

Sequential measurements use FastAPI's validated request/response boundary through its in-process
test transport. Concurrency measurements call the same loaded orchestrator directly so worker
timing is not confused with client-thread scheduling. Every response is revalidated with the
shared strict contract. Active requests must resolve to the promoted stage without degradation,
and every serving component must be ready.

## Reference-Machine and Offline Gates

The release host must report:

- macOS (`Darwin`) on `arm64`;
- exactly `Apple M3` and 8 GiB physical memory;
- AC power;
- Python 3.11 under the locked environment;
- a clean 40-character lowercase Git revision;
- `runtime.offline: true`.

Hardware capture retains only system, architecture, chip, model class, memory, CPU count, OS,
Python, power source, and available-memory facts. Serial number, platform UUID, UDID, and other
device identifiers returned by macOS tools are never modeled or persisted.

During bundle load, warmup, sequential measurement, and concurrency measurement, Hugging Face and
Transformers offline flags are forced and Python socket connection attempts raise immediately.
The guard is restored afterward. Startup remains a load-only path: no model acquisition,
embedding, index construction, training, artifact discovery, or repair is permitted.

## Targets and Evidence

| Gate | Target |
|---|---:|
| Cold bundle startup | ≤30,000 ms |
| Per-mode request p95 | ≤1,000 ms |
| Two-request concurrency p95 | ≤1,500 ms |
| Query parse p95 | ≤50 ms |
| BM25 p95 | ≤200 ms |
| Dense encode/search p95 | ≤200 ms |
| Fusion p95 | ≤50 ms |
| Pair features p95 | ≤250 ms |
| Ranker p95 | ≤50 ms |
| Process peak RSS | ≤5,632 MiB |

The report includes p50/p95/p99/maximum and sample count per mode, concurrent latency, stage
latency distributions, cold startup, process peak RSS, component state, safe hardware facts,
locked package versions, config/bundle/code lineage, workload bounds, operator conditions, and a
canonical sorted check list.

## Promotion and Failure Semantics

A passing run publishes:

```text
release-qualification/<dataset-version>/<profile>/release-qualification-v1/<config-sha256>/
├── release-qualification.json
├── release-qualification.md
├── manifest.json
└── _SUCCESS
```

The serving bundle is the exact artifact dependency and is recursively checksum-verified before
measurement. Compatible passing evidence is immutable and reusable. Payload corruption, bundle
lineage drift, report-schema drift, or dependency mismatch fails through the artifact protocol.

A failed measurement never creates `_SUCCESS`. Its strict JSON evidence is retained under
`reports/generated/qualification/failed/` so resource or host failures remain diagnosable without
being mistaken for a release candidate.

## Hardening Coverage

- API Host headers are allowlisted to loopback names in addition to loopback-only binding.
- Request body, query, top-K, deadline, concurrency, debug, and response bounds remain enforced.
- Error responses and hardware reports are tested for local-path and device-identifier redaction.
- Bundle and qualification payload corruption fail checksum verification before loading.
- A network-attempt fixture proves the offline socket guard fails immediately and restores state.
- The persisted tiny-bundle integration test covers load → readiness → six-mode API workload →
  concurrency → report → failed evidence → passing promotion → reuse → corruption rejection.
- The full existing regression suite continues to cover deterministic data, retrieval, features,
  ranking, evaluation, serving degradation, and the Streamlit unavailable state.

## Qualification Status

Goldfish 015 implements and fixture-validates the complete qualification boundary. No production
ESCI serving bundle is present in this workspace, so this Goldfish does **not** claim final
portfolio latency, RSS, or release-candidate passage. Goldfish 016 must build/freeze the portfolio
lineage and run this command from a clean checkout on the reference machine before publishing any
numbers or screenshots.

## Out of Scope

- creating or tuning the final portfolio artifacts;
- changing retrieval K, model parameters, catalog membership, or latency targets based on test;
- public deployment, auth/TLS, container qualification, Colab qualification, or production load;
- neural/diversity implementation or their optional latency targets;
- final scientific tables, screenshots, or portfolio narrative.

## Acceptance Criteria

1. Qualification accepts only one explicit recursively verified serving bundle.
2. The exact M3/8 GB, AC-power, clean-revision, offline, readiness, and non-degradation gates apply.
3. Safe hardware capture excludes unique device identifiers.
4. The frozen bounded workload covers all core modes and modest concurrency.
5. Cold startup, mode/stage/concurrent latency, and process RSS have checked-in targets.
6. Socket attempts fail immediately during the complete measured lifecycle.
7. Only an all-green report is promoted as an immutable artifact.
8. Failed evidence is retained separately and cannot appear as `_SUCCESS`.
9. Corrupt or incompatible qualification evidence fails closed.
10. Runbook, recovery, limitations, and reproduction commands are documented.
11. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove the qualification configuration/module/CLI/tests/docs and loopback Host middleware. Remove
only generated `release-qualification` artifacts and `reports/generated/qualification` evidence;
Goldfishes 001–014 remain operable. Configuration-hashed upstream artifacts must be rebuilt after
removing the Goldfish 015 configuration fields.

