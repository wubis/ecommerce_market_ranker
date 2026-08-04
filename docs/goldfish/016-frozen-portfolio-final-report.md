# Goldfish 016 — Frozen Portfolio Experiments and Final Report

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 016 |
| Parent design | `ELEPHANT.md`, Sections 25–27, 31, 37, 41–44, 47–48, 51–52 |
| Milestone | M15 — Frozen project-test evaluation and core portfolio release package |

## Objective

Close the core project without leaking test evidence into selection. Goldfish 016 consumes one
portfolio validation evaluation, its exact serving bundle, a passing M3/8 GB qualification, clean
reproduction evidence, and three demo screenshots. It evaluates the untouched project-test split
using the already-selected relevance stage, produces protocol-separated final evidence and an
honest narrative, and publishes one immutable portfolio-release artifact.

No finalizer argument can change a model, feature, gain, cutoff, candidate population, retrieval
depth, champion, or metric. Any semantic change alters the resolved configuration hash and requires
a new complete lineage.

## Frozen Test Boundary

Goldfish 012 remains the only champion-selection stage. Its validation-selected active contract is
loaded unchanged. Goldfish 016 then reads the existing portfolio feature matrices and fixed hybrid
candidate population using `project_split == "test"`, scores RRF, pointwise, and LambdaMART, marks
the frozen champion for presentation, and performs no reselection.

Two populations remain separate:

- `closed_pool_task1_v1` evaluates ranking over each complete supplied judged pool and permits
  official-gain NDCG, thresholded Precision/MAP/MRR, and Exact Hit;
- `end_to_end_diagnostic_v1` reorders the fixed retrieved union and permits only retrieval-aware
  judged diagnostics, never naive Precision, MAP, or NDCG.

The inherited catalog-retrieval table is filtered to the project-test slice under
`retrieval_catalog_task1_us_v1`. The finalizer revalidates prohibited-metric rules before writing
any report. When Goldfish 016A compact mode is active, the generated report and limitations name
`esci_task1_us_compact_catalog_v1` and prohibit full-catalog interpretation.

## Required Evidence

The initial finalization requires:

1. one explicit `ranking-evaluation/.../portfolio/...` artifact with `test_evaluated=false`;
2. the exact serving bundle that embeds that evaluation manifest hash;
3. a passing `release-qualification/...` artifact depending on that serving bundle;
4. clean reproduction evidence for the same config and clean Git revision;
5. three structurally valid, checksum-verified PNG screenshots at the configured dimensions:
   `ranking-comparison.png`, `product-provenance.png`, and `dataset-limitations.png`.

Screenshot files are checked for regular-file status, bounded bytes, PNG signature, canonical IHDR,
chunk boundaries, CRCs, IDAT, IEND, dimensions, and trailing content. The artifact records their
hashes and copies them under `screenshots/`. The named views deliberately include limitations as
evidence, not decorative marketing imagery.

## Clean Reproduction

After committing the intended release code, run:

```bash
uv run market-rank portfolio verify-reproduction \
  --output reports/generated/clean-reproduction.json
```

The command requires an empty worktree and clean 40-character revision, then runs the locked
environment check, Ruff format/lint, strict mypy, and the complete pytest suite. It records command
identities, durations, passing exit status, config/revision, offline state, and observed test count;
it does not retain potentially sensitive command output.

## Finalization

```bash
uv run market-rank portfolio finalize \
  --ranking-evaluation-id ranking-evaluation/<dataset>/portfolio/ranking-eval-v1/<config> \
  --serving-bundle-id serving-bundle/<dataset>/portfolio/serving-bundle-v1/<config> \
  --qualification-id release-qualification/<dataset>/portfolio/release-qualification-v1/<config> \
  --reproduction-evidence reports/generated/clean-reproduction.json \
  --screenshots-dir reports/generated/screenshots
```

The command recursively verifies every artifact and exact direct relationship before test scoring.
Missing screenshots, reproduction drift, qualification failure, wrong profile/config, dirty code,
or serving lineage mismatch fails before test access.

## Portfolio Release Artifact

```text
portfolio-release/<dataset-version>/portfolio/portfolio-release-v1/<config-sha256>/
├── portfolio-release.json
├── FINAL_REPORT.md
├── LIMITATIONS.md
├── lineage.json
├── reproduction.json
├── retrieval-test.csv
├── ranking-test.csv
├── ranking-test-slices.csv
├── ablations-test.csv
├── resources.csv
├── retrieval-recall.svg
├── ranking-ndcg.svg
├── serving-latency.svg
├── predictions.parquet
├── query-metrics.parquet
├── metrics.parquet
├── comparisons.parquet
├── failure-analysis.parquet
├── screenshots/
│   ├── ranking-comparison.png
│   ├── product-provenance.png
│   └── dataset-limitations.png
├── manifest.json
└── _SUCCESS
```

The three direct dependencies are the validation ranking evaluation, serving bundle, and passing
qualification. Recursive verification supplies the complete data/retrieval/feature/model DAG.
Compatible completion is immutable and reused rather than re-reading test.

## Final Narrative Contract

`FINAL_REPORT.md` fills measured project-test closed-pool NDCG and grouped confidence intervals for
the frozen active stage. It explains the system, distinguishes catalog retrieval from judged-pool
ranking, links all tables/plots/screenshots, reports qualification startup and RSS, discloses every
configured slice, and names every non-positive or confidence-interval-inconclusive comparison.

The report never claims live Amazon performance, treats unknown products as irrelevant, presents
end-to-end diagnostics as official NDCG, hides a negative ablation, or interprets brand/list
composition as fairness or business performance. Optional neural/diversity stages remain future
work rather than being implied in the core result.

## M3/8 GB Contract

The frozen test scorer loads persisted compact matrices and models sequentially and must remain
under the same 5,632 MiB process-RSS ceiling. Final resource CSV rows include retrieval evaluation,
ranking evaluation, and serving qualification artifact/RSS evidence. The exact serving runtime was
already measured independently by Goldfish 015 so offline evaluation frames and live indexes are
never intentionally retained together in the qualified service process.

## Current Publication Status

The complete path is fixture-validated, including a real persisted tiny artifact DAG and untouched
fixture test split. The workspace contains no production ESCI portfolio artifacts, passing
production qualification, or captured production demo screenshots. Therefore no final numeric
portfolio report is published or claimed in Git by this implementation turn. Running the documented
workflow on those real inputs is an explicit operational step, not permission to manufacture data.

## Out of Scope

- tuning or champion changes after project-test access;
- optional neural reranking or diversity algorithms;
- notebook-only calculations or hand-edited headline values;
- public hosting, production load/SLO claims, or online experimentation;
- business, causal, personalization, fairness, or live-marketplace claims.

## Acceptance Criteria

1. Validation remains the sole selection split and project test performs no reselection.
2. Frozen test predictions and metrics cover identical stage cohorts with complete rank lineage.
3. Closed-pool, end-to-end, and catalog-retrieval protocols remain visibly separated.
4. Final tables include baselines, ABL-01–05 evidence, all configured slices, and failure examples.
5. Non-positive/inconclusive comparisons are derived and disclosed automatically.
6. Passing M3/8 GB qualification and clean reproduction evidence match config/code lineage.
7. Three validated screenshots cover comparison, provenance, and limitations.
8. Report, tables, plots, raw metric Parquet, lineage, limitations, and reproduction are immutable.
9. Corruption, missing evidence, wrong profile, dirty revision, or lineage drift fails closed.
10. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove the `portfolio_report` configuration, portfolio module/CLI/tests/docs, and frozen-test helper.
Remove generated `portfolio-release` artifacts and reports only. Goldfishes 001–015 remain intact,
although configuration-hashed upstream artifacts must be rebuilt after removing Goldfish 016
fields.
