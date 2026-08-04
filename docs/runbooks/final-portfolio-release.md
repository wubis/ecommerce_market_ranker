# Final Portfolio Release Runbook

## 1. Freeze

Commit the intended code and configuration before exposing project test. Record the clean revision.
Do not change features, models, gains, cutoffs, retrieval depths, selection policy, or report targets
after this point without declaring a new evaluation generation.

## 2. Build the portfolio lineage

Run each stage in its own fresh process so macOS can reclaim memory:

```bash
uv run market-rank data build-esci-profiles
uv run market-rank data build-esci-foundation
uv run market-rank retrieval build-bm25
uv run market-rank retrieval build-dense
uv run market-rank retrieval evaluate-hybrid --profile portfolio
uv run market-rank features build-ranking --profile portfolio
uv run market-rank ranking train --profile portfolio
uv run market-rank ranking evaluate --profile portfolio
uv run market-rank serving promote --profile portfolio
```

Record the printed ranking-evaluation and serving-bundle IDs. Do not use `latest` aliases.

## 3. Qualify serving

Follow `docs/runbooks/local-release-qualification.md` on AC power with nonessential applications
closed. Record the passing release-qualification ID.

## 4. Capture demo evidence

Start the explicit API and Streamlit demo. Use the frozen example/query and capture three PNGs at
least 1000x600 pixels:

- `ranking-comparison.png`: mode summary and signed rank-change table;
- `product-provenance.png`: one result list with retrieval provenance and latency/lineage panel;
- `dataset-limitations.png`: the expanded always-available limitations content.

Place them in `reports/generated/screenshots/`. Do not crop away degraded/fallback banners,
limitations, or reproduction identifiers that materially affect interpretation. Never substitute
mock data or a different bundle.

## 5. Verify clean reproduction

Stop API/demo processes, return to the clean checkout, and run:

```bash
uv run market-rank portfolio verify-reproduction \
  --output reports/generated/clean-reproduction.json
```

If the worktree is dirty, resolve intended source changes through a reviewed commit. Generated
files belong only under ignored data/artifact/report roots.

## 6. Finalize

```bash
uv run market-rank portfolio finalize \
  --ranking-evaluation-id <exact-ranking-evaluation-id> \
  --serving-bundle-id <exact-serving-bundle-id> \
  --qualification-id <exact-release-qualification-id> \
  --reproduction-evidence reports/generated/clean-reproduction.json \
  --screenshots-dir reports/generated/screenshots
```

This is the first project-test scoring step. It must not be followed by tuning. A successful rerun
reuses the immutable release artifact.

## 7. Review before presenting

- Confirm `selection_split=validation` and `final_evaluation_split=test`.
- Confirm retrieval, closed-pool ranking, and end-to-end diagnostic tables are separate.
- Read every automatically disclosed negative/inconclusive finding.
- Verify screenshot hashes and exact lineage IDs.
- Verify the final report states dataset/business limitations.
- Use only measured values from the artifact in README, portfolio, or resume material.

## Recovery

| Failure | Recovery |
|---|---|
| Missing/corrupt parent | Rebuild the owning stage from verified parents; never edit checksums. |
| Ranking/serving mismatch | Promote a serving bundle from the exact ranking evaluation. |
| Qualification mismatch/failure | Re-run Goldfish 015 on the exact serving bundle after fixing the measured issue. |
| Screenshot invalid/missing | Recapture the named real demo view; do not create placeholder evidence. |
| Dirty reproduction | Commit intended code or remove generated files from tracked locations, then rerun gates. |
| Test finalization RSS failure | Preserve the failure, profile load order/representations, and create a reviewed new generation; do not tune quality from test. |
| Report artifact corruption | Remove only the corrupt generated release artifact and rebuild from its exact verified parents. |
| Unexpected weak/negative test result | Publish it honestly. Do not change the champion or hide the comparison. Future work requires a new named generation. |

