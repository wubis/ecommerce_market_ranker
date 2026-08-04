# Goldfish 011 — Pointwise and LambdaMART Rankers

| Field | Value |
|---|---|
| Status | Implemented |
| Roadmap authority | `docs/goldfish/consolidated-roadmap.md`, Goldfish 011 |
| Parent design | `ELEPHANT.md`, Sections 17–20, 25–27, 30–32, 36–43, 48–50; D-006–D-008 |
| Milestone | M8 — Exact grouped populations and persisted supervised rankers |

## Objective

Consume one compatible Goldfish 010 `ltr_core_v1` artifact, construct an exact audited grouped
training population, and train two directly comparable CPU models: pointwise LightGBM and
LambdaMART. Both models must use identical rows, feature order, labels, and group arrays;
perform validation-only early stopping; persist complete lineage and training evidence; and
produce identical predictions/ranks after model-text reload under the Apple M3/8 GB RSS gate.

```bash
uv run market-rank ranking train
uv run market-rank ranking train --profile portfolio
```

The default profile is `development`. Training requires a compatible promoted Goldfish 010
feature artifact. It performs no download, feature generation, index/model rebuild, test-set
evaluation, tuning search, or champion promotion.

## Runtime Dependency

Goldfish 011 uses actual CPU LightGBM, not a substitute learner. `pyproject.toml` constrains
`lightgbm>=4.7,<5`; `uv.lock` currently resolves 4.7.0, which provides a Python 3.11/macOS ARM64
wheel. macOS requires OpenMP:

```bash
brew install libomp
uv sync --group dev --python 3.11
```

LightGBM is imported during explicit ranking workflows and model loading, never to download or
train at application startup.

## Exact Training Population

The source is Goldfish 010's labeled `closed_judged` matrix. Its fixed definition is
`esci_task1_us_catalog_eligible_closed_train_validation_v1`:

- US ESCI Task 1 small-version judgments only;
- fixed-catalog-eligible query-product pairs only;
- project `train` and `validation` rows only;
- project `test` is predicate-filtered before materialization and never used for fitting,
  early stopping, feature selection, or model comparison;
- exactly one row per query/product and no unjudged rows;
- exact ordered `ltr_core_v1` columns with finite, non-null values;
- label IDs `0/1/2/3` remain distinct from verified official gains `0/0.01/0.1/1`;
- normalized-query groups cannot cross train/validation;
- rows sort by query ID then product ID and group sizes are contiguous positive arrays.

Within each split, groups with fewer than two rows or fewer than two distinct labels are excluded
from model fitting/early-stopping inputs and recorded with the fixed reasons `too_few_rows` or
`single_label`. They remain in the immutable Goldfish 010 feature artifact for Goldfish 012's
defined evaluation. Rows are never sampled to satisfy configured limits: an over-limit exact
population blocks.

`training-population.json` records predicates, feature/categorical columns, official mapping,
eligible query IDs, exact group arrays, observed/eligible rows and groups, exclusion counts,
normalized-group overlap, and SHA-256 identities over row keys, float32 feature matrices, int32
labels, and group arrays. `population-audit.parquet` records every considered train/validation
query and its disposition.

## Shared Model Inputs

Both objectives receive:

- the same contiguous float32 matrix in registry order;
- the same int32 ESCI label IDs;
- the same train and validation group arrays;
- `brand_code` and `color_code` as the same registered categorical features;
- the same deterministic seed, 1.0 feature/bagging fractions, column-wise algorithm, and bounded
  thread count;
- the same official `label_gain` and validation NDCG cutoffs.

Raw product IDs remain metadata keys and are never model features. Prediction ties sort by
product ID, independently inside each query.

## Pointwise Contract

The pointwise baseline uses LightGBM `regression_l2` over official label IDs. It deliberately
does not consume group structure in its objective, but the validation dataset retains exact
groups so early stopping uses official-gain NDCG at the configured cutoffs. This isolates the
effect of supervised tree scoring without a ranking-aware training objective.

## LambdaMART Contract

The primary ranker uses LightGBM `lambdarank` with the identical rows/features/labels and exact
query group arrays. The ordered label gain is `[0.0, 0.01, 0.1, 1.0]`. It cannot recover products
missed by retrieval; later end-to-end evaluation must keep retrieval and ranking populations
separate.

## Validation-Only Early Stopping

Both models allow at most 300 boosting rounds by default and stop after 30 validation rounds
without improvement. Only validation NDCG@10/@20 selects the persisted best iteration. Training
NDCG is not an early-stopping input, and project test results are unavailable to this command.

The complete per-iteration validation history is persisted. The model summary records actual
executed iterations separately from the selected best iteration and validation NDCG at every
configured cutoff.

## Model Artifact

```text
ranking-models/<dataset-version>/<profile>/lightgbm-rankers-v1/<config-sha256>/
├── training-population.json
├── population-audit.parquet
├── pointwise-lightgbm.txt
├── lambdamart.txt
├── validation-history.parquet
├── feature-importance.parquet
├── explanation-contributions.parquet
├── reload-parity.parquet
├── ranking-models.json
├── manifest.json
└── _SUCCESS
```

The single exact parent is the profile-scoped ranking-feature artifact; its four parents are
recursively verified by the Goldfish 003 artifact protocol. The model manifest additionally
pins the feature artifact/manifest, registry/state hashes, ordered feature names/dtypes,
categorical fields, feature-set/population IDs, gain mapping, LightGBM version, objective and
canonical parameters, model checksums/bytes, training durations, validation results, resource
measurements, and `rrf-on-model-failure-v1` fallback contract.

LightGBM text is the canonical portable model format. Compatible artifact reuse occurs before
matrix reads or training.

## Explainability and Reload Parity

`feature-importance.parquet` stores split and gain importance for every registered feature and
both objectives. `explanation-contributions.parquet` stores a bounded validation sample of
LightGBM contribution values, including the bias term. They are associations and developer
diagnostics, not causal explanations.

After sequential training, both text models are cold-loaded together. A bounded set of complete
validation query groups is predicted before and after reload. `reload-parity.parquet` stores
scores, absolute deltas, and within-query ranks. Promotion requires maximum prediction delta at
most `1e-12` and exact rank parity. Loading also requires model feature names/order to match the
manifest exactly; artifact checksum corruption fails before LightGBM parsing.

## M3/8 GB Resource Contract

Only projected train/validation feature columns are materialized. Pointwise and LambdaMART train
sequentially with at most four configured threads, compact matrices, at most 63 histogram bins,
and no parallel hyperparameter trials. Peak process RSS is measured after matrix construction,
after sequential training, and after simultaneous cold reload of both models.

The maximum must remain at or below `runtime.rss_limit_mb` (5,632 MiB). Matrix overage blocks
staging; training/reload overage discards completed staged models and cannot publish `_SUCCESS`.
Goldfish 015 owns final reference-machine timing/RSS qualification.

## Configuration

```yaml
ranker_training:
  component_version: lightgbm-rankers-v1
  population_version: training-population-v1
  pointwise_objective: regression_l2
  lambdamart_objective: lambdarank
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  min_data_in_leaf: 20
  max_bin: 63
  lambda_l1: 0.0
  lambda_l2: 1.0
  max_boost_rounds: 300
  early_stopping_rounds: 30
  ndcg_eval_at: [10, 20]
  min_group_rows: 2
  min_distinct_labels: 2
  max_train_rows: 500000
  max_validation_rows: 200000
  reload_parity_rows: 128
  explanation_rows: 64
```

Configuration rejects invalid/unsorted cutoffs and early-stopping patience greater than or equal
to the maximum rounds. Profile row limits are hard failure gates, not sampling targets.

## Out of Scope

- hyperparameter search or selection among trials;
- project-test or frozen portfolio evaluation;
- pointwise-versus-LambdaMART quality claims or champion selection;
- closed-pool/end-to-end ablations, confidence intervals, slices, or failure analysis;
- serving-bundle promotion, API fallback execution, or Streamlit integration;
- optional neural or diversity reranking.

Goldfish 012 owns ranking evaluation, required ablations, validation-only champion selection, and
the promoted active-relevance contract.

## Acceptance Criteria

1. Training consumes one exact compatible Goldfish 010 artifact and recursively verified
   lineage.
2. Only catalog-eligible judged train/validation rows are materialized; project test is absent.
3. Keys, locale, population, feature set/order/dtypes, nulls, finiteness, labels, and official
   gains are strict.
4. Normalized-query groups never cross train/validation.
5. Groups are complete, sorted, contiguous, and sum exactly to matrix rows.
6. Too-small and single-label groups are excluded without row sampling and audited by query ID.
7. Pointwise and LambdaMART use identical matrix and group hashes.
8. Pointwise uses `regression_l2`; LambdaMART uses `lambdarank` with official label gains.
9. Both models use deterministic parameters, bounded threads, and validation-only NDCG early
   stopping.
10. Model text, objectives, canonical parameters, iterations, histories, lineage, and fallback
    compatibility are persisted.
11. Feature importance covers every feature/model; contribution samples are bounded.
12. Cold reload preserves ordered feature names, predictions within `1e-12`, and exact
    within-query ranks.
13. Corrupt model payloads and incompatible feature dimensions fail safely.
14. Compatible artifacts reuse before matrix reads or training.
15. Matrix/training/simultaneous-reload RSS passes before immutable promotion; failure rolls back.
16. Ordinary imports/tests perform no download, production training, or test evaluation.
17. Lock, Ruff, strict mypy, pytest, and pre-commit gates pass.

## Rollback

Remove the LightGBM dependency/lock entry, `ranker_training` configuration, ranking package,
ranking CLI, tests, this document, and local `ranking-models` artifacts. Goldfish 006–010 logic
remains intact, although configuration-hashed parent artifacts must be rebuilt after removing the
Goldfish 011 fields.
