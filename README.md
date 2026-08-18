# Retention / Forgetting-Curve Regression (Duolingo Half-Life Regression)

A model of how language learners forget what they've learned over time, built on
Duolingo's public Half-Life Regression dataset (~12.85M learning-trace rows across
train/val/test). Replicates the Leitner-style baseline from Duolingo's own published
research (Settles & Meeder, 2016, ACL) and benchmarks a from-scratch half-life
regression model directly against it — not just reporting accuracy in isolation.

## Why this project

Most portfolio ML projects report a single accuracy number with nothing to compare
it to. This one instead: handles a dataset too large for naive pandas, engineers
features from raw event logs, splits data by user (not by row) to avoid leakage,
reimplements a real published baseline methodology, and benchmarks the final model
against both that baseline and the original paper's own reported numbers — including
an honest look at *where* the improvement comes from and where it doesn't.

## Tech stack

DuckDB · Python / pandas · NumPy · scikit-learn (`roc_auc_score`) · SQL (via DuckDB)

## Pipeline

| Module | What it does | Status |
|--------|---------------|--------|
| 1 — Data Loading & Exploration | Load 12.85M-row dataset via DuckDB, inspect schema, visualize the forgetting curve | ✅ Done |
| 2 — Feature Engineering | Engineer features, split by user to prevent leakage | ✅ Done |
| 3 — Baseline Replication | Leitner-style bucketed baseline, evaluated on val/test | ✅ Done |
| 4 — Model Building | Half-life regression (gradient descent) on `sqrt(history_seen/correct/incorrect)` features | ✅ Done |
| 5 — Evaluation & Benchmarking | AUC + MAE vs. baseline vs. published paper numbers, per-segment error analysis | ✅ Done |
| 6 — Write-up | This README + resume numbers | ✅ Done |

## Dataset

- **1,309 MB** raw CSV, **12,854,226** rows, no nulls in `p_recall`, no rows dropped
  during feature building (all had `delta > 0`) — 0% data loss during cleaning.
- **115,222** unique learners, **18,781** unique lexemes (word × language pairs).
- Date range: 2013-02-28 → 2013-03-12. Mean `p_recall` 0.896, mean lag between
  reviews (`delta`) 8.44 days.
- DuckDB full-table scan/profile: 2.18s. Feature-table build: 6.58s.
- `corr(h_estimate, p_recall)` = 0.1123 — a useful sanity check that the target
  variable and the half-life estimate used to fit it aren't trivially the same signal.
- **5 engineered features** built in Module 2: `delta_days`, `running_accuracy`,
  `h_estimate`, `log_delta`, `log_h_estimate`. Of these, `running_accuracy` (Leitner
  bucket assignment) and `h_estimate` (HLR training target) feed the downstream
  models directly; `log_delta`/`log_h_estimate` were built but not used by the final
  baseline or model scripts.

**Split (80/10/10, by user — not by row, to prevent leakage):**

| Split | Rows | Users |
|-------|------|-------|
| Train | 10,272,997 | 91,893 |
| Val   | 1,294,912  | 11,775 |
| Test  | 1,286,317  | 11,554 |

### Reproducing the data locally

Raw and processed data files (`data/*.csv`, `data/*.parquet`, `data/*.duckdb`) are
**not** committed to this repo — they're large (the raw CSV alone is 1.3GB) and
regenerable, so keeping them out of git is standard practice. To reproduce:

1. Download `settles.acl16.learning_traces.13m.csv` from the [official Duolingo
   half-life regression dataset release](https://github.com/duolingo/halflife-regression)
   and place it in `data/`.
2. Run `python scripts/01_data_loading.py` — builds `data/duolingo.duckdb` and the
   EDA outputs in `results/`.
3. Run `python scripts/02_feature_engineering.py` — builds
   `data/features_{train,val,test}.parquet`.
4. Run `scripts/03_baseline_leitner.py` and `scripts/04_model_hlr.py` in order.

## Results

| Split | Model    | MAE    | RMSE   | AUC    | Rows       |
|-------|----------|--------|--------|--------|------------|
| Val   | Leitner  | 0.1795 | 0.2782 | 0.5639 | 1,294,912  |
| Val   | HLR      | 0.1219 | 0.2878 | 0.5481 | 1,294,912  |
| Test  | Leitner  | 0.1811 | 0.2804 | 0.5643 | 1,286,317  |
| Test  | HLR      | 0.1232 | 0.2905 | 0.5427 | 1,286,317  |
| Paper (Settles & Meeder 2016) | Leitner | 0.235 | — | higher than HLR | — |
| Paper (Settles & Meeder 2016) | HLR (best variant, w/ lexeme tags) | 0.128 | — | lower than Leitner | — |

**HLR cuts test MAE by 32.0% vs. our Leitner baseline** (0.1811 → 0.1232), training
in **195.58s** on 10,272,997 rows over 200 gradient-descent iterations with 4
features (bias + 3 sqrt-transformed history features). The paper's own best variant
(with lexeme-tag features we didn't implement) reports a larger ~45% reduction
(0.235 → 0.128) — an honest gap explained by feature scope, not a methodology error.

### The AUC/MAE split (and why it's not a red flag)

Leitner scores a *higher* AUC than HLR on both splits, despite losing decisively on
MAE — the same pattern the original paper reports. AUC measures ranking quality;
MAE measures calibration. With most items already easy to recall (mean p_recall
0.896), there's little room for ranking discrimination, so AUC clusters near 0.5–0.6
for every method regardless of calibration quality. HLR's real advantage is in *how
close* its probability estimate is, not whether it ranks recall likelihood correctly.

### Where HLR actually wins (error analysis)

| Leitner bucket | Test rows | Leitner MAE | HLR MAE | Improvement |
|-----------------|-----------|-------------|---------|-------------|
| 0 (new/no history) | 10,246  | 0.437 | 0.239 | +45.4% |
| 1 | 115,297 | 0.348 | 0.175 | +49.6% |
| 2 | 214,996 | 0.263 | 0.143 | +45.8% |
| 3 | 273,575 | 0.181 | 0.107 | +40.8% |
| 4 (well-learned) | 672,203 | 0.122 | 0.113 | +7.9%  |

HLR's edge is concentrated in buckets 1–3 — learners with a partial, inconsistent
track record, where Leitner's flat per-bucket guess is weakest. For bucket 4
(long, mostly-correct history), Leitner's fixed 0.95 prediction is already close to
true recall, so HLR's improvement shrinks to single digits. That's also why blended
AUC understates HLR: bucket 4 holds over half the test rows and is where the two
models agree most.

## What this would inform in a real product

A calibrated per-item recall probability like this is the core input to a
spaced-repetition scheduler: instead of a fixed review interval per Leitner box,
review timing could be set directly from the predicted half-life, tightening
schedules for items in the "inconsistent history" segment where this model shows
its biggest edge — exactly where a real scheduler has the most to gain from better
calibration.

## Resume bullet

> Built and benchmarked a half-life regression model for spaced-repetition recall
> prediction against a reimplemented Leitner baseline and the original published
> methodology, on 12.85M Duolingo learning-trace rows (115K learners, split by
> user to prevent leakage); reduced test MAE by 32.0% vs. baseline (0.181 → 0.123)
> and identified via per-segment error analysis that the model's advantage
> concentrates in learners with inconsistent review history (+40–50% MAE
> reduction) versus well-learned items (+8%), explaining a counterintuitive
> AUC/MAE divergence consistent with the original research.

## Repo structure

```
project2_duolingo/
├── data/               # features_{train,val,test}.parquet (gitignored — large files)
├── scripts/
│   ├── 01_load_explore.py
│   ├── 02_feature_engineering.py
│   ├── 03_baseline_leitner.py
│   └── 04_model_hlr.py
├── results/
│   ├── leitner_baseline_metrics.json
│   └── hlr_model_results.json
├── module_notes/        # per-module notes (tech, concepts, bugs, time taken)
├── module5_notes.md
└── README.md
```