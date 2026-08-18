"""
Significance testing + bucket breakdown for Leitner vs HLR (RecallIQ) comparison.

WHAT THIS DOES
1. Bootstrap CI on the MAE difference (Leitner vs HLR)   -> is the 32% gap real?
2. Bootstrap test on the AUC difference (Leitner vs HLR) -> is the AUC gap noise?
3. HLR's per-bucket MAE, using the same bucket definition as Leitner's

HOW TO USE
Run 03_baseline_leitner.py, then 04_model_hlr.py, then merge_predictions.py.
That produces results/predictions.csv with these columns:
    y_true_p_recall -> actual observed p_recall (continuous) for each test row
    y_true_binary   -> y_true_p_recall >= 0.5, as 0/1 (used for AUC)
    y_pred_leitner  -> Leitner's predicted p_recall for that row
    y_pred_hlr      -> HLR's predicted p_recall for that row
    bucket          -> the same consistency-bucket label used in Module 3/4
Then just run this script — no config editing needed if you used merge_predictions.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error

# ---- CONFIG ----
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_CSV = RESULTS_DIR / "predictions.csv"

COL_Y_TRUE_CONT = "y_true_p_recall"   # continuous target, used for MAE
COL_Y_TRUE_BIN = "y_true_binary"      # 0/1 target, used for AUC
COL_LEITNER = "y_pred_leitner"
COL_HLR = "y_pred_hlr"
COL_BUCKET = "bucket"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
# ----------------


def bootstrap_mae_diff(y_true, pred_a, pred_b, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Paired bootstrap on the MAE difference (pred_a - pred_b), e.g. Leitner - HLR.
    Positive values mean model A has higher error than model B.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    point_diff = mean_absolute_error(y_true, pred_a) - mean_absolute_error(y_true, pred_b)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        mae_a = mean_absolute_error(y_true[idx], pred_a[idx])
        mae_b = mean_absolute_error(y_true[idx], pred_b[idx])
        diffs[i] = mae_a - mae_b

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return point_diff, (lower, upper), diffs


def bootstrap_auc_diff(y_true_bin, pred_a, pred_b, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Paired bootstrap on the AUC difference (pred_a - pred_b).
    y_true_bin must be 0/1.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true_bin)
    y_true_bin = np.asarray(y_true_bin)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    point_diff = roc_auc_score(y_true_bin, pred_a) - roc_auc_score(y_true_bin, pred_b)

    diffs = []
    tries = 0
    while len(diffs) < n_boot and tries < n_boot * 3:
        tries += 1
        idx = rng.integers(0, n, n)
        yt = y_true_bin[idx]
        if yt.min() == yt.max():
            continue  # skip resamples with only one class present
        auc_a = roc_auc_score(yt, pred_a[idx])
        auc_b = roc_auc_score(yt, pred_b[idx])
        diffs.append(auc_a - auc_b)

    diffs = np.array(diffs)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return point_diff, (lower, upper), diffs


def bucket_mae(df, col_true, col_pred, col_bucket):
    """Per-bucket MAE, same grouping logic used for the Leitner breakdown."""
    out = (
        df.groupby(col_bucket)
        .apply(lambda g: mean_absolute_error(g[col_true], g[col_pred]))
        .rename("mae")
        .reset_index()
        .sort_values(col_bucket)
    )
    return out


def interpret_ci(lower, upper, label):
    if lower > 0 or upper < 0:
        print(f"  -> {label}: CI does NOT cross zero. Difference is statistically significant.")
    else:
        print(f"  -> {label}: CI crosses zero. Difference could be noise at 95% confidence.")


def main():
    df = pd.read_csv(INPUT_CSV)

    print("=" * 60)
    print("1. MAE DIFFERENCE: Leitner vs HLR (bootstrap, 95% CI)")
    print("=" * 60)
    diff, (lo, hi), _ = bootstrap_mae_diff(
        df[COL_Y_TRUE_CONT], df[COL_LEITNER], df[COL_HLR]
    )
    print(f"  Point estimate (Leitner MAE - HLR MAE): {diff:.4f}")
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")
    interpret_ci(lo, hi, "MAE difference")

    print()
    print("=" * 60)
    print("2. AUC DIFFERENCE: Leitner vs HLR (bootstrap, 95% CI)")
    print("=" * 60)
    diff, (lo, hi), _ = bootstrap_auc_diff(
        df[COL_Y_TRUE_BIN], df[COL_LEITNER], df[COL_HLR]
    )
    print(f"  Point estimate (Leitner AUC - HLR AUC): {diff:.4f}")
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")
    interpret_ci(lo, hi, "AUC difference")

    print()
    print("=" * 60)
    print("3. HLR PER-BUCKET MAE (compare directly to your Leitner table)")
    print("=" * 60)
    hlr_buckets = bucket_mae(df, COL_Y_TRUE_CONT, COL_HLR, COL_BUCKET)
    print(hlr_buckets.to_string(index=False))
    print()
    print("  Paste this next to your existing Leitner bucket table in the README.")


if __name__ == "__main__":
    main()