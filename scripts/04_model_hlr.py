"""
Module 4 — HLR Model (Half-Life Regression)
Duolingo Half-Life Regression project

Implements the core model from Settles & Meeder (2016): predicts a memory
half-life h_hat = 2^(w . x) from learner-history features, then converts
that to a recall probability via the exponential forgetting curve
p_hat = 2^(-delta / h_hat). Trained with full-batch vectorized gradient
descent on a combined loss (recall error + half-life error + L2 reg),
then evaluated against the Module 3 Leitner baseline on val/test —
including a per-Leitner-bucket MAE breakdown, so the two models are
compared at the same granularity.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Loss weighting and regularization (paper defaults: alpha=1, lambda small)
ALPHA = 1.0       # weight on half-life term vs recall term
LAMBDA_REG = 0.01 # L2 regularization on weights
LR = 0.001        # learning rate
N_ITERS = 200
H_MIN, H_MAX = 15.0 / (24 * 60), 274.0  # clamp half-life in days (paper's bounds, ~15min to 9mo)
EPS = 1e-9

# Same bucket definition as Module 3's Leitner baseline (03_baseline_leitner.py),
# duplicated here (rather than imported, since the filename starts with a digit)
# so HLR's error breakdown lines up against the exact same buckets.
BUCKET_EDGES = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]  # 5 buckets, last edge > 1 to catch 1.0
BUCKET_LABELS = [0, 1, 2, 3, 4]


def section(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"features_{name}.parquet"
    df = pd.read_parquet(path)
    print(f"  Loaded {name}: {len(df):,} rows from {path.name}")
    return df


def assign_leitner_bucket(df: pd.DataFrame) -> pd.Series:
    """
    Bucket each row by running_accuracy, identical logic to Module 3's
    assign_buckets — used here only to group HLR's errors, not to predict.
    """
    acc = df["running_accuracy"].fillna(0.0)
    return pd.cut(
        acc, bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=False
    ).astype(int)


def build_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Feature set follows the paper's approach: sqrt-transformed history counts,
    which compress the long tail while staying well-behaved near zero.
    Columns: [bias, sqrt(history_seen), sqrt(history_correct), sqrt(history_seen - history_correct)]
    """
    seen = df["history_seen"].fillna(0).clip(lower=0)
    correct = df["history_correct"].fillna(0).clip(lower=0)
    incorrect = (seen - correct).clip(lower=0)

    X = np.column_stack([
        np.ones(len(df)),
        np.sqrt(seen + 1),
        np.sqrt(correct + 1),
        np.sqrt(incorrect + 1),
    ])
    return X


def train_hlr(X: np.ndarray, delta: np.ndarray, p: np.ndarray, h_target: np.ndarray) -> np.ndarray:
    """
    Full-batch vectorized gradient descent on:
      L = (p_hat - p)^2 + ALPHA * (log2(h_hat) - log2(h_target))^2 + LAMBDA_REG * ||w||^2
    where h_hat = clip(2^(w.x), H_MIN, H_MAX) and p_hat = 2^(-delta / h_hat).
    """
    section("Training HLR model (gradient descent)")

    n, d = X.shape
    w = np.zeros(d)
    h_target_clipped = np.clip(h_target, H_MIN, H_MAX)
    log2_h_target = np.log2(h_target_clipped)

    for it in range(N_ITERS):
        raw_h = X @ w
        h_hat = np.clip(np.power(2.0, raw_h), H_MIN, H_MAX)
        p_hat = np.power(2.0, -delta / h_hat)

        # dL/dw via chain rule (see Settles & Meeder 2016, Sec. 3)
        err_p = p_hat - p
        err_h = np.log2(h_hat) - log2_h_target

        # d(p_hat)/d(raw_h) = p_hat * ln(2) * delta / h_hat   (only where h_hat not clipped)
        not_clipped = (raw_h > np.log2(H_MIN)) & (raw_h < np.log2(H_MAX))
        dp_draw = np.where(not_clipped, p_hat * np.log(2) * delta / h_hat, 0.0)

        grad_p_term = 2 * err_p * dp_draw
        grad_h_term = 2 * ALPHA * err_h * np.where(not_clipped, 1.0, 0.0)

        grad_raw = grad_p_term + grad_h_term
        grad_w = (X.T @ grad_raw) / n + 2 * LAMBDA_REG * w

        w -= LR * grad_w

        if it % 20 == 0 or it == N_ITERS - 1:
            loss = np.mean(err_p ** 2) + ALPHA * np.mean(err_h ** 2) + LAMBDA_REG * np.sum(w ** 2)
            mae = np.mean(np.abs(err_p))
            print(f"  iter {it:>3}  loss={loss:.5f}  train_MAE={mae:.5f}")

    return w


def evaluate(w: np.ndarray, df: pd.DataFrame, split_name: str) -> dict:
    X = build_design_matrix(df)
    delta = df["delta_days"].to_numpy()
    p = df["p_recall"].to_numpy()

    raw_h = X @ w
    h_hat = np.clip(np.power(2.0, raw_h), H_MIN, H_MAX)
    p_hat = np.power(2.0, -delta / h_hat)

    err = p_hat - p
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    # AUC: same convention as the Leitner baseline — binary label from
    # p_recall >= 0.5, ranking score is the model's predicted p_hat.
    y_true = (p >= 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, p_hat)

    # Per-Leitner-bucket MAE breakdown, mirroring Module 3's by_bucket table,
    # so HLR's errors can be compared segment-by-segment against Leitner's.
    bucket_df = pd.DataFrame({
        "leitner_bucket": assign_leitner_bucket(df),
        "abs_err": np.abs(err),
    })
    by_bucket = (
        bucket_df.groupby("leitner_bucket")
        .agg(row_count=("abs_err", "size"), mae=("abs_err", "mean"))
        .reset_index()
    )
    by_bucket["leitner_bucket"] = by_bucket["leitner_bucket"].astype(int)
    by_bucket["mae"] = by_bucket["mae"].round(6)

    print(f"\n  [{split_name}] HLR  MAE = {mae:.4f}   RMSE = {rmse:.4f}   AUC = {auc:.4f}")
    print(by_bucket.to_string(index=False))

    return {
        "split": split_name,
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "row_count": int(len(df)),
        "by_bucket": by_bucket.to_dict(orient="records"),
    }


def compare_to_leitner(hlr_results: list[dict]) -> None:
    baseline_path = RESULTS_DIR / "leitner_baseline_metrics.json"
    if not baseline_path.exists():
        print("  [!] leitner_baseline_metrics.json not found — run Module 3 first for a comparison.")
        return

    with open(baseline_path) as f:
        leitner_results = {r["split"]: r for r in json.load(f)}

    section("HLR vs Leitner baseline")
    for r in hlr_results:
        split = r["split"]
        if split in leitner_results:
            base = leitner_results[split]
            base_mae = base["mae"]
            delta_mae = base_mae - r["mae"]
            pct_improvement = (delta_mae / base_mae) * 100
            print(f"  {split}: Leitner MAE={base_mae}  HLR MAE={r['mae']}  "
                  f"Δ={delta_mae:.4f} ({pct_improvement:+.1f}%)")

            # Per-bucket comparison: is HLR's edge over Leitner biggest where
            # Leitner is weakest (bucket 0) or more evenly spread?
            leitner_by_bucket = {b["leitner_bucket"]: b["mae"] for b in base.get("by_bucket", [])}
            hlr_by_bucket = {b["leitner_bucket"]: b["mae"] for b in r["by_bucket"]}
            print(f"  {split} — MAE by bucket (Leitner vs HLR vs improvement):")
            for bucket in sorted(set(leitner_by_bucket) | set(hlr_by_bucket)):
                lm = leitner_by_bucket.get(bucket)
                hm = hlr_by_bucket.get(bucket)
                if lm is None or hm is None:
                    continue
                bucket_delta = lm - hm
                bucket_pct = (bucket_delta / lm) * 100 if lm else float("nan")
                print(f"    bucket {bucket}: Leitner={lm:.4f}  HLR={hm:.4f}  "
                      f"Δ={bucket_delta:.4f} ({bucket_pct:+.1f}%)")


def save_results(w: np.ndarray, hlr_results: list[dict], train_seconds: float, train_rows: int) -> None:
    out = {
        "weights": w.tolist(),
        "feature_names": ["bias", "sqrt_history_seen", "sqrt_history_correct", "sqrt_history_incorrect"],
        "hyperparams": {"alpha": ALPHA, "lambda_reg": LAMBDA_REG, "lr": LR, "n_iters": N_ITERS},
        "training": {"train_seconds": round(train_seconds, 2), "train_rows": train_rows},
        "results": hlr_results,
    }
    out_path = RESULTS_DIR / "hlr_model_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    section("Loading feature splits")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train = build_design_matrix(train_df)
    delta_train = train_df["delta_days"].to_numpy()
    p_train = train_df["p_recall"].to_numpy()
    h_target_train = train_df["h_estimate"].to_numpy()

    train_start = time.perf_counter()
    w = train_hlr(X_train, delta_train, p_train, h_target_train)
    train_seconds = time.perf_counter() - train_start
    print(f"\n  Training time: {train_seconds:.2f}s for {len(train_df):,} rows, {N_ITERS} iters")

    section("Evaluating on val / test")
    hlr_results = [
        evaluate(w, val_df, "val"),
        evaluate(w, test_df, "test"),
    ]

    compare_to_leitner(hlr_results)
    save_results(w, hlr_results, train_seconds, len(train_df))

    print("\n  [Numbers to log in resume_numbers.md]")
    print(f"    training: {train_seconds:.2f}s on {len(train_df):,} rows, {N_ITERS} iters, "
          f"{X_train.shape[1]} features")
    for r in hlr_results:
        print(f"    {r['split']}: HLR MAE = {r['mae']}, RMSE = {r['rmse']} ({r['row_count']:,} rows)")

    print("\n✓  Module 4 complete.")