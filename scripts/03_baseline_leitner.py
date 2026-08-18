"""
Module 3 — Leitner Baseline
Duolingo Half-Life Regression project

Builds a deterministic, zero-parameter baseline that buckets each review
by the learner's recent accuracy streak and predicts a fixed p_recall per
bucket. Evaluates MAE/RMSE against actual p_recall on val/test, so Module 4
(HLR model) has a real number to beat.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # add this import at the top of the file


def evaluate(df: pd.DataFrame, split_name: str) -> dict:
    err = df["p_recall_pred"] - df["p_recall"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())

    y_true = (df["p_recall"] >= 0.5).astype(int)
    if y_true.nunique() < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, df["p_recall_pred"])

    # --- ADD THIS BLOCK: save row-level Leitner predictions for the test split ---
    if split_name == "test":
        preds_df = pd.DataFrame({
            "y_true_binary": y_true,
            "y_true_p_recall": df["p_recall"],
            "y_pred_leitner": df["p_recall_pred"],
            "bucket": df["leitner_bucket"],
        })
        preds_path = RESULTS_DIR / "leitner_test_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"  Saved row-level Leitner test predictions → {preds_path}")
    # -----------------------------------------------------------------------

    by_bucket = (
        df.groupby("leitner_bucket")
        .apply(lambda g: pd.Series({
            "row_count": len(g),
            "mae": (g["p_recall_pred"] - g["p_recall"]).abs().mean(),
        }))
        .reset_index()
    )

    print(f"\n  [{split_name}] MAE = {mae:.4f}   RMSE = {rmse:.4f}   AUC = {auc:.4f}")
    print(by_bucket.to_string(index=False))

    return {
        "split": split_name,
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "row_count": int(len(df)),
        "by_bucket": by_bucket.assign(
            leitner_bucket=lambda d: d["leitner_bucket"].astype(int)
        ).to_dict(orient="records"),
    }
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Bucket boundaries on running_accuracy (history_correct / history_seen),
# mapped to a fixed predicted p_recall per bucket — the core Leitner idea:
# more consistent recent success -> higher predicted recall, longer interval.
BUCKET_EDGES = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]  # 5 buckets, last edge > 1 to catch 1.0
BUCKET_LABELS = [0, 1, 2, 3, 4]
BUCKET_PREDICTIONS = {
    0: 0.50,
    1: 0.65,
    2: 0.78,
    3: 0.88,
    4: 0.95,
}


def section(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"features_{name}.parquet"
    df = pd.read_parquet(path)
    print(f"  Loaded {name}: {len(df):,} rows from {path.name}")
    return df


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket each row by running_accuracy. Rows with no prior history
    (running_accuracy is NULL, i.e. history_seen == 0) fall back to
    bucket 0 (treated as an unknown/new item -> lowest confidence).
    """
    acc = df["running_accuracy"].fillna(0.0)
    df = df.copy()
    df["leitner_bucket"] = pd.cut(
        acc, bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=False
    ).astype(int)
    df["p_recall_pred"] = df["leitner_bucket"].map(BUCKET_PREDICTIONS)
    return df





def save_results(results: list[dict]) -> None:
    out_path = RESULTS_DIR / "leitner_baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    section("Loading feature splits")
    val_df = load_split("val")
    test_df = load_split("test")

    section("Assigning Leitner buckets")
    val_df = assign_buckets(val_df)
    test_df = assign_buckets(test_df)

    section("Evaluating baseline")
    results = [
        evaluate(val_df, "val"),
        evaluate(test_df, "test"),
    ]

    section("Saving results")
    save_results(results)

    print("\n  [Numbers to log in resume_numbers.md]")
    for r in results:
        print(f"    {r['split']}: MAE = {r['mae']}, RMSE = {r['rmse']} ({r['row_count']:,} rows)")

    print("\n✓  Module 3 complete.")