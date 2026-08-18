"""
Merge Leitner + HLR row-level test predictions into one file
for significance_analysis.py.

Run this AFTER both:
    python 03_baseline_leitner.py
    python 04_model_hlr.py
have produced:
    results/leitner_test_predictions.csv
    results/hlr_test_predictions.csv
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

leitner = pd.read_csv(RESULTS_DIR / "leitner_test_predictions.csv")
hlr = pd.read_csv(RESULTS_DIR / "hlr_test_predictions.csv")

assert len(leitner) == len(hlr), (
    f"Row count mismatch: leitner={len(leitner)} hlr={len(hlr)}. "
    "Both scripts must load features_test.parquet without shuffling."
)

merged = pd.DataFrame({
    "y_true_p_recall": leitner["y_true_p_recall"],
    "y_true_binary": leitner["y_true_binary"],
    "y_pred_leitner": leitner["y_pred_leitner"],
    "y_pred_hlr": hlr["y_pred_hlr"],
    "bucket": leitner["bucket"],
})

# sanity check: buckets should match row-for-row since both scripts use
# the same bucket definition on the same test set
mismatches = (merged["bucket"] != hlr["bucket"]).sum()
if mismatches > 0:
    print(f"  [!] WARNING: {mismatches} rows have mismatched buckets between "
          "the two files — check row order / shuffling in 03 and 04.")

out_path = RESULTS_DIR / "predictions.csv"
merged.to_csv(out_path, index=False)
print(f"Saved merged predictions → {out_path} ({len(merged):,} rows)")