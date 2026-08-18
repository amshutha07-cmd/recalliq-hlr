"""
Module 2 — Feature Engineering
Duolingo Half-Life Regression project

Builds the engineered feature table on top of the `traces` DuckDB view
created in Module 1, and writes a user-stratified train/val/test split
to Parquet for Modules 3 (Leitner baseline) and 4 (HLR model).
"""

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# --- Reuse the same paths/config pattern as 01_data_loading.py ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DB_PATH = DATA_DIR / "duolingo.duckdb"

EPS = 1e-9  # avoids log2(0) when p_recall == 0


def section(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def connect_db() -> duckdb.DuckDBPyConnection:
    """Reopen the persistent DB from Module 1 — the `traces` view already exists there."""
    con = duckdb.connect(str(DB_PATH))
    print(f"[DuckDB] Connected — DB at {DB_PATH}")
    return con


def build_features(con: duckdb.DuckDBPyConnection) -> None:
    """
    Compute h_estimate (half-life proxy), log-transformed delta/h_estimate,
    and running accuracy per user-lexeme pair. Materializes a new DuckDB
    table `features` so downstream modules can query it directly.
    """
    section("Building engineered features")
    t0 = time.perf_counter()

    con.execute(f"""
        CREATE OR REPLACE TABLE features AS
        SELECT
            user_id,
            lexeme_string,
            timestamp,
            delta,
            delta / 86400.0                                        AS delta_days,
            p_recall,
            history_seen,
            history_correct,
            CASE WHEN history_seen > 0
                 THEN history_correct * 1.0 / history_seen
                 ELSE NULL
            END                                                     AS running_accuracy,
            -- half-life proxy: h = -delta / log2(p_recall_clamped)
            -- p_recall is clamped into (eps, 1-eps) so log2() stays negative:
            -- at p_recall = 1.0 exactly, log2(1 + eps) flips positive and blows up h_estimate.
            -delta / LOG2(LEAST(GREATEST(p_recall, {EPS}), 1 - {EPS}))          AS h_estimate,
            LOG2(delta + 1)                                                     AS log_delta,
            LOG2((-delta / LOG2(LEAST(GREATEST(p_recall, {EPS}), 1 - {EPS}))) + 1) AS log_h_estimate
        FROM traces
        WHERE delta > 0  -- drop first-review rows (delta = 0), no prior interval to learn from
    """)

    elapsed = time.perf_counter() - t0
    row_count = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    print(f"  Rows in features table (delta > 0 only): {row_count:,}")
    print(f"  Build time: {elapsed:.2f}s")


def split_train_val_test(con: duckdb.DuckDBPyConnection, seed: int = 42) -> dict:
    """
    Stratify the split by user_id (not by row) so a given learner's history
    doesn't leak across train/val/test. Uses a deterministic hash of user_id
    for reproducibility instead of a random sample each run.
    """
    section("Splitting train / val / test (80 / 10 / 10, by user)")

    con.execute(f"""
        CREATE OR REPLACE TABLE user_splits AS
        SELECT DISTINCT
            user_id,
            CASE
                WHEN (ABS(HASH(user_id)) % 100) < 80 THEN 'train'
                WHEN (ABS(HASH(user_id)) % 100) < 90 THEN 'val'
                ELSE 'test'
            END AS split
        FROM traces
    """)

    con.execute("""
        CREATE OR REPLACE TABLE features_split AS
        SELECT f.*, s.split
        FROM features f
        JOIN user_splits s USING (user_id)
    """)

    counts = con.execute("""
        SELECT split, COUNT(*) AS row_count, COUNT(DISTINCT user_id) AS user_count
        FROM features_split
        GROUP BY split
        ORDER BY split
    """).df()

    print(counts.to_string(index=False))
    return counts.set_index("split").to_dict(orient="index")


def save_outputs(con: duckdb.DuckDBPyConnection, split_counts: dict) -> None:
    """Write the split table to Parquet (one file per split) for Modules 3/4."""
    section("Saving Parquet outputs")

    for split_name in ("train", "val", "test"):
        out_path = DATA_DIR / f"features_{split_name}.parquet"
        con.execute(f"""
            COPY (SELECT * FROM features_split WHERE split = '{split_name}')
            TO '{out_path.as_posix()}' (FORMAT PARQUET)
        """)
        print(f"  Saved → {out_path}")

    # Correlation between h_estimate and p_recall, for resume_numbers.md
    corr = con.execute("""
        SELECT CORR(h_estimate, p_recall) AS corr_h_precall
        FROM features
    """).fetchone()[0]

    summary = {
        "split_counts": split_counts,
        "corr_h_estimate_p_recall": round(corr, 4) if corr is not None else None,
    }

    out_json = RESULTS_DIR / "module2_feature_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {out_json}")

    print()
    print("  [Numbers to log in resume_numbers.md]")
    for split_name, stats in split_counts.items():
        print(f"    {split_name}: {stats['row_count']:,} rows, {stats['user_count']:,} users")
    print(f"    corr(h_estimate, p_recall): {summary['corr_h_estimate_p_recall']}")


if __name__ == "__main__":
    con = connect_db()
    build_features(con)
    split_counts = split_train_val_test(con)
    save_outputs(con, split_counts)
    print("\n✓  Module 2 complete.")