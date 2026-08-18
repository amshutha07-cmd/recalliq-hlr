"""
Module 1 — Data Loading & Exploratory Data Analysis
=====================================================
Uses DuckDB to stream the 13 M-row CSV without loading it entirely into RAM,
then exports a tidy summary DataFrame for downstream modules.

Run:
    python scripts/01_data_loading.py

Outputs:
    results/eda_summary.txt          – printed stats captured to file
    results/forgetting_curve.png     – p_recall vs Δt plot
    results/p_recall_dist.png        – histogram of p_recall
    data/duolingo.duckdb             – persistent DuckDB database (gitignored)
"""

import os
import sys
import time
import json
from pathlib import Path

import duckdb
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CSV_PATH    = DATA_DIR / "settles.acl16.learning_traces.13m.csv"
DB_PATH     = DATA_DIR / "duolingo.duckdb"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_csv() -> None:
    if not CSV_PATH.exists():
        print(f"""
[ERROR] Dataset not found at:
    {CSV_PATH}

Download it from:
  https://www.kaggle.com/datasets/duolingo/spaced-repetition-data

Then place the CSV as:
  data/settles.acl16.learning_traces.13m.csv
""")
        sys.exit(1)
    size_mb = CSV_PATH.stat().st_size / 1e6
    print(f"[OK] CSV found — {size_mb:.0f} MB")


# ── Step 1: Connect to DuckDB and register CSV ─────────────────────────────────

def connect_db() -> duckdb.DuckDBPyConnection:
    """
    Open (or create) a persistent DuckDB database and register the CSV
    as a virtual table called 'traces'.  DuckDB streams the CSV in chunks
    so peak RAM stays well under 1 GB even for 13 M rows.
    """
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"""
        CREATE OR REPLACE VIEW traces AS
        SELECT *
        FROM read_csv_auto('{CSV_PATH.as_posix()}', header=True, parallel=True)
    """)
    print(f"[DuckDB] Connected — DB at {DB_PATH}")
    return con


# ── Step 2: Profile the dataset ───────────────────────────────────────────────

def profile_dataset(con: duckdb.DuckDBPyConnection) -> dict:
    """Run aggregate queries over the full 13 M rows via DuckDB."""
    section("Dataset Profile")
 
    t0 = time.perf_counter()
 
    stats = con.execute("""
        SELECT
            COUNT(*)                        AS total_rows,
            COUNT(DISTINCT user_id)         AS unique_users,
            COUNT(DISTINCT lexeme_string)   AS unique_lexemes,
            MIN(timestamp)                  AS ts_min,
            MAX(timestamp)                  AS ts_max,
            ROUND(AVG(p_recall), 6)         AS mean_p_recall,
            ROUND(STDDEV(p_recall), 6)      AS std_p_recall,
            ROUND(MIN(p_recall), 6)         AS min_p_recall,
            ROUND(MAX(p_recall), 6)         AS max_p_recall,
            ROUND(AVG(delta) / 86400.0, 2)  AS mean_delta_days,
            ROUND(MAX(delta) / 86400.0, 2)  AS max_delta_days,
            SUM(CASE WHEN p_recall IS NULL THEN 1 ELSE 0 END) AS null_p_recall
        FROM traces
    """).df().iloc[0]
 
    elapsed = time.perf_counter() - t0
 
    # Convert Unix timestamps → readable dates
    ts_min = pd.Timestamp(stats["ts_min"], unit="s", tz="UTC")
    ts_max = pd.Timestamp(stats["ts_max"], unit="s", tz="UTC")
 
    result = {
        "total_rows":      int(stats["total_rows"]),
        "unique_users":    int(stats["unique_users"]),
        "unique_lexemes":  int(stats["unique_lexemes"]),
        "date_range":      f"{ts_min.date()} → {ts_max.date()}",
        "mean_p_recall":   float(stats["mean_p_recall"]),
        "std_p_recall":    float(stats["std_p_recall"]),
        "min_p_recall":    float(stats["min_p_recall"]),
        "max_p_recall":    float(stats["max_p_recall"]),
        "mean_delta_days": float(stats["mean_delta_days"]),
        "max_delta_days":  float(stats["max_delta_days"]),
        "null_p_recall":   int(stats["null_p_recall"]),
        "query_time_sec":  round(elapsed, 2),
    }
 
    for k, v in result.items():
        print(f"  {k:<22}: {v}")
 
    return result


# ── Step 3: p_recall distribution ─────────────────────────────────────────────

def plot_p_recall_distribution(con: duckdb.DuckDBPyConnection) -> None:
    """Histogram of p_recall sampled at 200 k rows for speed."""
    section("Plotting p_recall distribution")

    df = con.execute("""
        SELECT p_recall
        FROM traces
        WHERE p_recall IS NOT NULL
        USING SAMPLE 200000 (reservoir)
    """).df()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["p_recall"], bins=50, color="#4f86c6", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("p_recall", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Recall Probability  (200 k sample)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    fig.tight_layout()

    out = RESULTS_DIR / "p_recall_dist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")
# ── Step 4: Forgetting curve ───────────────────────────────────────────────────

def plot_forgetting_curve(con: duckdb.DuckDBPyConnection) -> None:
    """
    Mean p_recall vs Δt (days since last review), split by history_seen bins.
    Groups Δt into log-spaced buckets so the x-axis is readable.
    Uses DuckDB to aggregate 13 M rows server-side.
    """
    section("Plotting forgetting curve")

    df = con.execute("""
        SELECT
            -- log-spaced Δt bucket (days)
            FLOOR(LOG(GREATEST(delta, 1) + 1) / LOG(1.5)) AS log_bucket,
            -- history_seen quintile (0–4)
            CASE
                WHEN history_seen <= 1  THEN '1'
                WHEN history_seen <= 3  THEN '2-3'
                WHEN history_seen <= 7  THEN '4-7'
                WHEN history_seen <= 15 THEN '8-15'
                ELSE '16+'
            END AS seen_bin,
            ROUND(AVG(p_recall), 4)  AS mean_p_recall,
            COUNT(*)                  AS n,
            -- representative Δt for x-axis label
            ROUND(EXP(log_bucket * LOG(1.5)) - 1, 1) AS delta_repr
        FROM traces
        WHERE delta IS NOT NULL
          AND p_recall IS NOT NULL
          AND delta <= 365
        GROUP BY log_bucket, seen_bin
        ORDER BY seen_bin, log_bucket
    """).df()

    palette = {
        "1":    "#e63946",
        "2-3":  "#f4a261",
        "4-7":  "#2a9d8f",
        "8-15": "#457b9d",
        "16+":  "#6a0572",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for bin_label, grp in df.groupby("seen_bin"):
        # Filter to buckets with enough data
        grp = grp[grp["n"] > 50].sort_values("delta_repr")
        ax.plot(
            grp["delta_repr"],
            grp["mean_p_recall"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"seen={bin_label}",
            color=palette.get(bin_label, "grey"),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Δt — Days since last review  (log scale)", fontsize=12)
    ax.set_ylabel("Mean p_recall", fontsize=12)
    ax.set_title("Forgetting Curve by Review History\n(Duolingo 13 M traces)", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend(title="Times seen", fontsize=9, title_fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = RESULTS_DIR / "forgetting_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Step 5: Persist processed stats ───────────────────────────────────────────

def save_stats(stats: dict) -> None:
    out = RESULTS_DIR / "eda_summary.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Saved] {out}")


# ── Step 6: Print resume checklist ────────────────────────────────────────────

RESUME_FIELDS = [
    ("total_rows",      "Total rows in dataset"),
    ("unique_users",    "Unique learners"),
    ("unique_lexemes",  "Unique lexemes (word × language)"),
    ("date_range",      "Date range"),
    ("mean_p_recall",   "Mean p_recall (overall)"),
    ("mean_delta_days", "Mean Δt (days between reviews)"),
    ("query_time_sec",  "DuckDB full-scan query time (sec)"),
]

def print_resume_numbers(stats: dict) -> None:
    section("Numbers to log in resume_numbers.md  ← copy these")
    for key, label in RESUME_FIELDS:
        print(f"  [{label}]  →  {stats[key]}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_csv()
    con     = connect_db()
    stats   = profile_dataset(con)
    plot_p_recall_distribution(con)
    plot_forgetting_curve(con)
    save_stats(stats)
    print_resume_numbers(stats)
    con.close()
    print("\n✓  Module 1 complete.\n")
