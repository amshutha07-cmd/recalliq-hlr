# IMPLEMENTATION PLAN — Duolingo HLR Retention Regression

## Paper
Settles, B. & Meeder, B. (2016). *A Trainable Spaced Repetition Model for Language Learning*. ACL.  
https://aclanthology.org/P16-1174/

---

## Module 1 — Data Loading & EDA
**Goal:** Load 13 M rows efficiently; profile the dataset; plot forgetting curves.

### Steps
1. Install DuckDB + pandas + matplotlib.
2. Load CSV via DuckDB (streaming, < 2 GB RAM).
3. Compute summary statistics: row count, unique users, unique lexemes, date range, p_recall distribution.
4. Plot forgetting curve: mean p_recall vs Δt (time since last review), grouped by history bins.
5. Log key numbers → `resume_numbers.md`.

### Key columns
| Column | Description |
|--------|-------------|
| `p_recall` | Probability of correct recall (target) |
| `timestamp` | Unix time of review |
| `delta` | Days since last review (Δt) |
| `history_seen` | Total times word seen |
| `history_correct` | Total correct reviews |
| `lexeme_string` | Word + language tag |
| `user_id` | Anonymised learner ID |

---

## Module 2 — Feature Engineering
- Compute `h_estimate` (half-life proxy) = `-delta / log2(p_recall + ε)`
- Log-transform skewed features
- Train/val/test split (80/10/10) stratified by user

## Module 3 — Leitner Baseline
- Deterministic spaced-repetition schedule
- Evaluate MAE / RMSE on p_recall prediction

## Module 4 — HLR Model
- Loss: `L = (p̂ - p)² + α(log₂ ĥ - log₂ h)²`
- Gradient descent with regularisation
- Compare vs Leitner baseline → log Δ MAE in `resume_numbers.md`
