# Resume Numbers — Duolingo HLR Project

Fill in the blanks after running each module script.
All numbers should come from **actual script output**, not guesses.

---

## Module 1 — Data Loading & EDA

| Metric | Value |
|--------|-------|
| Total rows (13 M CSV) | _____ |
| Unique learners (`user_id`) | _____ |
| Unique lexemes (`lexeme_string`) | _____ |
| Date range | _____ |
| Mean p_recall (overall) | _____ |
| Std p_recall | _____ |
| Mean Δt between reviews (days) | _____ |
| DuckDB full-scan query time | _____ s |
| Peak RAM during profiling | < _____ MB |

**Resume bullet template:**
> "Ingested 13 M Duolingo spaced-repetition traces with DuckDB in **X s** (<300 MB RAM);
> explored recall distributions across **Y learners × Z lexemes** spanning **DATE RANGE**;
> visualised Ebbinghaus forgetting curves stratified by review history
> (mean p_recall = **W**)."

---

## Module 2 — Feature Engineering  *(fill after Module 2)*

| Metric | Value |
|--------|-------|
| Train rows | _____ |
| Validation rows | _____ |
| Test rows | _____ |
| Features engineered | _____ |
| % rows with delta = 0 (dropped) | _____ % |

---

## Module 3 — Leitner Baseline  *(fill after Module 3)*

| Metric | Value |
|--------|-------|
| MAE (test) | _____ |
| RMSE (test) | _____ |

---

## Module 4 — HLR Model  *(fill after Module 4)*

| Metric | Value |
|--------|-------|
| MAE (test) | _____ |
| RMSE (test) | _____ |
| Δ MAE vs Leitner | _____ (↓ improvement) |
| Training time | _____ s |
| Regularisation α | _____ |

**Resume bullet template:**
> "Implemented Half-Life Regression (Settles & Meeder 2016); beat Leitner baseline by
> **Δ MAE** on p_recall prediction; trained on 10 M+ traces in **X s**."
