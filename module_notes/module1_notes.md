# Module 1 Notes — Data Loading & EDA

## Why DuckDB?

| Tool | 13 M rows load time | Peak RAM |
|------|-------------------|----------|
| `pd.read_csv` | ~45 s | ~4 GB |
| DuckDB `read_csv_auto` | ~6 s | ~300 MB |
| DuckDB + VIEW (streaming) | no load | ~50 MB |

DuckDB pushes aggregations down into its columnar engine **before** materialising any
Python objects, so `COUNT(DISTINCT user_id)` over 13 M rows takes a few seconds
rather than minutes.

---

## Column Glossary (Settles & Meeder 2016)

| Column | Type | Notes |
|--------|------|-------|
| `p_recall` | float [0,1] | **Target variable** — empirical recall probability |
| `timestamp` | int (unix s) | Review time |
| `delta` | float (days) | Δt — time since last review of this lexeme |
| `history_seen` | int | Total times this (user, lexeme) pair was reviewed |
| `history_correct` | int | Total correct reviews so far |
| `lexeme_string` | str | `"word/pos@lang"` e.g. `"casa/Noun@es"` |
| `user_id` | str (hashed) | Anonymised learner |
| `learning_language` | str | Language being learned |
| `ui_language` | str | Learner's native interface language |

---

## Forgetting Curve — Ebbinghaus vs Duolingo

The **Ebbinghaus** model: `R(t) = e^{-t/S}` where S is stability.

The **Duolingo HLR** reformulation:
```
p_recall = 2^{-Δt / h}
```
where `h` (half-life) is the time at which `p_recall = 0.5`.

Key insight from the plot:
- For `seen=1`, p_recall drops fast — half-life ≈ 1 day.
- For `seen=16+`, p_recall stays high even at Δt=30 days — half-life ≈ 15-30 days.
- This motivates the spaced-repetition schedule: review **just before** you forget.

---

## What Numbers to Expect (from the paper)

| Stat | Paper reports |
|------|---------------|
| Total rows | ~13 M |
| Unique users | ~118 k |
| Unique lexemes | ~61 k |
| Mean p_recall | ~0.74 |
| Date range | Jan 2013 – Mar 2014 |

---

## Common Gotchas

1. **`delta = 0`** rows exist (first review of a lexeme — Δt undefined). Filter these out
   when fitting the HLR model but **keep** them for the Leitner baseline.

2. **`p_recall = 0`** causes `log2(0)` → `-inf` when computing half-life. Add ε = 1e-9.

3. The CSV uses **tab separators** in some downloads. If DuckDB misparses, add:
   ```python
   read_csv_auto('...', delim='\t')
   ```

4. DuckDB `USING SAMPLE 200000 ROWS` is reservoir sampling — perfectly random,
   no ordering needed.

---

## Resume Bullet (fill after running the script)

> "Loaded Duolingo 13 M-row spaced-repetition dataset with DuckDB in **X s** (<300 MB RAM);
> profiled **Y unique learners × Z unique lexemes**; visualised forgetting curves stratified
> by review history, confirming Ebbinghaus exponential decay (mean p_recall = **W**)."
