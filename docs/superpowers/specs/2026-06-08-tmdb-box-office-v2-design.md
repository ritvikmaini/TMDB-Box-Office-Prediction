# TMDB Box Office Prediction v2 — Design Spec

**Date:** 2026-06-08
**Repo:** https://github.com/ritvikmaini/TMDB-Box-Office-Prediction
**Competition:** [TMDB Box Office Prediction](https://www.kaggle.com/c/tmdb-box-office-prediction) (metric: RMSLE)

## Problem

Predict movie box-office `revenue` from metadata available before release (budget,
cast, crew, genres, release date, production companies, etc.). Submissions are scored
on **Root Mean Squared Logarithmic Error (RMSLE)**.

## Baseline (current `TMDB.ipynb`) and its defects

1. **Score-defining bug:** computes `logRevenue = log1p(revenue)` but trains the model
   on the raw `revenue` column. The metric is RMSLE, so training must be on the log
   target with `expm1` inversion. This is the single biggest fix.
2. Single `RandomForestRegressor` on only 13 features; discards most parsed JSON data.
3. CV uses `neg_root_mean_squared_error` (RMSE), not RMSLE — reported scores are
   misleading and not comparable to the leaderboard.
4. `SimpleImputer(strategy='mean')` applied across the whole frame including id-like
   columns; `fit_transform` called separately on train and test (inconsistent stats).

## Goals

- Correct, honest evaluation: a single `rmsle()` used everywhere, OOF CV on fixed folds.
- Strong, measurable accuracy gain via heavy feature engineering + a gradient-boosting
  ensemble (LightGBM + XGBoost + CatBoost), blended.
- Clean, published portfolio artifact: reusable `src/` modules, a narrative notebook,
  a one-command `train.py`, and a README with a before/after results table.

## Non-goals

- External data (IMDb scrapes, inflation tables). Out of scope for v2.
- Deep learning / NLP embeddings of overviews. Counts/flags only for v2.
- Hyperparameter search beyond sensible hand-tuned defaults (optional light tuning).

## Architecture

```
TMDB-Box-Office-Prediction/
├── data/                  # train.csv, test.csv (gitignored), submission.csv output
├── src/
│   ├── __init__.py
│   ├── features.py        # pure feature-engineering functions
│   ├── models.py          # model factory + K-fold OOF training + blend
│   └── evaluate.py        # rmsle(), OOF scoring helpers
├── notebooks/
│   └── TMDB_v2.ipynb      # EDA → features → train → ensemble → submit narrative
├── TMDB.ipynb             # original kept for before/after comparison
├── train.py               # end-to-end CLI: data/ -> data/submission.csv
├── requirements.txt
├── .gitignore             # data CSVs, caches, checkpoints
└── README.md
```

### Data flow

`data/train.csv,test.csv` → `features.build_features(train, test)` (fit on combined,
return engineered train X / y=log1p(revenue) / test X with aligned columns) →
`models.train_oof(X, y, X_test)` per algorithm (5-fold OOF) → blend OOF to pick weights
→ `expm1` + clip(0) test predictions → `data/submission.csv`.

## Components

### `src/features.py`
Pure functions, no global state (the original relied on globals). Key contract:
`build_features(train_df, test_df) -> (X_train, y_train, X_test, feature_names)`.

- **Target:** `y = log1p(revenue)`; inversion (`expm1`, clip ≥0) lives in `train.py`.
- **Dates:** robust parse of `release_date` with the 2-digit-year fix
  (`<=19 -> +2000`, `>19 & <100 -> +1900`); derive year, month, day, dayofweek,
  quarter, and `years_since_release` relative to a fixed reference (max year in data).
- **Budget:** `log1p(budget)`, `budget_is_zero` flag, `budget/popularity`,
  `budget/runtime`, budget ÷ mean-budget-of-release-year (inflation proxy).
- **JSON counts** (parse with a safe `ast.literal_eval`): num_genres, num_cast,
  num_crew, num_production_companies, num_production_countries,
  num_spoken_languages, num_keywords, cast/crew gender counts (0/1/2).
- **Flags / text:** has_homepage, has_collection, has_tagline, is_english,
  title/overview/tagline word & char counts.
- **Categoricals:** top-N one-hot for genres; frequency encoding for production
  companies/countries and spoken languages; rare-category pruning
  (keep names appearing ≥10× across combined train+test).
- Missing values imputed deterministically (median for numerics, 0 for counts/flags);
  imputer fit on combined frame so train/test share statistics.

### `src/models.py`
- `make_models()` returns configured LightGBM, XGBoost, CatBoost regressors with
  log-target-appropriate RMSE objectives and sane defaults
  (e.g. LGBM: ~2000 trees, lr 0.01, num_leaves 31, subsample/colsample 0.8, early stop;
  XGB and CatBoost analogous).
- `train_oof(model_name, X, y, X_test, folds=5, seed=42) -> (oof_pred, test_pred, cv_rmsle)`:
  KFold; train per fold with early stopping on the held-out fold; collect OOF preds and
  average test preds across folds; report OOF RMSLE (on revenue scale via expm1).
- `blend(oof_dict, y, test_dict)`: start with simple average; if a non-negative-weighted
  ridge meta-learner on OOF preds lowers OOF RMSLE, use it (stacking). Pick the lower.

### `src/evaluate.py`
- `rmsle(y_true_revenue, y_pred_revenue)`: clip preds ≥0, `sqrt(mean((log1p(p)-log1p(t))^2))`.
- Helper to assemble and print the model comparison table.

### `train.py`
CLI orchestrator: load data → build features → train each model OOF → blend →
write `data/submission.csv` (columns `id,revenue`) → print results table. Fails
clearly with an actionable message if `data/train.csv` / `data/test.csv` are absent.

### `notebooks/TMDB_v2.ipynb`
Narrative version importing from `src/`: brief EDA, feature construction, per-model CV,
ensemble, submission, and a markdown results table. Keeps logic in `src/` (notebook is
presentation, not the source of truth).

## Evaluation & success criteria

- All evaluation via the single `rmsle()` on out-of-fold predictions over fixed folds.
- README results table compares: RF baseline (correctly re-scored on log target for a
  fair fight) vs LightGBM vs XGBoost vs CatBoost vs blend — all OOF RMSLE on identical folds.
- **Success:** the blended ensemble's OOF RMSLE is materially below the baseline's, and
  `train.py` runs end-to-end producing a valid `submission.csv`. Target band ~1.8–1.9
  RMSLE (exact numbers filled in after running on the provided CSVs).

## Testing

- `src/features.py` and `src/evaluate.py` get lightweight unit tests on tiny synthetic
  frames (date-year fix, count extraction, rmsle correctness, no NaNs/inf in output,
  train/test column alignment). Run with `pytest`.
- End-to-end smoke: `train.py` on a small sampled frame produces a submission with the
  right shape and no NaNs. Full run deferred until real CSVs are provided.

## Risks / open items

- **Data not yet present.** All code is written to be runnable; final RMSLE numbers and
  any tuning are completed once `data/train.csv` and `data/test.csv` are dropped in.
- CatBoost/XGB runtime on full data is minutes-scale; acceptable. If too slow, reduce
  trees or drop to LightGBM + XGBoost blend (noted as a fallback, not default).
