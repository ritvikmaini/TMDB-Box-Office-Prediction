# TMDB Box Office Prediction

Predicting movie box-office revenue for the
[Kaggle TMDB Box Office Prediction](https://www.kaggle.com/c/tmdb-box-office-prediction)
competition. Metric: **RMSLE**.

## What's here

- `src/` — reusable modules: `features.py` (feature engineering), `models.py`
  (K-fold OOF gradient-boosting ensemble), `evaluate.py` (RMSLE).
- `train.py` — one command: `data/{train,test}.csv` → `data/submission.csv`.
- `notebooks/TMDB_v2.ipynb` — narrative walkthrough.
- `TMDB.ipynb` — the original baseline, kept for comparison.

## Approach

The original notebook trained a single Random Forest on **raw revenue**, while the
competition is scored on RMSLE. v2 fixes this by training on `log1p(revenue)` and
inverting with `expm1`, then adds:

- 57 features from the JSON columns (cast/crew/genre/company counts, gender splits,
  collection & homepage flags, date parts) plus non-leaky inflation/interaction
  features (budget ÷ release-year-mean budget, log-popularity, budget×popularity).
- LightGBM + XGBoost + CatBoost, each with 5-fold out-of-fold (OOF) predictions.
- A blended ensemble that picks the lowest-OOF option among a simple mean, a
  non-negative ridge stack, and the best single model.

## Results (5-fold OOF RMSLE, lower is better)

The headline improvement is correctness: the original notebook computed `logRevenue`
but trained on **raw revenue**, optimizing the wrong target for an RMSLE metric.
Fixing that target alone removes ~0.4 RMSLE; features + a tuned ensemble take it further.

| Stage | OOF RMSLE | Notes |
|-------|-----------|-------|
| Original approach (RandomForest, **raw** revenue target) | **2.638** | the baseline's effective score on the competition metric |
| Same features, log-target fix only | 2.233 | the single biggest correctness fix |
| LightGBM, full feature set | 2.116 | |
| XGBoost, full feature set | 2.113 | |
| CatBoost, full feature set (tuned) | **2.064** | strongest single model |
| **Blended ensemble (v2)** | **2.064** | ridge stack ≈ 96% CatBoost; weak models down-weighted |

**Net: 2.638 → 2.064, a ~21.8% reduction in RMSLE**, using only the provided
competition data.

> **Note on the ceiling.** Public solutions that score below ~1.9 almost all rely on
> an *external* "TMDB additional features" dataset (`rating`, `totalVotes`,
> `popularity2`). Using only the provided data, ~2.0 is the realistic neighborhood;
> in-fold target encoding of high-cardinality fields (cast/crew/company) was tested
> and *overfit* on the 3,000-row training set, so it is deliberately excluded.

## Run it

```bash
pip install -r requirements.txt
# place train.csv and test.csv in ./data/
python train.py        # writes data/submission.csv, prints the OOF table
python -m pytest -q    # run the unit tests
```
