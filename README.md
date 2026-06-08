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

- Rich features from the JSON columns (cast/crew/genre/company counts, gender splits,
  collection & homepage flags, budget ratios, date parts).
- LightGBM + XGBoost + CatBoost, each with 5-fold out-of-fold predictions.
- A blended ensemble (mean or non-negative ridge stack, whichever scores lower OOF).

## Results (5-fold OOF RMSLE)

<!-- filled in by Task 6 after running on real data -->
| Model | OOF RMSLE |
|-------|-----------|
| LightGBM | _pending full run_ |
| XGBoost | _pending full run_ |
| CatBoost | _pending full run_ |
| Ensemble | _pending full run_ |

## Run it

```bash
pip install -r requirements.txt
# place train.csv and test.csv in ./data/
python train.py
```
