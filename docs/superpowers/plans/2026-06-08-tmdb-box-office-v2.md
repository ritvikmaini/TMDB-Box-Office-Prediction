# TMDB Box Office Prediction v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single mis-targeted Random Forest with a log-target gradient-boosting ensemble (LightGBM + XGBoost + CatBoost) plus rich feature engineering, packaged as reusable `src/` modules, a notebook, a one-command `train.py`, and a README with a before/after RMSLE table.

**Architecture:** Pure feature-engineering functions in `src/features.py` produce aligned train/test matrices with a `log1p(revenue)` target. `src/models.py` trains each booster with 5-fold out-of-fold (OOF) predictions and blends them. `src/evaluate.py` provides the single `rmsle()` used for all scoring. `train.py` orchestrates end-to-end; `notebooks/TMDB_v2.ipynb` is the narrative presentation.

**Tech Stack:** Python 3.13, pandas, numpy, scikit-learn 1.8, lightgbm 4.6, xgboost 3.2, catboost 1.2, pytest.

---

## File Structure

- `src/__init__.py` — package marker.
- `src/evaluate.py` — `rmsle()` and the results-table printer. No deps on other src modules.
- `src/features.py` — `build_features(train_df, test_df)` and helpers. Depends on pandas/numpy only.
- `src/models.py` — `make_models()`, `train_oof()`, `blend()`. Depends on evaluate.py.
- `train.py` — CLI orchestrator at repo root. Depends on all src modules.
- `tests/test_evaluate.py`, `tests/test_features.py` — unit tests on synthetic frames.
- `notebooks/TMDB_v2.ipynb` — narrative notebook importing from `src/`.
- `requirements.txt`, `.gitignore`, `README.md` — project scaffolding/docs.

Tasks are ordered so each produces self-contained, testable changes. Tasks 1–5 do not require the real CSVs (they use synthetic frames). Task 6 (full run + fill in numbers) requires `data/train.csv` and `data/test.csv`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
pytest>=7.0
jupyter
```

- [ ] **Step 2: Create `.gitignore`**

```
data/*.csv
data/submission.csv
__pycache__/
*.pyc
.ipynb_checkpoints/
catboost_info/
.DS_Store
```

- [ ] **Step 3: Create empty `src/__init__.py` and `tests/__init__.py`**

Both files are empty (package markers).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore src/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding for v2"
```

---

## Task 2: RMSLE evaluation (`src/evaluate.py`)

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
from src.evaluate import rmsle


def test_rmsle_zero_when_equal():
    y = np.array([100.0, 5000.0, 1_000_000.0])
    assert rmsle(y, y) == 0.0


def test_rmsle_matches_manual():
    y_true = np.array([100.0, 1000.0])
    y_pred = np.array([110.0, 900.0])
    expected = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    assert abs(rmsle(y_true, y_pred) - expected) < 1e-12


def test_rmsle_clips_negative_predictions():
    # negative predictions must be clipped to 0, not crash on log1p
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([-50.0, 200.0])
    expected = np.sqrt(np.mean((np.log1p([0.0, 200.0]) - np.log1p(y_true)) ** 2))
    assert abs(rmsle(y_true, y_pred) - expected) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'rmsle'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluate.py
import numpy as np
import pandas as pd


def rmsle(y_true_revenue, y_pred_revenue):
    """Root Mean Squared Logarithmic Error on the revenue scale.

    Predictions are clipped to >= 0 before taking log1p (the competition metric).
    """
    y_true = np.asarray(y_true_revenue, dtype=float)
    y_pred = np.clip(np.asarray(y_pred_revenue, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def results_table(scores: dict) -> str:
    """Render {model_name: oof_rmsle} as a sorted markdown table (best first)."""
    rows = sorted(scores.items(), key=lambda kv: kv[1])
    lines = ["| Model | OOF RMSLE |", "|-------|-----------|"]
    for name, score in rows:
        lines.append(f"| {name} | {score:.5f} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add rmsle metric and results table"
```

---

## Task 3: Feature engineering (`src/features.py`)

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
import pandas as pd
from src.features import parse_year, safe_json, build_features


def _synthetic_frame(with_revenue=True):
    rows = {
        "id": [1, 2, 3],
        "budget": [1000000, 0, 5000000],
        "popularity": [10.0, 2.5, 50.0],
        "runtime": [120.0, 90.0, np.nan],
        "release_date": ["3/15/05", "12/1/99", "7/4/18"],
        "homepage": ["http://x.com", np.nan, np.nan],
        "belongs_to_collection": [
            "[{'id': 1, 'name': 'C'}]", np.nan, np.nan],
        "genres": [
            "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Drama'}]",
            "[{'id': 2, 'name': 'Drama'}]",
            np.nan],
        "cast": ["[{'id': 1, 'name': 'A', 'gender': 2}]", "[]", np.nan],
        "crew": ["[{'id': 1, 'name': 'B', 'gender': 1}]", np.nan, np.nan],
        "production_companies": ["[{'id': 1, 'name': 'Co'}]", np.nan, np.nan],
        "production_countries": ["[{'iso_3166_1': 'US', 'name': 'USA'}]", np.nan, np.nan],
        "spoken_languages": ["[{'iso_639_1': 'en', 'name': 'English'}]", np.nan, np.nan],
        "Keywords": ["[{'id': 1, 'name': 'k'}]", np.nan, np.nan],
        "original_language": ["en", "fr", "en"],
        "tagline": ["A tagline", np.nan, "Hi"],
        "title": ["Movie One", "Deux", "Three"],
        "overview": ["An overview here", np.nan, "Short"],
    }
    if with_revenue:
        rows["revenue"] = [10000000, 500000, 80000000]
    return pd.DataFrame(rows)


def test_parse_year_two_digit_fix():
    assert parse_year(5) == 2005
    assert parse_year(99) == 1999
    assert parse_year(18) == 2018
    assert parse_year(2015) == 2015


def test_safe_json_handles_nan_and_bad():
    assert safe_json(np.nan) == []
    assert safe_json("not json") == []
    assert safe_json("[{'name': 'x'}]") == [{"name": "x"}]


def test_build_features_shapes_and_no_nan():
    train = _synthetic_frame(with_revenue=True)
    test = _synthetic_frame(with_revenue=False)
    X_train, y_train, X_test, feature_names = build_features(train, test)
    # target is log1p(revenue)
    assert np.allclose(y_train.values, np.log1p(train["revenue"].values))
    # train/test column-aligned
    assert list(X_train.columns) == list(X_test.columns) == list(feature_names)
    # no NaN / inf leaked through
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert np.isfinite(X_train.to_numpy()).all()
    # engineered count features exist
    for col in ["num_genres", "num_cast", "num_crew", "budget_is_zero",
                "has_homepage", "has_collection", "release_year"]:
        assert col in X_train.columns
    # budget_is_zero correctly flags the zero-budget row
    assert X_train["budget_is_zero"].tolist() == [0, 1, 0]
    # num_genres counts correctly (2, 1, 0)
    assert X_train["num_genres"].tolist() == [2, 1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_year'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/features.py
import ast
import numpy as np
import pandas as pd

JSON_COLS = ["genres", "cast", "crew", "production_companies",
             "production_countries", "spoken_languages", "Keywords"]


def safe_json(s):
    """Parse a stringified list-of-dicts; return [] on NaN / malformed input."""
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_year(y):
    """Map a possibly 2-digit release year to a 4-digit year."""
    y = int(y)
    if y >= 100:
        return y
    return y + 2000 if y <= 19 else y + 1900


def _count(series):
    return series.map(lambda s: len(safe_json(s)))


def _gender_count(series, gender):
    return series.map(lambda s: sum(1 for d in safe_json(s)
                                     if d.get("gender") == gender))


def _add_features(df):
    """Return a new DataFrame of engineered numeric features for df."""
    out = pd.DataFrame(index=df.index)

    # --- dates ---
    parts = df["release_date"].astype(str).str.split("/", expand=True)
    month = pd.to_numeric(parts[0], errors="coerce")
    day = pd.to_numeric(parts[1], errors="coerce")
    year = pd.to_numeric(parts[2], errors="coerce")
    out["release_month"] = month.fillna(0).astype(int)
    out["release_day"] = day.fillna(0).astype(int)
    out["release_year"] = year.fillna(0).astype(int).map(
        lambda y: parse_year(y) if y > 0 else 0)
    dt = pd.to_datetime(df["release_date"], errors="coerce")
    out["release_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    out["release_quarter"] = dt.dt.quarter.fillna(0).astype(int)
    ref_year = out.loc[out["release_year"] > 0, "release_year"].max()
    out["years_since_release"] = np.where(
        out["release_year"] > 0, ref_year - out["release_year"], 0)

    # --- budget / popularity / runtime ---
    budget = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    out["budget"] = budget
    out["log_budget"] = np.log1p(budget)
    out["budget_is_zero"] = (budget == 0).astype(int)
    out["popularity"] = pop
    out["runtime"] = runtime.fillna(runtime.median())
    out["budget_per_popularity"] = budget / (pop + 1)
    out["budget_per_runtime"] = budget / (out["runtime"] + 1)

    # --- JSON counts ---
    for col in JSON_COLS:
        out[f"num_{col}"] = _count(df[col]) if col in df else 0
    for g in (0, 1, 2):
        out[f"cast_gender_{g}"] = _gender_count(df["cast"], g) if "cast" in df else 0
        out[f"crew_gender_{g}"] = _gender_count(df["crew"], g) if "crew" in df else 0

    # --- flags / text ---
    out["has_homepage"] = df.get("homepage", pd.Series(index=df.index)).notna().astype(int)
    out["has_collection"] = df.get(
        "belongs_to_collection", pd.Series(index=df.index)).notna().astype(int)
    out["has_tagline"] = df.get("tagline", pd.Series(index=df.index)).notna().astype(int)
    out["is_english"] = (df.get("original_language", "") == "en").astype(int)
    out["title_len"] = df.get("title", pd.Series(index=df.index)).fillna("").str.len()
    out["overview_word_count"] = df.get(
        "overview", pd.Series(index=df.index)).fillna("").str.split().map(len)
    out["tagline_word_count"] = df.get(
        "tagline", pd.Series(index=df.index)).fillna("").str.split().map(len)

    return out


def _frequency_encode(train_lists, test_lists, min_count=10):
    """Top categories (by combined count >= min_count) -> per-row count features."""
    from collections import Counter
    counter = Counter()
    for lst in list(train_lists) + list(test_lists):
        for d in lst:
            name = d.get("name")
            if name:
                counter[name] += 1
    keep = {name for name, c in counter.items() if c >= min_count}
    return keep


def build_features(train_df, test_df):
    """Build aligned numeric feature matrices.

    Returns (X_train, y_train, X_test, feature_names).
    y_train is log1p(revenue).
    """
    y_train = np.log1p(pd.to_numeric(train_df["revenue"], errors="coerce").fillna(0))
    y_train.name = "logRevenue"

    X_train = _add_features(train_df)
    X_test = _add_features(test_df)

    # one-hot top genres (combined frequency >= 10)
    keep_genres = _frequency_encode(
        train_df["genres"].map(safe_json), test_df["genres"].map(safe_json), min_count=10)
    for name in sorted(keep_genres):
        col = f"genre_{name}"
        X_train[col] = train_df["genres"].map(
            lambda s: int(any(d.get("name") == name for d in safe_json(s))))
        X_test[col] = test_df["genres"].map(
            lambda s: int(any(d.get("name") == name for d in safe_json(s))))

    # align columns, impute deterministically (median from combined)
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    medians = pd.concat([X_train, X_test]).median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    feature_names = list(X_train.columns)
    return X_train, y_train, X_test, feature_names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -v`
Expected: PASS (3 passed). If column-alignment ordering causes the `list(X_train.columns) == list(X_test.columns)` check to pass but feature-name list mismatches, ensure `align` is applied before `feature_names` is read (it is).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: feature engineering with log target and JSON mining"
```

---

## Task 4: Models and ensemble (`src/models.py`)

**Files:**
- Create: `src/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import numpy as np
import pandas as pd
from src.models import make_models, train_oof, blend


def _toy_data(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    # log-revenue target linear in features + noise, kept positive
    y = pd.Series(10 + 2 * X["a"] - 1.5 * X["b"] + rng.normal(scale=0.3, size=n))
    return X, y


def test_make_models_has_three():
    models = make_models()
    assert set(models.keys()) == {"lightgbm", "xgboost", "catboost"}


def test_train_oof_returns_aligned_predictions():
    X, y = _toy_data()
    X_test = X.iloc[:20].copy()
    oof, test_pred, cv = train_oof("lightgbm", X, y, X_test, folds=3, seed=1)
    assert oof.shape == (len(X),)
    assert test_pred.shape == (len(X_test),)
    assert np.isfinite(oof).all() and np.isfinite(test_pred).all()
    assert cv > 0  # RMSLE on revenue scale


def test_blend_not_worse_than_best_single():
    X, y = _toy_data()
    X_test = X.iloc[:20].copy()
    oof_dict, test_dict = {}, {}
    for name in ("lightgbm", "xgboost"):
        oof, tp, _ = train_oof(name, X, y, X_test, folds=3, seed=1)
        oof_dict[name] = oof
        test_dict[name] = tp
    blended_oof, blended_test, weights = blend(oof_dict, y, test_dict)
    from src.evaluate import rmsle
    y_rev = np.expm1(y.values)
    best_single = min(rmsle(y_rev, np.expm1(v)) for v in oof_dict.values())
    blended_score = rmsle(y_rev, np.expm1(blended_oof))
    assert blended_score <= best_single + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/models.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.evaluate import rmsle


def make_models(seed=42):
    """Return configured gradient-boosting regressors keyed by name.

    All train on the log1p(revenue) target with an RMSE objective, which is
    exactly RMSLE on the revenue scale.
    """
    return {
        "lightgbm": LGBMRegressor(
            n_estimators=3000, learning_rate=0.01, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            random_state=seed, n_jobs=-1, verbose=-1),
        "xgboost": XGBRegressor(
            n_estimators=3000, learning_rate=0.01, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=seed, n_jobs=-1, eval_metric="rmse",
            early_stopping_rounds=100, verbosity=0),
        "catboost": CatBoostRegressor(
            iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
            random_seed=seed, loss_function="RMSE", eval_metric="RMSE",
            early_stopping_rounds=100, verbose=False),
    }


def _fit_one(name, model, X_tr, y_tr, X_val, y_val):
    """Fit a single fold with early stopping where supported."""
    if name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(100, verbose=False),
                             log_evaluation(0)])
    elif name == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:  # catboost
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    return model


def train_oof(name, X, y, X_test, folds=5, seed=42):
    """K-fold out-of-fold training for one model.

    Returns (oof_log_pred, test_log_pred, cv_rmsle). Predictions are on the
    log scale; cv_rmsle is computed on the revenue scale.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for tr_idx, val_idx in kf.split(X):
        model = make_models(seed)[name]
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = _fit_one(name, model, X_tr, y_tr, X_val, y_val)
        oof[val_idx] = model.predict(X_val)
        test_pred += model.predict(X_test) / folds
    cv = rmsle(np.expm1(y.values), np.expm1(oof))
    return oof, test_pred, cv


def blend(oof_dict, y, test_dict):
    """Blend OOF predictions. Try non-negative ridge stacking; fall back to mean.

    Returns (blended_oof_log, blended_test_log, weights_dict).
    """
    names = list(oof_dict.keys())
    y_rev = np.expm1(np.asarray(y))
    oof_mat = np.column_stack([oof_dict[n] for n in names])
    test_mat = np.column_stack([test_dict[n] for n in names])

    # baseline: simple average
    mean_oof = oof_mat.mean(axis=1)
    mean_test = test_mat.mean(axis=1)
    best_oof, best_test = mean_oof, mean_test
    best_score = rmsle(y_rev, np.expm1(mean_oof))
    best_weights = {n: 1.0 / len(names) for n in names}

    # candidate: ridge meta-learner (clip negative weights, renormalize)
    ridge = Ridge(alpha=1.0, positive=True)
    ridge.fit(oof_mat, np.asarray(y))
    w = np.clip(ridge.coef_, 0, None)
    if w.sum() > 0:
        w = w / w.sum()
        stack_oof = oof_mat @ w
        stack_test = test_mat @ w
        stack_score = rmsle(y_rev, np.expm1(stack_oof))
        if stack_score < best_score:
            best_oof, best_test, best_score = stack_oof, stack_test, stack_score
            best_weights = {n: float(wi) for n, wi in zip(names, w)}

    return best_oof, best_test, best_weights
```

Note: `make_models()` is called per fold so each fold gets a fresh estimator. The `early_stopping_rounds` set in the XGB/CatBoost constructors are honored by their `fit`; LightGBM uses callbacks.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (3 passed). Runtime ~30–60s (toy data, small folds).

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: k-fold OOF gradient-boosting ensemble with blending"
```

---

## Task 5: End-to-end orchestrator (`train.py`)

**Files:**
- Create: `train.py`

- [ ] **Step 1: Write `train.py`**

```python
# train.py
"""End-to-end: data/{train,test}.csv -> data/submission.csv with CV report."""
import os
import sys
import numpy as np
import pandas as pd

from src.features import build_features
from src.models import make_models, train_oof, blend
from src.evaluate import rmsle, results_table

DATA_DIR = "data"


def main():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        sys.exit(f"ERROR: place train.csv and test.csv in ./{DATA_DIR}/ first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Loaded train={train_df.shape} test={test_df.shape}")

    X, y, X_test, feats = build_features(train_df, test_df)
    print(f"Built {len(feats)} features")

    scores, oof_dict, test_dict = {}, {}, {}
    for name in make_models().keys():
        print(f"Training {name} (5-fold OOF)...")
        oof, test_pred, cv = train_oof(name, X, y, X_test, folds=5)
        scores[name] = cv
        oof_dict[name] = oof
        test_dict[name] = test_pred
        print(f"  {name} OOF RMSLE = {cv:.5f}")

    blended_oof, blended_test, weights = blend(oof_dict, y, test_dict)
    scores["ensemble"] = rmsle(np.expm1(y.values), np.expm1(blended_oof))
    print(f"Blend weights: {weights}")
    print("\n" + results_table(scores))

    preds = np.clip(np.expm1(blended_test), 0, None)
    out = pd.DataFrame({"id": test_df["id"].astype(int), "revenue": preds})
    out_path = os.path.join(DATA_DIR, "submission.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test on a sampled frame (only if real data present; otherwise skip)**

Run (skip gracefully if no data):
```bash
python -c "
import os, pandas as pd
if os.path.exists('data/train.csv'):
    df = pd.read_csv('data/train.csv').sample(200, random_state=0)
    df.to_csv('data/_smoke_train.csv', index=False)
    print('smoke data ready')
else:
    print('no data; skipping smoke test')
"
```
Expected: either "smoke data ready" or "no data; skipping smoke test". (Full `python train.py` run happens in Task 6 when real CSVs are present.)

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: end-to-end training CLI producing submission.csv"
```

---

## Task 6: Run on real data, fill in results (REQUIRES `data/train.csv`, `data/test.csv`)

**Files:**
- Modify: `README.md` (created in Task 7) — fill in the real RMSLE numbers.

- [ ] **Step 1: Run the full pipeline**

Run: `python train.py`
Expected: per-model OOF RMSLE lines, a blend-weights line, a results table, and `data/submission.csv` written. Ensemble RMSLE expected in roughly the 1.8–1.9 band.

- [ ] **Step 2: Validate the submission file**

Run:
```bash
python -c "
import pandas as pd
s = pd.read_csv('data/submission.csv')
assert list(s.columns) == ['id', 'revenue'], s.columns
assert s['revenue'].notna().all() and (s['revenue'] >= 0).all()
print('submission OK', s.shape)
"
```
Expected: `submission OK (4398, 2)` (test set has 4398 rows).

- [ ] **Step 3: Record numbers**

Capture the printed results table; these numbers go into the README table in Task 7.
If data is NOT yet available, this task is blocked — leave the README table with a
clearly-marked "pending full run" note and surface to the user.

- [ ] **Step 4: Commit (only if numbers were produced)**

```bash
git add README.md
git commit -m "docs: record measured OOF RMSLE results"
```

---

## Task 7: Notebook and README

**Files:**
- Create: `notebooks/TMDB_v2.ipynb`
- Create: `README.md`

- [ ] **Step 1: Create `notebooks/TMDB_v2.ipynb`**

Build with `jupyter nbconvert` from a script, or via a small Python script using
`nbformat`. The notebook must contain these cells (importing from `src/`, not
re-implementing logic):

1. (markdown) Title + competition link + one-paragraph summary.
2. (code) `import sys; sys.path.append('..')` then imports from `src`, pandas, numpy.
3. (code) Load `../data/train.csv`, `../data/test.csv`; `train.shape`, `train.head()`.
4. (markdown) "Feature engineering" — note the log-target fix vs the original.
5. (code) `X, y, X_test, feats = build_features(train, test)`; show `len(feats)`.
6. (markdown) "Models & cross-validation".
7. (code) loop over `make_models()` calling `train_oof`, collecting scores.
8. (code) `blend(...)`, compute ensemble RMSLE, `print(results_table(scores))`.
9. (markdown) "Results" table + brief discussion.
10. (code) write `../data/submission.csv`.

Generate it with this helper script (run once, then delete the script):
```python
# build_notebook.py  (temporary)
import nbformat as nbf
nb = nbf.v4.new_notebook()
c = []
c.append(nbf.v4.new_markdown_cell(
    "# TMDB Box Office Prediction v2\n\n"
    "[Kaggle competition](https://www.kaggle.com/c/tmdb-box-office-prediction) "
    "— metric: RMSLE. Log-target gradient-boosting ensemble "
    "(LightGBM + XGBoost + CatBoost)."))
c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.append('..')\n"
    "import numpy as np, pandas as pd\n"
    "from src.features import build_features\n"
    "from src.models import make_models, train_oof, blend\n"
    "from src.evaluate import rmsle, results_table"))
c.append(nbf.v4.new_code_cell(
    "train = pd.read_csv('../data/train.csv')\n"
    "test = pd.read_csv('../data/test.csv')\n"
    "print(train.shape, test.shape)\n"
    "train.head()"))
c.append(nbf.v4.new_markdown_cell(
    "## Feature engineering\n"
    "Train on `log1p(revenue)` (the original trained on raw revenue — the core "
    "bug). Mine counts/flags from the JSON columns the original discarded."))
c.append(nbf.v4.new_code_cell(
    "X, y, X_test, feats = build_features(train, test)\n"
    "print(f'{len(feats)} features')"))
c.append(nbf.v4.new_markdown_cell("## Models & 5-fold cross-validation"))
c.append(nbf.v4.new_code_cell(
    "scores, oof_dict, test_dict = {}, {}, {}\n"
    "for name in make_models().keys():\n"
    "    oof, tp, cv = train_oof(name, X, y, X_test, folds=5)\n"
    "    scores[name], oof_dict[name], test_dict[name] = cv, oof, tp\n"
    "    print(name, round(cv, 5))"))
c.append(nbf.v4.new_code_cell(
    "blended_oof, blended_test, weights = blend(oof_dict, y, test_dict)\n"
    "scores['ensemble'] = rmsle(np.expm1(y.values), np.expm1(blended_oof))\n"
    "print('weights', weights)\n"
    "print(results_table(scores))"))
c.append(nbf.v4.new_markdown_cell("## Results"))
c.append(nbf.v4.new_code_cell(
    "preds = np.clip(np.expm1(blended_test), 0, None)\n"
    "pd.DataFrame({'id': test['id'].astype(int), 'revenue': preds})"
    ".to_csv('../data/submission.csv', index=False)\n"
    "print('submission written')"))
nb["cells"] = c
nbf.write(nb, "notebooks/TMDB_v2.ipynb")
print("notebook written")
```
Run: `python build_notebook.py && rm build_notebook.py`
Expected: `notebook written`, and `notebooks/TMDB_v2.ipynb` exists.

- [ ] **Step 2: Create `README.md`**

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/TMDB_v2.ipynb README.md
git commit -m "docs: add v2 notebook and README"
```

---

## Self-Review notes

- **Spec coverage:** scaffolding (T1) → evaluate.py/rmsle (T2) → features.py incl. log
  target + JSON mining + alignment/imputation (T3) → models.py OOF ensemble + blend (T4)
  → train.py orchestrator (T5) → real-data run + numbers (T6) → notebook + README (T7).
  All spec sections map to a task. Unit tests cover evaluate/features/models per the
  spec's testing section.
- **Data dependency:** Tasks 1–5 and 7 (except filling numbers) run without the CSVs;
  Task 6 and the README numbers require the user's data. This is the one known blocker
  and is called out explicitly.
- **Type consistency:** `build_features -> (X_train, y_train, X_test, feature_names)`;
  `train_oof(name, X, y, X_test, folds, seed) -> (oof, test_pred, cv)`;
  `blend(oof_dict, y, test_dict) -> (oof, test, weights)`; `rmsle(y_true, y_pred)`.
  Names used consistently across train.py, notebook, and tests.
- **Fallback:** if CatBoost/XGB are too slow on full data, drop to LightGBM+XGBoost
  blend (noted in spec risks); not the default.
