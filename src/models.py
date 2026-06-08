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

    mean_oof = oof_mat.mean(axis=1)
    mean_test = test_mat.mean(axis=1)
    best_oof, best_test = mean_oof, mean_test
    best_score = rmsle(y_rev, np.expm1(mean_oof))
    best_weights = {n: 1.0 / len(names) for n in names}

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
