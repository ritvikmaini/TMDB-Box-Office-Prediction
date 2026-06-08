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
    assert cv > 0


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
