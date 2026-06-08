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
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([-50.0, 200.0])
    expected = np.sqrt(np.mean((np.log1p([0.0, 200.0]) - np.log1p(y_true)) ** 2))
    assert abs(rmsle(y_true, y_pred) - expected) < 1e-12
