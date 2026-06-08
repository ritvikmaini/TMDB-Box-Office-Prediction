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
