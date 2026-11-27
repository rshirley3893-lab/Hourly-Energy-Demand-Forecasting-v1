from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard forecasting error metrics (MAE, RMSE, MAPE, WAPE, sMAPE, R2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), eps) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps))
        * 100.0
    )
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE_MW": round(mae, 2),
        "RMSE_MW": round(rmse, 2),
        "MAPE_pct": round(mape, 3),
        "WAPE_pct": round(wape, 3),
        "sMAPE_pct": round(smape, 3),
        "R2": round(r2, 5),
    }
