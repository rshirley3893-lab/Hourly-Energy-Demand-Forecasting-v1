from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig


def _us_fixed_holiday_flag(dt: pd.Series) -> pd.Series:
    """Lightweight, dependency-free US major holiday indicator for PJM grid load."""
    m = dt.dt.month
    d = dt.dt.day
    dow = dt.dt.dayofweek
    is_hol = (
        ((m == 1) & (d == 1))                              # New Year's Day
        | ((m == 7) & (d == 4))                            # Independence Day
        | ((m == 12) & (d == 25))                          # Christmas Day
        | ((m == 11) & (dow == 3) & (d >= 22) & (d <= 28)) # Thanksgiving
        | ((m == 5) & (dow == 0) & (d >= 25))              # Memorial Day
        | ((m == 9) & (dow == 0) & (d <= 7))               # Labor Day
    )
    return is_hol.astype(int)


def build_features(
    df: pd.DataFrame,
    config: ProjectConfig,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create strictly causal calendar, cyclical, lag, and rolling features.

    Leakage Guarantee:
      Every lag and rolling statistic is computed using target values at <= t - 1
      (via `.shift(1)` prior to rolling windows), so target y[t] never leaks into X[t].
    """
    out = df.copy().sort_values(config.datetime_col).reset_index(drop=True)
    dt = pd.to_datetime(out[config.datetime_col])
    out[config.datetime_col] = dt
    y = out[config.target_col].astype(float)

    # 1. Calendar features
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["dayofmonth"] = dt.dt.day
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["dayofyear"] = dt.dt.dayofyear
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out["is_us_holiday"] = _us_fixed_holiday_flag(dt)

    # 2. Cyclical Fourier encodings (helps Linear & Ridge models capture periodicity)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.25)

    # 3. Strict causal lag features (t - k, k >= 1)
    for lag in config.lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 to avoid leakage, got {lag}")
        out[f"lag_{lag}"] = y.shift(lag)

    # 4. Strict causal rolling statistics (shift(1) BEFORE rolling)
    y_past = y.shift(1)
    for window in config.rolling_windows:
        roll = y_past.rolling(window=window, min_periods=window)
        out[f"roll_mean_{window}"] = roll.mean()
        out[f"roll_std_{window}"] = roll.std()
        out[f"roll_min_{window}"] = roll.min()
        out[f"roll_max_{window}"] = roll.max()

    # 5. Momentum / trend deltas (strictly past differences)
    out["diff_1_24"] = out["lag_1"] - out["lag_24"]
    out["diff_24_168"] = out["lag_24"] - out["lag_168"]

    feature_cols = [c for c in out.columns if c not in (config.datetime_col, config.target_col)]

    if drop_na:
        out = out.dropna(subset=feature_cols + [config.target_col]).reset_index(drop=True)

    return out, feature_cols
