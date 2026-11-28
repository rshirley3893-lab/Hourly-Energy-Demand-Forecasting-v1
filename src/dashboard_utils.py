from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.evaluate import compute_metrics
from src.features import _us_fixed_holiday_flag, build_features


def _single_step_feature_row(
    next_ts: pd.Timestamp,
    history_values: List[float],
    config: ProjectConfig,
) -> Dict[str, float]:
    """Compute the exact 31 causal features for a single future timestamp in O(1)."""
    if len(history_values) < max(max(config.lags), max(config.rolling_windows)):
        raise ValueError(
            f"Need at least {max(max(config.lags), max(config.rolling_windows))} prior hourly values, "
            f"got {len(history_values)}"
        )

    dt_series = pd.Series([next_ts])
    hour = int(next_ts.hour)
    dow = int(next_ts.dayofweek)
    dom = int(next_ts.day)
    month = int(next_ts.month)
    quarter = int(next_ts.quarter)
    doy = int(next_ts.dayofyear)

    row: Dict[str, float] = {
        "hour": hour,
        "dayofweek": dow,
        "dayofmonth": dom,
        "month": month,
        "quarter": quarter,
        "dayofyear": doy,
        "is_weekend": int(dow >= 5),
        "is_us_holiday": int(_us_fixed_holiday_flag(dt_series).iloc[0]),
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        "doy_sin": float(np.sin(2.0 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2.0 * np.pi * doy / 365.25)),
    }

    for lag in config.lags:
        row[f"lag_{lag}"] = float(history_values[-lag])

    for window in config.rolling_windows:
        arr = np.asarray(history_values[-window:], dtype=float)
        row[f"roll_mean_{window}"] = float(np.mean(arr))
        row[f"roll_std_{window}"] = float(np.std(arr, ddof=1))
        row[f"roll_min_{window}"] = float(np.min(arr))
        row[f"roll_max_{window}"] = float(np.max(arr))

    row["diff_1_24"] = row["lag_1"] - row["lag_24"]
    row["diff_24_168"] = row["lag_24"] - row["lag_168"]
    return row


def recursive_forecast_fast(
    history_df: pd.DataFrame,
    artifact: Dict[str, Any],
    config: ProjectConfig,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Fast O(H) recursive multi-step forecaster matching `src/features.py` bit-for-bit."""
    model = artifact["model"]
    feature_cols: List[str] = artifact["feature_cols"]

    work = history_df[[config.datetime_col, config.target_col]].copy()
    work[config.datetime_col] = pd.to_datetime(work[config.datetime_col])
    work = work.sort_values(config.datetime_col).tail(400).reset_index(drop=True)

    history_vals = work[config.target_col].astype(float).tolist()
    last_ts = pd.Timestamp(work[config.datetime_col].iloc[-1])

    preds: List[Dict[str, Any]] = []
    for step in range(1, horizon_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)
        feat_dict = _single_step_feature_row(next_ts, history_vals, config)
        X_next = pd.DataFrame([[feat_dict[c] for c in feature_cols]], columns=feature_cols)
        y_hat = float(model.predict(X_next)[0])
        history_vals.append(y_hat)
        preds.append(
            {
                "step": step,
                "datetime": next_ts,
                "predicted_mw": round(y_hat, 2),
            }
        )

    return pd.DataFrame(preds)


def find_stress_regimes(holdout_only_df: pd.DataFrame, config: ProjectConfig) -> Dict[str, Dict[str, Any]]:
    """Auto-detect meaningful operational stress regimes inside the 1-year holdout."""
    df = holdout_only_df.copy().sort_values(config.datetime_col).reset_index(drop=True)
    df[config.datetime_col] = pd.to_datetime(df[config.datetime_col])

    # Leave at least 72h after the selected cutoff so 72h backtest always has ground truth
    valid_end = len(df) - 73
    if valid_end < 48:
        ts = df[config.datetime_col].iloc[0]
        return {"Default Holdout Start": {"cutoff_ts": ts, "description": "Start of available holdout"}}

    scoped = df.iloc[24:valid_end].copy()
    regimes: Dict[str, Dict[str, Any]] = {}

    # 1. Summer Heatwave Peak (18h before annual peak load)
    summer = scoped[scoped[config.datetime_col].dt.month.isin([6, 7, 8, 9])]
    peak_pool = summer if len(summer) > 24 else scoped
    peak_idx = int(peak_pool[config.target_col].idxmax())
    cutoff_peak = df.loc[max(24, peak_idx - 18), config.datetime_col]
    regimes["Summer Heatwave Peak"] = {
        "cutoff_ts": pd.Timestamp(cutoff_peak),
        "description": f"Leads into annual peak load ({df.loc[peak_idx, config.target_col]:,.0f} MW on {df.loc[peak_idx, config.datetime_col].strftime('%Y-%m-%d %H:%M')})",
    }

    # 2. Winter Cold Snap Peak
    winter = scoped[scoped[config.datetime_col].dt.month.isin([12, 1, 2])]
    if len(winter) > 24:
        w_idx = int(winter[config.target_col].idxmax())
        cutoff_w = df.loc[max(24, w_idx - 18), config.datetime_col]
        regimes["Winter Cold Snap"] = {
            "cutoff_ts": pd.Timestamp(cutoff_w),
            "description": f"Leads into winter surge ({df.loc[w_idx, config.target_col]:,.0f} MW on {df.loc[w_idx, config.datetime_col].strftime('%Y-%m-%d %H:%M')})",
        }

    # 3. Steepest Demand Ramp
    ramps = scoped[config.target_col].diff().abs()
    if ramps.notna().any():
        r_idx = int(ramps.idxmax())
        cutoff_r = df.loc[max(24, r_idx - 12), config.datetime_col]
        regimes["Steepest Grid Ramp"] = {
            "cutoff_ts": pd.Timestamp(cutoff_r),
            "description": f"Captures largest 1-hour load swing ({ramps.loc[r_idx]:,.0f} MW/h on {df.loc[r_idx, config.datetime_col].strftime('%Y-%m-%d %H:%M')})",
        }

    # 4. Holiday Demand Drop
    hol_mask = _us_fixed_holiday_flag(scoped[config.datetime_col]) == 1
    hols = scoped[hol_mask]
    if len(hols) > 0:
        h_idx = int(hols.index[0])
        cutoff_h = df.loc[max(24, h_idx - 12), config.datetime_col]
        regimes["Federal Holiday Regime"] = {
            "cutoff_ts": pd.Timestamp(cutoff_h),
            "description": f"Holiday load drop starting near {df.loc[h_idx, config.datetime_col].strftime('%Y-%m-%d')}",
        }

    # 5. Mild Shoulder Season (lowest daily peak)
    shoulder = scoped[scoped[config.datetime_col].dt.month.isin([4, 5, 10])]
    if len(shoulder) > 24:
        s_idx = int(shoulder[config.target_col].idxmin())
        cutoff_s = df.loc[max(24, s_idx - 12), config.datetime_col]
        regimes["Mild Shoulder Season"] = {
            "cutoff_ts": pd.Timestamp(cutoff_s),
            "description": f"Low-demand trough ({df.loc[s_idx, config.target_col]:,.0f} MW on {df.loc[s_idx, config.datetime_col].strftime('%Y-%m-%d')})",
        }

    return regimes


def run_backtest_comparison(
    slice_df: pd.DataFrame,
    cutoff_ts: pd.Timestamp,
    horizon_hours: int,
    best_artifact: Dict[str, Any],
    ridge_artifact: Dict[str, Any] | None,
    config: ProjectConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run live recursive backtest from `cutoff_ts` for `horizon_hours` against ground truth.

    Returns:
      - comparison_df: hourly actual vs. recursive model forecasts & naive baselines
      - summary_metrics_df: aggregate MAE/RMSE/MAPE/WAPE across the selected horizon
      - horizon_curve_df: step-by-step error compounding from step 1 to horizon_hours
    """
    df = slice_df[[config.datetime_col, config.target_col]].copy()
    df[config.datetime_col] = pd.to_datetime(df[config.datetime_col])
    df = df.sort_values(config.datetime_col).reset_index(drop=True)

    cutoff = pd.Timestamp(cutoff_ts)
    history = df[df[config.datetime_col] <= cutoff].copy()
    future = df[df[config.datetime_col] > cutoff].head(horizon_hours).copy().reset_index(drop=True)

    if len(history) < 340:
        raise ValueError("Selected cutoff does not have 340+ prior hours of history for causal lags.")
    if len(future) == 0:
        raise ValueError("Selected cutoff has no future ground-truth hours to evaluate.")

    actual_horizon = len(future)
    best_name = str(best_artifact.get("model_name", "BestModel"))

    best_fcst = recursive_forecast_fast(history, best_artifact, config, horizon_hours=actual_horizon)
    comp = future.rename(columns={config.target_col: "actual_mw"}).copy()
    comp["step"] = np.arange(1, actual_horizon + 1)
    comp[best_name] = best_fcst["predicted_mw"].to_numpy()

    model_cols = [best_name]

    if ridge_artifact is not None:
        ridge_fcst = recursive_forecast_fast(history, ridge_artifact, config, horizon_hours=actual_horizon)
        comp["RidgeCV"] = ridge_fcst["predicted_mw"].to_numpy()
        model_cols.append("RidgeCV")

    # Strictly causal Seasonal Naive 168h & 24h (using only history <= cutoff)
    hist_vals = history[config.target_col].astype(float).to_numpy()
    naive_168 = [hist_vals[-168 + ((h - 1) % 168)] for h in range(1, actual_horizon + 1)]
    naive_24 = [hist_vals[-24 + ((h - 1) % 24)] for h in range(1, actual_horizon + 1)]
    comp["SeasonalNaive_168h"] = np.round(naive_168, 2)
    comp["SeasonalNaive_24h"] = np.round(naive_24, 2)
    model_cols.extend(["SeasonalNaive_24h", "SeasonalNaive_168h"])

    # Aggregate metrics over the forecast horizon
    metric_rows = []
    y_true = comp["actual_mw"].to_numpy()
    for col in model_cols:
        m = compute_metrics(y_true, comp[col].to_numpy())
        metric_rows.append({"model": col, **m})
    summary_metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE_MW", ascending=True).reset_index(drop=True)

    # Cumulative error compounding curve by step h = 1 .. actual_horizon
    curve_rows = []
    for h in range(1, actual_horizon + 1):
        yt = y_true[:h]
        row: Dict[str, Any] = {"horizon_step": h}
        for col in model_cols:
            yp = comp[col].to_numpy()[:h]
            row[f"{col}_RMSE"] = round(float(np.sqrt(np.mean((yt - yp) ** 2))), 2)
            row[f"{col}_MAPE"] = round(float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), 1e-8))) * 100.0), 3)
        curve_rows.append(row)
    horizon_curve_df = pd.DataFrame(curve_rows)

    return comp, summary_metrics_df, horizon_curve_df


def simulate_grid_operations(
    pred_df: pd.DataFrame,
    actual_col: str,
    model_pred_col: str,
    baseline_pred_col: str,
    reserve_buffer_mw: float = 500.0,
    under_cost_per_mwh: float = 180.0,
    over_cost_per_mwh: float = 35.0,
    capacity_alert_mw: float = 45000.0,
) -> Dict[str, Any]:
    """Simulate asymmetric grid dispatch cost, shortfall risk, and peak alert capture."""
    actual = pred_df[actual_col].to_numpy(dtype=float)

    def _eval_one(pred_arr: np.ndarray) -> Dict[str, Any]:
        committed = pred_arr + reserve_buffer_mw
        shortfall = np.maximum(0.0, actual - committed)
        excess = np.maximum(0.0, committed - actual)

        shortfall_mwh = float(np.sum(shortfall))
        excess_mwh = float(np.sum(excess))
        shortfall_hours = int(np.sum(shortfall > 0.0))

        under_cost = shortfall_mwh * under_cost_per_mwh
        over_cost = excess_mwh * over_cost_per_mwh
        total_cost = under_cost + over_cost

        # Peak capacity alert confusion metrics
        true_peak = actual >= capacity_alert_mw
        pred_peak = pred_arr >= capacity_alert_mw
        tp = int(np.sum(true_peak & pred_peak))
        fp = int(np.sum((~true_peak) & pred_peak))
        fn = int(np.sum(true_peak & (~pred_peak)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "shortfall_mwh": round(shortfall_mwh, 1),
            "excess_mwh": round(excess_mwh, 1),
            "shortfall_hours": shortfall_hours,
            "under_cost_usd": round(under_cost, 2),
            "over_cost_usd": round(over_cost, 2),
            "total_cost_usd": round(total_cost, 2),
            "true_peak_hours": int(np.sum(true_peak)),
            "peak_precision": round(precision * 100.0, 2),
            "peak_recall": round(recall * 100.0, 2),
            "peak_f1": round(f1 * 100.0, 2),
            "shortfall_series": shortfall,
            "excess_series": excess,
        }

    m_stats = _eval_one(pred_df[model_pred_col].to_numpy(dtype=float))
    b_stats = _eval_one(pred_df[baseline_pred_col].to_numpy(dtype=float))

    cost_saved = b_stats["total_cost_usd"] - m_stats["total_cost_usd"]
    cost_saved_pct = (cost_saved / b_stats["total_cost_usd"] * 100.0) if b_stats["total_cost_usd"] > 0 else 0.0

    # Hourly risk profile
    dt = pd.to_datetime(pred_df["datetime"])
    hourly_df = pd.DataFrame(
        {
            "hour": dt.dt.hour,
            "model_shortfall_hr": (m_stats["shortfall_series"] > 0).astype(int),
            "baseline_shortfall_hr": (b_stats["shortfall_series"] > 0).astype(int),
            "model_shortfall_mwh": m_stats["shortfall_series"],
            "baseline_shortfall_mwh": b_stats["shortfall_series"],
        }
    ).groupby("hour", as_index=False).sum()

    return {
        "model": m_stats,
        "baseline": b_stats,
        "cost_saved_usd": round(cost_saved, 2),
        "cost_saved_pct": round(cost_saved_pct, 2),
        "shortfall_hours_avoided": b_stats["shortfall_hours"] - m_stats["shortfall_hours"],
        "shortfall_mwh_avoided": round(b_stats["shortfall_mwh"] - m_stats["shortfall_mwh"], 1),
        "hourly_risk_df": hourly_df,
    }


def parse_and_clean_uploaded_csv(file_bytes: bytes, config: ProjectConfig) -> pd.DataFrame:
    """Parse an uploaded 2-column hourly demand CSV, deduplicate, and ensure continuous hourly grid."""
    raw = pd.read_csv(io.BytesIO(file_bytes))
    if raw.shape[1] < 2:
        raise ValueError(f"CSV must have at least 2 columns (timestamp, load MW); found {list(raw.columns)}")

    dt_col, val_col = raw.columns[0], raw.columns[1]
    df = pd.DataFrame(
        {
            config.datetime_col: pd.to_datetime(raw[dt_col], errors="coerce"),
            config.target_col: pd.to_numeric(raw[val_col], errors="coerce"),
        }
    ).dropna()

    if len(df) < 340:
        raise ValueError(
            f"Need at least 340 hourly rows (14+ days) to compute causal 336h lag features; got {len(df)}."
        )

    df = (
        df.groupby(config.datetime_col, as_index=False)[config.target_col]
        .mean()
        .sort_values(config.datetime_col)
        .set_index(config.datetime_col)
    )
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = config.datetime_col
    df[config.target_col] = df[config.target_col].interpolate(method="time", limit=3).ffill().bfill()
    return df.reset_index()
