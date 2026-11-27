from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cv_splits(
    dates: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    holdout_start_idx: int,
    total_len: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    for i, (tr, val) in enumerate(splits):
        ax.barh(i, len(tr), left=tr[0], height=0.55, color="#3b82f6", label="Train" if i == 0 else "")
        ax.barh(i, len(val), left=val[0], height=0.55, color="#f59e0b", label="Validation" if i == 0 else "")

    ax.barh(
        len(splits),
        holdout_start_idx,
        left=0,
        height=0.55,
        color="#1d4ed8",
        label="Final Train",
    )
    ax.barh(
        len(splits),
        total_len - holdout_start_idx,
        left=holdout_start_idx,
        height=0.55,
        color="#10b981",
        label="Final Holdout Test",
    )

    yticks = list(range(len(splits) + 1))
    ylabels = [f"CV Fold {i + 1}" for i in range(len(splits))] + ["Final Holdout"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Chronological Hourly Sample Index")
    ax.set_title("Expanding-Window TimeSeriesSplit + Untouched Final Holdout")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_benchmark(holdout_df: pd.DataFrame, out_path: Path) -> None:
    df = holdout_df.sort_values("RMSE_MW", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    colors = ["#10b981" if i == 0 else "#475569" for i in range(len(df))]
    bars = ax.barh(df["model"], df["RMSE_MW"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Holdout RMSE (MW) — Lower is Better")
    ax.set_title("Hourly Demand Forecasting Model Benchmark (Holdout Test Set)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val, mape in zip(bars, df["RMSE_MW"], df["MAPE_pct"]):
        ax.text(
            bar.get_width() + max(df["RMSE_MW"]) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.1f} MW ({mape:.2f}% MAPE)",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0, max(df["RMSE_MW"]) * 1.28)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
    hours_to_show: int = 24 * 14,
) -> None:
    sl = slice(-hours_to_show, None)
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax.plot(dates.iloc[sl], y_true[sl], label="Actual Demand (MW)", color="#0f172a", linewidth=1.6)
    ax.plot(
        dates.iloc[sl],
        y_pred[sl],
        label=f"{model_name} Forecast (MW)",
        color="#ef4444",
        linestyle="--",
        linewidth=1.4,
    )
    ax.set_title(f"Actual vs. Predicted Hourly Load (Last {hours_to_show // 24} Days of Holdout)")
    ax.set_ylabel("Megawatts (MW)")
    ax.set_xlabel("Timestamp")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_residual_diagnostics(
    hours: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
) -> None:
    residuals = y_true - y_pred
    abs_err = np.abs(residuals)
    err_by_hour = pd.DataFrame({"hour": hours.values, "abs_err": abs_err}).groupby("hour")["abs_err"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    axes[0].hist(residuals, bins=50, color="#3b82f6", alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="#ef4444", linestyle="--", linewidth=1.3)
    axes[0].set_title(f"{model_name} Residual Distribution (Actual - Pred)")
    axes[0].set_xlabel("Error (MW)")
    axes[0].set_ylabel("Hour Count")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].bar(err_by_hour.index, err_by_hour.values, color="#f59e0b", alpha=0.9)
    axes[1].set_title(f"{model_name} Mean Absolute Error by Hour of Day")
    axes[1].set_xlabel("Hour of Day (0-23)")
    axes[1].set_ylabel("MAE (MW)")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(imp_df: pd.DataFrame, model_name: str, out_path: Path, top_n: int = 15) -> None:
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=140)
    ax.barh(top["feature"], top["importance"], color="#2563eb")
    ax.set_title(f"Top {len(top)} Features — {model_name}")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
