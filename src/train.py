from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig
from src.evaluate import compute_metrics
from src.plots import (
    plot_actual_vs_predicted,
    plot_cv_splits,
    plot_feature_importance,
    plot_model_benchmark,
    plot_residual_diagnostics,
)


def build_model_registry(random_state: int = 42) -> Dict[str, Callable[[], Any]]:
    """Return factory functions for all benchmarked models."""
    registry: Dict[str, Callable[[], Any]] = {
        "LinearRegression": lambda: Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "RidgeCV": lambda: Pipeline(
            [("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-2, 4, 25)))]
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=30,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBRegressor  # type: ignore

        registry["XGBoost"] = lambda: XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    except ImportError:
        print("[warn] xgboost not installed — skipping XGBoost (install via requirements.txt)")

    try:
        from lightgbm import LGBMRegressor  # type: ignore

        registry["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        print("[warn] lightgbm not installed — skipping LightGBM (install via requirements.txt)")

    return registry


def _extract_feature_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: List[str],
    random_state: int,
) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "coef_"):
        importances = np.abs(np.asarray(model.named_steps["model"].coef_, dtype=float))
    else:
        sample_n = min(len(X_val), 3000)
        perm = permutation_importance(
            model,
            X_val.iloc[-sample_n:],
            y_val[-sample_n:],
            n_repeats=5,
            random_state=random_state,
            scoring="neg_root_mean_squared_error",
        )
        importances = np.maximum(perm.importances_mean, 0.0)

    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_training_and_evaluation(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    config: ProjectConfig,
    source_label: str,
    sql_engine: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run expanding-window TimeSeriesSplit CV + untouched final holdout evaluation."""
    config.ensure_dirs()

    n_total = len(feat_df)
    holdout_hours = min(config.holdout_hours, max(24 * 30, n_total // 5))
    dev_df = feat_df.iloc[:-holdout_hours].copy().reset_index(drop=True)
    test_df = feat_df.iloc[-holdout_hours:].copy().reset_index(drop=True)

    X_dev = dev_df[feature_cols]
    y_dev = dev_df[config.target_col].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df[config.target_col].to_numpy()

    # Configure expanding-window TimeSeriesSplit
    max_test_size = max(24 * 14, len(dev_df) // (config.cv_splits + 2))
    cv_test_size = min(config.cv_test_hours, max_test_size)
    tscv = TimeSeriesSplit(n_splits=config.cv_splits, test_size=cv_test_size)
    splits = list(tscv.split(X_dev))

    plot_cv_splits(
        dates=feat_df[config.datetime_col],
        splits=splits,
        holdout_start_idx=len(dev_df),
        total_len=n_total,
        out_path=config.figures_dir / "01_cv_splits.png",
    )

    cv_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []

    # 1. Seasonal Naive Baselines (lag_24 and lag_168)
    for baseline_name, lag_col in [("SeasonalNaive_24h", "lag_24"), ("SeasonalNaive_168h", "lag_168")]:
        for fold_idx, (_, val_idx) in enumerate(splits, start=1):
            m = compute_metrics(y_dev[val_idx], X_dev.iloc[val_idx][lag_col].to_numpy())
            cv_rows.append({"model": baseline_name, "fold": fold_idx, **m})
        hm = compute_metrics(y_test, X_test[lag_col].to_numpy())
        holdout_rows.append({"model": baseline_name, **hm})

    # 2. ML & Linear Models
    registry = build_model_registry(random_state=config.random_state)
    fitted_models: Dict[str, Any] = {}
    holdout_preds: Dict[str, np.ndarray] = {}

    for name, factory in registry.items():
        print(f"[train] Running {config.cv_splits}-fold expanding-window CV for {name}...")
        for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
            model = factory()
            model.fit(X_dev.iloc[tr_idx], y_dev[tr_idx])
            preds = model.predict(X_dev.iloc[val_idx])
            m = compute_metrics(y_dev[val_idx], preds)
            cv_rows.append({"model": name, "fold": fold_idx, **m})

        # Fit on full development window and evaluate once on untouched holdout
        final_model = factory()
        final_model.fit(X_dev, y_dev)
        test_pred = final_model.predict(X_test)
        hm = compute_metrics(y_test, test_pred)
        holdout_rows.append({"model": name, **hm})
        fitted_models[name] = final_model
        holdout_preds[name] = test_pred
        print(f"[holdout] {name:20s} | RMSE: {hm['RMSE_MW']:8.2f} MW | MAE: {hm['MAE_MW']:7.2f} MW | MAPE: {hm['MAPE_pct']:.3f}%")

    cv_df = pd.DataFrame(cv_rows)
    cv_summary = (
        cv_df.groupby("model")[["MAE_MW", "RMSE_MW", "MAPE_pct", "WAPE_pct", "R2"]]
        .agg(["mean", "std"])
        .round(3)
    )
    holdout_df = pd.DataFrame(holdout_rows).sort_values("RMSE_MW", ascending=True).reset_index(drop=True)

    # Identify best trainable model by CV mean RMSE (prevents holdout-based selection bias)
    trainable_cv = (
        cv_df[cv_df["model"].isin(registry.keys())]
        .groupby("model")["RMSE_MW"]
        .mean()
        .sort_values()
    )
    best_model_name = str(trainable_cv.index[0])
    best_model = fitted_models[best_model_name]
    best_pred = holdout_preds[best_model_name]

    # Improvement over SeasonalNaive_168h
    naive_168_rmse = float(
        holdout_df.loc[holdout_df["model"] == "SeasonalNaive_168h", "RMSE_MW"].iloc[0]
    )
    best_rmse = float(holdout_df.loc[holdout_df["model"] == best_model_name, "RMSE_MW"].iloc[0])
    rmse_gain_pct = round((1.0 - best_rmse / naive_168_rmse) * 100.0, 2)

    # Save tables, serialized model, metadata & plots
    cv_df.to_csv(config.reports_dir / "cv_fold_results.csv", index=False)
    cv_summary.to_csv(config.reports_dir / "cv_summary.csv")
    holdout_df.to_csv(config.reports_dir / "holdout_metrics.csv", index=False)

    imp_df = _extract_feature_importance(
        best_model, X_test, y_test, feature_cols, random_state=config.random_state
    )
    imp_df.to_csv(config.reports_dir / "feature_importance.csv", index=False)

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_cols": feature_cols,
        "source_label": source_label,
        "sql_engine": sql_engine,
        "holdout_metrics": holdout_df.loc[holdout_df["model"] == best_model_name].iloc[0].to_dict(),
        "rmse_reduction_vs_naive168_pct": rmse_gain_pct,
    }
    joblib.dump(artifact, config.models_dir / "best_model.joblib")

    metadata = {
        "source_dataset": source_label,
        "sql_engine": sql_engine,
        "total_feature_rows": n_total,
        "dev_rows": len(dev_df),
        "holdout_rows": len(test_df),
        "cv_splits": config.cv_splits,
        "cv_test_hours": cv_test_size,
        "selected_best_model_by_cv": best_model_name,
        "rmse_reduction_vs_seasonal_naive_168h_pct": rmse_gain_pct,
        "holdout_leaderboard": holdout_df.to_dict(orient="records"),
    }
    (config.reports_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Generate diagnostic plots
    plot_model_benchmark(holdout_df, config.figures_dir / "02_model_benchmark.png")
    plot_actual_vs_predicted(
        test_df[config.datetime_col],
        y_test,
        best_pred,
        best_model_name,
        config.figures_dir / "03_actual_vs_predicted.png",
    )
    plot_residual_diagnostics(
        test_df["hour"],
        y_test,
        best_pred,
        best_model_name,
        config.figures_dir / "04_residual_diagnostics.png",
    )
    plot_feature_importance(
        imp_df,
        best_model_name,
        config.figures_dir / "05_feature_importance.png",
    )

    # Generate copy-paste Markdown summary
    md_lines = [
        "# Benchmark Summary",
        "",
        f"- **Dataset:** `{source_label}`",
        f"- **SQL Engine:** `{sql_engine}`",
        f"- **Validation:** {config.cv_splits}-fold expanding-window `TimeSeriesSplit` ({cv_test_size:,}h/fold) + untouched {len(test_df):,}h final holdout",
        f"- **Selected Model (by CV RMSE):** `{best_model_name}` (**{rmse_gain_pct}%** holdout RMSE reduction vs. 168h Seasonal Naive baseline)",
        "",
        "## Holdout Test Set Leaderboard",
        "",
        "| Model | RMSE (MW) | MAE (MW) | MAPE (%) | WAPE (%) | R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in holdout_df.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['RMSE_MW']:,.2f} | {row['MAE_MW']:,.2f} | "
            f"{row['MAPE_pct']:.3f}% | {row['WAPE_pct']:.3f}% | {row['R2']:.5f} |"
        )
    (config.reports_dir / "RESULTS_SUMMARY.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return cv_df, holdout_df, best_model_name
