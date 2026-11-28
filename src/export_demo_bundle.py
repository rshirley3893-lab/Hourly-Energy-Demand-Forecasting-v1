from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig
from src.features import build_features


def export_bundle(config: ProjectConfig | None = None) -> Path:
    """Package trained artifacts + lightweight 1-year holdout slice into `demo_bundle/` for cloud hosting.

    Run this AFTER `python run_pipeline.py` so Streamlit Community Cloud / Hugging Face Spaces
    can run real live LightGBM & RidgeCV inference without committing large raw CSVs.
    """
    cfg = config or ProjectConfig()
    bundle_dir = cfg.root_dir / "demo_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    clean_path = cfg.processed_data_dir / "clean_hourly_demand.csv"
    best_model_path = cfg.models_dir / "best_model.joblib"

    if not clean_path.exists() or not best_model_path.exists():
        raise FileNotFoundError(
            "Missing processed data or trained model. Run `python run_pipeline.py` first."
        )

    print("[bundle] Loading processed hourly demand & saved winning model...")
    clean_df = pd.read_csv(clean_path)
    clean_df[cfg.datetime_col] = pd.to_datetime(clean_df[cfg.datetime_col])
    clean_df = clean_df.sort_values(cfg.datetime_col).reset_index(drop=True)

    best_artifact = joblib.load(best_model_path)
    best_model = best_artifact["model"]
    best_name = best_artifact["model_name"]
    feature_cols = best_artifact["feature_cols"]

    # Build exact causal features and reproduce the exact dev/holdout split
    feat_df, _ = build_features(clean_df, cfg, drop_na=True)
    holdout_hours = min(cfg.holdout_hours, max(24 * 30, len(feat_df) // 5))
    dev_df = feat_df.iloc[:-holdout_hours].copy().reset_index(drop=True)
    test_df = feat_df.iloc[-holdout_hours:].copy().reset_index(drop=True)

    # Fit companion RidgeCV on exact dev split (<1s) for live head-to-head comparison in app
    print("[bundle] Fitting companion RidgeCV on development split for live comparison...")
    ridge_pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-2, 4, 25)))]
    )
    ridge_pipe.fit(dev_df[feature_cols], dev_df[cfg.target_col].to_numpy())
    ridge_artifact = {
        "model_name": "RidgeCV",
        "model": ridge_pipe,
        "feature_cols": feature_cols,
        "source_label": best_artifact.get("source_label", "PJME_hourly.csv"),
    }

    # Save compressed models (keeps GitHub repo lightweight, well below Git LFS limits)
    joblib.dump(best_artifact, bundle_dir / "best_model.joblib", compress=3)
    joblib.dump(ridge_artifact, bundle_dir / "ridge_model.joblib", compress=3)

    # Save holdout slice WITH 400h leading history buffer so hour-0 recursive backtesting works
    holdout_start_ts = test_df[cfg.datetime_col].iloc[0]
    buffer_start_ts = holdout_start_ts - pd.Timedelta(hours=400)
    slice_df = clean_df[clean_df[cfg.datetime_col] >= buffer_start_ts].copy()
    slice_df.to_csv(bundle_dir / "holdout_slice.csv.gz", index=False, compression="gzip")

    # Save precomputed 1-hour-ahead holdout predictions for instant Risk Simulator & Diagnostics
    X_test = test_df[feature_cols]
    pred_df = pd.DataFrame(
        {
            "datetime": test_df[cfg.datetime_col],
            "actual_mw": np.round(test_df[cfg.target_col].to_numpy(), 2),
            best_name: np.round(best_model.predict(X_test), 2),
            "RidgeCV": np.round(ridge_pipe.predict(X_test), 2),
            "SeasonalNaive_24h": np.round(X_test["lag_24"].to_numpy(), 2),
            "SeasonalNaive_168h": np.round(X_test["lag_168"].to_numpy(), 2),
        }
    )
    pred_df.to_csv(bundle_dir / "holdout_predictions.csv.gz", index=False, compression="gzip")

    # Copy lightweight CSV/JSON reports
    for fname in [
        "holdout_metrics.csv",
        "cv_summary.csv",
        "feature_importance.csv",
        "monthly_summary.csv",
        "annual_summary.csv",
        "run_metadata.json",
    ]:
        src_file = cfg.reports_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, bundle_dir / fname)

    # Write manifest with exact holdout boundary
    manifest = {
        "best_model_name": best_name,
        "source_dataset": best_artifact.get("source_label", "PJME_hourly.csv"),
        "holdout_start": str(holdout_start_ts),
        "holdout_end": str(test_df[cfg.datetime_col].iloc[-1]),
        "holdout_hours": int(len(test_df)),
        "slice_hours": int(len(slice_df)),
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_kb = sum(p.stat().st_size for p in bundle_dir.iterdir() if p.is_file()) / 1024.0
    print(f"[bundle] Exported deployable bundle -> {bundle_dir} ({total_kb:,.1f} KB total)")
    return bundle_dir


if __name__ == "__main__":
    export_bundle()
