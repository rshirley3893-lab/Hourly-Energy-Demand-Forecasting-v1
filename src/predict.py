from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import ProjectConfig
from src.features import build_features


def recursive_forecast(
    history_df: pd.DataFrame,
    artifact: dict,
    config: ProjectConfig,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Generate multi-step recursive hourly forecasts using the saved best model."""
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # Keep only the recent tail needed to compute max lag (336h + buffer)
    work = history_df[[config.datetime_col, config.target_col]].copy()
    work[config.datetime_col] = pd.to_datetime(work[config.datetime_col])
    work = work.sort_values(config.datetime_col).tail(400).reset_index(drop=True)

    preds = []
    last_ts = work[config.datetime_col].iloc[-1]

    for step in range(1, horizon_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)
        new_row = pd.DataFrame({config.datetime_col: [next_ts], config.target_col: [float("nan")]})
        extended = pd.concat([work, new_row], ignore_index=True)

        feat_ext, _ = build_features(extended, config, drop_na=False)
        X_next = feat_ext.iloc[[-1]][feature_cols]
        y_hat = float(model.predict(X_next)[0])

        work = pd.concat(
            [work, pd.DataFrame({config.datetime_col: [next_ts], config.target_col: [y_hat]})],
            ignore_index=True,
        )
        preds.append({"datetime": next_ts, "predicted_mw": round(y_hat, 2)})

    return pd.DataFrame(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate next-N-hour demand forecast.")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours (default: 24)")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to saved joblib model artifact",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    model_path = args.model_path or (config.models_dir / "best_model.joblib")
    data_path = config.processed_data_dir / "clean_hourly_demand.csv"

    artifact = joblib.load(model_path)
    history_df = pd.read_csv(data_path)
    fcst = recursive_forecast(history_df, artifact, config, horizon_hours=args.horizon)

    out_path = config.reports_dir / f"forecast_next_{args.horizon}h.csv"
    fcst.to_csv(out_path, index=False)
    print(f"[predict] Model: {artifact['model_name']} | Saved {args.horizon}h forecast -> {out_path}")
    print(fcst.to_string(index=False))


if __name__ == "__main__":
    main()
