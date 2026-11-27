from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ProjectConfig
from src.data_loader import load_raw_dataframe
from src.features import build_features
from src.sql_layer import run_sql_prep
from src.train import run_training_and_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hourly Energy Demand Forecasting v1 — End-to-End Pipeline"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to local PJM CSV (e.g., data/raw/PJME_hourly.csv). If omitted, downloads automatically.",
    )
    parser.add_argument(
        "--sql-engine",
        choices=["auto", "duckdb", "sqlite"],
        default="auto",
        help="SQL engine for preparation and aggregation layer (default: auto -> duckdb else sqlite).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run offline smoke test on synthetic PJM-scale data (do not use for resume metrics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    config.ensure_dirs()

    print("=" * 72)
    print("Hourly Energy Demand Forecasting v1")
    print("=" * 72)

    raw_df, source_label = load_raw_dataframe(
        csv_path=args.csv,
        use_synthetic=args.synthetic,
        config=config,
    )
    clean_df, _, engine_used = run_sql_prep(raw_df, config, engine=args.sql_engine)
    feat_df, feature_cols = build_features(clean_df, config, drop_na=True)
    print(f"[features] Engineered {len(feature_cols)} leakage-free features over {len(feat_df):,} hourly rows")

    _, holdout_df, best_model = run_training_and_evaluation(
        feat_df=feat_df,
        feature_cols=feature_cols,
        config=config,
        source_label=source_label,
        sql_engine=engine_used,
    )

    print("\n" + "=" * 72)
    print(f"DONE | Best Model (by expanding-window CV): {best_model}")
    print("=" * 72)
    print(holdout_df.to_string(index=False))
    print(f"\nArtifacts saved under: {config.reports_dir} and {config.models_dir}")
    if args.synthetic:
        print(
            "\n[NOTE] You ran in --synthetic smoke-test mode. "
            "Run `python run_pipeline.py` (without --synthetic) to compute real PJM benchmarks."
        )


if __name__ == "__main__":
    main()
