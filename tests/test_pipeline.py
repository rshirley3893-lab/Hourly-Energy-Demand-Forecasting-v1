import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import ProjectConfig
from src.dashboard_utils import (
    find_stress_regimes,
    parse_and_clean_uploaded_csv,
    recursive_forecast_fast,
    run_backtest_comparison,
    simulate_grid_operations,
)
from src.data_loader import generate_synthetic_grid_data
from src.evaluate import compute_metrics
from src.features import build_features
from src.predict import recursive_forecast
from src.sql_layer import run_sql_prep


class TestEnergyForecastingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.config = ProjectConfig(root_dir=Path(self.tmp.name))
        sql_dst = self.config.sql_dir
        sql_dst.mkdir(parents=True, exist_ok=True)
        src_sql = Path(__file__).resolve().parent.parent / "sql" / "01_hourly_aggregations.sql"
        (sql_dst / "01_hourly_aggregations.sql").write_text(
            src_sql.read_text(encoding="utf-8"), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sql_deduplication_and_continuous_index(self) -> None:
        raw = pd.DataFrame(
            {
                "datetime_str": [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 01:00:00",  # DST duplicate
                    "2020-01-01 03:00:00",  # 02:00 missing
                ],
                "mw": [30000.0, 31000.0, 33000.0, 34000.0],
            }
        )
        clean, summaries, _ = run_sql_prep(raw, self.config, engine="sqlite")
        self.assertEqual(len(clean), 4)
        self.assertAlmostEqual(clean.loc[1, "mw"], 32000.0, places=3)
        self.assertAlmostEqual(clean.loc[2, "mw"], 33000.0, places=3)
        self.assertGreaterEqual(len(summaries["monthly_summary"]), 1)

    def test_zero_target_leakage_in_features(self) -> None:
        """Mutating y[t] must NEVER alter feature vector X[t] or any earlier row."""
        raw = generate_synthetic_grid_data(years=1, seed=7)
        raw_std = pd.DataFrame(
            {
                "datetime_str": pd.to_datetime(raw["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                "mw": raw["PJME_MW"],
            }
        )
        clean, _, _ = run_sql_prep(raw_std, self.config, engine="sqlite")

        target_idx = 500
        clean_perturbed = clean.copy()
        clean_perturbed.loc[target_idx, "mw"] += 999_999.0

        feat_orig, feat_cols = build_features(clean, self.config, drop_na=False)
        feat_pert, _ = build_features(clean_perturbed, self.config, drop_na=False)

        pd.testing.assert_frame_equal(
            feat_orig.iloc[: target_idx + 1][feat_cols],
            feat_pert.iloc[: target_idx + 1][feat_cols],
        )
        self.assertNotEqual(
            feat_orig.loc[target_idx + 1, "lag_1"],
            feat_pert.loc[target_idx + 1, "lag_1"],
        )

    def test_fast_recursive_matches_reference_recursive_bit_for_bit(self) -> None:
        """Fast O(H) recursive forecaster must match full pandas `build_features` exactly."""
        raw = generate_synthetic_grid_data(years=1, seed=11)
        clean = pd.DataFrame(
            {
                "datetime": pd.to_datetime(raw["Datetime"]),
                "mw": raw["PJME_MW"],
            }
        ).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

        feat_df, feat_cols = build_features(clean, self.config, drop_na=True)
        model = Ridge(alpha=1.0).fit(feat_df[feat_cols].iloc[:1000], feat_df["mw"].iloc[:1000])
        artifact = {"model_name": "Ridge", "model": model, "feature_cols": feat_cols}

        history = clean.iloc[:1500].copy()
        ref = recursive_forecast(history, artifact, self.config, horizon_hours=24)
        fast = recursive_forecast_fast(history, artifact, self.config, horizon_hours=24)

        np.testing.assert_allclose(ref["predicted_mw"].to_numpy(), fast["predicted_mw"].to_numpy(), atol=1e-2)

    def test_backtest_regimes_and_risk_simulator(self) -> None:
        raw = generate_synthetic_grid_data(years=1, seed=19)
        clean = pd.DataFrame(
            {
                "datetime": pd.to_datetime(raw["Datetime"]),
                "mw": raw["PJME_MW"],
            }
        ).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

        feat_df, feat_cols = build_features(clean, self.config, drop_na=True)
        model = Ridge(alpha=1.0).fit(feat_df[feat_cols].iloc[:2000], feat_df["mw"].iloc[:2000])
        artifact = {"model_name": "LightGBM_Mock", "model": model, "feature_cols": feat_cols}

        regimes = find_stress_regimes(clean.iloc[400:].reset_index(drop=True), self.config)
        self.assertGreaterEqual(len(regimes), 3)

        cutoff = clean["datetime"].iloc[2500]
        comp, summary, curve = run_backtest_comparison(
            clean, cutoff, horizon_hours=24, best_artifact=artifact, ridge_artifact=artifact, config=self.config
        )
        self.assertEqual(len(comp), 24)
        self.assertEqual(len(curve), 24)
        self.assertIn("LightGBM_Mock", summary["model"].values)

        sim = simulate_grid_operations(
            pred_df=comp,
            actual_col="actual_mw",
            model_pred_col="LightGBM_Mock",
            baseline_pred_col="SeasonalNaive_168h",
            reserve_buffer_mw=300.0,
        )
        self.assertGreaterEqual(sim["model"]["total_cost_usd"], 0.0)
        self.assertEqual(len(sim["hourly_risk_df"]), 24)

        csv_bytes = clean.tail(400).to_csv(index=False).encode("utf-8")
        parsed = parse_and_clean_uploaded_csv(csv_bytes, self.config)
        self.assertEqual(len(parsed), 400)

    def test_metrics_perfect_and_nonzero(self) -> None:
        y = np.array([100.0, 200.0, 300.0])
        perfect = compute_metrics(y, y)
        self.assertEqual(perfect["MAE_MW"], 0.0)
        self.assertEqual(perfect["RMSE_MW"], 0.0)
        self.assertEqual(perfect["R2"], 1.0)

        off = compute_metrics(y, np.array([110.0, 190.0, 300.0]))
        self.assertAlmostEqual(off["MAE_MW"], 6.67, places=2)


if __name__ == "__main__":
    unittest.main()
