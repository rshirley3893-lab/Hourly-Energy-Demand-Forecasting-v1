import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.data_loader import generate_synthetic_grid_data
from src.evaluate import compute_metrics
from src.features import build_features
from src.sql_layer import run_sql_prep


class TestEnergyForecastingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.config = ProjectConfig(root_dir=Path(self.tmp.name))
        # Copy SQL file into temporary project root
        sql_dst = self.config.sql_dir
        sql_dst.mkdir(parents=True, exist_ok=True)
        src_sql = Path(__file__).resolve().parent.parent / "sql" / "01_hourly_aggregations.sql"
        (sql_dst / "01_hourly_aggregations.sql").write_text(src_sql.read_text(encoding="utf-8"), encoding="utf-8")

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
        # Duplicate 01:00 averaged to 32,000; missing 02:00 linearly interpolated to 33,000
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

        # Perturb a single timestamp's target massively
        target_idx = 500
        clean_perturbed = clean.copy()
        clean_perturbed.loc[target_idx, "mw"] += 999_999.0

        feat_orig, feat_cols = build_features(clean, self.config, drop_na=False)
        feat_pert, _ = build_features(clean_perturbed, self.config, drop_na=False)

        # Row target_idx and all rows BEFORE target_idx must have identical features
        pd.testing.assert_frame_equal(
            feat_orig.iloc[: target_idx + 1][feat_cols],
            feat_pert.iloc[: target_idx + 1][feat_cols],
        )
        # Row target_idx + 1 SHOULD reflect the change in lag_1
        self.assertNotEqual(
            feat_orig.loc[target_idx + 1, "lag_1"],
            feat_pert.loc[target_idx + 1, "lag_1"],
        )

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
