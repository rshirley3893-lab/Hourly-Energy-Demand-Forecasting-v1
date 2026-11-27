from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    target_col: str = "mw"
    datetime_col: str = "datetime"
    random_state: int = 42

    # Validation strategy: final 1-year holdout + 5-fold expanding-window CV
    holdout_hours: int = 24 * 365
    cv_splits: int = 5
    cv_test_hours: int = 24 * 90  # 90-day validation block per fold

    # Feature engineering settings (all strictly causal / shifted >= 1)
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 24, 48, 168, 336])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])

    # Public mirrors for PJM East Hourly Energy Consumption (PJME_hourly.csv)
    pjm_mirrors: List[str] = field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv",
            "https://raw.githubusercontent.com/ArpitaSatsangi/Time-Series-Forecasting-for-Energy-Consumption/main/PJME_hourly.csv",
        ]
    )

    @property
    def raw_data_dir(self) -> Path:
        return self.root_dir / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def sql_dir(self) -> Path:
        return self.root_dir / "sql"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    def ensure_dirs(self) -> None:
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
