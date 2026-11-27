from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig


def download_pjme_csv(config: ProjectConfig, dest_path: Path | None = None) -> Path:
    """Download PJM East hourly load CSV (PJME_hourly.csv) from public mirrors."""
    config.ensure_dirs()
    target = dest_path or (config.raw_data_dir / "PJME_hourly.csv")
    if target.exists() and target.stat().st_size > 100_000:
        print(f"[data] Using existing raw file: {target}")
        return target

    last_err: Exception | None = None
    for url in config.pjm_mirrors:
        try:
            print(f"[data] Downloading PJME_hourly.csv from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 100_000:
                raise ValueError(f"Downloaded file unexpectedly small ({len(data)} bytes)")
            target.write_bytes(data)
            print(f"[data] Saved {len(data):,} bytes -> {target}")
            return target
        except Exception as exc:  # pragma: no cover
            print(f"[data] Mirror failed ({exc}); trying next mirror...")
            last_err = exc

    raise RuntimeError(
        "Could not download PJME_hourly.csv automatically. "
        "Download it manually from Kaggle (robikscube/hourly-energy-consumption) "
        f"and pass --csv path/to/PJME_hourly.csv. Last error: {last_err}"
    )


def generate_synthetic_grid_data(
    years: int = 5,
    seed: int = 42,
    start_date: str = "2014-01-01",
) -> pd.DataFrame:
    """Generate realistic PJM-scale synthetic hourly grid load for offline testing/CI.

    NOTE: Synthetic data is strictly for pipeline verification and unit tests.
    Always run on real PJM data (--download or --csv) before publishing benchmarks.
    """
    rng = np.random.default_rng(seed)
    periods = years * 365 * 24
    idx = pd.date_range(start=start_date, periods=periods, freq="h")

    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    # Realistic PJM-scale load (~32,000 MW base with dual summer/winter peaks)
    base = 31_000.0
    annual = 6_200.0 * np.cos(2 * np.pi * (doy - 205) / 365.25) + 3_500.0 * np.cos(
        4 * np.pi * (doy - 20) / 365.25
    )
    diurnal = (
        -4_800.0 * np.cos(2 * np.pi * (hour - 4) / 24.0)
        + 1_600.0 * np.sin(4 * np.pi * (hour - 7) / 24.0)
    )
    weekend_drop = np.where(dow >= 5, -2_800.0, 0.0)

    # Autocorrelated AR(1) weather-like shocks
    eps = rng.normal(0.0, 620.0, size=periods)
    ar_shock = np.zeros(periods)
    for t in range(1, periods):
        ar_shock[t] = 0.92 * ar_shock[t - 1] + eps[t]

    mw = np.clip(base + annual + diurnal + weekend_drop + ar_shock, 14_000.0, 62_000.0)
    df = pd.DataFrame({"Datetime": idx.strftime("%Y-%m-%d %H:%M:%S"), "PJME_MW": np.round(mw, 1)})

    # Inject 2 synthetic DST-style duplicate timestamps to exercise SQL deduplication
    dups = df.iloc[[500, 1500]].copy()
    dups["PJME_MW"] = dups["PJME_MW"] + 250.0
    return pd.concat([df, dups], ignore_index=True)


def load_raw_dataframe(
    csv_path: Path | None,
    use_synthetic: bool,
    config: ProjectConfig,
) -> Tuple[pd.DataFrame, str]:
    """Load raw hourly energy demand CSV into standardized columns (datetime_str, mw)."""
    if use_synthetic:
        raw = generate_synthetic_grid_data(years=5, seed=config.random_state)
        source_label = "SYNTHETIC_SMOKE_TEST"
    else:
        path = csv_path if csv_path is not None else download_pjme_csv(config)
        raw = pd.read_csv(path)
        source_label = path.name

    if raw.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns (timestamp, load), got {list(raw.columns)}")

    dt_col, val_col = raw.columns[0], raw.columns[1]
    standardized = pd.DataFrame(
        {
            "datetime_str": pd.to_datetime(raw[dt_col]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "mw": pd.to_numeric(raw[val_col], errors="coerce"),
        }
    )
    return standardized, source_label
