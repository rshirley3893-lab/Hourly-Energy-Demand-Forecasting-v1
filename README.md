# Hourly Energy Demand Forecasting (v1)

End-to-end hourly electricity load forecasting pipeline built on **PJM East Interconnection (`PJME_hourly.csv`)** grid demand data. Designed with strict temporal hygiene: SQL deduplication & window profiling, strictly causal feature engineering, expanding-window `TimeSeriesSplit` cross-validation, and an untouched 1-year holdout test set.

## Architecture

```text
Raw PJM Hourly CSV (MW)
        │
        ▼
[1] SQL Layer (DuckDB / SQLite — sql/01_hourly_aggregations.sql)
    ├── Deduplicate DST fall-back timestamps via AVG(mw)
    ├── Generate monthly, annual, & SQL window-function ramp tables
    └── Reindex onto continuous 1-hour grid (causal gap handling)
        │
        ▼
[2] Leakage-Free Feature Engineering (Pandas — src/features.py)
    ├── Calendar + US federal holiday indicators
    ├── Cyclical Fourier encodings (hour, day-of-week, day-of-year)
    ├── Causal lags: t-1, t-2, t-3, t-24, t-48, t-168, t-336
    └── Shifted rolling stats (shift(1) before 24h & 168h windows)
        │
        ▼
[3] Expanding-Window CV & Benchmarking (src/train.py)
    ├── Baselines: Seasonal Naive (t-24h, t-168h)
    ├── Linear: Linear Regression, RidgeCV (StandardScaler inside folds)
    ├── Gradient Boosting: XGBoost, LightGBM, HistGradientBoosting
    └── Model selected strictly by CV RMSE -> evaluated once on 1-yr holdout
        │
        ▼
[4] Diagnostics & Recursive Forecasting (src/plots.py, src/predict.py)
    ├── CV splits, model benchmark, 14-day forecast overlay, residual-by-hour
    └── CLI 24h recursive forecaster (models/best_model.joblib)
```

## Leakage-Prevention Checklist

- **Chronological partitioning only:** No random train/test shuffles anywhere in the codebase.
- **Strictly causal features:** Every lag uses $k \ge 1$, and rolling statistics apply `.shift(1)` before `.rolling(...)`. Verified automatically by `test_zero_target_leakage_in_features` in `tests/test_pipeline.py`.
- **In-fold scaling & hyperparameter selection:** `StandardScaler` and `RidgeCV` fit strictly inside each expanding CV fold.
- **Untouched holdout:** The final 8,760 hours (1 full year covering all four seasons) are held out until after model selection.

## Quickstart

### 1. Environment Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Unit Tests (Verify Zero Leakage & SQL Prep)

```powershell
python -m unittest discover -s tests -v
```

### 3. Run Full Pipeline on Real PJM Hourly Data

```powershell
# Auto-downloads PJME_hourly.csv, runs DuckDB/SQLite prep, 5-fold CV, holdout eval & plots
python run_pipeline.py
```

*(Optional)* If you downloaded `PJME_hourly.csv` manually from Kaggle, pass it directly:

```powershell
python run_pipeline.py --csv data/raw/PJME_hourly.csv
```

*(Optional)* Run a 24-hour recursive forecast using the saved winning model:

```powershell
python -m src.predict --horizon 24
```

## Generated Artifacts

After running `python run_pipeline.py`, check:
- `reports/RESULTS_SUMMARY.md` — Copy-paste Markdown benchmark table (RMSE, MAE, MAPE, WAPE, $R^2$)
- `reports/cv_fold_results.csv` & `reports/holdout_metrics.csv` — Full fold-by-fold and holdout metrics
- `reports/monthly_summary.csv` & `reports/annual_summary.csv` — SQL aggregation outputs
- `reports/figures/01_cv_splits.png` through `05_feature_importance.png` — Diagnostic plots
- `models/best_model.joblib` — Serialized winning model artifact

## Benchmark Results (PJM East — 1-Year Holdout)

- **Dataset:** `PJME_hourly.csv`
- **SQL Engine:** `duckdb`
- **Validation:** 5-fold expanding-window `TimeSeriesSplit` (2,160h/fold) + untouched 8,760h final holdout
- **Selected Model (by CV RMSE):** `LightGBM` (**93.31%** holdout RMSE reduction vs. 168h Seasonal Naive baseline)

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | WAPE (%) | R2 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 324.49 | 245.17 | 0.775% | 0.786% | 0.99713 |
| XGBoost | 334.75 | 251.54 | 0.794% | 0.806% | 0.99695 |
| HistGradientBoosting | 355.14 | 268.41 | 0.848% | 0.860% | 0.99656 |
| LinearRegression | 533.52 | 400.30 | 1.270% | 1.283% | 0.99224 |
| RidgeCV | 533.58 | 400.38 | 1.271% | 1.283% | 0.99224 |
| SeasonalNaive_24h | 3,146.15 | 2,306.80 | 7.320% | 7.394% | 0.73019 |
| SeasonalNaive_168h | 4,852.33 | 3,581.81 | 11.218% | 11.481% | 0.35819 |
