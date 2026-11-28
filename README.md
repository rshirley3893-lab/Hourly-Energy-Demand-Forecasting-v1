# Hourly Energy Demand Forecasting & Grid Operations Console (v1)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)

End-to-end hourly electricity load forecasting pipeline and interactive **Grid Operations Console** built on **PJM East Interconnection (`PJME_hourly.csv`)** grid demand data. Engineered with strict temporal hygiene: DuckDB SQL deduplication & window profiling, 31 strictly causal features verified against target leakage, 5-fold expanding-window `TimeSeriesSplit` cross-validation, an untouched 8,760-hour (1-year) holdout, and live multi-step recursive inference.

## Benchmark Results (PJM East — Untouched 1-Year / 8,760h Holdout)

- **SQL Engine:** `duckdb`
- **Validation:** 5-fold expanding-window `TimeSeriesSplit` (2,160h/fold) + untouched 8,760h final holdout
- **Selected Model (by CV RMSE):** `LightGBM` (**39.2%** RMSE reduction vs. Linear/RidgeCV; **93.3%** vs. 168h Seasonal Naive)

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | WAPE (%) | R² |
|---|---:|---:|---:|---:|---:|
| **LightGBM (Selected)** | **324.49** | **245.17** | **0.775%** | **0.786%** | **0.99713** |
| XGBoost | 334.75 | 251.54 | 0.794% | 0.806% | 0.99695 |
| HistGradientBoosting | 355.14 | 268.41 | 0.848% | 0.860% | 0.99656 |
| LinearRegression | 533.52 | 400.30 | 1.270% | 1.283% | 0.99224 |
| RidgeCV | 533.58 | 400.38 | 1.271% | 1.283% | 0.99224 |
| SeasonalNaive_24h | 3,146.15 | 2,306.80 | 7.320% | 7.394% | 0.73019 |
| SeasonalNaive_168h | 4,852.33 | 3,581.81 | 11.218% | 11.481% | 0.35819 |

## Interactive Grid Operations Console (`app.py`)

Instead of static charts, `app.py` executes the real serialized **LightGBM** and **RidgeCV** models live across four workflows:
1. **Live Time-Travel Backtest (1h–72h Recursive):** Pick any date/hour inside the untouched 1-year holdout (or jump to auto-detected Summer Heatwave, Winter Cold Snap, Holiday, or Steepest Ramp regimes) and watch live recursive error compound across 1–72 hours without peeking at future actuals.
2. **Grid Risk & Reserve Simulator:** Adjust spinning reserve buffers (MW), asymmetric under-forecast peaker penalties vs. over-forecast reserve costs, and peak capacity thresholds to quantify shortfall hours avoided and peak-alert F1 scores.
3. **Leaderboard & Diagnostics:** Interactive model comparison, top-15 feature importances, hour-of-day residual profiles, and temporal hygiene audit.
4. **Custom CSV Live Forecaster:** Upload any 2-column hourly demand CSV (340+ rows) or load a sample PJM tail to generate and download a next-N-hour forecast CSV.

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
[4] Deployment Bundle & Live Console (src/export_demo_bundle.py, app.py)
    ├── Compresses trained models + 1-yr holdout slice into demo_bundle/
    └── Powers interactive Streamlit + Plotly Grid Operations Console
```

## Quickstart & Live Deployment

### 1. Install Dependencies (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Zero Leakage & Run Full Pipeline on Real PJM Data

```powershell
python -m unittest discover -s tests -v
python run_pipeline.py
```

### 3. Export Lightweight Cloud Bundle & Launch Console Locally

```powershell
# Packages compressed LightGBM + RidgeCV + 1-yr holdout slice into demo_bundle/ (~2-3 MB)
python -m src.export_demo_bundle

# Launch interactive Grid Operations Console
streamlit run app.py
```

### 4. Deploy Free to Streamlit Community Cloud (~2 Minutes)

1. Commit and push the repository (including `demo_bundle/`—already whitelisted in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your GitHub repo, branch `main`, and main file path `app.py`.
3. Click **Deploy**, copy your live URL, and replace `https://YOUR-APP-NAME.streamlit.app` in the badge at the top of this README.
