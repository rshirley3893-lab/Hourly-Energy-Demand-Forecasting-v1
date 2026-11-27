# Benchmark Summary

- **Dataset:** `PJME_hourly.csv`
- **SQL Engine:** `duckdb`
- **Validation:** 5-fold expanding-window `TimeSeriesSplit` (2,160h/fold) + untouched 8,760h final holdout
- **Selected Model (by CV RMSE):** `LightGBM` (**93.31%** holdout RMSE reduction vs. 168h Seasonal Naive baseline)

## Holdout Test Set Leaderboard

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | WAPE (%) | R2 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 324.49 | 245.17 | 0.775% | 0.786% | 0.99713 |
| XGBoost | 334.75 | 251.54 | 0.794% | 0.806% | 0.99695 |
| HistGradientBoosting | 355.14 | 268.41 | 0.848% | 0.860% | 0.99656 |
| LinearRegression | 533.52 | 400.30 | 1.270% | 1.283% | 0.99224 |
| RidgeCV | 533.58 | 400.38 | 1.271% | 1.283% | 0.99224 |
| SeasonalNaive_24h | 3,146.15 | 2,306.80 | 7.320% | 7.394% | 0.73019 |
| SeasonalNaive_168h | 4,852.33 | 3,581.81 | 11.218% | 11.481% | 0.35819 |
