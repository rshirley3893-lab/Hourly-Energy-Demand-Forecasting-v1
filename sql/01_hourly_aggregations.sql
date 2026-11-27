-- =============================================================================
-- Hourly Energy Demand Forecasting v1 — SQL Preparation & Profiling Layer
-- Compatible with both DuckDB and SQLite (standard ANSI SQL + window functions)
-- =============================================================================

-- 1. Deduplicate timestamps (handles Daylight Saving Time fall-back duplicates)
DROP TABLE IF EXISTS clean_hourly_demand;
CREATE TABLE clean_hourly_demand AS
SELECT
    datetime_str,
    AVG(mw) AS mw,
    COUNT(*) AS raw_record_count
FROM raw_hourly_demand
WHERE datetime_str IS NOT NULL
  AND mw IS NOT NULL
  AND mw > 0
GROUP BY datetime_str
ORDER BY datetime_str ASC;

-- 2. Monthly demand profile (average, peak, trough, and volatility proxy)
DROP TABLE IF EXISTS monthly_summary;
CREATE TABLE monthly_summary AS
SELECT
    SUBSTR(datetime_str, 1, 7) AS year_month,
    COUNT(*) AS hours_observed,
    ROUND(AVG(mw), 2) AS avg_mw,
    ROUND(MIN(mw), 2) AS min_mw,
    ROUND(MAX(mw), 2) AS max_mw,
    ROUND(MAX(mw) - MIN(mw), 2) AS peak_to_trough_mw
FROM clean_hourly_demand
GROUP BY SUBSTR(datetime_str, 1, 7)
ORDER BY year_month ASC;

-- 3. Annual demand & peak load trend
DROP TABLE IF EXISTS annual_summary;
CREATE TABLE annual_summary AS
SELECT
    SUBSTR(datetime_str, 1, 4) AS year,
    COUNT(*) AS hours_observed,
    ROUND(AVG(mw), 2) AS avg_mw,
    ROUND(MIN(mw), 2) AS min_mw,
    ROUND(MAX(mw), 2) AS max_mw
FROM clean_hourly_demand
GROUP BY SUBSTR(datetime_str, 1, 4)
ORDER BY year ASC;

-- 4. Hourly load curve with SQL window-function ramp rate & 24h rolling mean
DROP TABLE IF EXISTS hourly_with_sql_metrics;
CREATE TABLE hourly_with_sql_metrics AS
SELECT
    datetime_str,
    mw,
    mw - LAG(mw, 1) OVER (ORDER BY datetime_str) AS hourly_ramp_mw,
    AVG(mw) OVER (
        ORDER BY datetime_str
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    ) AS sql_prev_24h_avg_mw
FROM clean_hourly_demand
ORDER BY datetime_str ASC;
