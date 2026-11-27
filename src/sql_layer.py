from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.config import ProjectConfig


def _split_sql_statements(sql_text: str) -> list[str]:
    statements = []
    for chunk in sql_text.split(";"):
        lines = [ln for ln in chunk.splitlines() if not ln.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            statements.append(cleaned)
    return statements


def run_sql_prep(
    raw_df: pd.DataFrame,
    config: ProjectConfig,
    engine: str = "auto",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], str]:
    """Execute SQL deduplication, aggregations, and window metrics via DuckDB or SQLite."""
    config.ensure_dirs()
    sql_path = config.sql_dir / "01_hourly_aggregations.sql"
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql_text)

    use_duckdb = False
    if engine in ("auto", "duckdb"):
        try:
            import duckdb  # type: ignore

            use_duckdb = True
        except ImportError:
            if engine == "duckdb":
                raise ImportError("duckdb is not installed. Run `pip install duckdb` or use --sql-engine sqlite.")
            use_duckdb = False

    if use_duckdb:
        import duckdb  # type: ignore

        db_path = config.processed_data_dir / "energy_demand.duckdb"
        if db_path.exists():
            db_path.unlink()
        con = duckdb.connect(str(db_path))
        con.register("raw_df_view", raw_df)
        con.execute("CREATE TABLE raw_hourly_demand AS SELECT * FROM raw_df_view")
        for stmt in statements:
            con.execute(stmt)

        clean_df = con.execute("SELECT datetime_str, mw FROM clean_hourly_demand ORDER BY datetime_str").df()
        summaries = {
            "monthly_summary": con.execute("SELECT * FROM monthly_summary").df(),
            "annual_summary": con.execute("SELECT * FROM annual_summary").df(),
            "sql_window_sample": con.execute("SELECT * FROM hourly_with_sql_metrics LIMIT 168").df(),
        }
        con.close()
        engine_used = "duckdb"
    else:
        db_path = config.processed_data_dir / "energy_demand.sqlite"
        if db_path.exists():
            db_path.unlink()
        con = sqlite3.connect(str(db_path))
        raw_df.to_sql("raw_hourly_demand", con, if_exists="replace", index=False)
        cur = con.cursor()
        for stmt in statements:
            cur.execute(stmt)
        con.commit()

        clean_df = pd.read_sql_query(
            "SELECT datetime_str, mw FROM clean_hourly_demand ORDER BY datetime_str", con
        )
        summaries = {
            "monthly_summary": pd.read_sql_query("SELECT * FROM monthly_summary", con),
            "annual_summary": pd.read_sql_query("SELECT * FROM annual_summary", con),
            "sql_window_sample": pd.read_sql_query("SELECT * FROM hourly_with_sql_metrics LIMIT 168", con),
        }
        con.close()
        engine_used = "sqlite3"

    # Reindex to a strict continuous hourly grid (handles DST spring-forward missing hours)
    clean_df["datetime"] = pd.to_datetime(clean_df["datetime_str"])
    clean_df = clean_df.drop(columns=["datetime_str"]).sort_values("datetime").set_index("datetime")

    full_index = pd.date_range(clean_df.index.min(), clean_df.index.max(), freq="h")
    missing_hours = int(len(full_index) - len(clean_df))
    clean_df = clean_df.reindex(full_index)
    clean_df.index.name = "datetime"

    # Causal-safe short gap interpolation (max 3h linear, else forward-fill)
    clean_df["mw"] = clean_df["mw"].interpolate(method="time", limit=3).ffill().bfill()
    clean_df = clean_df.reset_index()

    # Persist SQL summary artifacts
    for name, sdf in summaries.items():
        sdf.to_csv(config.reports_dir / f"{name}.csv", index=False)
    clean_df.to_csv(config.processed_data_dir / "clean_hourly_demand.csv", index=False)

    print(
        f"[sql:{engine_used}] Raw rows: {len(raw_df):,} -> Deduplicated & reindexed hourly rows: "
        f"{len(clean_df):,} (missing timestamps filled: {missing_hours})"
    )
    return clean_df, summaries, engine_used
