from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ProjectConfig
from src.dashboard_utils import (
    find_stress_regimes,
    parse_and_clean_uploaded_csv,
    recursive_forecast_fast,
    run_backtest_comparison,
    simulate_grid_operations,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
_FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "favicon.png"

st.set_page_config(
    page_title="PJM Grid Load Forecasting Console",
    page_icon=str(_FAVICON_PATH) if _FAVICON_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 3.6rem; padding-bottom: 2.2rem; max-width: 1320px; }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.5rem 1.75rem;
        border-radius: 12px;
        margin-top: 0.35rem;
        margin-bottom: 1.25rem;
        border: 1px solid #334155;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.35rem;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.15;
        letter-spacing: -0.015em;
    }
    .hero-title { font-size: 1.65rem; font-weight: 700; margin: 0 0 0.3rem 0; letter-spacing: -0.01em; }
    .hero-sub { font-size: 0.95rem; color: #cbd5e1; margin: 0; }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.65rem;
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(147, 197, 253, 0.35);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.18);
        color: #6ee7b7;
        border: 1px solid rgba(110, 231, 183, 0.35);
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Artifact Loading (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained forecasting models & holdout data...")
def load_console_assets() -> Dict[str, Any]:
    config = ProjectConfig()
    bundle_dir = config.root_dir / "demo_bundle"

    if (bundle_dir / "best_model.joblib").exists() and (bundle_dir / "holdout_slice.csv.gz").exists():
        best_artifact = joblib.load(bundle_dir / "best_model.joblib")
        ridge_path = bundle_dir / "ridge_model.joblib"
        ridge_artifact = joblib.load(ridge_path) if ridge_path.exists() else None

        slice_df = pd.read_csv(bundle_dir / "holdout_slice.csv.gz")
        slice_df[config.datetime_col] = pd.to_datetime(slice_df[config.datetime_col])

        preds_df = pd.read_csv(bundle_dir / "holdout_predictions.csv.gz")
        preds_df["datetime"] = pd.to_datetime(preds_df["datetime"])

        holdout_metrics = pd.read_csv(bundle_dir / "holdout_metrics.csv")
        feat_imp = pd.read_csv(bundle_dir / "feature_importance.csv")
        manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
        metadata_path = bundle_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

        return {
            "ready": True,
            "mode": "bundle",
            "config": config,
            "best_artifact": best_artifact,
            "ridge_artifact": ridge_artifact,
            "slice_df": slice_df,
            "preds_df": preds_df,
            "holdout_metrics": holdout_metrics,
            "feature_importance": feat_imp,
            "manifest": manifest,
            "metadata": metadata,
        }

    return {"ready": False, "config": config}


def _build_synthetic_preview(config: ProjectConfig) -> Dict[str, Any]:
    """Optional fallback if someone launches app.py before running `export_demo_bundle`."""
    from src.data_loader import generate_synthetic_grid_data
    from src.features import build_features
    from src.sql_layer import run_sql_prep
    from src.train import run_training_and_evaluation
    from src.export_demo_bundle import export_bundle

    raw = generate_synthetic_grid_data(years=3, seed=42)
    raw_std = pd.DataFrame(
        {
            "datetime_str": pd.to_datetime(raw["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "mw": raw["PJME_MW"],
        }
    )
    clean_df, _, engine_used = run_sql_prep(raw_std, config, engine="auto")
    feat_df, feat_cols = build_features(clean_df, config, drop_na=True)
    run_training_and_evaluation(feat_df, feat_cols, config, "SYNTHETIC_PREVIEW", engine_used)
    export_bundle(config)
    st.cache_resource.clear()
    return load_console_assets()


assets = load_console_assets()

if not assets.get("ready"):
    st.warning(
        "**Deployable model bundle (`demo_bundle/`) not found yet.** "
        "Run these two commands locally on your real PJM dataset first:"
    )
    st.code("python run_pipeline.py\npython -m src.export_demo_bundle", language="bash")
    if st.button("Generate Quick Synthetic Preview (for UI testing only)", type="primary"):
        with st.spinner("Building synthetic preview bundle (~20s)..."):
            assets = _build_synthetic_preview(assets["config"])
            st.rerun()
    st.stop()


config: ProjectConfig = assets["config"]
best_artifact: Dict[str, Any] = assets["best_artifact"]
ridge_artifact: Dict[str, Any] | None = assets["ridge_artifact"]
slice_df: pd.DataFrame = assets["slice_df"]
preds_df: pd.DataFrame = assets["preds_df"]
holdout_metrics: pd.DataFrame = assets["holdout_metrics"]
feat_imp: pd.DataFrame = assets["feature_importance"]
manifest: Dict[str, Any] = assets["manifest"]
metadata: Dict[str, Any] = assets["metadata"]

best_name: str = manifest["best_model_name"]
source_dataset: str = manifest.get("source_dataset", "PJME_hourly.csv")
holdout_start = pd.Timestamp(manifest["holdout_start"])
holdout_end = pd.Timestamp(manifest["holdout_end"])
holdout_only_df = slice_df[slice_df[config.datetime_col] >= holdout_start].reset_index(drop=True)

if "SYNTHETIC" in source_dataset.upper():
    st.warning(
        "Currently running on **synthetic preview data**. Run `python run_pipeline.py` and "
        "`python -m src.export_demo_bundle` locally to deploy real PJM East (`PJME_hourly.csv`) benchmarks."
    )

# ---------------------------------------------------------------------------
# Sidebar Overview
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <svg width="34" height="34" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Grid Forecast Logo">
                <rect x="4" y="4" width="92" height="92" rx="22" fill="#0f172a" stroke="#334155" stroke-width="4"/>
                <path d="M20 32H80M20 52H80M20 72H80M34 20V80M54 20V80M74 20V80" stroke="#1e293b" stroke-width="3"/>
                <path d="M18 66L34 52L46 60L58 32" stroke="#38bdf8" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M58 32L70 44L82 26" stroke="#10b981" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="58" cy="32" r="5" fill="#f8fafc"/>
                <circle cx="82" cy="26" r="5.5" fill="#10b981"/>
            </svg>
            <div class="brand-title">Grid Forecast<br/>Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Interactive inference & decision-support layer over the Hourly Energy Demand Forecasting v1 pipeline.")
    st.divider()
    st.markdown("**Active Pipeline Spec**")
    st.markdown(
        f"- **Dataset:** `{source_dataset}`\n"
        f"- **SQL Layer:** `{metadata.get('sql_engine', 'duckdb')}`\n"
        f"- **Selected Model:** `{best_name}`\n"
        f"- **Features:** `31` strictly causal\n"
        f"- **Holdout Window:** `{holdout_start.strftime('%Y-%m-%d')}` to `{holdout_end.strftime('%Y-%m-%d')}` (`{manifest['holdout_hours']:,}h`)"
    )
    st.divider()
    st.markdown("**Temporal Hygiene Guarantees**")
    st.markdown(
        "1. Zero random shuffles (`TimeSeriesSplit`)\n"
        "2. All lags $k \\ge 1$; `.shift(1)` before rolling stats\n"
        "3. `StandardScaler` fit strictly inside CV folds\n"
        "4. Unit-tested empirical zero target leakage"
    )

# ---------------------------------------------------------------------------
# Hero Header + Executive KPI Strip
# ---------------------------------------------------------------------------
best_row = holdout_metrics.loc[holdout_metrics["model"] == best_name].iloc[0]
ridge_row = (
    holdout_metrics.loc[holdout_metrics["model"] == "RidgeCV"].iloc[0]
    if "RidgeCV" in holdout_metrics["model"].values
    else None
)
naive_row = (
    holdout_metrics.loc[holdout_metrics["model"] == "SeasonalNaive_168h"].iloc[0]
    if "SeasonalNaive_168h" in holdout_metrics["model"].values
    else None
)

vs_ridge_pct = (
    (1.0 - float(best_row["RMSE_MW"]) / float(ridge_row["RMSE_MW"])) * 100.0
    if ridge_row is not None
    else 0.0
)
vs_naive_pct = (
    (1.0 - float(best_row["RMSE_MW"]) / float(naive_row["RMSE_MW"])) * 100.0
    if naive_row is not None
    else 0.0
)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">Hourly Energy Demand Forecasting — Grid Operations Console</div>
        <div class="hero-sub">
            Live multi-step recursive inference, regime stress-testing, and asymmetric dispatch cost simulation
            backed by CV-selected <b>{best_name}</b> on PJM East Interconnection load data.
        </div>
        <div>
            <span class="badge badge-green">Live {best_name} Inference</span>
            <span class="badge">Dataset: {source_dataset}</span>
            <span class="badge">5-Fold Expanding TimeSeriesSplit</span>
            <span class="badge">Untouched {manifest['holdout_hours']:,}h Holdout</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    f"{best_name} 1h Holdout RMSE",
    f"{best_row['RMSE_MW']:,.2f} MW",
    delta=f"-{vs_ridge_pct:.1f}% vs. RidgeCV",
    delta_color="inverse",
)
k2.metric(
    "1h Holdout MAPE / WAPE",
    f"{best_row['MAPE_pct']:.2f}% / {best_row['WAPE_pct']:.2f}%",
    delta=f"MAE: {best_row['MAE_MW']:,.1f} MW",
    delta_color="off",
)
k3.metric(
    "Holdout R²",
    f"{best_row['R2']:.4f}",
    delta=f"-{vs_naive_pct:.1f}% RMSE vs. 168h Naive",
    delta_color="inverse",
)
k4.metric(
    "Causal Feature Set",
    f"{len(best_artifact['feature_cols'])} Features",
    delta="Zero-Leakage Verified",
    delta_color="off",
)

st.write("")

# ---------------------------------------------------------------------------
# Main 4-Tab Interactive Workspace
# ---------------------------------------------------------------------------
tab_backtest, tab_risk, tab_bench, tab_custom = st.tabs(
    [
        "1. Live Time-Travel Backtest (1h–72h)",
        "2. Grid Risk & Reserve Simulator",
        "3. Leaderboard & Diagnostics",
        "4. Custom CSV Forecaster",
    ]
)

# ===========================================================================
# TAB 1: LIVE TIME-TRAVEL BACKTEST (RECURSIVE MULTI-STEP)
# ===========================================================================
with tab_backtest:
    st.subheader("Live Multi-Step Recursive Backtest on Untouched Holdout")
    st.caption(
        "Pick any cutoff hour inside the 1-year holdout (or jump to an auto-detected stress regime). "
        "The app runs live recursive inference without peeking at future actuals—revealing how error compounds across 1–72 hours."
    )

    regimes = find_stress_regimes(holdout_only_df, config)
    regime_names = list(regimes.keys())

    if "selected_cutoff_ts" not in st.session_state:
        st.session_state["selected_cutoff_ts"] = regimes[regime_names[0]]["cutoff_ts"]

    preset_cols = st.columns(len(regime_names))
    for col, rname in zip(preset_cols, regime_names):
        if col.button(rname, use_container_width=True, help=regimes[rname]["description"]):
            st.session_state["selected_cutoff_ts"] = regimes[rname]["cutoff_ts"]

    min_date = holdout_start.date()
    max_date = (holdout_end - pd.Timedelta(hours=72)).date()
    current_cutoff = pd.Timestamp(st.session_state["selected_cutoff_ts"])
    current_cutoff = min(max(current_cutoff, holdout_start), holdout_end - pd.Timedelta(hours=72))

    c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.6, 1.5])
    chosen_date = c1.date_input(
        "Forecast Origin Date (Cutoff)",
        value=current_cutoff.date(),
        min_value=min_date,
        max_value=max_date,
    )
    chosen_hour = c2.selectbox("Origin Hour (0–23)", options=list(range(24)), index=int(current_cutoff.hour))
    horizon_hours = c3.slider("Recursive Forecast Horizon (Hours)", min_value=6, max_value=72, value=36, step=6)
    show_models = c4.multiselect(
        "Models to Overlay",
        options=[best_name, "RidgeCV", "SeasonalNaive_168h", "SeasonalNaive_24h"],
        default=[best_name, "RidgeCV", "SeasonalNaive_168h"],
    )

    cutoff_ts = pd.Timestamp(chosen_date) + pd.Timedelta(hours=int(chosen_hour))
    st.session_state["selected_cutoff_ts"] = cutoff_ts

    comp_df, win_metrics, curve_df = run_backtest_comparison(
        slice_df=slice_df,
        cutoff_ts=cutoff_ts,
        horizon_hours=horizon_hours,
        best_artifact=best_artifact,
        ridge_artifact=ridge_artifact,
        config=config,
    )

    # Past 48h context before cutoff
    context_df = slice_df[
        (slice_df[config.datetime_col] > cutoff_ts - pd.Timedelta(hours=48))
        & (slice_df[config.datetime_col] <= cutoff_ts)
    ]

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=context_df[config.datetime_col],
            y=context_df[config.target_col],
            mode="lines",
            name="Historical Load (Past 48h)",
            line=dict(color="#64748b", width=2),
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=comp_df["datetime"],
            y=comp_df["actual_mw"],
            mode="lines",
            name="Actual Future Load (Ground Truth)",
            line=dict(color="#0f172a", width=2.8),
        )
    )

    palette = {
        best_name: ("#10b981", "solid", 2.6),
        "RidgeCV": ("#f59e0b", "dash", 2.2),
        "SeasonalNaive_168h": ("#94a3b8", "dot", 1.9),
        "SeasonalNaive_24h": ("#8b5cf6", "dashdot", 1.8),
    }
    for m_col in show_models:
        if m_col in comp_df.columns:
            color, dash, width = palette.get(m_col, ("#2563eb", "solid", 2.0))
            fig_bt.add_trace(
                go.Scatter(
                    x=comp_df["datetime"],
                    y=comp_df[m_col],
                    mode="lines",
                    name=f"{m_col} (Recursive)",
                    line=dict(color=color, dash=dash, width=width),
                )
            )

    fig_bt.add_vline(x=cutoff_ts, line_width=1.5, line_dash="dash", line_color="#ef4444")
    fig_bt.add_annotation(
        x=cutoff_ts,
        y=1.02,
        yref="paper",
        text=f"Forecast Origin ({cutoff_ts.strftime('%Y-%m-%d %H:%M')})",
        showarrow=False,
        font=dict(color="#ef4444", size=11),
    )
    fig_bt.update_layout(
        title=f"Live {horizon_hours}-Hour Recursive Forecast vs. Actual PJM Load",
        xaxis_title="Timestamp",
        yaxis_title="Grid Load (MW)",
        hovermode="x unified",
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    col_tbl, col_drift = st.columns([1.05, 1.25])
    with col_tbl:
        st.markdown(f"**Window Error Leaderboard (Steps 1–{horizon_hours}h)**")
        st.dataframe(
            win_metrics[["model", "RMSE_MW", "MAE_MW", "MAPE_pct", "WAPE_pct"]].rename(
                columns={
                    "model": "Model",
                    "RMSE_MW": "RMSE (MW)",
                    "MAE_MW": "MAE (MW)",
                    "MAPE_pct": "MAPE (%)",
                    "WAPE_pct": "WAPE (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "Notice how recursive forecasts feed each predicted hour back into `lag_1`..`lag_3` for subsequent steps. "
            "Tree ensembles preserve non-linear diurnal & weekend interactions better than linear models as the horizon stretches."
        )

    with col_drift:
        st.markdown("**Cumulative RMSE Compounding by Horizon Step**")
        fig_drift = go.Figure()
        for m_col in show_models:
            col_rmse = f"{m_col}_RMSE"
            if col_rmse in curve_df.columns:
                color, dash, width = palette.get(m_col, ("#2563eb", "solid", 2.0))
                fig_drift.add_trace(
                    go.Scatter(
                        x=curve_df["horizon_step"],
                        y=curve_df[col_rmse],
                        mode="lines+markers",
                        name=m_col,
                        line=dict(color=color, dash=dash, width=width),
                        marker=dict(size=4),
                    )
                )
        fig_drift.update_layout(
            xaxis_title="Forecast Horizon Step (Hours Ahead)",
            yaxis_title="Cumulative RMSE (MW)",
            hovermode="x unified",
            height=300,
            margin=dict(l=15, r=15, t=15, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_drift, use_container_width=True)


# ===========================================================================
# TAB 2: GRID RISK & RESERVE SIMULATOR (BUSINESS IMPACT)
# ===========================================================================
with tab_risk:
    st.subheader("Asymmetric Grid Dispatch Cost & Reserve Margin Simulator")
    st.caption(
        "Under-forecasting electricity demand forces expensive emergency peaker dispatch or outage risk, "
        "while over-forecasting wastes spinning reserve capacity. Adjust illustrative dispatch economics below to quantify model impact over the 1-year holdout."
    )

    s1, s2, s3, s4, s5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    baseline_choice = s1.selectbox("Compare Against Baseline", options=["RidgeCV", "SeasonalNaive_168h", "SeasonalNaive_24h"])
    season_filter = s2.selectbox("Season Scope", options=["Full Year (8,760h)", "Summer (Jun–Aug)", "Winter (Dec–Feb)", "Shoulder (Mar–May, Sep–Nov)"])
    reserve_mw = s3.slider("Spinning Reserve Buffer (MW)", min_value=0, max_value=2000, value=400, step=50)
    under_cost = s4.number_input("Under-Forecast Peaker Penalty ($/MWh)", min_value=20, max_value=1000, value=180, step=10)
    over_cost = s5.number_input("Over-Forecast Reserve Cost ($/MWh)", min_value=5, max_value=300, value=35, step=5)

    p95_load = float(np.percentile(preds_df["actual_mw"], 95))
    cap_threshold = st.slider(
        "Peak Capacity Alert Threshold (MW) — Default is 95th Percentile Load",
        min_value=int(preds_df["actual_mw"].min()),
        max_value=int(preds_df["actual_mw"].max()),
        value=int(round(p95_load, -2)),
        step=500,
    )

    sim_df = preds_df.copy()
    months = sim_df["datetime"].dt.month
    if season_filter.startswith("Summer"):
        sim_df = sim_df[months.isin([6, 7, 8])]
    elif season_filter.startswith("Winter"):
        sim_df = sim_df[months.isin([12, 1, 2])]
    elif season_filter.startswith("Shoulder"):
        sim_df = sim_df[months.isin([3, 4, 5, 9, 10, 11])]

    sim = simulate_grid_operations(
        pred_df=sim_df,
        actual_col="actual_mw",
        model_pred_col=best_name,
        baseline_pred_col=baseline_choice,
        reserve_buffer_mw=float(reserve_mw),
        under_cost_per_mwh=float(under_cost),
        over_cost_per_mwh=float(over_cost),
        capacity_alert_mw=float(cap_threshold),
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Illustrative Dispatch Cost Saved",
        f"${sim['cost_saved_usd']:,.0f}",
        delta=f"{sim['cost_saved_pct']:.1f}% lower than {baseline_choice}",
    )
    r2.metric(
        "Shortfall Risk Hours Avoided",
        f"{sim['shortfall_hours_avoided']:,} hrs",
        delta=f"{sim['model']['shortfall_hours']} vs. {sim['baseline']['shortfall_hours']} hrs",
    )
    r3.metric(
        "Emergency Peaker Energy Avoided",
        f"{sim['shortfall_mwh_avoided']:,.0f} MWh",
        delta=f"{best_name}: {sim['model']['shortfall_mwh']:,.0f} MWh",
        delta_color="off",
    )
    r4.metric(
        "Peak Capacity Alert F1 Score",
        f"{sim['model']['peak_f1']:.1f}%",
        delta=f"{baseline_choice}: {sim['baseline']['peak_f1']:.1f}% (Recall {sim['model']['peak_recall']:.1f}%)",
        delta_color="off",
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        fig_cost = go.Figure(
            data=[
                go.Bar(
                    name="Emergency Shortfall Cost ($)",
                    x=[best_name, baseline_choice],
                    y=[sim["model"]["under_cost_usd"], sim["baseline"]["under_cost_usd"]],
                    marker_color="#ef4444",
                ),
                go.Bar(
                    name="Excess Spinning Reserve Cost ($)",
                    x=[best_name, baseline_choice],
                    y=[sim["model"]["over_cost_usd"], sim["baseline"]["over_cost_usd"]],
                    marker_color="#3b82f6",
                ),
            ]
        )
        fig_cost.update_layout(
            barmode="stack",
            title=f"Illustrative Total Dispatch Imbalance Cost ({season_filter})",
            yaxis_title="USD ($)",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with rc2:
        hr_df = sim["hourly_risk_df"]
        fig_hr = go.Figure()
        fig_hr.add_trace(
            go.Bar(
                x=hr_df["hour"],
                y=hr_df["baseline_shortfall_hr"],
                name=f"{baseline_choice} Shortfall Hours",
                marker_color="#f59e0b",
                opacity=0.85,
            )
        )
        fig_hr.add_trace(
            go.Bar(
                x=hr_df["hour"],
                y=hr_df["model_shortfall_hr"],
                name=f"{best_name} Shortfall Hours",
                marker_color="#10b981",
            )
        )
        fig_hr.update_layout(
            barmode="group",
            title="Shortfall Risk Hours by Hour of Day (0–23)",
            xaxis_title="Hour of Day",
            yaxis_title="Hours Exceeding Committed Capacity",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    st.caption(
        "Note: Dollar figures use user-adjustable illustrative grid imbalance cost parameters to demonstrate how RMSE gains "
        "translate to asymmetric dispatch risk—not official PJM Locational Marginal Pricing (LMP) tariffs."
    )


# ===========================================================================
# TAB 3: MODEL LEADERBOARD & DIAGNOSTICS
# ===========================================================================
with tab_bench:
    st.subheader("Full Model Benchmark & Residual Diagnostics")
    lb_col, chart_col = st.columns([1.15, 1.15])

    with lb_col:
        st.markdown("**1-Year Untouched Holdout Leaderboard (1h-Ahead)**")
        st.dataframe(
            holdout_metrics.rename(
                columns={
                    "model": "Model",
                    "RMSE_MW": "RMSE (MW)",
                    "MAE_MW": "MAE (MW)",
                    "MAPE_pct": "MAPE (%)",
                    "WAPE_pct": "WAPE (%)",
                    "sMAPE_pct": "sMAPE (%)",
                    "R2": "R²",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with chart_col:
        sorted_hm = holdout_metrics.sort_values("RMSE_MW", ascending=True)
        bar_colors = ["#10b981" if m == best_name else "#475569" for m in sorted_hm["model"]]
        fig_lb = go.Figure(
            go.Bar(
                x=sorted_hm["RMSE_MW"],
                y=sorted_hm["model"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:,.1f} MW ({m:.2f}%)" for v, m in zip(sorted_hm["RMSE_MW"], sorted_hm["MAPE_pct"])],
                textposition="outside",
            )
        )
        fig_lb.update_layout(
            title="Holdout RMSE (MW) — Lower is Better",
            xaxis_title="RMSE (MW)",
            yaxis=dict(autorange="reversed"),
            height=330,
            margin=dict(l=15, r=90, t=45, b=20),
        )
        st.plotly_chart(fig_lb, use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        top_fi = feat_imp.head(15).iloc[::-1]
        fig_fi = go.Figure(
            go.Bar(
                x=top_fi["importance"],
                y=top_fi["feature"],
                orientation="h",
                marker_color="#2563eb",
            )
        )
        fig_fi.update_layout(
            title=f"Top 15 Feature Importances — {best_name}",
            xaxis_title="Importance Score",
            height=400,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with d2:
        res_df = preds_df.copy()
        res_df["hour"] = res_df["datetime"].dt.hour
        res_df["abs_err_best"] = (res_df["actual_mw"] - res_df[best_name]).abs()
        res_df["abs_err_ridge"] = (res_df["actual_mw"] - res_df["RidgeCV"]).abs()
        by_hr = res_df.groupby("hour", as_index=False)[["abs_err_best", "abs_err_ridge"]].mean()

        fig_res = go.Figure()
        fig_res.add_trace(
            go.Scatter(
                x=by_hr["hour"],
                y=by_hr["abs_err_ridge"],
                mode="lines+markers",
                name="RidgeCV MAE (MW)",
                line=dict(color="#f59e0b", width=2.3, dash="dash"),
            )
        )
        fig_res.add_trace(
            go.Scatter(
                x=by_hr["hour"],
                y=by_hr["abs_err_best"],
                mode="lines+markers",
                name=f"{best_name} MAE (MW)",
                line=dict(color="#10b981", width=2.6),
            )
        )
        fig_res.update_layout(
            title="Mean Absolute Error by Hour of Day (Morning Ramp & Evening Peak)",
            xaxis_title="Hour of Day (0–23)",
            yaxis_title="MAE (MW)",
            hovermode="x unified",
            height=400,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with st.expander("Leakage-Prevention & Temporal Validation Audit"):
        st.markdown(
            """
            - **SQL Deduplication (`sql/01_hourly_aggregations.sql`):** Aggregates Daylight Saving Time fall-back duplicate timestamps via `AVG(mw)` and reindexes onto a continuous 1-hour grid before feature creation.
            - **Strictly Causal Features (`src/features.py`):** Every lag uses $k \ge 1$, and rolling statistics apply `.shift(1)` before `.rolling(window)`.
            - **Automated Perturbation Test (`tests/test_pipeline.py`):** `test_zero_target_leakage_in_features` perturbs $y_t$ by $+999{,}999\text{ MW}$ and asserts feature vector $X_t$ and all earlier rows remain bit-for-bit identical.
            - **Zero Selection Bias (`src/train.py`):** Model selection relies strictly on 5-fold expanding-window `TimeSeriesSplit` CV RMSE (`2,160h` per fold); the final 8,760h (1-year) holdout is evaluated once.
            """
        )


# ===========================================================================
# TAB 4: CUSTOM CSV LIVE FORECASTER
# ===========================================================================
with tab_custom:
    st.subheader("Instant Custom Hourly Load Forecaster")
    st.caption(
        "Upload any 2-column hourly demand CSV (`timestamp, load_mw`) with at least 340 hourly rows (14+ days), "
        "or load a sample PJM holdout window to run live recursive forecasting and export predictions."
    )

    u1, u2 = st.columns([1.8, 1.0])
    uploaded = u1.file_uploader("Upload Hourly Demand CSV (2 columns: Datetime, MW)", type=["csv"])
    custom_horizon = u2.slider("Forecast Horizon (Hours)", min_value=6, max_value=72, value=24, step=6, key="custom_h")

    use_sample = st.button("Load Sample PJM Holdout Tail (400 Hours)", type="secondary")

    target_df = None
    label_src = ""
    if uploaded is not None:
        try:
            target_df = parse_and_clean_uploaded_csv(uploaded.getvalue(), config)
            label_src = uploaded.name
        except Exception as exc:
            st.error(f"Could not parse uploaded CSV: {exc}")
    elif use_sample or "sample_loaded" in st.session_state:
        if use_sample:
            st.session_state["sample_loaded"] = True
        target_df = slice_df.tail(400).copy().reset_index(drop=True)
        label_src = f"Sample tail of {source_dataset} (last 400 hours)"

    if target_df is not None:
        st.success(f"Loaded `{label_src}` ({len(target_df):,} clean hourly records ending `{target_df[config.datetime_col].iloc[-1]}`).")
        fcst_df = recursive_forecast_fast(target_df, best_artifact, config, horizon_hours=custom_horizon)

        recent = target_df.tail(72)
        fig_c = go.Figure()
        fig_c.add_trace(
            go.Scatter(
                x=recent[config.datetime_col],
                y=recent[config.target_col],
                mode="lines",
                name="Recent Observed Load (72h)",
                line=dict(color="#0f172a", width=2.4),
            )
        )
        fig_c.add_trace(
            go.Scatter(
                x=fcst_df["datetime"],
                y=fcst_df["predicted_mw"],
                mode="lines+markers",
                name=f"{best_name} Forecast (+{custom_horizon}h)",
                line=dict(color="#10b981", width=2.6),
                marker=dict(size=5),
            )
        )
        fig_c.update_layout(
            title=f"Next {custom_horizon}-Hour Recursive Electricity Load Forecast",
            xaxis_title="Timestamp",
            yaxis_title="Predicted Load (MW)",
            hovermode="x unified",
            height=390,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_c, use_container_width=True)

        dl_col, tbl_col = st.columns([1.0, 2.0])
        with dl_col:
            st.metric("Forecast Peak Load", f"{fcst_df['predicted_mw'].max():,.1f} MW")
            st.metric("Forecast Minimum Load", f"{fcst_df['predicted_mw'].min():,.1f} MW")
            st.download_button(
                label=f"Download {custom_horizon}h Forecast CSV",
                data=fcst_df.to_csv(index=False).encode("utf-8"),
                file_name=f"pjm_forecast_next_{custom_horizon}h.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with tbl_col:
            st.dataframe(fcst_df, use_container_width=True, hide_index=True, height=260)
