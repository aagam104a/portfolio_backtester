"""
Portfolio Backtesting & Crash Simulation Tool
──────────────────────────────────────────────
Author : Personal use only
Disclaimer: Educational backtesting tool. Not financial advice.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.backtester import run_sip_backtest
from src.crash_simulator import (
    CrashScenario,
    run_crash_simulation,
    replay_historical_crash,
)
from src.currency import prices_in_inr
from src.data_fetcher import build_price_matrix
from src.metrics import rolling_returns
from src.plotting import (
    allocation_pie,
    asset_contribution_bar,
    asset_lines_chart,
    crash_recovery_chart,
    drawdown_chart,
    monthly_returns_heatmap,
    portfolio_value_chart,
    rolling_returns_chart,
    strategy_comparison_chart,
)
from src.portfolio import compare_strategies, load_example_strategies
from src.utils import (
    ASSET_UNIVERSE,
    DEFAULT_RISK_FREE_RATE,
    DISCLAIMER,
    HISTORICAL_CRASHES,
    TICKER_META,
)

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 4px 0;
    border-left: 4px solid #4f8df5;
}
.metric-label { font-size: 0.78rem; color: #9ca3af; text-transform: uppercase; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #f0f0f0; }
.metric-sub   { font-size: 0.72rem; color: #6b7280; }
.good  { border-left-color: #5ec26b !important; }
.bad   { border-left-color: #e05252 !important; }
.warn  { border-left-color: #f5a44f !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_inr(v: float) -> str:
    if np.isnan(v):
        return "N/A"
    if abs(v) >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"


def _fmt_pct(v: float, decimals: int = 2) -> str:
    if np.isnan(v):
        return "N/A"
    return f"{v*100:.{decimals}f}%"


def _metric_card(label: str, value: str, sub: str = "", cls: str = "") -> None:
    st.markdown(
        f"""<div class="metric-card {cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _ticker_label(t: str) -> str:
    return TICKER_META.get(t, {}).get("label", t)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar: investment assumptions
# ══════════════════════════════════════════════════════════════════════════════

def sidebar() -> dict:
    with st.sidebar:
        st.title("⚙️ Investment Settings")

        st.subheader("📅 SIP Parameters")
        monthly_sip = st.number_input(
            "Monthly SIP (₹)", min_value=500, max_value=10_000_000,
            value=12_000, step=500,
        )
        sip_start = st.date_input(
            "SIP Start Month", value=date(2026, 7, 1),
            help="Only relevant for future SIP projections. Backtest uses historical data.",
        )
        backtest_years = st.slider(
            "Backtest Period (years)", 1, 15, 10,
        )
        end_date = date.today()
        start_date = date(end_date.year - backtest_years, end_date.month, 1)

        st.subheader("⚖️ Rebalancing")
        rebalance = st.selectbox(
            "Rebalancing Frequency",
            ["none", "monthly", "quarterly", "yearly"],
            index=0,
        )

        st.subheader("📊 Risk Parameters")
        risk_free_rate = st.number_input(
            "Risk-Free Rate (annual %)",
            min_value=0.0, max_value=20.0, value=7.0, step=0.5,
        ) / 100

        st.subheader("💸 Fees")
        transaction_fee = st.number_input(
            "Transaction Fee (%)", min_value=0.0, max_value=2.0,
            value=0.0, step=0.01, format="%.3f",
        ) / 100

        st.markdown("---")
        st.markdown(
            "<small>📌 Base currency: INR. USD assets converted using live FX data.</small>",
            unsafe_allow_html=True,
        )

    return {
        "monthly_sip": monthly_sip,
        "sip_start": sip_start,
        "backtest_years": backtest_years,
        "start_date": start_date,
        "end_date": end_date,
        "rebalance": rebalance,
        "risk_free_rate": risk_free_rate,
        "transaction_fee": transaction_fee,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Asset Selector + Allocation Editor
# ══════════════════════════════════════════════════════════════════════════════

def asset_allocation_section(settings: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Returns (allocations_pct, expense_ratios)."""

    st.header("🏦 Asset Allocation")

    # ── Load example strategy ────────────────────────────────────────────────
    strategies = load_example_strategies(ROOT / "configs" / "example_strategies.json")
    strategy_names = ["Custom"] + [s["name"] for s in strategies]
    chosen_strategy = st.selectbox("Load Example Strategy", strategy_names, index=0)

    preset_alloc: dict[str, float] = {}
    if chosen_strategy != "Custom":
        for s in strategies:
            if s["name"] == chosen_strategy:
                preset_alloc = s.get("allocations", {})
                break

    # ── Asset selector ───────────────────────────────────────────────────────
    st.subheader("Select Assets")
    selected_tickers: list[str] = []

    for category, assets in ASSET_UNIVERSE.items():
        with st.expander(f"📂 {category}", expanded=(category == "Indian Market")):
            cols = st.columns(2)
            for i, (ticker, meta) in enumerate(assets.items()):
                default_checked = ticker in preset_alloc
                checked = cols[i % 2].checkbox(
                    f"{meta['label']} ({ticker})",
                    value=default_checked,
                    key=f"asset_{ticker}",
                )
                if checked:
                    selected_tickers.append(ticker)

    # Custom ticker input
    st.subheader("➕ Add Custom Tickers")
    custom_raw = st.text_input(
        "Custom tickers (comma-separated, e.g. VTI, BABA, PLTR)",
        value="",
    )
    custom_tickers = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
    selected_tickers.extend(custom_tickers)
    selected_tickers = list(dict.fromkeys(selected_tickers))  # deduplicate, preserve order

    if not selected_tickers:
        st.warning("Please select at least one asset.")
        return {}, {}

    # ── Allocation editor ────────────────────────────────────────────────────
    st.subheader("📐 Set Allocation %")

    allocations: dict[str, float] = {}
    expense_ratios: dict[str, float] = {}

    with st.container():
        header_cols = st.columns([3, 2, 2])
        header_cols[0].markdown("**Asset**")
        header_cols[1].markdown("**Allocation %**")
        header_cols[2].markdown("**Expense Ratio %**")

        for ticker in selected_tickers:
            label = _ticker_label(ticker)
            default_alloc = float(preset_alloc.get(ticker, round(100.0 / len(selected_tickers), 1)))
            row = st.columns([3, 2, 2])
            row[0].markdown(f"`{ticker}` — {label}")
            alloc = row[1].number_input(
                f"Alloc {ticker}", min_value=0.0, max_value=100.0,
                value=default_alloc, step=0.5, label_visibility="collapsed",
                key=f"alloc_{ticker}",
            )
            exp_ratio = row[2].number_input(
                f"ER {ticker}", min_value=0.0, max_value=5.0,
                value=0.0, step=0.01, format="%.2f", label_visibility="collapsed",
                key=f"exp_{ticker}",
            )
            allocations[ticker] = alloc
            expense_ratios[ticker] = exp_ratio / 100

    # Allocation validation
    total_alloc = sum(allocations.values())
    if abs(total_alloc - 100.0) > 0.01:
        st.error(f"⚠️ Allocations sum to **{total_alloc:.1f}%** — must equal **100%**.")
    else:
        st.success(f"✅ Allocations sum to 100%")

    # Allocation pie preview
    if allocations and abs(total_alloc - 100.0) < 0.01:
        fig = allocation_pie(allocations, {t: _ticker_label(t) for t in allocations})
        st.plotly_chart(fig, use_container_width=True)

    return allocations, expense_ratios


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Section
# ══════════════════════════════════════════════════════════════════════════════

def backtest_section(settings: dict, allocations: dict, expense_ratios: dict) -> dict | None:
    """Runs backtest and displays results. Returns result dict or None."""

    st.header("🔬 Backtest Results")

    total_alloc = sum(allocations.values())
    if not allocations or abs(total_alloc - 100.0) > 0.01:
        st.info("Configure allocations summing to 100% and click Run Backtest.")
        return None

    run_btn = st.button("▶️ Run Backtest", type="primary", use_container_width=True)
    if not run_btn and "backtest_result" not in st.session_state:
        return None

    if run_btn or "backtest_result" not in st.session_state:
        tickers = list(allocations.keys())
        norm_alloc = {t: v / 100.0 for t, v in allocations.items()}

        # ── Fetch data ───────────────────────────────────────────────────────
        with st.spinner("📡 Fetching historical data..."):
            progress_bar = st.progress(0)
            fetch_warnings: dict[str, str] = {}

            def progress_cb(i, total, ticker):
                progress_bar.progress(int((i + 1) / total * 100))

            price_matrix, fetch_warnings = build_price_matrix(
                tickers,
                start=str(settings["start_date"]),
                end=str(settings["end_date"]),
                status_callback=progress_cb,
            )
            progress_bar.empty()

        if price_matrix.empty:
            st.error("No data could be fetched. Check tickers and date range.")
            return None

        # Show data warnings
        if fetch_warnings:
            for ticker, msg in fetch_warnings.items():
                st.warning(f"**{ticker}**: {msg}")

        # ── Convert to INR ───────────────────────────────────────────────────
        if "INR=X" not in price_matrix.columns:
            st.error("Could not fetch USD/INR exchange rate. Check internet connection.")
            return None

        usd_inr = price_matrix["INR=X"]
        prices_inr_df = prices_in_inr(price_matrix, TICKER_META, usd_inr)

        # ── Run backtest ─────────────────────────────────────────────────────
        with st.spinner("⚙️ Running SIP simulation..."):
            try:
                result = run_sip_backtest(
                    prices_inr=prices_inr_df,
                    allocations=norm_alloc,
                    monthly_sip_inr=settings["monthly_sip"],
                    expense_ratios=expense_ratios,
                    transaction_fee_pct=settings["transaction_fee"],
                    rebalance=settings["rebalance"],
                    risk_free_annual=settings["risk_free_rate"],
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                return None

        st.session_state["backtest_result"] = result
        st.session_state["usd_inr"] = usd_inr
        st.session_state["prices_inr_df"] = prices_inr_df

    result = st.session_state["backtest_result"]
    m = result.metrics

    # ── Metric Cards ─────────────────────────────────────────────────────────
    st.subheader("📊 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Final Portfolio Value", _fmt_inr(m["final_value"]), "", "good")
        _metric_card("Total Invested", _fmt_inr(m["total_invested"]))
    with c2:
        gain_cls = "good" if m["abs_gain"] >= 0 else "bad"
        _metric_card("Absolute Gain", _fmt_inr(m["abs_gain"]), "", gain_cls)
        _metric_card("Absolute Return", _fmt_pct(m["abs_return_pct"]))
    with c3:
        xirr_val = m["xirr"]
        _metric_card(
            "XIRR", _fmt_pct(xirr_val) if not np.isnan(xirr_val) else "N/A",
            "Most accurate SIP metric", "good" if (not np.isnan(xirr_val) and xirr_val > 0) else "warn",
        )
        _metric_card("CAGR", _fmt_pct(m["cagr"]))
    with c4:
        dd = m["max_drawdown"]
        _metric_card("Max Drawdown", _fmt_pct(-dd), f"Recovery: {m['recovery_months'] or 'Not yet'} months", "bad")
        _metric_card("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}" if not np.isnan(m['sharpe_ratio']) else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        _metric_card("Annualised Volatility", _fmt_pct(m["annualized_volatility"]))
    with c6:
        _metric_card("Best Month", _fmt_pct(m["best_month_return"]),
                     str(m["best_month_date"])[:7] if m["best_month_date"] else "", "good")
    with c7:
        _metric_card("Worst Month", _fmt_pct(m["worst_month_return"]),
                     str(m["worst_month_date"])[:7] if m["worst_month_date"] else "", "bad")
    with c8:
        n_months = len(result.portfolio_values)
        _metric_card("Backtest Length", f"{n_months} months", f"≈ {n_months//12} years")

    # ── Backtest warnings ────────────────────────────────────────────────────
    if result.warnings:
        with st.expander("⚠️ Backtest Warnings"):
            for w in result.warnings:
                st.warning(w)

    # ── Charts ───────────────────────────────────────────────────────────────
    st.subheader("📈 Charts")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Portfolio Value", "Drawdown", "Asset Breakdown",
        "Rolling Returns", "Monthly Heatmap", "Individual Assets"
    ])

    with tab1:
        fig = portfolio_value_chart(result.portfolio_values, result.monthly_invested)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = drawdown_chart(result.portfolio_values)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = allocation_pie(allocations, {t: _ticker_label(t) for t in allocations})
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = asset_contribution_bar(result.asset_contributions, {t: _ticker_label(t) for t in allocations})
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = rolling_returns_chart(result.portfolio_values)
        st.plotly_chart(fig, use_container_width=True)

        roll_df = pd.DataFrame({
            "1Y Rolling Return": rolling_returns(result.portfolio_values, 12),
            "3Y Rolling Return": rolling_returns(result.portfolio_values, 36),
            "5Y Rolling Return": rolling_returns(result.portfolio_values, 60),
        }).dropna(how="all") * 100
        if not roll_df.empty:
            st.caption("Rolling return stats (%)")
            st.dataframe(
                roll_df.describe().round(2).rename(
                    index={"mean": "Mean", "std": "Std Dev", "min": "Min", "max": "Max",
                           "50%": "Median"}
                ),
                use_container_width=True,
            )

    with tab5:
        fig = monthly_returns_heatmap(result.portfolio_values)
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        label_map = {t: _ticker_label(t) for t in result.asset_values.columns}
        fig = asset_lines_chart(result.asset_values, label_map)
        st.plotly_chart(fig, use_container_width=True)

    # ── CSV Export ───────────────────────────────────────────────────────────
    st.subheader("📥 Export")
    export_df = pd.DataFrame({
        "portfolio_value_inr": result.portfolio_values,
        "cumulative_invested_inr": result.monthly_invested.cumsum(),
    }).join(result.asset_values)

    csv_bytes = export_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download Portfolio Time-Series CSV",
        data=csv_bytes,
        file_name="portfolio_backtest.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Metrics CSV
    metrics_df = pd.DataFrame([m]).T.rename(columns={0: "value"})
    metrics_csv = metrics_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download Metrics Summary CSV",
        data=metrics_csv,
        file_name="portfolio_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )

    return {"result": result}


# ══════════════════════════════════════════════════════════════════════════════
# Crash Simulation Section
# ══════════════════════════════════════════════════════════════════════════════

def crash_simulation_section(settings: dict) -> None:
    st.header("💥 Crash Simulation")
    st.caption("Simulate a sudden market shock and estimate recovery time.")

    if "backtest_result" not in st.session_state:
        st.info("Run a backtest first to enable crash simulation.")
        return

    result = st.session_state["backtest_result"]
    last_asset_values = result.asset_values.iloc[-1].dropna()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Quick Drop Scenarios")
        global_drop = st.select_slider(
            "Portfolio-wide instant drop",
            options=[0, 10, 20, 30, 40, 50],
            value=30,
            format_func=lambda x: f"{x}%",
        )

        st.subheader("🎯 Asset-Class Custom Shocks")
        us_tech_drop = st.slider("US Tech drop (%)", 0, 80, 0)
        india_drop   = st.slider("Indian Equities drop (%)", 0, 60, 0)
        gold_change  = st.slider("Gold change (negative = drop) (%)", -50, 50, 0)
        silver_change = st.slider("Silver change (%)", -50, 50, 0)
        fx_change    = st.slider("INR depreciation vs USD (%)", -20, 30, 0)

        st.subheader("🔧 Recovery Assumptions")
        recovery_cagr = st.number_input("Recovery CAGR assumption (%)", 5.0, 30.0, 12.0, 1.0) / 100
        sip_paused = st.checkbox("Pause SIP during recovery", value=False)

    with col2:
        st.subheader("📌 Individual Stock Shocks")
        individual_drops: dict[str, float] = {}
        available_stocks = [t for t in last_asset_values.index if t in TICKER_META]
        if available_stocks:
            selected_stocks_for_shock = st.multiselect(
                "Select stocks to shock individually",
                available_stocks,
                format_func=_ticker_label,
            )
            for stk in selected_stocks_for_shock:
                d = st.slider(f"{_ticker_label(stk)} drop (%)", 0, 100, 20, key=f"shock_{stk}")
                individual_drops[stk] = float(d)

    run_crash = st.button("💥 Run Crash Simulation", type="primary", use_container_width=True)

    if run_crash:
        # Build asset-specific drops
        asset_drops: dict[str, float] = {**individual_drops}

        # US Tech
        us_tech_tickers = {"NVDA", "AMD", "MSFT", "AAPL", "GOOGL", "META", "AMZN",
                           "TSLA", "AVGO", "TSM", "MU", "ASML", "SMCI", "QCOM", "ARM",
                           "QQQ", "SOXX", "SMH"}
        if us_tech_drop > 0:
            for t in us_tech_tickers:
                asset_drops[t] = max(asset_drops.get(t, 0), us_tech_drop)

        # Indian equities
        india_tickers = {"^NSEI", "^BSESN", "NIFTYNXT50.NS", "NIFTYMIDCAP150.NS"}
        if india_drop > 0:
            for t in india_tickers:
                asset_drops[t] = max(asset_drops.get(t, 0), india_drop)

        # Gold / Silver (can rise too)
        for t, change in [("GLD", gold_change), ("SLV", silver_change)]:
            if change != 0:
                # Negative change = drop, positive = rise → we use "drop" as % to subtract
                # but allow negative drop (i.e., asset rises)
                asset_drops[t] = -change  # invert: positive slider = rise = negative drop

        scenario = CrashScenario(
            name=f"Custom Crash ({global_drop}% global)",
            global_drop_pct=float(global_drop),
            asset_drops=asset_drops,
            fx_change_pct=float(fx_change),
        )

        crash_result = run_crash_simulation(
            last_asset_values,
            scenario,
            monthly_sip_inr=settings["monthly_sip"],
            recovery_annual_cagr=recovery_cagr,
            sip_paused=sip_paused,
        )

        # ── Crash metrics ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💣 Crash Impact")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pre-Crash Value", _fmt_inr(crash_result.pre_crash_value))
        m2.metric("Post-Crash Value", _fmt_inr(crash_result.post_crash_value),
                  delta=f"-{_fmt_inr(crash_result.drawdown_amount)}", delta_color="inverse")
        m3.metric("Drawdown", f"-{crash_result.drawdown_pct:.1f}%")
        if crash_result.months_to_recover is not None:
            m4.metric("Months to Recover", crash_result.months_to_recover,
                      help="Based on your recovery CAGR assumption and SIP settings")
        else:
            m4.metric("Months to Recover", "Not recovered (within 20 years)")

        st.metric(
            "SIP Invested During Recovery" if not sip_paused else "SIP During Recovery",
            _fmt_inr(crash_result.invested_during_recovery) if not sip_paused else "₹0 (Paused)",
        )

        fig = crash_recovery_chart(
            crash_result.recovery_values,
            crash_result.pre_crash_value,
            crash_result.post_crash_value,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Historical Crash Replay
# ══════════════════════════════════════════════════════════════════════════════

def historical_crash_section() -> None:
    st.header("📜 Historical Crash Replay")
    st.caption("Replay major historical crashes against your backtested portfolio.")

    if "backtest_result" not in st.session_state:
        st.info("Run a backtest first to enable historical crash replay.")
        return

    result = st.session_state["backtest_result"]
    pv = result.portfolio_values

    for crash_name, crash_info in HISTORICAL_CRASHES.items():
        with st.expander(f"📉 {crash_name}"):
            st.caption(crash_info["description"])
            replay = replay_historical_crash(
                pv,
                crash_info["start"],
                crash_info["trough"],
                crash_info["end"],
            )
            if replay["data_available"]:
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Pre-Crash Value", _fmt_inr(replay["pre_crash_value"]))
                rc2.metric("Trough Value", _fmt_inr(replay["trough_value"]))
                rc3.metric("Max Drawdown", f"-{replay['drawdown_pct']:.1f}%")
                rc4.metric(
                    "Recovery",
                    f"{replay['recovery_months']} months" if replay["recovery_months"] else "Not yet",
                )
                st.caption(f"Trough date: {str(replay['trough_date'])[:10]}")
            else:
                st.warning(f"⚠️ {replay['reason']}")


# ══════════════════════════════════════════════════════════════════════════════
# Strategy Comparison Section
# ══════════════════════════════════════════════════════════════════════════════

def strategy_comparison_section(settings: dict) -> None:
    st.header("⚖️ Strategy Comparison")
    st.caption("Compare pre-built allocation strategies side by side using your backtest period.")

    run_compare = st.button(
        "🔁 Compare All Example Strategies", type="secondary", use_container_width=True
    )
    if not run_compare and "comparison_results" not in st.session_state:
        return

    if run_compare:
        strategies = load_example_strategies(ROOT / "configs" / "example_strategies.json")
        strategy_results = {}

        comparison_progress = st.progress(0)
        for i, strategy in enumerate(strategies):
            name = strategy["name"]
            raw_alloc = strategy.get("allocations", {})
            if not raw_alloc:
                continue

            tickers = list(raw_alloc.keys())
            norm_alloc = {t: v / sum(raw_alloc.values()) for t, v in raw_alloc.items()}

            with st.spinner(f"Fetching data for: {name}..."):
                price_matrix, _ = build_price_matrix(
                    tickers,
                    start=str(settings["start_date"]),
                    end=str(settings["end_date"]),
                )

            if price_matrix.empty or "INR=X" not in price_matrix.columns:
                st.warning(f"Skipping {name}: insufficient data.")
                continue

            usd_inr = price_matrix["INR=X"]
            prices_inr_df = prices_in_inr(price_matrix, TICKER_META, usd_inr)

            try:
                res = run_sip_backtest(
                    prices_inr=prices_inr_df,
                    allocations=norm_alloc,
                    monthly_sip_inr=settings["monthly_sip"],
                    risk_free_annual=settings["risk_free_rate"],
                )
                strategy_results[name] = res
            except Exception as e:
                st.warning(f"{name}: backtest error — {e}")
                continue

            comparison_progress.progress(int((i + 1) / len(strategies) * 100))

        st.session_state["comparison_results"] = strategy_results
        comparison_progress.empty()

    comp_results = st.session_state.get("comparison_results", {})
    if not comp_results:
        st.warning("No comparison results available.")
        return

    comp_df = compare_strategies(comp_results)

    st.subheader("📋 Comparison Table")
    fmt_df = comp_df.copy()
    for col in fmt_df.columns:
        if "₹" in col or "Value" in col or "Invested" in col or "Gain" in col:
            fmt_df[col] = fmt_df[col].apply(_fmt_inr)
        elif "%" in col or "Return" in col or "Drawdown" in col or "Volatility" in col:
            fmt_df[col] = fmt_df[col].apply(lambda v: f"{v:.2f}%" if not np.isnan(v) else "N/A")
        elif "Sharpe" in col or "Ratio" in col:
            fmt_df[col] = fmt_df[col].apply(lambda v: f"{v:.2f}" if not np.isnan(v) else "N/A")
        elif "Month" in col and "Return" not in col:
            fmt_df[col] = fmt_df[col].apply(lambda v: f"{int(v)}" if not np.isnan(v) else "Not yet")

    st.dataframe(fmt_df, use_container_width=True)

    st.subheader("📊 Visual Comparison")
    fig = strategy_comparison_chart(comp_df)
    st.plotly_chart(fig, use_container_width=True)

    # ── Portfolio value lines ────────────────────────────────────────────────
    fig2 = go_lines_comparison(comp_results)
    st.plotly_chart(fig2, use_container_width=True)

    # Download
    comp_csv = comp_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download Comparison CSV",
        data=comp_csv,
        file_name="strategy_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )


def go_lines_comparison(comp_results: dict) -> object:
    import plotly.graph_objects as go
    fig = go.Figure()
    colors = [
        "#4f8df5", "#f5a44f", "#5ec26b", "#e05252", "#a855f7", "#ec4899"
    ]
    for i, (name, result) in enumerate(comp_results.items()):
        fig.add_trace(
            go.Scatter(
                x=result.portfolio_values.index,
                y=result.portfolio_values.values,
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    fig.update_layout(
        title="Strategy Portfolio Values Over Time (₹)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (INR ₹)",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ───────────────────────────────────────────────────────────────
    st.title("📈 Personal Investment Portfolio Backtester")
    st.markdown(DISCLAIMER)
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    settings = sidebar()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_backtest, tab_crash, tab_history, tab_compare = st.tabs([
        "🔬 Backtest",
        "💥 Crash Simulation",
        "📜 Historical Crashes",
        "⚖️ Strategy Comparison",
    ])

    with tab_backtest:
        allocations, expense_ratios = asset_allocation_section(settings)
        backtest_section(settings, allocations, expense_ratios)

    with tab_crash:
        crash_simulation_section(settings)

    with tab_history:
        historical_crash_section()

    with tab_compare:
        strategy_comparison_section(settings)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<center><small>Built for personal research. "
        "Data sourced from Yahoo Finance. "
        "All figures in INR. "
        "Not SEBI-registered advice.</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
