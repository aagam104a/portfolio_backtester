"""
Plotly chart builders for the Streamlit UI.
All functions return plotly Figure objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PALETTE = px.colors.qualitative.Plotly

# ── Portfolio Value Chart ─────────────────────────────────────────────────────

def portfolio_value_chart(
    portfolio_values: pd.Series,
    monthly_invested: pd.Series,
    title: str = "Portfolio Value Over Time",
) -> go.Figure:
    cumulative_invested = monthly_invested.cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            name="Portfolio Value",
            line=dict(color="#4f8df5", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(79,141,245,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_invested.index,
            y=cumulative_invested.values,
            name="Total Invested",
            line=dict(color="#f5a44f", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value (INR ₹)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Drawdown Chart ────────────────────────────────────────────────────────────

def drawdown_chart(portfolio_values: pd.Series) -> go.Figure:
    rolling_max = portfolio_values.cummax()
    dd = (portfolio_values - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="#e05252"),
            fillcolor="rgba(224,82,82,0.2)",
        )
    )
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Asset Allocation Pie ──────────────────────────────────────────────────────

def allocation_pie(
    allocations: dict[str, float],
    labels: dict[str, str] | None = None,
) -> go.Figure:
    labels = labels or {}
    tickers = list(allocations.keys())
    display_labels = [labels.get(t, t) for t in tickers]
    values = [allocations[t] for t in tickers]

    fig = go.Figure(
        data=[go.Pie(labels=display_labels, values=values, hole=0.4)]
    )
    fig.update_layout(
        title="Target Allocation",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Asset Contribution Bar ────────────────────────────────────────────────────

def asset_contribution_bar(
    asset_contributions: pd.DataFrame,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    labels = labels or {}
    df = asset_contributions.copy()
    df["display"] = df["ticker"].apply(lambda t: labels.get(t, t))
    df.sort_values("final_value_inr", ascending=True, inplace=True)

    fig = go.Figure(
        go.Bar(
            x=df["final_value_inr"],
            y=df["display"],
            orientation="h",
            marker_color=_PALETTE[: len(df)],
            text=df["portfolio_share_pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Final Asset Contribution (₹)",
        xaxis_title="Final Value (INR ₹)",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Rolling Returns Chart ─────────────────────────────────────────────────────

def rolling_returns_chart(portfolio_values: pd.Series) -> go.Figure:
    from .metrics import rolling_returns

    r1 = rolling_returns(portfolio_values, 12) * 100
    r3 = rolling_returns(portfolio_values, 36) * 100
    r5 = rolling_returns(portfolio_values, 60) * 100

    fig = go.Figure()
    for label, series, color in [
        ("1-Year Rolling Return", r1, "#4f8df5"),
        ("3-Year Rolling Return", r3, "#f5a44f"),
        ("5-Year Rolling Return", r5, "#5ec26b"),
    ]:
        valid = series.dropna()
        if len(valid) > 0:
            fig.add_trace(
                go.Scatter(x=valid.index, y=valid.values, name=label, line=dict(color=color))
            )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Rolling Annualised Returns",
        xaxis_title="Date",
        yaxis_title="Annualised Return (%)",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Asset Value Lines Chart ───────────────────────────────────────────────────

def asset_lines_chart(
    asset_values: pd.DataFrame,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    labels = labels or {}
    fig = go.Figure()
    for i, col in enumerate(asset_values.columns):
        series = asset_values[col].dropna()
        if len(series) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=labels.get(col, col),
                line=dict(color=_PALETTE[i % len(_PALETTE)]),
            )
        )
    fig.update_layout(
        title="Individual Asset Values (₹)",
        xaxis_title="Date",
        yaxis_title="Value (INR ₹)",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Monthly Returns Heatmap ───────────────────────────────────────────────────

def monthly_returns_heatmap(portfolio_values):
    import pandas as pd
    import plotly.express as px

    pv = portfolio_values.copy()
    pv.index = pd.to_datetime(pv.index)
    pv = pv.sort_index()
    pv = pv.dropna()

    # Correct monthly returns
    monthly_returns = pv.pct_change()

    # Remove first month because it has no previous month to compare against
    monthly_returns = monthly_returns.dropna()

    heatmap_df = pd.DataFrame({
        "Year": monthly_returns.index.year,
        "Month": monthly_returns.index.month,
        "Return": monthly_returns.values * 100,
    })

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    heatmap_df["Month Name"] = heatmap_df["Month"].map(month_names)

    pivot = heatmap_df.pivot(
        index="Year",
        columns="Month Name",
        values="Return",
    )

    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    pivot = pivot.reindex(columns=month_order)

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Monthly Returns Heatmap (%)",
        labels=dict(color="Return (%)"),
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Year",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ── Crash Recovery Chart ──────────────────────────────────────────────────────

def crash_recovery_chart(
    recovery_values: pd.Series,
    pre_crash_value: float,
    post_crash_value: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(recovery_values))),
            y=recovery_values.values,
            name="Portfolio Value",
            line=dict(color="#4f8df5", width=2.5),
        )
    )
    fig.add_hline(
        y=pre_crash_value,
        line_dash="dash",
        line_color="#5ec26b",
        annotation_text="Pre-Crash Level",
        annotation_position="right",
    )
    fig.add_hline(
        y=post_crash_value,
        line_dash="dash",
        line_color="#e05252",
        annotation_text="Post-Crash Level",
        annotation_position="right",
    )
    fig.update_layout(
        title="Crash Recovery Simulation",
        xaxis_title="Months After Crash",
        yaxis_title="Portfolio Value (₹)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Strategy Comparison Chart ─────────────────────────────────────────────────

def strategy_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    metrics = ["XIRR (%)", "CAGR (%)", "Max Drawdown (%)", "Volatility (%)"]
    available = [m for m in metrics if m in comparison_df.columns]

    fig = make_subplots(
        rows=1, cols=len(available),
        subplot_titles=available,
    )
    for j, metric in enumerate(available, start=1):
        vals = comparison_df[metric]
        fig.add_trace(
            go.Bar(
                x=comparison_df.index.tolist(),
                y=vals.values,
                name=metric,
                marker_color=_PALETTE[j - 1],
                showlegend=False,
            ),
            row=1,
            col=j,
        )

    fig.update_layout(
        title="Strategy Comparison",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
