"""
SIP Backtesting Engine.

Simulates monthly SIP investments into a multi-asset portfolio.
Handles:
  - Fractional unit purchases
  - INR-to-USD conversion for international assets
  - Expense ratio deduction (applied monthly)
  - Transaction fee per purchase
  - Rebalancing (none / monthly / quarterly / yearly)
  - XIRR-compatible cash-flow tracking
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics
from .utils import TICKER_META

RebalanceFreq = Literal["none", "monthly", "quarterly", "yearly"]


class SIPBacktestResult:
    """Container for all backtest outputs."""

    def __init__(
        self,
        portfolio_values: pd.Series,
        asset_values: pd.DataFrame,
        units_held: pd.DataFrame,
        monthly_invested: pd.Series,
        cash_flows: list[float],
        cf_dates: list[date],
        total_invested: float,
        metrics: dict,
        asset_contributions: pd.DataFrame,
        warnings: list[str],
    ):
        self.portfolio_values = portfolio_values
        self.asset_values = asset_values
        self.units_held = units_held
        self.monthly_invested = monthly_invested
        self.cash_flows = cash_flows
        self.cf_dates = cf_dates
        self.total_invested = total_invested
        self.metrics = metrics
        self.asset_contributions = asset_contributions
        self.warnings = warnings


def run_sip_backtest(
    prices_inr: pd.DataFrame,
    allocations: dict[str, float],
    monthly_sip_inr: float,
    expense_ratios: dict[str, float] | None = None,
    transaction_fee_pct: float = 0.0,
    rebalance: RebalanceFreq = "none",
    risk_free_annual: float = 0.07,
) -> SIPBacktestResult:
    """
    Run a SIP backtest.

    Parameters
    ----------
    prices_inr     : Monthly prices in INR, index = month-end dates.
    allocations    : {ticker: weight} where weights sum to 1.0.
    monthly_sip_inr: INR invested per month.
    expense_ratios : {ticker: annual_expense_ratio} e.g. {"SPY": 0.0009}.
    transaction_fee_pct : Flat fee on each purchase (e.g. 0.001 = 0.1%).
    rebalance      : Rebalancing frequency.
    risk_free_annual: For Sharpe ratio.

    Returns
    -------
    SIPBacktestResult
    """
    expense_ratios = expense_ratios or {}

    # Validate tickers
    available_tickers = [t for t in allocations if t in prices_inr.columns]
    warnings_list: list[str] = []
    for t in allocations:
        if t not in prices_inr.columns:
            warnings_list.append(
                f"{t}: not in price matrix — excluded from backtest."
            )

    if not available_tickers:
        raise ValueError("No valid tickers found in the price matrix.")

    # Normalise allocations to sum to 1.0 among available tickers
    raw_alloc = {t: allocations[t] for t in available_tickers}
    total_alloc = sum(raw_alloc.values())
    norm_alloc = {t: w / total_alloc for t, w in raw_alloc.items()}

    prices = prices_inr[available_tickers].copy()

    # Monthly expense deduction factor per ticker
    monthly_expense = {
        t: (expense_ratios.get(t, 0.0) / 12) for t in available_tickers
    }

    # ── State ────────────────────────────────────────────────────────────────
    units: dict[str, float] = {t: 0.0 for t in available_tickers}
    records: list[dict] = []
    cash_flows: list[float] = []
    cf_dates: list[date] = []
    total_invested = 0.0

    for i, (month, row) in enumerate(prices.iterrows()):
        try:
            month_date = month.date()
        except Exception:
            month_date = date(int(month.year), int(month.month), 28)

        # ── Rebalance check ──────────────────────────────────────────────────
        do_rebalance = False
        if rebalance == "monthly" and i > 0:
            do_rebalance = True
        elif rebalance == "quarterly" and i > 0 and (i % 3 == 0):
            do_rebalance = True
        elif rebalance == "yearly" and i > 0 and (i % 12 == 0):
            do_rebalance = True

        if do_rebalance:
            # Current portfolio value
            current_val = sum(
                units[t] * row[t] for t in available_tickers if not np.isnan(row[t])
            )
            # Target units for each ticker
            for t in available_tickers:
                if np.isnan(row[t]) or row[t] <= 0:
                    continue
                target_units = (current_val * norm_alloc[t]) / row[t]
                units[t] = target_units

        # ── Buy this month's SIP ─────────────────────────────────────────────
        month_invested = 0.0
        for t in available_tickers:
            price = row[t]
            if np.isnan(price) or price <= 0:
                continue

            allocated_inr = monthly_sip_inr * norm_alloc[t]
            fee = allocated_inr * transaction_fee_pct
            investable = allocated_inr - fee
            new_units = investable / price
            units[t] += new_units
            month_invested += allocated_inr

        total_invested += month_invested
        cash_flows.append(-month_invested)
        cf_dates.append(month_date)

        # ── Apply monthly expense ratio ──────────────────────────────────────
        for t in available_tickers:
            units[t] *= 1.0 - monthly_expense[t]

        # ── Portfolio value ──────────────────────────────────────────────────
        asset_vals = {}
        for t in available_tickers:
            price = row[t]
            asset_vals[t] = units[t] * price if not np.isnan(price) else np.nan
        portfolio_val = sum(v for v in asset_vals.values() if not np.isnan(v))

        records.append(
            {
                "date": month,
                "portfolio_value": portfolio_val,
                "month_invested": month_invested,
                **{f"val_{t}": asset_vals[t] for t in available_tickers},
                **{f"units_{t}": units[t] for t in available_tickers},
            }
        )

    if not records:
        raise ValueError("No records generated — check date range.")

    df = pd.DataFrame(records).set_index("date")

    portfolio_values = df["portfolio_value"]
    asset_values = df[[f"val_{t}" for t in available_tickers]].rename(
        columns={f"val_{t}": t for t in available_tickers}
    )
    units_df = df[[f"units_{t}" for t in available_tickers]].rename(
        columns={f"units_{t}": t for t in available_tickers}
    )
    monthly_invested = df["month_invested"]

    # Final value as inflow for XIRR
    final_cf = portfolio_values.iloc[-1]
    cash_flows.append(final_cf)
    cf_dates.append(cf_dates[-1])

    metrics = compute_all_metrics(
        portfolio_values, cash_flows, cf_dates, total_invested, risk_free_annual
    )

    # ── Asset contribution: share of final portfolio value ───────────────────
    final_asset_vals = asset_values.iloc[-1].fillna(0.0)
    total_final = final_asset_vals.sum()
    asset_contributions = pd.DataFrame(
        {
            "ticker": final_asset_vals.index,
            "final_value_inr": final_asset_vals.values,
            "portfolio_share_pct": (final_asset_vals / total_final * 100).values
            if total_final > 0
            else np.zeros(len(final_asset_vals)),
        }
    )

    return SIPBacktestResult(
        portfolio_values=portfolio_values,
        asset_values=asset_values,
        units_held=units_df,
        monthly_invested=monthly_invested,
        cash_flows=cash_flows,
        cf_dates=cf_dates,
        total_invested=total_invested,
        metrics=metrics,
        asset_contributions=asset_contributions,
        warnings=warnings_list,
    )
