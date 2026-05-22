"""
Crash Simulation Module.

Applies instant shocks to a portfolio and estimates recovery time
under continued SIP investment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TICKER_META


@dataclass
class CrashScenario:
    """Definition of a crash scenario."""
    name: str
    global_drop_pct: float = 0.0                  # e.g. 30.0 = drop 30%
    asset_drops: dict[str, float] = field(default_factory=dict)  # {ticker: drop%}
    fx_change_pct: float = 0.0                     # INR depreciation (+) or appreciation (-)
    description: str = ""


@dataclass
class CrashResult:
    pre_crash_value: float
    post_crash_value: float
    drawdown_amount: float
    drawdown_pct: float
    months_to_recover: Optional[int]
    recovery_values: pd.Series
    invested_during_recovery: float
    sip_paused: bool


def apply_crash(
    portfolio_value: float,
    asset_values: pd.Series,
    scenario: CrashScenario,
    ticker_meta: dict[str, dict] | None = None,
) -> tuple[float, pd.Series]:
    """
    Apply crash shocks to a portfolio snapshot.

    Returns (post_crash_total, post_crash_asset_series).
    """
    ticker_meta = ticker_meta or TICKER_META

    new_asset_values = asset_values.copy().fillna(0.0)

    # Apply asset-specific drops
    for ticker, drop_pct in scenario.asset_drops.items():
        if ticker in new_asset_values.index:
            new_asset_values[ticker] *= 1 - drop_pct / 100

    # Apply global drop to remaining (not already shocked) assets
    if scenario.global_drop_pct != 0:
        for ticker in new_asset_values.index:
            if ticker not in scenario.asset_drops:
                new_asset_values[ticker] *= 1 - scenario.global_drop_pct / 100

    # Apply FX shock: INR depreciation increases USD asset values in INR
    if scenario.fx_change_pct != 0:
        for ticker in new_asset_values.index:
            meta = ticker_meta.get(ticker, {})
            if meta.get("currency", "USD") == "USD":
                # INR weakens → USD assets worth more in INR
                new_asset_values[ticker] *= 1 + scenario.fx_change_pct / 100

    post_crash_total = float(new_asset_values.sum())
    return post_crash_total, new_asset_values


def simulate_recovery(
    post_crash_value: float,
    pre_crash_value: float,
    monthly_sip_inr: float,
    monthly_recovery_rate: float,
    max_months: int = 120,
    sip_paused: bool = False,
) -> tuple[pd.Series, float]:
    """
    Simulate portfolio recovery from post-crash value back to pre-crash peak.

    Parameters
    ----------
    post_crash_value     : Portfolio value immediately after crash.
    pre_crash_value      : Target recovery value (pre-crash peak).
    monthly_sip_inr      : Monthly SIP in INR (0 if paused).
    monthly_recovery_rate: Monthly return rate (e.g. 0.007 = ~8.4% annual).
    max_months           : Stop simulating after this many months.
    sip_paused           : If True, no SIP during recovery.

    Returns
    -------
    (recovery_series, total_sip_invested_during_recovery)
    """
    values = [post_crash_value]
    current = post_crash_value
    sip_invested = 0.0

    effective_sip = 0.0 if sip_paused else monthly_sip_inr

    for _ in range(max_months):
        current = current * (1 + monthly_recovery_rate) + effective_sip
        sip_invested += effective_sip
        values.append(current)
        if current >= pre_crash_value:
            break

    return pd.Series(values), sip_invested


def run_crash_simulation(
    last_asset_values: pd.Series,
    scenario: CrashScenario,
    monthly_sip_inr: float,
    recovery_annual_cagr: float = 0.12,
    sip_paused: bool = False,
    ticker_meta: dict[str, dict] | None = None,
) -> CrashResult:
    """
    End-to-end crash simulation.

    Parameters
    ----------
    last_asset_values    : Asset values (INR) at the time of crash.
    scenario             : CrashScenario definition.
    monthly_sip_inr      : Monthly SIP amount.
    recovery_annual_cagr : Annual return assumption during recovery phase.
    sip_paused           : Whether to pause SIP during recovery.
    ticker_meta          : Asset metadata (currency, etc.).

    Returns
    -------
    CrashResult
    """
    ticker_meta = ticker_meta or TICKER_META
    pre_crash_value = float(last_asset_values.sum())

    post_crash_total, _ = apply_crash(
        pre_crash_value, last_asset_values, scenario, ticker_meta
    )

    drawdown_amount = pre_crash_value - post_crash_total
    drawdown_pct = drawdown_amount / pre_crash_value * 100 if pre_crash_value > 0 else 0.0

    # Monthly recovery rate from annual CAGR
    monthly_rate = (1 + recovery_annual_cagr) ** (1 / 12) - 1

    recovery_series, invested_during_recovery = simulate_recovery(
        post_crash_total,
        pre_crash_value,
        monthly_sip_inr,
        monthly_rate,
        max_months=240,
        sip_paused=sip_paused,
    )

    # Did it recover?
    recovered = recovery_series.iloc[-1] >= pre_crash_value
    months_to_recover = (
        int((recovery_series >= pre_crash_value).idxmax())
        if recovered
        else None
    )

    return CrashResult(
        pre_crash_value=pre_crash_value,
        post_crash_value=post_crash_total,
        drawdown_amount=drawdown_amount,
        drawdown_pct=drawdown_pct,
        months_to_recover=months_to_recover,
        recovery_values=recovery_series,
        invested_during_recovery=invested_during_recovery,
        sip_paused=sip_paused,
    )


# ── Historical Crash Replay (operates on actual price data) ──────────────────

def replay_historical_crash(
    portfolio_values: pd.Series,
    crash_start: str,
    crash_trough: str,
    crash_end: str,
) -> dict:
    """
    Slice a backtest result and measure crash/recovery metrics from real data.
    Returns a dict of stats, or notes if data is insufficient.
    """
    try:
        pre = portfolio_values[portfolio_values.index <= crash_start].iloc[-1]
        trough_slice = portfolio_values[(portfolio_values.index >= crash_start) & (portfolio_values.index <= crash_trough)]
        trough = trough_slice.min()
        trough_date = trough_slice.idxmin()

        dd_pct = (pre - trough) / pre * 100 if pre > 0 else 0
        recovery_slice = portfolio_values[(portfolio_values.index >= trough_date) & (portfolio_values.index <= crash_end)]
        recovered = recovery_slice[recovery_slice >= pre]
        recovery_months = len(portfolio_values[(portfolio_values.index >= trough_date) & (portfolio_values.index <= recovered.index[0])]) if not recovered.empty else None

        return {
            "pre_crash_value": pre,
            "trough_value": trough,
            "trough_date": trough_date,
            "drawdown_pct": dd_pct,
            "recovery_months": recovery_months,
            "data_available": True,
        }
    except (KeyError, IndexError, TypeError):
        return {"data_available": False, "reason": "Insufficient data for this crash period."}
