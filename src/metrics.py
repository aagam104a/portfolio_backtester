"""
Financial metrics calculations.

All functions operate on pandas Series / numpy arrays and are
unit-tested independently of the Streamlit UI.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import numpy_financial as npf
import pandas as pd
from scipy.optimize import brentq


# ── CAGR ──────────────────────────────────────────────────────────────────────

def cagr(start_value: float, end_value: float, years: float) -> float:
    """Compound Annual Growth Rate."""
    if years <= 0 or start_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


# ── XIRR ──────────────────────────────────────────────────────────────────────

def xirr(cash_flows: list[float], dates: list[date]) -> float:
    """
    Extended Internal Rate of Return for irregular cash flows.

    cash_flows: list of floats. Negative = outflows (investments),
                positive = inflow (final portfolio value).
    dates     : matching list of date objects.

    Returns annualised rate as a decimal (0.12 = 12%).
    Returns NaN on failure.
    """
    if len(cash_flows) != len(dates):
        return np.nan
    if len(cash_flows) < 2:
        return np.nan

    def _npv(rate: float) -> float:
        t0 = dates[0]
        total = 0.0
        for cf, d in zip(cash_flows, dates):
            t = (d - t0).days / 365.25
            total += cf / (1 + rate) ** t
        return total

    try:
        result = brentq(_npv, -0.9999, 100.0, maxiter=1000, xtol=1e-8)
        return result
    except (ValueError, RuntimeError):
        return np.nan


# ── Volatility ────────────────────────────────────────────────────────────────

def annualized_volatility(monthly_returns: pd.Series) -> float:
    """Annualised std of monthly returns (×√12)."""
    if len(monthly_returns) < 2:
        return np.nan
    return float(monthly_returns.std() * np.sqrt(12))


# ── Sharpe Ratio ──────────────────────────────────────────────────────────────

def sharpe_ratio(
    monthly_returns: pd.Series,
    risk_free_annual: float = 0.07,
) -> float:
    """Sharpe ratio using annualised excess return / annualised vol."""
    vol = annualized_volatility(monthly_returns)
    if np.isnan(vol) or vol == 0:
        return np.nan
    ann_return = (1 + monthly_returns.mean()) ** 12 - 1
    return (ann_return - risk_free_annual) / vol


# ── Drawdown ──────────────────────────────────────────────────────────────────

def drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    """Return the drawdown at each point (negative values)."""
    rolling_max = portfolio_values.cummax()
    dd = (portfolio_values - rolling_max) / rolling_max
    return dd


def max_drawdown(portfolio_values: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.35 = 35% drawdown)."""
    dd = drawdown_series(portfolio_values)
    return float(abs(dd.min()))


def drawdown_recovery_months(portfolio_values: pd.Series) -> int | None:
    """
    Find the number of months it took to recover from the maximum drawdown.
    Returns None if the portfolio had not recovered by the end of the series.
    """
    dd = drawdown_series(portfolio_values)
    trough_idx = int(dd.argmin())
    peak_val = float(portfolio_values.iloc[: trough_idx + 1].max())

    post_trough = portfolio_values.iloc[trough_idx:]
    recovery_mask = post_trough >= peak_val
    if not recovery_mask.any():
        return None
    recovery_pos = int(recovery_mask.argmax())
    return recovery_pos  # months from trough to recovery


# ── Rolling Returns ───────────────────────────────────────────────────────────

def rolling_returns(
    portfolio_values: pd.Series,
    window_months: int,
) -> pd.Series:
    """
    Annualised rolling return over *window_months* months.
    Returns NaN where insufficient history exists.
    """
    years = window_months / 12
    rolled = portfolio_values.pct_change(periods=window_months)
    annualised = (1 + rolled) ** (1 / years) - 1
    return annualised


# ── Monthly Returns ───────────────────────────────────────────────────────────

def monthly_returns(portfolio_values: pd.Series) -> pd.Series:
    """Simple month-over-month returns."""
    return portfolio_values.pct_change().dropna()


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_all_metrics(
    portfolio_values: pd.Series,
    cash_flows: list[float],
    cf_dates: list[date],
    total_invested: float,
    risk_free_annual: float = 0.07,
) -> dict:
    """
    Compute and return all metrics as a dict.
    """
    rets = monthly_returns(portfolio_values)
    final_value = float(portfolio_values.iloc[-1])
    abs_gain = final_value - total_invested
    abs_return_pct = abs_gain / total_invested if total_invested > 0 else np.nan

    years = len(portfolio_values) / 12
    _cagr = cagr(total_invested, final_value, years)
    _xirr = xirr(cash_flows, cf_dates)
    _vol = annualized_volatility(rets)
    _sharpe = sharpe_ratio(rets, risk_free_annual)
    _max_dd = max_drawdown(portfolio_values)
    _recovery = drawdown_recovery_months(portfolio_values)

    best_month_val = float(rets.max()) if len(rets) else np.nan
    worst_month_val = float(rets.min()) if len(rets) else np.nan
    best_month_date = rets.idxmax() if len(rets) else None
    worst_month_date = rets.idxmin() if len(rets) else None

    return {
        "final_value": final_value,
        "total_invested": total_invested,
        "abs_gain": abs_gain,
        "abs_return_pct": abs_return_pct,
        "cagr": _cagr,
        "xirr": _xirr,
        "annualized_volatility": _vol,
        "sharpe_ratio": _sharpe,
        "max_drawdown": _max_dd,
        "recovery_months": _recovery,
        "best_month_return": best_month_val,
        "best_month_date": best_month_date,
        "worst_month_return": worst_month_val,
        "worst_month_date": worst_month_date,
    }
