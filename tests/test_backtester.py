"""
Unit tests for backtester.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
import numpy as np
import pandas as pd
import pytest

from src.backtester import run_sip_backtest


def _make_prices(tickers, months=24, start_price=100.0, monthly_growth=0.01):
    """Create a synthetic price matrix with steady growth."""
    idx = pd.date_range("2022-01-31", periods=months, freq="ME")
    data = {}
    for t in tickers:
        data[t] = [start_price * ((1 + monthly_growth) ** i) for i in range(months)]
    return pd.DataFrame(data, index=idx)


class TestSIPBasics:
    def test_total_invested(self):
        prices = _make_prices(["SPY"], months=12)
        result = run_sip_backtest(prices, {"SPY": 1.0}, monthly_sip_inr=12000)
        assert abs(result.total_invested - 12 * 12000) < 1.0  # within ₹1

    def test_portfolio_grows(self):
        prices = _make_prices(["SPY"], months=24)
        result = run_sip_backtest(prices, {"SPY": 1.0}, monthly_sip_inr=12000)
        assert result.portfolio_values.iloc[-1] > result.total_invested

    def test_multi_asset(self):
        prices = _make_prices(["SPY", "GLD"], months=24)
        result = run_sip_backtest(
            prices,
            {"SPY": 0.6, "GLD": 0.4},
            monthly_sip_inr=12000,
        )
        assert "SPY" in result.asset_values.columns
        assert "GLD" in result.asset_values.columns

    def test_expense_ratio(self):
        prices = _make_prices(["SPY"], months=24, monthly_growth=0.0)
        # With no market growth but expense ratio, portfolio should be < invested
        r_no_exp = run_sip_backtest(prices, {"SPY": 1.0}, monthly_sip_inr=12000)
        r_with_exp = run_sip_backtest(
            prices, {"SPY": 1.0}, monthly_sip_inr=12000,
            expense_ratios={"SPY": 0.10}  # 10% annual expense (extreme, for test)
        )
        assert r_with_exp.portfolio_values.iloc[-1] < r_no_exp.portfolio_values.iloc[-1]

    def test_metrics_present(self):
        prices = _make_prices(["NVDA"], months=36)
        result = run_sip_backtest(prices, {"NVDA": 1.0}, monthly_sip_inr=12000)
        for key in ["cagr", "xirr", "max_drawdown", "sharpe_ratio"]:
            assert key in result.metrics

    def test_allocation_normalisation(self):
        """Weights summing to != 1.0 should be normalised internally."""
        prices = _make_prices(["A", "B"], months=12)
        result = run_sip_backtest(prices, {"A": 60, "B": 40}, monthly_sip_inr=12000)
        assert result.total_invested > 0  # Should not raise

    def test_unknown_ticker_skipped(self):
        prices = _make_prices(["SPY"], months=12)
        result = run_sip_backtest(
            prices, {"SPY": 0.7, "UNKNOWN": 0.3}, monthly_sip_inr=12000
        )
        assert any("UNKNOWN" in w for w in result.warnings)


class TestRebalancing:
    def test_rebalance_monthly(self):
        prices = _make_prices(["A", "B"], months=24)
        result = run_sip_backtest(
            prices, {"A": 0.5, "B": 0.5},
            monthly_sip_inr=12000, rebalance="monthly"
        )
        assert result.portfolio_values.iloc[-1] > 0

    def test_rebalance_quarterly(self):
        prices = _make_prices(["A", "B"], months=36)
        result = run_sip_backtest(
            prices, {"A": 0.5, "B": 0.5},
            monthly_sip_inr=12000, rebalance="quarterly"
        )
        assert result.portfolio_values.iloc[-1] > 0
