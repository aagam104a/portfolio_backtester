"""
Unit tests for metrics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    cagr,
    xirr,
    annualized_volatility,
    sharpe_ratio,
    drawdown_series,
    max_drawdown,
    drawdown_recovery_months,
    rolling_returns,
    monthly_returns,
)


class TestCAGR:
    def test_basic(self):
        # 100 → 121 in 2 years = 10% CAGR
        assert abs(cagr(100, 121, 2) - 0.10) < 1e-6

    def test_zero_years(self):
        assert np.isnan(cagr(100, 200, 0))

    def test_zero_start(self):
        assert np.isnan(cagr(0, 200, 5))

    def test_negative_return(self):
        result = cagr(100, 64, 4)
        # (64/100)^(1/4) - 1 ≈ -0.1056
        assert result < 0
        assert abs(result - (-0.1056)) < 0.001


class TestXIRR:
    def test_simple_investment(self):
        # Invest 1200 for 12 months, receive 13200 at end (~10% XIRR approx)
        cfs = [-1000] * 12 + [13200]
        dates = [date(2020, i, 28) for i in range(1, 13)] + [date(2020, 12, 28)]
        result = xirr(cfs, dates)
        assert not np.isnan(result)
        assert result > 0

    def test_mismatched_lengths(self):
        assert np.isnan(xirr([100, -200], [date(2020, 1, 1)]))

    def test_single_flow(self):
        assert np.isnan(xirr([-1000], [date(2020, 1, 1)]))


class TestVolatility:
    def test_constant_returns(self):
        rets = pd.Series([0.01] * 24)
        assert abs(annualized_volatility(rets)) < 1e-10  # no volatility

    def test_reasonable_value(self):
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.01, 0.04, 60))
        vol = annualized_volatility(rets)
        assert 0.05 < vol < 0.30  # typical equity range


class TestSharpe:
    def test_positive_sharpe(self):
        rets = pd.Series([0.015] * 60)  # 1.5%/month ≈ 19.6% annual
        sr = sharpe_ratio(rets, risk_free_annual=0.07)
        assert sr > 0

    def test_zero_vol(self):
        rets = pd.Series([0.01] * 24)
        sr = sharpe_ratio(rets)
        assert np.isnan(sr)


class TestDrawdown:
    def test_no_drawdown(self):
        vals = pd.Series([100, 110, 120, 130])
        assert max_drawdown(vals) == 0.0

    def test_known_drawdown(self):
        vals = pd.Series([100, 90, 80, 70, 80, 90, 100])
        dd = max_drawdown(vals)
        assert abs(dd - 0.30) < 1e-6

    def test_recovery_months(self):
        vals = pd.Series([100, 90, 80, 90, 100, 110])
        recovery = drawdown_recovery_months(vals)
        assert recovery is not None
        assert recovery > 0

    def test_no_recovery(self):
        vals = pd.Series([100, 80, 70, 60])
        recovery = drawdown_recovery_months(vals)
        assert recovery is None


class TestRollingReturns:
    def test_shape(self):
        vals = pd.Series(range(100, 160), dtype=float)
        roll = rolling_returns(vals, 12)
        assert len(roll) == len(vals)

    def test_positive_trend(self):
        vals = pd.Series([100 * (1.01 ** i) for i in range(48)])
        roll = rolling_returns(vals, 12)
        valid = roll.dropna()
        assert all(v > 0 for v in valid)


class TestMonthlyReturns:
    def test_basic(self):
        vals = pd.Series([100, 110, 99, 108])
        rets = monthly_returns(vals)
        assert len(rets) == 3
        assert abs(rets.iloc[0] - 0.10) < 1e-9
