"""
Unit tests for crash_simulator.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.crash_simulator import (
    apply_crash,
    simulate_recovery,
    run_crash_simulation,
    CrashScenario,
)


class TestApplyCrash:
    def test_global_drop(self):
        asset_vals = pd.Series({"SPY": 500000.0, "GLD": 200000.0})
        scenario = CrashScenario(name="30% drop", global_drop_pct=30.0)
        post_total, post_vals = apply_crash(700000.0, asset_vals, scenario)
        assert abs(post_total - 490000.0) < 1.0

    def test_asset_specific_drop(self):
        asset_vals = pd.Series({"NVDA": 300000.0, "GLD": 100000.0})
        scenario = CrashScenario(
            name="Tech crash",
            asset_drops={"NVDA": 50.0},
        )
        post_total, post_vals = apply_crash(400000.0, asset_vals, scenario)
        assert abs(post_vals["NVDA"] - 150000.0) < 1.0
        assert abs(post_vals["GLD"] - 100000.0) < 1.0

    def test_fx_depreciation(self):
        """INR depreciation should raise value of USD assets."""
        asset_vals = pd.Series({"SPY": 500000.0})
        scenario = CrashScenario(name="INR weak", fx_change_pct=10.0)
        ticker_meta = {"SPY": {"currency": "USD"}}
        _, post_vals = apply_crash(500000.0, asset_vals, scenario, ticker_meta)
        assert post_vals["SPY"] > 500000.0


class TestSimulateRecovery:
    def test_recovers_eventually(self):
        recovery_series, invested = simulate_recovery(
            post_crash_value=700000.0,
            pre_crash_value=1000000.0,
            monthly_sip_inr=12000,
            monthly_recovery_rate=0.01,
            max_months=120,
        )
        assert recovery_series.iloc[-1] >= 1000000.0
        assert invested > 0

    def test_sip_paused(self):
        _, invested_paused = simulate_recovery(
            post_crash_value=700000.0,
            pre_crash_value=1000000.0,
            monthly_sip_inr=12000,
            monthly_recovery_rate=0.01,
            max_months=120,
            sip_paused=True,
        )
        assert invested_paused == 0.0

    def test_mild_crash_recovers_fast(self):
        recovery_series, _ = simulate_recovery(
            post_crash_value=950000.0,
            pre_crash_value=1000000.0,
            monthly_sip_inr=12000,
            monthly_recovery_rate=0.01,
        )
        months = len(recovery_series)
        assert months < 20  # Mild crash recovers quickly


class TestRunCrashSimulation:
    def test_basic_run(self):
        asset_vals = pd.Series({"SPY": 800000.0, "GLD": 200000.0})
        scenario = CrashScenario(name="Test 20%", global_drop_pct=20.0)
        result = run_crash_simulation(
            asset_vals, scenario, monthly_sip_inr=12000,
            recovery_annual_cagr=0.12
        )
        assert result.pre_crash_value == 1000000.0
        assert result.post_crash_value < result.pre_crash_value
        assert result.drawdown_pct > 0

    def test_no_crash(self):
        asset_vals = pd.Series({"SPY": 1000000.0})
        scenario = CrashScenario(name="No crash", global_drop_pct=0.0)
        result = run_crash_simulation(
            asset_vals, scenario, monthly_sip_inr=12000
        )
        assert abs(result.drawdown_pct) < 0.01
