"""
Portfolio strategy management and comparison utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .backtester import SIPBacktestResult, run_sip_backtest
from .metrics import rolling_returns


def load_example_strategies(path: Path) -> list[dict]:
    """Load example strategies from JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("strategies", [])
    except Exception:
        return []


def compare_strategies(
    results: dict[str, SIPBacktestResult],
) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple backtest results.

    Parameters
    ----------
    results : {strategy_name: SIPBacktestResult}

    Returns
    -------
    DataFrame with one row per strategy and key metrics as columns.
    """
    rows = []
    for name, result in results.items():
        m = result.metrics

        # Rolling returns
        roll_1y = rolling_returns(result.portfolio_values, 12)
        roll_3y = rolling_returns(result.portfolio_values, 36)
        roll_5y = rolling_returns(result.portfolio_values, 60)

        rows.append(
            {
                "Strategy": name,
                "Final Value (₹)": m["final_value"],
                "Total Invested (₹)": m["total_invested"],
                "Abs. Gain (₹)": m["abs_gain"],
                "Abs. Return (%)": m["abs_return_pct"] * 100,
                "CAGR (%)": m["cagr"] * 100 if not pd.isna(m["cagr"]) else float("nan"),
                "XIRR (%)": m["xirr"] * 100 if not pd.isna(m["xirr"]) else float("nan"),
                "Volatility (%)": m["annualized_volatility"] * 100,
                "Sharpe Ratio": m["sharpe_ratio"],
                "Max Drawdown (%)": m["max_drawdown"] * 100,
                "Recovery Months": m["recovery_months"],
                "Best Month (%)": m["best_month_return"] * 100,
                "Worst Month (%)": m["worst_month_return"] * 100,
                "Best 12M Return (%)": roll_1y.max() * 100 if len(roll_1y.dropna()) > 0 else float("nan"),
                "Worst 12M Return (%)": roll_1y.min() * 100 if len(roll_1y.dropna()) > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows).set_index("Strategy")
