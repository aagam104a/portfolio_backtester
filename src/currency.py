"""
Currency conversion helpers.
Converts USD-denominated asset prices to INR using the fetched INR=X series.
"""

from __future__ import annotations

import pandas as pd


def align_fx(
    price_matrix: pd.DataFrame,
    usd_inr: pd.Series,
) -> pd.Series:
    """
    Return a USD/INR rate series aligned to price_matrix's index.
    Missing FX dates are forward-filled (short gaps only), then
    back-filled for the very first observation.
    """
    aligned = usd_inr.reindex(price_matrix.index)
    aligned = aligned.ffill().bfill()
    return aligned


def convert_to_inr(
    price_usd: pd.Series,
    usd_inr: pd.Series,
) -> pd.Series:
    """
    Multiply a USD price series by the FX rate to get INR prices.
    Both series must share the same index (already aligned via align_fx).
    """
    return price_usd * usd_inr


def prices_in_inr(
    price_matrix: pd.DataFrame,
    ticker_meta: dict[str, dict],
    usd_inr: pd.Series,
) -> pd.DataFrame:
    """
    Return a copy of price_matrix where all USD columns are converted to INR.
    INR=X column is dropped from the result.
    """
    result = price_matrix.drop(columns=["INR=X"], errors="ignore").copy()
    fx = align_fx(price_matrix, usd_inr)

    for ticker in result.columns:
        meta = ticker_meta.get(ticker, {})
        if meta.get("currency", "USD") == "USD":
            result[ticker] = result[ticker] * fx

    return result
