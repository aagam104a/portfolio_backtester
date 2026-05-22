"""
Data fetching module.
Downloads monthly adjusted-close prices via yfinance, caches to parquet,
aligns all tickers on a common monthly DatetimeIndex, and warns on gaps.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .utils import CACHE_DIR, PROBLEMATIC_TICKERS

# ── Cache TTL: refresh if cache is older than this many days ──────────────────
CACHE_TTL_DAYS = 1


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("=", "FX_").replace(".", "_")
    return CACHE_DIR / f"{safe}_monthly.parquet"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).days < CACHE_TTL_DAYS


def fetch_ticker_monthly(
    ticker: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.Series | None:
    """
    Return a monthly close price Series for *ticker* between *start* and *end*.
    Returns None and warns if data is unavailable or too sparse.
    """
    if ticker in PROBLEMATIC_TICKERS:
        warnings.warn(
            f"{ticker} is no longer actively traded / delisted. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None

    cache_file = _cache_path(ticker)

    # ── Try cache ─────────────────────────────────────────────────────────────
    if not force_refresh and _is_cache_fresh(cache_file):
        try:
            cached = pd.read_parquet(cache_file)
            # Slice to requested window
            series = cached["Close"].loc[start:end]
            if len(series) > 0:
                return series
        except Exception:
            pass  # Rebuild cache on any read error

    # ── Download from yfinance ────────────────────────────────────────────────
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1mo",
            auto_adjust=True,
            progress=False,
            show_errors=False,
        )
    except Exception as exc:
        warnings.warn(f"Could not download {ticker}: {exc}", UserWarning, stacklevel=2)
        return None

    if raw is None or raw.empty:
        warnings.warn(
            f"No data returned for {ticker}. It may be delisted or the ticker is wrong.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" not in raw.columns:
        warnings.warn(f"No 'Close' column for {ticker}.", UserWarning, stacklevel=2)
        return None

    series = raw["Close"].copy()
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp("M")
    series.name = ticker

    # Persist full downloaded range to cache
    pd.DataFrame({"Close": series}).to_parquet(cache_file)

    return series.loc[start:end]


def fetch_usd_inr(start: str, end: str) -> pd.Series | None:
    """Return monthly USD/INR rate."""
    return fetch_ticker_monthly("INR=X", start=start, end=end)


def build_price_matrix(
    tickers: list[str],
    start: str,
    end: str,
    status_callback=None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch all tickers and return:
      - price_matrix: DataFrame indexed by month-end dates, columns = tickers
      - warnings_map: {ticker: warning_message}

    Tickers with zero data are excluded from the matrix.
    """
    warnings_map: dict[str, str] = {}
    series_dict: dict[str, pd.Series] = {}

    all_tickers = list(set(tickers + ["INR=X"]))

    for i, ticker in enumerate(all_tickers):
        if status_callback:
            status_callback(i, len(all_tickers), ticker)

        s = fetch_ticker_monthly(ticker, start, end)

        if s is None or s.empty:
            if ticker != "INR=X":
                warnings_map[ticker] = "No data available for this ticker."
            continue

        # ── Gap check ─────────────────────────────────────────────────────────
        total_months = len(pd.period_range(start=start, end=end, freq="M"))
        coverage = len(s) / max(total_months, 1)
        if coverage < 0.5 and ticker != "INR=X":
            warnings_map[ticker] = (
                f"Only {len(s)}/{total_months} months of data available "
                f"({coverage:.0%} coverage). Results may be unreliable."
            )

        # Check for consecutive NaN runs > 3 months
        nan_run = s.isna().astype(int)
        max_consecutive_nans = (
            nan_run.groupby((nan_run != nan_run.shift()).cumsum()).sum().max()
        )
        if max_consecutive_nans > 3:
            warnings_map[ticker] = warnings_map.get(ticker, "") + (
                f" Warning: {max_consecutive_nans} consecutive missing months detected."
            )

        series_dict[ticker] = s

    if not series_dict:
        return pd.DataFrame(), warnings_map

    # Align on common monthly index (outer join, so we can warn later)
    price_matrix = pd.DataFrame(series_dict)
    price_matrix.index = pd.to_datetime(price_matrix.index)
    price_matrix.sort_index(inplace=True)

    # Forward-fill small gaps only (≤2 months), never more
    price_matrix = price_matrix.fillna(method="ffill", limit=2)

    return price_matrix, warnings_map
