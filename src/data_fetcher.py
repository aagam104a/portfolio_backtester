"""
Data fetching module.
Downloads monthly adjusted-close prices via yfinance, caches to parquet,
aligns all tickers on a common monthly DatetimeIndex, and warns on gaps.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from .utils import CACHE_DIR, PROBLEMATIC_TICKERS

CACHE_TTL_DAYS = 1


def _cache_path(ticker: str) -> Path:
    safe = (
        ticker.replace("^", "IDX_")
        .replace("=", "FX_")
        .replace(".", "_")
        .replace("/", "_")
    )
    return CACHE_DIR / f"{safe}_monthly.parquet"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False

    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).days < CACHE_TTL_DAYS


def _normalize_date(value: str | datetime) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _month_end_index(index) -> pd.DatetimeIndex:
    return pd.to_datetime(index).to_period("M").to_timestamp("M")


def fetch_ticker_monthly(
    ticker: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.Series | None:
    """
    Return monthly close price Series for ticker between start and end.
    Returns None if data is unavailable.
    """
    start = _normalize_date(start)
    end = _normalize_date(end)

    if ticker in PROBLEMATIC_TICKERS:
        warnings.warn(
            f"{ticker} is known to be unavailable or unreliable on Yahoo Finance. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if pd.to_datetime(start) >= pd.to_datetime(end):
        warnings.warn(
            f"Invalid date range for {ticker}: start={start}, end={end}.",
            UserWarning,
            stacklevel=2,
        )
        return None

    cache_file = _cache_path(ticker)

    if not force_refresh and _is_cache_fresh(cache_file):
        try:
            cached = pd.read_parquet(cache_file)
            if "Close" in cached.columns:
                series = cached["Close"]
                series.index = pd.to_datetime(series.index)
                series = series.loc[start:end]
                series = series.dropna()
                if not series.empty:
                    series.name = ticker
                    return series
        except Exception as exc:
            print(f"Cache read error for {ticker}: {exc}")

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1mo",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        warnings.warn(
            f"Could not download {ticker}: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None

    if raw is None or raw.empty:
        warnings.warn(
            f"No data returned for {ticker}. Ticker may be invalid, delisted, or unsupported.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    price_col = None

    if "Close" in raw.columns:
        price_col = "Close"
    elif "Adj Close" in raw.columns:
        price_col = "Adj Close"

    if price_col is None:
        warnings.warn(
            f"No usable price column for {ticker}. Columns found: {list(raw.columns)}",
            UserWarning,
            stacklevel=2,
        )
        return None

    series = raw[price_col].copy()
    series.index = _month_end_index(series.index)
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[~series.index.duplicated(keep="last")]
    series.name = ticker

    if series.empty:
        warnings.warn(
            f"Downloaded data for {ticker} became empty after cleaning.",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        pd.DataFrame({"Close": series}).to_parquet(cache_file)
    except Exception as exc:
        print(f"Cache write error for {ticker}: {exc}")

    return series.loc[start:end]


def fetch_usd_inr(start: str, end: str) -> pd.Series | None:
    return fetch_ticker_monthly("INR=X", start=start, end=end)


def build_price_matrix(
    tickers: list[str],
    start: str,
    end: str,
    status_callback=None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch all tickers and return:
    - price_matrix: monthly prices
    - warnings_map: ticker-level warnings
    """
    start = _normalize_date(start)
    end = _normalize_date(end)

    warnings_map: dict[str, str] = {}
    series_dict: dict[str, pd.Series] = {}

    clean_tickers = []
    for ticker in tickers:
        if ticker and ticker not in clean_tickers:
            clean_tickers.append(ticker.strip().upper())

    all_tickers = sorted(set(clean_tickers + ["INR=X"]))

    for i, ticker in enumerate(all_tickers):
        if status_callback:
            status_callback(i, len(all_tickers), ticker)

        s = fetch_ticker_monthly(ticker, start, end)

        if s is None or s.empty:
            if ticker != "INR=X":
                warnings_map[ticker] = "No data available for this ticker."
            continue

        total_months = len(pd.period_range(start=start, end=end, freq="M"))
        coverage = len(s) / max(total_months, 1)

        if coverage < 0.5 and ticker != "INR=X":
            warnings_map[ticker] = (
                f"Only {len(s)}/{total_months} months of data available "
                f"({coverage:.0%} coverage). Results may be unreliable."
            )

        nan_run = s.isna().astype(int)
        max_consecutive_nans = (
            nan_run.groupby((nan_run != nan_run.shift()).cumsum()).sum().max()
        )

        if pd.notna(max_consecutive_nans) and max_consecutive_nans > 3:
            warnings_map[ticker] = warnings_map.get(ticker, "") + (
                f" Warning: {int(max_consecutive_nans)} consecutive missing months detected."
            )

        series_dict[ticker] = s

    if not series_dict:
        print("No ticker data fetched.")
        print("Requested tickers:", all_tickers)
        print("Warnings:", warnings_map)
        return pd.DataFrame(), warnings_map

    price_matrix = pd.DataFrame(series_dict)
    price_matrix.index = pd.to_datetime(price_matrix.index)
    price_matrix.sort_index(inplace=True)

    price_matrix = price_matrix.ffill(limit=2)

    print("Fetched tickers:", list(series_dict.keys()))
    print("Price matrix shape:", price_matrix.shape)
    print("Warnings:", warnings_map)

    return price_matrix, warnings_map
