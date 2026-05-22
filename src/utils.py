"""
Shared utilities, constants, and asset universe definitions.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Asset Universe ─────────────────────────────────────────────────────────────
ASSET_UNIVERSE = {
    # ── Indian Market ──────────────────────────────────────────────────────────
    "Indian Market": {
        "^NSEI":               {"label": "NIFTY 50",            "currency": "INR"},
        "^BSESN":              {"label": "Sensex",              "currency": "INR"},
        "NIFTYNXT50.NS":       {"label": "NIFTY Next 50",       "currency": "INR"},
        "NIFTYMIDCAP150.NS":   {"label": "NIFTY Midcap 150",    "currency": "INR"},
    },
    # ── US Indices & ETFs ──────────────────────────────────────────────────────
    "US Indices & ETFs": {
        "SPY":   {"label": "S&P 500 ETF (SPY)",     "currency": "USD"},
        "QQQ":   {"label": "Nasdaq 100 ETF (QQQ)",  "currency": "USD"},
        "DIA":   {"label": "Dow Jones ETF (DIA)",   "currency": "USD"},
        "SOXX":  {"label": "Semiconductor ETF (SOXX)", "currency": "USD"},
        "SMH":   {"label": "Semiconductor ETF (SMH)",  "currency": "USD"},
    },
    # ── Commodities ───────────────────────────────────────────────────────────
    "Commodities": {
        "GLD":   {"label": "Gold ETF (GLD)",   "currency": "USD"},
        "SLV":   {"label": "Silver ETF (SLV)", "currency": "USD"},
    },
    # ── US Tech / AI / Semiconductor ──────────────────────────────────────────
    "US Tech / AI / Semis": {
        "NVDA":  {"label": "NVIDIA",             "currency": "USD"},
        "AMD":   {"label": "AMD",                "currency": "USD"},
        "MSFT":  {"label": "Microsoft",          "currency": "USD"},
        "AAPL":  {"label": "Apple",              "currency": "USD"},
        "GOOGL": {"label": "Alphabet (Google)",  "currency": "USD"},
        "META":  {"label": "Meta Platforms",     "currency": "USD"},
        "AMZN":  {"label": "Amazon",             "currency": "USD"},
        "TSLA":  {"label": "Tesla",              "currency": "USD"},
        "AVGO":  {"label": "Broadcom",           "currency": "USD"},
        "TSM":   {"label": "TSMC",               "currency": "USD"},
        "MU":    {"label": "Micron Technology",  "currency": "USD"},
        "ASML":  {"label": "ASML",               "currency": "USD"},
        "SMCI":  {"label": "Super Micro Computer","currency": "USD"},
        "QCOM":  {"label": "Qualcomm",           "currency": "USD"},
        "ARM":   {"label": "Arm Holdings",       "currency": "USD"},
        "RTX":   {"label": "RTX (Raytheon)",     "currency": "USD"},
    },
}

# Flat ticker → meta lookup
TICKER_META: dict[str, dict] = {}
for _cat, _tickers in ASSET_UNIVERSE.items():
    for _ticker, _meta in _tickers.items():
        TICKER_META[_ticker] = {**_meta, "category": _cat}

# Tickers that historically had data issues / may be delisted
PROBLEMATIC_TICKERS = {"SNDK"}   # Acquired by WD; ticker gone

# ── Historical Crash Presets ───────────────────────────────────────────────────
HISTORICAL_CRASHES = {
    "COVID-19 Crash (Feb–Mar 2020)": {
        "start": "2020-02-01",
        "trough": "2020-03-01",
        "end": "2020-12-01",
        "description": "Global markets fell ~35% in 5 weeks; recovered within ~6 months.",
    },
    "2022 Tech Selloff (Jan–Dec 2022)": {
        "start": "2021-12-01",
        "trough": "2022-10-01",
        "end": "2023-06-01",
        "description": "Rising rates crushed growth / tech stocks. QQQ fell ~35%.",
    },
    "2008 Financial Crisis (Synthetic)": {
        "start": "2007-10-01",
        "trough": "2009-03-01",
        "end": "2013-01-01",
        "description": "S&P 500 fell ~57%. Full recovery took ~5 years.",
    },
}

# ── Risk-free rate default (approx. India 10-yr G-Sec yield) ──────────────────
DEFAULT_RISK_FREE_RATE = 0.07   # 7% per annum

DISCLAIMER = (
    "⚠️ **Disclaimer:** This tool is for educational backtesting only. "
    "Past performance does not guarantee future returns. "
    "This is **not financial advice**. Always consult a SEBI-registered "
    "investment advisor before making investment decisions."
)
