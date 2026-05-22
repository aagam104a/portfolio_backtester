"""
Shared utilities, constants, and asset universe definitions.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Asset Universe ─────────────────────────────────────────────────────────────
ASSET_UNIVERSE = {
    # ── Indian Market ──────────────────────────────────────────────────────────
    "Indian Market": {
        "^NSEI": {
            "label": "NIFTY 50",
            "currency": "INR",
        },
        "^BSESN": {
            "label": "Sensex",
            "currency": "INR",
        },

        # NOTE:
        # Yahoo Finance does not reliably support NIFTY Next 50 / NIFTY Midcap 150
        # as direct index tickers.
        # Removed invalid tickers:
        # - NIFTYNXT50.NS
        # - NIFTYMIDCAP150.NS
        #
        # Use ETF or mutual-fund proxies manually if needed.
        # Example custom NSE tickers may work only if Yahoo supports them.
    },

    # ── US Indices & ETFs ──────────────────────────────────────────────────────
    "US Indices & ETFs": {
        "SPY": {
            "label": "S&P 500 ETF (SPY)",
            "currency": "USD",
        },
        "QQQ": {
            "label": "Nasdaq 100 ETF (QQQ)",
            "currency": "USD",
        },
        "DIA": {
            "label": "Dow Jones ETF (DIA)",
            "currency": "USD",
        },
        "SOXX": {
            "label": "Semiconductor ETF (SOXX)",
            "currency": "USD",
        },
        "SMH": {
            "label": "Semiconductor ETF (SMH)",
            "currency": "USD",
        },
    },

    # ── Commodities ───────────────────────────────────────────────────────────
    "Commodities": {
        "GLD": {
            "label": "Gold ETF (GLD)",
            "currency": "USD",
        },
        "SLV": {
            "label": "Silver ETF (SLV)",
            "currency": "USD",
        },
    },

    # ── US Tech / AI / Semiconductor ──────────────────────────────────────────
    "US Tech / AI / Semis": {
        "NVDA": {
            "label": "NVIDIA",
            "currency": "USD",
        },
        "AMD": {
            "label": "AMD",
            "currency": "USD",
        },
        "MSFT": {
            "label": "Microsoft",
            "currency": "USD",
        },
        "AAPL": {
            "label": "Apple",
            "currency": "USD",
        },
        "GOOGL": {
            "label": "Alphabet (Google)",
            "currency": "USD",
        },
        "META": {
            "label": "Meta Platforms",
            "currency": "USD",
        },
        "AMZN": {
            "label": "Amazon",
            "currency": "USD",
        },
        "TSLA": {
            "label": "Tesla",
            "currency": "USD",
        },
        "AVGO": {
            "label": "Broadcom",
            "currency": "USD",
        },
        "TSM": {
            "label": "TSMC",
            "currency": "USD",
        },
        "MU": {
            "label": "Micron Technology",
            "currency": "USD",
        },
        "ASML": {
            "label": "ASML",
            "currency": "USD",
        },
        "SMCI": {
            "label": "Super Micro Computer",
            "currency": "USD",
        },
        "QCOM": {
            "label": "Qualcomm",
            "currency": "USD",
        },
        "ARM": {
            "label": "Arm Holdings",
            "currency": "USD",
        },
        "RTX": {
            "label": "RTX",
            "currency": "USD",
        },
    },
}

# ── Flat ticker → metadata lookup ─────────────────────────────────────────────
TICKER_META: dict[str, dict] = {}

for category, tickers in ASSET_UNIVERSE.items():
    for ticker, meta in tickers.items():
        TICKER_META[ticker] = {
            **meta,
            "category": category,
        }

# ── Tickers that are known to fail / delisted / unavailable ───────────────────
PROBLEMATIC_TICKERS = {
    "SNDK",              # Old SanDisk ticker; acquired by Western Digital
    "NIFTYNXT50.NS",     # Not reliably available on Yahoo Finance
    "NIFTYMIDCAP150.NS", # Not reliably available on Yahoo Finance
}

# ── Historical Crash Presets ──────────────────────────────────────────────────
HISTORICAL_CRASHES = {
    "COVID-19 Crash (Feb-Mar 2020)": {
        "start": "2020-02-01",
        "trough": "2020-03-01",
        "end": "2020-12-01",
        "description": "Global markets fell sharply in early 2020 and recovered over the following months.",
    },
    "2022 Tech Selloff (Jan-Dec 2022)": {
        "start": "2021-12-01",
        "trough": "2022-10-01",
        "end": "2023-06-01",
        "description": "Rising rates hurt growth and tech stocks. Nasdaq-heavy portfolios saw large drawdowns.",
    },
    "2008 Financial Crisis (Synthetic)": {
        "start": "2007-10-01",
        "trough": "2009-03-01",
        "end": "2013-01-01",
        "description": "Synthetic scenario inspired by the global financial crisis.",
    },
}

# ── Risk-free rate default ────────────────────────────────────────────────────
DEFAULT_RISK_FREE_RATE = 0.07

# ── Disclaimer ────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "**Disclaimer:** This tool is for educational backtesting only. "
    "Past performance does not guarantee future returns. "
    "This is not financial advice. Always consult a SEBI-registered "
    "investment advisor before making investment decisions."
)
