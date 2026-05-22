# 📈 Portfolio Backtesting & Crash Simulation Tool

A personal investment research tool for backtesting SIP-style portfolios across Indian and global assets, with crash simulation and strategy comparison.

> ⚠️ **Disclaimer:** This tool is for educational backtesting only. Past performance does not guarantee future returns. This is **not financial advice**.

---

## Features

| Feature | Description |
|--------|-------------|
| **SIP Backtesting** | Simulate monthly SIP investments over up to 15 years of history |
| **Multi-Asset Support** | Indian indices, US ETFs, commodities, AI/tech stocks, semiconductors |
| **INR Base Currency** | All USD assets auto-converted using live USD/INR FX data |
| **Metrics** | XIRR, CAGR, Sharpe Ratio, Max Drawdown, Volatility, Rolling Returns |
| **Rebalancing** | None / Monthly / Quarterly / Yearly |
| **Crash Simulation** | Instant portfolio shocks with SIP-based recovery projection |
| **Historical Replays** | COVID crash, 2022 tech selloff, 2008-style scenario |
| **Strategy Comparison** | Side-by-side comparison of 6 pre-built strategies |
| **CSV Export** | Download full portfolio time-series and metrics |

---

## Quick Start

### 1. Clone / copy this folder

```bash
cd portfolio_backtester
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Folder Structure

```
portfolio_backtester/
├── app.py                        # Main Streamlit app
├── requirements.txt
├── README.md
├── data/
│   └── cache/                    # Cached yfinance parquet files (auto-created)
├── configs/
│   └── example_strategies.json   # Pre-built strategy definitions
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Asset universe, constants, disclaimer
│   ├── data_fetcher.py           # yfinance download + caching
│   ├── currency.py               # USD/INR conversion
│   ├── backtester.py             # SIP simulation engine
│   ├── metrics.py                # CAGR, XIRR, Sharpe, Drawdown, etc.
│   ├── crash_simulator.py        # Crash scenarios + recovery projection
│   ├── portfolio.py              # Strategy loading + comparison
│   └── plotting.py               # All Plotly chart builders
└── tests/
    ├── test_metrics.py
    ├── test_backtester.py
    └── test_crash_simulator.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 35 unit tests should pass.

---

## Supported Assets

### Indian Market
- NIFTY 50, Sensex, NIFTY Next 50, NIFTY Midcap 150

### US Indices & ETFs
- SPY (S&P 500), QQQ (Nasdaq 100), DIA (Dow Jones), SOXX, SMH

### Commodities
- GLD (Gold), SLV (Silver)

### US Tech / AI / Semis
- NVDA, AMD, MSFT, AAPL, GOOGL, META, AMZN, TSLA
- AVGO (Broadcom), TSM (TSMC), MU (Micron)
- ASML, SMCI, QCOM, ARM, RTX

### Custom
- Enter any valid Yahoo Finance ticker symbol

---

## Key Design Decisions

### XIRR vs CAGR
XIRR is the primary metric for SIP portfolios because it correctly accounts for the time value of money across irregular cash flows. CAGR is shown alongside as a secondary reference.

### INR Conversion
USD-denominated assets are converted to INR each month using the actual historical USD/INR exchange rate. This means you benefit from INR depreciation on USD assets (as typically happens over long periods).

### Data Gaps
- Forward-fill is limited to ≤2 consecutive missing months
- Assets with <50% data coverage generate warnings
- Newer assets (e.g., ARM) will have limited history — the backtest runs on available data only

### Rebalancing
- **None (default):** Allocation drifts with market performance
- **Monthly/Quarterly/Yearly:** Portfolio rebalanced to target weights at set intervals

---

## Example Strategies (configs/example_strategies.json)

| Strategy | Focus |
|----------|-------|
| India-Heavy | 85% Indian equities, 10% gold, 5% S&P 500 |
| US-Heavy | 80% US indices, 10% gold, 10% Nifty |
| Equal Global | Balanced mix across 7 asset classes |
| AI & Semis | NVDA, AMD, SOXX, SMH, TSM, MU, MSFT, GOOGL |
| Gold/Silver Hedge | 60% precious metals, 40% equities |
| Conservative Balanced | SPY + Nifty + Gold + blue chips |

---

## Notes for Aagam

- Default SIP is ₹12,000/month starting July 2026
- The backtest uses **historical data** from the past 10 years (configurable)
- Since you're investing in USD assets, a weak INR actually boosts your returns — USD assets have historically appreciated in INR terms
- XIRR accounts for the fact that earlier SIP installments have been invested longer than later ones
- Run the **Strategy Comparison** tab to see how different allocations would have performed before committing to one
