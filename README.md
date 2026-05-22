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

```bash
cd portfolio_backtester
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Folder Structure
portfolio_backtester/
├── app.py                        # Main Streamlit app
├── requirements.txt
├── README.md
├── data/
│   └── cache/                    # Cached yfinance parquet files (auto-created)
├── configs/
│   └── example_strategies.json   # Pre-built strategy definitions
├── src/
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

---

## Running Tests

```bash
pytest tests/ -v
```

All 35 unit tests pass.

---

## Supported Assets

**Indian Market** — NIFTY 50, Sensex, NIFTY Next 50, NIFTY Midcap 150

**US Indices & ETFs** — SPY, QQQ, DIA, SOXX, SMH

**Commodities** — GLD (Gold), SLV (Silver)

**US Tech / AI / Semis** — NVDA, AMD, MSFT, AAPL, GOOGL, META, AMZN, TSLA, AVGO, TSM, MU, ASML, SMCI, QCOM, ARM, RTX

**Custom** — any valid Yahoo Finance ticker

---

## Key Design Decisions

**XIRR vs CAGR** — XIRR is the primary metric because it correctly accounts for the time value of money across recurring SIP cash flows. CAGR is shown as a secondary reference.

**INR Conversion** — USD assets are converted to INR each month using actual historical USD/INR rates, so INR depreciation (which has averaged ~4–5%/year) naturally boosts your USD exposure returns.

**Data Gaps** — forward-fill is capped at ≤2 consecutive missing months; assets with <50% coverage generate visible warnings; newer tickers like ARM run on whatever history is available.

**Rebalancing** — None (allocation drifts), Monthly, Quarterly, or Yearly, all supported.

---

## Example Strategies

| Strategy | Focus |
|----------|-------|
| India-Heavy | 85% Indian equities, 10% gold, 5% S&P 500 |
| US-Heavy | 80% US indices, 10% gold, 10% Nifty |
| Equal Global | Balanced mix across 7 asset classes |
| AI & Semis | NVDA, AMD, SOXX, SMH, TSM, MU, MSFT, GOOGL |
| Gold/Silver Hedge | 60% precious metals, 40% equities |
| Conservative Balanced | SPY + Nifty + Gold + blue chips |

---

## Notes

- Default SIP is ₹12,000/month; backtest period is configurable up to 15 years
- Run the **Strategy Comparison** tab before committing to an allocation — it fetches real historical data for all 6 strategies and ranks them by XIRR, drawdown, Sharpe, and worst 12-month return
- The crash simulator lets you model multi-asset shocks simultaneously (e.g. NVDA −60%, Gold +15%, INR −8%) and projects how long recovery takes with or without continuing SIP
