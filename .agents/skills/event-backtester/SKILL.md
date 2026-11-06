---
name: event-backtester
description: Executes tick/1-minute event-driven backtesting incorporating statutory Indian trading costs (STT, NSE charges, SEBI fees, GST, Stamp Duty) and realistic execution slippage. Use when simulating trading strategies and evaluating CAGR/Sharpe.
---
# Indian Market Event Backtesting Protocol

### Objective
Simulate strategy execution while factoring in complete statutory deductions and realistic fill dynamics.

### Statutory Cost Table (NSE India)
- **STT (Securities Transaction Tax):** $0.025\%$ on Equity Intraday sell turnover; $0.125\%$ on Options sell premium.
- **Exchange Turnover Charges:** $0.00297\%$ (Equity) / $0.03503\%$ (Options premium).
- **SEBI Turnover Fee:** ₹10 per crore ($0.0001\%$).
- **Stamp Duty:** $0.003\%$ on buy turnover (Equity Intraday) / $0.002\%$ on Options buy.
- **GST:** $18\%$ applied to: `(Brokerage + Exchange Charges + SEBI Fees)`.
- **Brokerage:** Flat ₹20 per executed order.

### Execution Directives (`backend/app/backtest/engine.py`)
1. Model fills conservatively: Buys fill at `Ask + (Spread * 0.5) + Slippage`; Sells fill at `Bid - (Spread * 0.5) - Slippage`.
2. Compute metrics: Net CAGR, Annualized Sharpe Ratio, Max Drawdown (MDD), Profit Factor, and Total Statutory Taxes Paid.