---
name: indian-market-backtest
description: Run event-driven backtests for NSE strategies with realistic fills, statutory Indian costs (brokerage, STT, exchange charges, SEBI fee, stamp duty, GST), and metrics (CAGR, Sharpe, max drawdown, profit factor). Use when changing backend/app/backtest or evaluating whether a model signal survives costs.
---

# Indian Market Backtest

Spec: `.agents/skills/event-backtester/SKILL.md`. Code: `backend/app/backtest/` (`engine.py`, `strategies.py`, `taxes.py`). Tests: `backend/tests/test_backtest_costs.py`.

## Procedure

1. **Signals come in as data.** A strategy receives bars and model outputs up to `t`, and emits target positions for `t+1`. It never reads bar `t+1` to decide at `t`.
2. **Fills.** Fill at the next bar's open, or at the quote side when depth data exists:
   - Buy at `ask + 0.5 * spread + slippage`
   - Sell at `bid - 0.5 * spread - slippage`
   - Without depth data, use a fixed slippage in basis points and report it.
3. **Costs.** Apply the statutory table below to every fill. Keep each component as its own ledger line so the user can see where P&L went.
4. **Metrics.** Net CAGR, annualized Sharpe (state the risk-free rate used), max drawdown, profit factor, trade count, win rate, average holding time, total taxes and charges.
5. **Walk-forward.** Evaluate on the out-of-sample period only. Report the in-sample period separately and do not optimize on the test block.
6. **Sensitivity.** Re-run with slippage at 0.5× and 2×. A strategy that is profitable only at 0.5× slippage is not robust.

## Statutory cost table (verify before you hard-code)

| Component | Equity intraday | Options (premium) |
|---|---|---|
| STT | 0.025% on sell turnover | Verify. Spec says 0.125% on sell. Current rate appears to be 0.1% since Oct 2024. |
| Exchange transaction charge | 0.00297% | 0.03503% |
| SEBI turnover fee | ₹10 per crore (0.0001%) | same |
| Stamp duty | 0.003% on buy | 0.003% on buy (spec says 0.002%; verify) |
| Brokerage | ₹20 flat per executed order (FYERS spec) | same |
| GST | 18% on (brokerage + exchange charge + SEBI fee) | same |

Put the rates in one config table in `taxes.py`, with an `effective_from` date and a source link comment. Do not scatter them through the engine. Update the table when circulars change, and note the change in the test expectations.

## Checks before reporting a result

- Total costs are non-zero and roughly proportional to turnover. A zero-cost run is a bug.
- Position never exceeds capital. Margin use is explicit.
- No trade executes outside 09:15–15:30 IST unless the strategy explicitly targets a different session.
- Intraday positions are squared off by 15:20 IST, or the strategy documents why not.
- Equity curve is marked to market every bar, including flat periods.

## Output

Report results as a table: gross return, each cost component, net return, CAGR, Sharpe, max drawdown, trades, and the slippage sensitivity. State the period, symbols, and assumptions. Don't present a single number without its costs.
