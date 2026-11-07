"""Daily top-N swing backtest with Indian delivery (CNC) costs.

Each rebalance date, the strategy buys the top-N scored stocks at the next open and sells
at the close `H` trading days later. Rebalances are spaced H trading days apart so holding
periods never overlap. Positions are equal-weighted.

Delivery costs are NOT the intraday schedule in taxes.py. Delivery STT is charged on both buy
and sell. Every rate below is a named constant. Verify each one against current circulars
and your broker's schedule before trusting net results.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import polars as pl

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class DeliveryCostModel:
    """Round-trip cost of one equal-sized delivery position, as a fraction of notional."""

    # ASSUMPTION: FYERS charges no brokerage on equity delivery. Confirm on your plan.
    brokerage_per_order: float = 0.0
    # Securities Transaction Tax on delivery: 0.1% on buy AND sell.
    stt_rate: float = 0.001
    # Stamp duty on delivery buys: 0.015%.
    stamp_duty_buy_rate: float = 0.00015
    # NSE equity turnover charge: 0.00297%, both sides.
    exchange_rate: float = 0.0000297
    # SEBI turnover fee: ₹10 per crore, both sides.
    sebi_rate: float = 0.000001
    # GST: 18% on brokerage + exchange + SEBI.
    gst_rate: float = 0.18
    # Depositary (CDSL) charge per sell scrip, in ₹. ASSUMPTION: roughly ₹16 incl. GST. Verify.
    dp_charge_per_sell_scrip: float = 16.0

    def round_trip_fraction(self, notional: float) -> float:
        """Total charges for buying and later selling `notional` worth, as a fraction of notional."""
        if notional <= 0:
            raise ValueError("notional must be positive")
        gst_base_per_leg = self.brokerage_per_order + self.exchange_rate * notional + self.sebi_rate * notional
        gst_per_leg = self.gst_rate * gst_base_per_leg

        buy = (
            self.brokerage_per_order
            + self.stt_rate * notional
            + self.stamp_duty_buy_rate * notional
            + self.exchange_rate * notional
            + self.sebi_rate * notional
            + gst_per_leg
        )
        sell = (
            self.brokerage_per_order
            + self.stt_rate * notional
            + self.exchange_rate * notional
            + self.sebi_rate * notional
            + gst_per_leg
            + self.dp_charge_per_sell_scrip
        )
        return (buy + sell) / notional


def _max_drawdown(equity: List[float]) -> float:
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def backtest_top_n(
    predictions: pl.DataFrame,
    horizon: int,
    top_n: int = 5,
    costs: DeliveryCostModel = DeliveryCostModel(),
    notional_per_position: float = 100_000.0,
) -> Dict[str, Any]:
    """Evaluate the top-N strategy against the universe average and the Nifty 50.

    `predictions` needs: date, symbol, score, label_raw, bench_label.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    cost = costs.round_trip_fraction(notional_per_position)
    all_dates = sorted(predictions["date"].unique().to_list())
    rebalance_dates = all_dates[::horizon]

    records: List[Dict[str, Any]] = []
    for d in rebalance_dates:
        day = predictions.filter(pl.col("date") == d).filter(pl.col("label_raw").is_not_null())
        if day.height < top_n:
            continue
        picks = day.sort("score", descending=True).head(top_n)
        strategy_gross = float(picks["label_raw"].mean())
        universe_gross = float(day["label_raw"].mean())
        bench = float(day["bench_label"][0])
        records.append({
            "date": d,
            "picks": picks["symbol"].to_list(),
            "strategy_net": strategy_gross - cost,
            "universe_net": universe_gross - cost,
            "bench": bench,
        })

    if not records:
        raise ValueError("No rebalance dates had enough labelled stocks to form a portfolio")

    strat = [r["strategy_net"] for r in records]
    univ = [r["universe_net"] for r in records]
    bench = [r["bench"] for r in records]

    def equity(returns: List[float]) -> List[float]:
        curve = [1.0]
        for r in returns:
            curve.append(curve[-1] * (1.0 + r))
        return curve

    strat_eq = equity(strat)
    univ_eq = equity(univ)
    bench_eq = equity(bench)

    periods_per_year = TRADING_DAYS_PER_YEAR / horizon
    n = len(strat)
    years = n / periods_per_year

    def sharpe(returns: List[float]) -> float:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
        std = var ** 0.5
        return 0.0 if std == 0 else (mean / std) * (periods_per_year ** 0.5)

    def cagr(curve: List[float]) -> float:
        if years <= 0 or curve[-1] <= 0:
            return float("nan")
        return curve[-1] ** (1.0 / years) - 1.0

    excess = [s - b for s, b in zip(strat, bench)]
    return {
        "horizon_trading_days": horizon,
        "top_n": top_n,
        "round_trip_cost_pct": round(cost * 100, 4),
        "n_trades": n,
        "period_start": str(records[0]["date"]),
        "period_end": str(records[-1]["date"]),
        "strategy": {
            "total_return": strat_eq[-1] - 1.0,
            "cagr": cagr(strat_eq),
            "sharpe": sharpe(strat),
            "max_drawdown": _max_drawdown(strat_eq),
            "hit_rate": sum(1 for r in strat if r > 0) / n,
            "avg_return_per_trade": sum(strat) / n,
        },
        "universe_equal_weight": {
            "total_return": univ_eq[-1] - 1.0,
            "cagr": cagr(univ_eq),
            "sharpe": sharpe(univ),
            "max_drawdown": _max_drawdown(univ_eq),
        },
        "nifty50": {
            "total_return": bench_eq[-1] - 1.0,
            "cagr": cagr(bench_eq),
            "sharpe": sharpe(bench),
            "max_drawdown": _max_drawdown(bench_eq),
        },
        "strategy_vs_nifty_avg_excess_per_trade": sum(excess) / n,
        "sample_trades": [
            {"date": str(r["date"]), "picks": r["picks"], "net": round(r["strategy_net"], 5)}
            for r in records[-5:]
        ],
    }
