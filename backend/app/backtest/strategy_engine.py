"""Generic daily backtest engine for rule-based strategies (see `backend/app/strategies/`).

The engine knows nothing about any specific strategy. It consumes a signal frame
(`SIGNAL_COLUMNS`) and simulates a long-only portfolio:

* Signals are formed at the close of day t. Entries and signal exits fill at day t+1's open.
* Stops are intraday: if the low touches the stop, the fill is the stop price, or the open when the
  bar gaps below it. The stop is checked from the entry bar onward.
* `max_positions` equal slots, each sized from start-of-day equity / max_positions (capped by cash).
  When there are more entry signals than free slots, the highest `priority` wins.
* Slippage is applied against you on every fill. Delivery costs come from `DeliveryCostModel`,
  computed on each position's entry notional (an approximation: the sell notional differs a little).
* Positions still open on the last bar are closed at that close (`end_of_data`).
"""

from dataclasses import dataclass, field
from datetime import date
import math
from typing import Any, Dict, List, Optional

import polars as pl

from backend.app.backtest.swing_backtest import DeliveryCostModel
from backend.app.strategies.base import Strategy, validate_signal_frame

TRADING_DAYS_PER_YEAR = 252
MIN_TRADES_FOR_CONFIDENCE = 100


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    slippage: float = 0.0005  # 5 bps per fill, against you
    costs: DeliveryCostModel = field(default_factory=DeliveryCostModel)
    start: Optional[date] = None  # first trading date (history before it only warms up indicators)
    end: Optional[date] = None


@dataclass
class _Position:
    symbol: str
    entry_date: date
    entry_price: float
    shares: int
    stop: Optional[float]
    bars_held: int = 0


def simulate(
    signals: pl.DataFrame,
    config: EngineConfig,
    stop_atr_mult: Optional[float] = None,
    max_hold_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the portfolio simulation. Returns `trades` (list of dicts) and the `equity` curve."""
    validate_signal_frame(signals)
    if config.max_positions < 1:
        raise ValueError("max_positions must be at least 1")
    if signals.is_empty():
        raise ValueError("Signal frame is empty")

    frame = signals
    if config.start is not None:
        frame = frame.filter(pl.col("date") >= config.start)
    if config.end is not None:
        frame = frame.filter(pl.col("date") <= config.end)
    if frame.is_empty():
        raise ValueError("No bars inside the requested start/end range")

    slip = config.slippage
    cash = config.initial_capital
    positions: Dict[str, _Position] = {}
    last_close: Dict[str, float] = {}
    pending_exits: set = set()
    pending_entries: List[tuple] = []  # (symbol, atr), already ordered by priority
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    def close_position(sym: str, raw_price: float, d: date, reason: str) -> None:
        nonlocal cash
        pos = positions.pop(sym)
        px = raw_price * (1.0 - slip)
        proceeds = pos.shares * px
        entry_notional = pos.shares * pos.entry_price
        cost = config.costs.round_trip_fraction(entry_notional) * entry_notional
        cash += proceeds - cost
        pnl = proceeds - cost - entry_notional
        trades.append({
            "symbol": sym,
            "entry_date": pos.entry_date,
            "exit_date": d,
            "entry_price": pos.entry_price,
            "exit_price": px,
            "shares": pos.shares,
            "bars_held": pos.bars_held,
            "pnl": pnl,
            "return_net": pnl / entry_notional,
            "reason": reason,
        })

    by_date = frame.sort(["date", "symbol"]).partition_by("date", maintain_order=True, as_dict=False)
    last_date: Optional[date] = None
    for day_frame in by_date:
        d = day_frame["date"][0]
        last_date = d
        rows = {r["symbol"]: r for r in day_frame.iter_rows(named=True)}

        # 1) Signal exits fill at today's open. A symbol with no bar today (halt) stays pending.
        for sym in sorted(pending_exits):
            row = rows.get(sym)
            if sym in positions and row is not None:
                close_position(sym, row["open"], d, "signal")
        pending_exits = {s for s in pending_exits if s in positions}

        # 2) Entries fill at today's open into free slots.
        if pending_entries:
            equity_open = cash + sum(p.shares * last_close.get(p.symbol, p.entry_price) for p in positions.values())
            alloc = equity_open / config.max_positions
            for sym, atr_value in pending_entries:
                if len(positions) >= config.max_positions:
                    break
                row = rows.get(sym)
                if row is None or sym in positions or not row["open"] or row["open"] <= 0:
                    continue
                fill = row["open"] * (1.0 + slip)
                shares = int(min(alloc, cash) // fill)
                if shares <= 0:
                    continue
                cash -= shares * fill
                stop = None
                if stop_atr_mult is not None and atr_value is not None:
                    stop = fill - stop_atr_mult * atr_value
                positions[sym] = _Position(sym, d, fill, shares, stop)
            pending_entries = []

        # 3) Intraday stop, then time stop, then mark to market at the close.
        for sym in list(positions):
            row = rows.get(sym)
            if row is None:
                continue
            pos = positions[sym]
            pos.bars_held += 1
            if pos.stop is not None and row["low"] <= pos.stop:
                close_position(sym, min(row["open"], pos.stop), d, "stop")
            elif max_hold_days is not None and pos.bars_held >= max_hold_days:
                close_position(sym, row["close"], d, "time")
        for sym, row in rows.items():
            if sym in positions:
                last_close[sym] = row["close"]
        invested = sum(p.shares * last_close.get(p.symbol, p.entry_price) for p in positions.values())
        equity_rows.append({"date": d, "equity": cash + invested, "invested": invested})

        # 4) Form tomorrow's orders from today's close.
        candidates = []
        for sym, row in rows.items():
            if sym in positions:
                if row["exit_signal"]:
                    pending_exits.add(sym)
            elif row["entry_signal"]:
                candidates.append((-(row["priority"] or 0.0), sym, row["atr"]))
        candidates.sort(key=lambda t: (t[0], t[1]))
        pending_entries = [(sym, atr_value) for _, sym, atr_value in candidates]

    # End of data: liquidate at the last close.
    assert last_date is not None
    for sym in list(positions):
        close_position(sym, last_close.get(sym, positions[sym].entry_price), last_date, "end_of_data")
    equity_rows[-1]["equity"] = cash
    equity_rows[-1]["invested"] = 0.0
    return {"trades": trades, "equity": equity_rows}


def _max_drawdown(values: List[float]) -> float:
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    return worst


def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    std = (sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
    return 0.0 if std == 0 else mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _curve_stats(values: List[float], dates: List[date]) -> Dict[str, float]:
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    returns = [b / a - 1.0 for a, b in zip(values[:-1], values[1:])]
    total = values[-1] / values[0] - 1.0
    return {
        "total_return": total,
        "cagr": (values[-1] / values[0]) ** (1.0 / years) - 1.0 if values[-1] > 0 else float("nan"),
        "sharpe": _sharpe(returns),
        "max_drawdown": _max_drawdown(values),
    }


def summarise(
    sim: Dict[str, Any],
    config: EngineConfig,
    benchmark: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """Portfolio and trade statistics, plus buy-and-hold of the benchmark over the same dates."""
    equity = sim["equity"]
    trades = sim["trades"]
    dates = [r["date"] for r in equity]
    values = [r["equity"] for r in equity]
    result: Dict[str, Any] = {
        "period_start": str(dates[0]),
        "period_end": str(dates[-1]),
        "initial_capital": config.initial_capital,
        "final_equity": values[-1],
        "portfolio": _curve_stats(values, dates),
        "avg_exposure": sum(r["invested"] / r["equity"] for r in equity if r["equity"] > 0) / len(equity),
        "n_trades": len(trades),
    }
    if trades:
        rets = [t["return_net"] for t in trades]
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [-t["pnl"] for t in trades if t["pnl"] <= 0]
        by_year: Dict[int, List[float]] = {}
        for t in trades:
            by_year.setdefault(t["exit_date"].year, []).append(t["return_net"])
        reasons: Dict[str, int] = {}
        for t in trades:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        result["trades"] = {
            "hit_rate": len(wins) / len(trades),
            "avg_return": sum(rets) / len(rets),
            "avg_win": (sum(r for r in rets if r > 0) / max(len([r for r in rets if r > 0]), 1)),
            "avg_loss": (sum(r for r in rets if r <= 0) / max(len([r for r in rets if r <= 0]), 1)),
            "profit_factor": (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else None,
            "avg_bars_held": sum(t["bars_held"] for t in trades) / len(trades),
            "exit_reasons": reasons,
            "by_year": {
                str(y): {"n": len(v), "avg_return": sum(v) / len(v), "hit_rate": sum(1 for x in v if x > 0) / len(v)}
                for y, v in sorted(by_year.items())
            },
        }
    if benchmark is not None and not benchmark.is_empty():
        b = benchmark.sort("date").filter((pl.col("date") >= dates[0]) & (pl.col("date") <= dates[-1]))
        if b.height >= 2:
            result["benchmark_buy_hold"] = _curve_stats(b["close"].to_list(), b["date"].to_list())
    warnings = []
    if len(trades) < MIN_TRADES_FOR_CONFIDENCE:
        warnings.append(
            f"Only {len(trades)} trades. Below {MIN_TRADES_FOR_CONFIDENCE}, results are often noise. "
            "Use more history or a larger universe."
        )
    result["warnings"] = warnings
    return result


def run_strategy_backtest(
    strategy: Strategy,
    bars: pl.DataFrame,
    bench: pl.DataFrame,
    config: EngineConfig = EngineConfig(),
) -> Dict[str, Any]:
    """Prepare signals on the full history, then simulate inside [config.start, config.end].

    `bars`: symbol, date, open, high, low, close, volume. `bench`: date, close.
    Returns the summary dict plus raw `trades` and `equity` under `_trades` / `_equity`.
    """
    signals = strategy.prepare(bars, bench)
    sim = simulate(signals, config, strategy.stop_atr_mult, strategy.max_hold_days)
    summary = summarise(sim, config, bench)
    summary.update({
        "strategy": strategy.name,
        "params": strategy.params(),
        "stop_atr_mult": strategy.stop_atr_mult,
        "max_hold_days": strategy.max_hold_days,
        "n_symbols": signals["symbol"].n_unique(),
        "config": {
            "max_positions": config.max_positions,
            "slippage": config.slippage,
            "start": str(config.start) if config.start else None,
            "end": str(config.end) if config.end else None,
        },
        "_trades": sim["trades"],
        "_equity": sim["equity"],
    })
    return summary
