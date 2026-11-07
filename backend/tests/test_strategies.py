"""Strategy and engine tests on synthetic daily bars: signal sanity, no lookahead, fill mechanics."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import polars as pl
import pytest

from backend.app.backtest.strategy_engine import EngineConfig, run_strategy_backtest, simulate
from backend.app.backtest.swing_backtest import DeliveryCostModel
from backend.app.data.universe import load_universe
from backend.app.strategies import STRATEGIES, BreakoutMomentum, Rsi2Pullback, RsPullback, create_strategy

ZERO_COSTS = DeliveryCostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def weekdays(n: int, start: date = date(2024, 1, 1)) -> List[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_bars(symbol: str, closes, volumes=None, spread: float = 0.005) -> pl.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.r_[closes[0], closes[:-1]]
    return pl.DataFrame({
        "symbol": [symbol] * n,
        "date": weekdays(n),
        "open": opens,
        "high": np.maximum(opens, closes) * (1 + spread),
        "low": np.minimum(opens, closes) * (1 - spread),
        "close": closes,
        "volume": np.full(n, 1000.0) if volumes is None else np.asarray(volumes, dtype=float),
    })


def make_bench(closes) -> pl.DataFrame:
    return make_bars("BENCH", closes).select("date", "close")


# ---------- engine mechanics ----------

def sig_frame(rows: List[dict]) -> pl.DataFrame:
    base = {"entry_signal": False, "exit_signal": False, "priority": 0.0, "atr": 2.0}
    full = [{**base, **r} for r in rows]
    return pl.DataFrame(full).with_columns(pl.col("date").cast(pl.Date))


def bar(sym: str, i: int, o: float, h: float, l: float, c: float, **kw) -> dict:
    return {"symbol": sym, "date": weekdays(40)[i], "open": o, "high": h, "low": l, "close": c, **kw}


CFG = EngineConfig(initial_capital=100_000.0, max_positions=1, slippage=0.0, costs=ZERO_COSTS)


def test_entry_and_exit_fill_at_next_open():
    sig = sig_frame([
        bar("A", 0, 100, 101, 99, 100, entry_signal=True),
        bar("A", 1, 102, 104, 101, 103),                      # entry fills at this open
        bar("A", 2, 103, 106, 102, 105, exit_signal=True),
        bar("A", 3, 108, 109, 107, 108),                      # exit fills at this open
        bar("A", 4, 108, 109, 107, 108),
    ])
    sim = simulate(sig, CFG)
    (t,) = sim["trades"]
    assert t["entry_date"] == weekdays(40)[1] and t["entry_price"] == 102
    assert t["exit_date"] == weekdays(40)[3] and t["exit_price"] == 108
    assert t["reason"] == "signal"
    assert t["shares"] == 980  # int(100000 / 102)
    assert t["pnl"] == pytest.approx(980 * 6)


def test_stop_fills_at_stop_price_and_at_open_on_a_gap():
    intraday = sig_frame([
        bar("A", 0, 100, 101, 99, 100, entry_signal=True, atr=2.0),
        bar("A", 1, 100, 101, 99, 100),   # entry at 100, stop = 100 - 2*2 = 96, low 99
        bar("A", 2, 99, 100, 95, 97),     # low 95 <= 96 -> stopped at 96
        bar("A", 3, 97, 98, 96, 97),
    ])
    (t,) = simulate(intraday, CFG, stop_atr_mult=2.0)["trades"]
    assert t["reason"] == "stop" and t["exit_price"] == 96

    gap = sig_frame([
        bar("A", 0, 100, 101, 99, 100, entry_signal=True, atr=2.0),
        bar("A", 1, 100, 101, 99, 100),
        bar("A", 2, 90, 92, 88, 91),      # opens below the stop -> fills at the open
        bar("A", 3, 91, 92, 90, 91),
    ])
    (t,) = simulate(gap, CFG, stop_atr_mult=2.0)["trades"]
    assert t["reason"] == "stop" and t["exit_price"] == 90


def test_time_stop_and_end_of_data():
    rows = [bar("A", 0, 100, 101, 99, 100, entry_signal=True)] + [
        bar("A", i, 100, 101, 99, 100) for i in range(1, 6)
    ]
    (t,) = simulate(sig_frame(rows), CFG, max_hold_days=3)["trades"]
    assert t["reason"] == "time" and t["bars_held"] == 3
    (t,) = simulate(sig_frame(rows), CFG)["trades"]
    assert t["reason"] == "end_of_data"


def test_slots_are_limited_and_priority_decides():
    rows = []
    for sym, prio in (("A", 1.0), ("B", 5.0), ("C", 3.0)):
        rows += [
            bar(sym, 0, 100, 101, 99, 100, entry_signal=True, priority=prio),
            bar(sym, 1, 100, 101, 99, 100),
            bar(sym, 2, 100, 101, 99, 100),
        ]
    sim = simulate(sig_frame(rows), EngineConfig(100_000.0, 2, 0.0, ZERO_COSTS))
    assert sorted(t["symbol"] for t in sim["trades"]) == ["B", "C"]


def test_costs_and_slippage_reduce_pnl():
    rows = [
        bar("A", 0, 100, 101, 99, 100, entry_signal=True),
        bar("A", 1, 100, 101, 99, 100),
        bar("A", 2, 100, 101, 99, 100, exit_signal=True),
        bar("A", 3, 100, 101, 99, 100),
    ]
    flat = simulate(sig_frame(rows), CFG)["trades"][0]
    real = simulate(sig_frame(rows), EngineConfig(100_000.0, 1, 0.001, DeliveryCostModel()))["trades"][0]
    assert flat["pnl"] == pytest.approx(0.0)
    assert real["pnl"] < -200  # ~0.24% costs + 0.2% slippage on 100k


def test_equity_matches_trade_pnl():
    rows = [
        bar("A", 0, 100, 101, 99, 100, entry_signal=True),
        bar("A", 1, 100, 103, 99, 102),
        bar("A", 2, 102, 106, 101, 105, exit_signal=True),
        bar("A", 3, 106, 107, 105, 106),
    ]
    sim = simulate(sig_frame(rows), CFG)
    assert sim["equity"][-1]["equity"] == pytest.approx(100_000 + sum(t["pnl"] for t in sim["trades"]))


# ---------- strategy signals ----------

def test_registry_and_param_override():
    assert set(STRATEGIES) == {"rsi2_pullback", "breakout_momentum", "rs_pullback"}
    assert create_strategy("rsi2_pullback", rsi_entry_below=5.0).rsi_entry_below == 5.0
    with pytest.raises(ValueError):
        create_strategy("nope")
    with pytest.raises(TypeError):
        create_strategy("rsi2_pullback", not_a_param=1)


def uptrend_then_dip(n_up: int = 300, drops: int = 3, drop: float = 0.03):
    up = 100 * 1.002 ** np.arange(n_up)
    tail = up[-1] * (1 - drop) ** np.arange(1, drops + 1)
    return np.r_[up, tail]


def last_signal(df: pl.DataFrame, sym: str, col: str = "entry_signal") -> bool:
    return bool(df.filter(pl.col("symbol") == sym).sort("date")[col][-1])


def test_rsi2_fires_on_dip_in_uptrend_and_respects_regime():
    stock = make_bars("S", uptrend_then_dip())
    n = stock.height
    bench_up = make_bench(100 * 1.001 ** np.arange(n))
    bench_down = make_bench(300 * 0.998 ** np.arange(n))
    strat = Rsi2Pullback()
    assert last_signal(strat.prepare(stock, bench_up), "S")
    assert not last_signal(strat.prepare(stock, bench_down), "S")                       # regime filter blocks
    assert last_signal(Rsi2Pullback(use_regime_filter=False).prepare(stock, bench_down), "S")


def test_rsi2_exit_when_close_above_sma5():
    closes = uptrend_then_dip()
    stock = make_bars("S", np.r_[closes, closes[-1] * 1.05])  # bounce above the 5-day average
    bench = make_bench(100 * 1.001 ** np.arange(stock.height))
    assert last_signal(Rsi2Pullback().prepare(stock, bench), "S", "exit_signal")


def test_rsi2_no_trend_no_entry():
    down = 100 * 0.998 ** np.arange(320)
    stock = make_bars("S", down)  # RSI(2) is low but price is below its 200d SMA
    bench = make_bench(100 * 1.001 ** np.arange(stock.height))
    sig = Rsi2Pullback().prepare(stock, bench)
    assert not sig["entry_signal"].any()


def flat_noise(n: int, level: float = 100.0):
    return level + 0.5 * (np.arange(n) % 2)


def test_breakout_needs_volume_and_relative_strength():
    closes = np.r_[flat_noise(60), 110.0]
    vol_hi = np.r_[np.full(60, 1000.0), 3000.0]
    vol_lo = np.r_[np.full(60, 1000.0), 1100.0]
    bench = make_bench(np.full(61, 100.0))
    strat = BreakoutMomentum()
    assert last_signal(strat.prepare(make_bars("S", closes, vol_hi), bench), "S")
    assert not last_signal(strat.prepare(make_bars("S", closes, vol_lo), bench), "S")      # no volume
    bench_rocket = make_bench(100 * 1.01 ** np.arange(61))                                 # Nifty up ~82%
    assert not last_signal(strat.prepare(make_bars("S", closes, vol_hi), bench_rocket), "S")  # weaker than Nifty


def test_breakout_exit_on_10_day_low():
    closes = np.r_[flat_noise(60), 110.0, 111.0, 90.0]
    stock = make_bars("S", closes)
    out = BreakoutMomentum().prepare(stock, make_bench(np.full(len(closes), 100.0)))
    assert last_signal(out, "S", "exit_signal")
    assert BreakoutMomentum().stop_atr_mult == 2.0


def rs_universe(dip: bool = True) -> pl.DataFrame:
    leader = np.r_[100 * 1.004 ** np.arange(200)]
    if dip:
        leader = np.r_[leader, leader[-1] * 0.98 ** np.arange(1, 4)]
    n = len(leader)
    frames = [make_bars("LEADER", leader)]
    for i in range(19):
        frames.append(make_bars(f"FLAT{i:02d}", flat_noise(n, 100.0 + i)))
    return pl.concat(frames)


def test_rs_pullback_only_buys_dips_in_top_momentum():
    bars = rs_universe()
    bench = make_bench(np.full(bars.filter(pl.col("symbol") == "LEADER").height, 100.0))
    sig = RsPullback().prepare(bars, bench)
    assert last_signal(sig, "LEADER")
    last_day = sig["date"].max()
    others = sig.filter((pl.col("symbol") != "LEADER") & (pl.col("date") == last_day))
    assert not others["entry_signal"].any()
    # no pullback -> no entry even for the leader
    nodip = rs_universe(dip=False)
    sig2 = RsPullback().prepare(nodip, make_bench(np.full(200, 100.0)))
    assert not last_signal(sig2, "LEADER")


# ---------- no lookahead + end to end ----------

def random_universe(n_symbols: int = 25, n_days: int = 420, seed: int = 1):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_symbols):
        closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n_days)))
        vols = rng.integers(500, 3000, n_days).astype(float)
        frames.append(make_bars(f"S{i:02d}", closes, vols, spread=0.01))
    bench = make_bench(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days))))
    return pl.concat(frames), bench


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_signals_do_not_use_future_data(name: str):
    bars, bench = random_universe()
    strat = create_strategy(name)
    full = strat.prepare(bars, bench)
    cut = weekdays(420)[300]
    part = strat.prepare(bars.filter(pl.col("date") <= cut), bench.filter(pl.col("date") <= cut))
    cols = ["symbol", "date", "entry_signal", "exit_signal", "priority"]
    a = full.filter(pl.col("date") <= cut).select(cols).sort(["symbol", "date"])
    b = part.select(cols).sort(["symbol", "date"])
    assert a.equals(b)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_end_to_end_runs_and_reconciles(name: str):
    bars, bench = random_universe()
    res = run_strategy_backtest(create_strategy(name), bars, bench,
                                EngineConfig(500_000.0, 5, start=weekdays(420)[260]))
    pnl = sum(t["pnl"] for t in res["_trades"])
    assert res["final_equity"] == pytest.approx(500_000.0 + pnl, rel=1e-9)
    assert res["n_symbols"] == 25
    assert all(t["entry_date"] >= weekdays(420)[260] for t in res["_trades"])
    assert all(t["exit_date"] >= t["entry_date"] for t in res["_trades"])
    assert res["warnings"] or res["n_trades"] >= 100


def test_universe_loader_accepts_custom_names(tmp_path):
    (tmp_path / "bigset.csv").write_text("Symbol\nRELIANCE\nTCS\n")
    assert load_universe("bigset", tmp_path) == ["NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
    with pytest.raises(ValueError):
        load_universe("../etc/passwd", tmp_path)
