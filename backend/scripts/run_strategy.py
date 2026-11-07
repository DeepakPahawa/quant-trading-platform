"""Backtest a rule-based swing strategy on any archived universe.

Run from the repo root (daily bars must already be backfilled):

    .venv/bin/python -m backend.scripts.run_strategy --strategy all --universe nifty500
    .venv/bin/python -m backend.scripts.run_strategy --strategy rsi2_pullback --universe nifty50 \
        --param rsi_entry_below=5 --param use_regime_filter=false

`--universe NAME` reads backend/data/universe/NAME.csv, so bigger sets work by dropping in a new CSV
(backfill its daily bars first). Writes a JSON summary and a trades CSV to backend/data/research/.
"""

import argparse
import ast
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import polars as pl

from backend.app.backtest.strategy_engine import EngineConfig, run_strategy_backtest
from backend.app.data.daily_archive import DAILY_ARCHIVE_DIR, load_benchmark, load_daily_bars
from backend.app.data.storage import DuckDBStorage
from backend.app.data.universe import UNIVERSE_DIR, load_universe
from backend.app.features.daily_features import to_daily_frame
from backend.app.strategies import STRATEGIES, create_strategy

RESEARCH_DIR = Path("backend/data/research")


def parse_params(items: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--param expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        low = raw.strip().lower()
        if low in ("true", "false"):
            params[key.strip()] = low == "true"
        elif low in ("none", "null"):
            params[key.strip()] = None
        else:
            try:
                params[key.strip()] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                params[key.strip()] = raw
    return params


def fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backtest a rule-based swing strategy.")
    p.add_argument("--strategy", required=True, choices=sorted(STRATEGIES) + ["all"])
    p.add_argument("--universe", required=True, help="Name of a CSV in the universe dir (e.g. nifty50)")
    p.add_argument("--universe-dir", default=str(UNIVERSE_DIR))
    p.add_argument("--archive-dir", default=DAILY_ARCHIVE_DIR)
    p.add_argument("--start", default=None, help="First trading date YYYY-MM-DD (earlier bars only warm up)")
    p.add_argument("--end", default=None)
    p.add_argument("--max-positions", type=int, default=10)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--param", action="append", default=[], help="Strategy parameter override key=value")
    args = p.parse_args(argv)

    params = parse_params(args.param)
    if params and args.strategy == "all":
        print("--param needs a single --strategy (parameters differ per strategy).")
        return 2

    symbols = load_universe(args.universe, Path(args.universe_dir))
    with DuckDBStorage(archive_dir=args.archive_dir) as storage:
        bars = to_daily_frame(load_daily_bars(symbols, storage))
        bench = to_daily_frame(load_benchmark(storage)).select("date", "close")

    config = EngineConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        slippage=args.slippage_bps / 10_000.0,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
    )

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(STRATEGIES) if args.strategy == "all" else [args.strategy]
    print(f"Universe {args.universe}: {bars['symbol'].n_unique()} stocks, "
          f"{bars['date'].min()} .. {bars['date'].max()}. Slots {config.max_positions}, "
          f"slippage {args.slippage_bps:g} bps/side.")
    header = f"{'Strategy':20}{'Trades':>8}{'Hit':>8}{'AvgTrade':>10}{'Total':>9}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'Expo':>7}"
    print(header)
    for name in names:
        try:
            strategy = create_strategy(name, **params)
        except TypeError as exc:
            print(f"Bad --param for {name}: {exc}")
            return 2
        res = run_strategy_backtest(strategy, bars, bench, config)
        trades = res.pop("_trades")
        res.pop("_equity")
        res["universe"] = args.universe
        out = RESEARCH_DIR / f"strategy_{name}_{args.universe}.json"
        out.write_text(json.dumps(res, indent=2, default=str))
        if trades:
            pl.DataFrame(trades).write_csv(RESEARCH_DIR / f"strategy_{name}_{args.universe}_trades.csv")
        port, tr = res["portfolio"], res.get("trades", {})
        print(f"{name:20}{res['n_trades']:>8}{fmt(tr.get('hit_rate')):>8}{fmt(tr.get('avg_return')):>10}"
              f"{port['total_return']:>9.1%}{port['cagr']:>8.1%}{port['sharpe']:>8.2f}"
              f"{port['max_drawdown']:>8.1%}{res['avg_exposure']:>7.0%}")
        for w in res["warnings"]:
            print(f"  WARNING ({name}): {w}")
    bh = res.get("benchmark_buy_hold")
    if bh:
        print(f"{'Nifty 50 buy&hold':20}{'':>8}{'':>8}{'':>10}{bh['total_return']:>9.1%}{bh['cagr']:>8.1%}"
              f"{bh['sharpe']:>8.2f}{bh['max_drawdown']:>8.1%}")
    print(f"Reports in {RESEARCH_DIR}/ (strategy_<name>_{args.universe}.json and _trades.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
