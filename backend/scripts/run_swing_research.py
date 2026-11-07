"""Walk-forward research for swing trades: does the top-N ranking beat the index after costs?

Run from the repo root (daily bars must already be backfilled):

    .venv/bin/python -m backend.scripts.run_swing_research --universe nifty500 --horizon 10 --top-n 5

Writes a JSON report to backend/data/research/.
"""

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import List, Optional

from backend.app.backtest.swing_backtest import backtest_top_n
from backend.app.data.daily_archive import DAILY_ARCHIVE_DIR, load_benchmark, load_daily_bars
from backend.app.data.storage import DuckDBStorage
from backend.app.data.universe import load_universe
from backend.app.models.swing_dataset import build_labeled_dataset
from backend.app.models.swing_model import walk_forward_predictions

RESEARCH_DIR = Path("backend/data/research")


MIN_TRAIN_YEARS = 2  # the 52-week feature needs about a year, and the model needs more than that


def default_test_starts(first: date, last: date) -> List[date]:
    """One out-of-sample test period per calendar year, after at least MIN_TRAIN_YEARS of history."""
    return [date(y, 1, 1) for y in range(first.year + MIN_TRAIN_YEARS, last.year + 1)]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward swing research.")
    p.add_argument("--universe", required=True, choices=["nifty50", "nifty500"])
    p.add_argument("--horizon", type=int, default=10, help="Hold length in trading days (label horizon)")
    p.add_argument("--top-n", type=int, default=5, help="Stocks held per rebalance")
    p.add_argument("--test-starts", default=None, help="Comma-separated YYYY-MM-DD test-period starts")
    args = p.parse_args(argv)

    symbols = load_universe(args.universe)
    with DuckDBStorage(archive_dir=DAILY_ARCHIVE_DIR) as storage:
        bars = load_daily_bars(symbols, storage)
        bench = load_benchmark(storage)

    dataset = build_labeled_dataset(bars, bench, horizon=args.horizon)
    first = dataset["date"].min()
    last = dataset["date"].max()
    if args.test_starts:
        starts = [date.fromisoformat(s.strip()) for s in args.test_starts.split(",") if s.strip()]
    else:
        starts = default_test_starts(first, last)
    if not starts:
        print(f"Not enough history ({first} .. {last}) for a test period. Backfill more years of daily bars.")
        return 1

    predictions = walk_forward_predictions(dataset, starts)
    result = backtest_top_n(predictions, horizon=args.horizon, top_n=args.top_n)

    report = {
        "universe": args.universe,
        "n_symbols": bars["symbol"].n_unique(),
        "data_range": [str(first), str(last)],
        "test_starts": [str(s) for s in starts],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "result": result,
        "assumptions": [
            "Entry at next-day open, exit at close H trading days later.",
            "Delivery costs from backend/app/backtest/swing_backtest.py DeliveryCostModel. Verify the rates.",
            "Equal-weight positions. Rebalance every H trading days, so holding periods do not overlap.",
        ],
    }
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out = RESEARCH_DIR / f"swing_{args.universe}_h{args.horizon}_top{args.top_n}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    s, u, n = result["strategy"], result["universe_equal_weight"], result["nifty50"]
    print(f"Universe {args.universe}: {report['n_symbols']} stocks, {report['data_range'][0]} .. {report['data_range'][1]}")
    print(f"Horizon {args.horizon}d, top {args.top_n}, round-trip cost {result['round_trip_cost_pct']}%")
    print(f"{'':22}{'Strategy':>12}{'Universe avg':>14}{'Nifty 50':>12}")
    print(f"{'Total return':22}{s['total_return']:>12.1%}{u['total_return']:>14.1%}{n['total_return']:>12.1%}")
    print(f"{'CAGR':22}{s['cagr']:>12.1%}{u['cagr']:>14.1%}{n['cagr']:>12.1%}")
    print(f"{'Sharpe':22}{s['sharpe']:>12.2f}{u['sharpe']:>14.2f}{n['sharpe']:>12.2f}")
    print(f"{'Max drawdown':22}{s['max_drawdown']:>12.1%}{u['max_drawdown']:>14.1%}{n['max_drawdown']:>12.1%}")
    print(f"Hit rate {s['hit_rate']:.1%} over {result['n_trades']} trades")
    if result["n_trades"] < 100:
        print(f"WARNING: only {result['n_trades']} trades. Differences of this size are often noise. "
              "Treat the result as inconclusive until you have 100+ trades (more years, or a larger universe).")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
