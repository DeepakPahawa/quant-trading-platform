"""Rank the universe for the latest trading day and print the top-N swing candidates.

Trains on all fully-labelled history, then scores the most recent date.

    .venv/bin/python -m backend.scripts.swing_suggest --universe nifty500 --horizon 10 --top-n 10

Output is research, not investment advice. Check the signal against the backtest report first.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import polars as pl

from backend.app.data.daily_archive import DAILY_ARCHIVE_DIR, load_benchmark, load_daily_bars
from backend.app.data.storage import DuckDBStorage
from backend.app.data.universe import load_universe
from backend.app.models.swing_dataset import build_labeled_dataset
from backend.app.models.swing_model import fit_ranker, score

SUGGESTIONS_DIR = Path("backend/data/research")
DISCLAIMER = "Research output only. Not investment advice. Check the backtest before acting on it."


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Swing candidates for the latest trading day.")
    p.add_argument("--universe", required=True, choices=["nifty50", "nifty500"])
    p.add_argument("--horizon", type=int, default=10, help="Hold length in trading days")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args(argv)

    symbols = load_universe(args.universe)
    with DuckDBStorage(archive_dir=DAILY_ARCHIVE_DIR) as storage:
        bars = load_daily_bars(symbols, storage)
        bench = load_benchmark(storage)

    dataset = build_labeled_dataset(bars, bench, horizon=args.horizon)
    latest = dataset["date"].max()
    train = dataset.filter(pl.col("exit_date").is_not_null())  # every label has matured
    model = fit_ranker(train)

    today = dataset.filter(pl.col("date") == latest)
    today = today.with_columns(pl.Series("score", score(model, today)))
    picks = today.sort("score", descending=True).head(args.top_n)

    cols = ["symbol", "score", "close", "ret_5", "ret_20", "rsi_14", "volume_ratio_20", "rel_ret_20", "high_52w_prox"]
    table = picks.select(cols)
    out = SUGGESTIONS_DIR / f"suggestions_{args.universe}_h{args.horizon}_{latest}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)

    print(f"Latest trading day: {latest}  (universe {args.universe}, hold {args.horizon} trading days)")
    with pl.Config(tbl_rows=args.top_n, tbl_cols=len(cols), fmt_str_lengths=20):
        print(table)
    print(f"Saved: {out}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
