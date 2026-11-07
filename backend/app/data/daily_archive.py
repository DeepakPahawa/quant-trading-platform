"""Load daily bars from the daily archive (kept separate from the 1-minute archive)."""

from typing import List

import polars as pl

from backend.app.data.storage import DuckDBStorage
from backend.app.data.universe import BENCHMARK_SYMBOL

DAILY_ARCHIVE_DIR = "backend/data/parquet_daily"


def load_daily_bars(symbols: List[str], storage: DuckDBStorage) -> pl.DataFrame:
    """Daily bars for `symbols`. Symbols with no archived bars are skipped with a count."""
    parts = []
    missing = []
    for sym in symbols:
        bars = storage.query_bars(symbol=sym, resolution="D")
        if bars.is_empty():
            missing.append(sym)
            continue
        parts.append(bars.with_columns(pl.lit(sym).alias("symbol")))
    if missing:
        print(f"WARNING: {len(missing)} symbol(s) have no daily bars and were skipped: "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    if not parts:
        raise ValueError("No daily bars found. Run backfill_bars with --resolution D first.")
    return pl.concat(parts)


def load_benchmark(storage: DuckDBStorage) -> pl.DataFrame:
    bench = storage.query_bars(symbol=BENCHMARK_SYMBOL, resolution="D")
    if bench.is_empty():
        raise ValueError(f"No daily bars for benchmark {BENCHMARK_SYMBOL}. Backfill it with --resolution D.")
    return bench.with_columns(pl.lit(BENCHMARK_SYMBOL).alias("symbol"))
