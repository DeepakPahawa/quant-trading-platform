"""Backfill historical bars from FYERS into the local DuckDB/Parquet archive.

Run from the repo root, after logging in (see the FYERS login steps):

    .venv/bin/python -m backend.scripts.backfill_bars \
        --symbols NSE:SBIN-EQ,NSE:RELIANCE-EQ --start 2025-11-01

Notes:
- Intraday history (resolution 1..240) is limited by the FYERS plan. On the Standard plan
  that is roughly the last 365 days. Earlier starts are clamped and a warning is printed.
- Rejected chunks are reported with their date ranges. Re-running is safe: the archive
  de-duplicates on (symbol, timestamp), so only the missing ranges need another run.
- Exit code is 1 if any chunk failed, so you can tell a clean backfill from a partial one.
"""

import argparse
from datetime import date, timedelta
import logging
import sys
from typing import List, Optional

from backend.app.data.historical import HistoricalBarFetcher
from backend.app.data.storage import DuckDBStorage
from backend.app.data.universe import BENCHMARK_SYMBOL, load_universe

# Daily bars are archived separately. The 1-minute archive has no resolution column,
# so mixing resolutions in one archive would silently mix bar types.
DAILY_ARCHIVE_DIR = "backend/data/parquet_daily"

logger = logging.getLogger("backfill_bars")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill FYERS historical bars into the local archive.")
    p.add_argument("--symbols", default=None, help="Comma-separated, e.g. NSE:SBIN-EQ,NSE:RELIANCE-EQ")
    p.add_argument("--universe", default=None, choices=["nifty50", "nifty500"],
                   help="Load symbols from backend/data/universe/<name>.csv and add the Nifty 50 benchmark")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--resolution", default="1", help="1 for 1-minute bars, D for daily")
    p.add_argument(
        "--intraday-lookback-days", type=int, default=365,
        help="Earliest allowed start for intraday resolutions (Standard plan is about 365 days)",
    )
    return p.parse_args(argv)


def effective_start(start: date, end: date, resolution: str, lookback_days: int) -> date:
    """Clamp intraday starts to the plan window. Daily history is not clamped."""
    if resolution.upper() == "D":
        return start
    earliest = end - timedelta(days=lookback_days)
    if start < earliest:
        logger.warning(
            "Start %s is before the intraday window (%s). Using %s. "
            "Longer intraday history needs a higher FYERS plan.",
            start, earliest, earliest,
        )
        return earliest
    return start


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    if args.universe:
        symbols = load_universe(args.universe) + [BENCHMARK_SYMBOL]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        print("Give --symbols or --universe")
        return 2
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = effective_start(date.fromisoformat(args.start), end, args.resolution, args.intraday_lookback_days)
    if start > end:
        logger.error("Start %s is after end %s", start, end)
        return 2

    any_failed = False
    archive_dir = DAILY_ARCHIVE_DIR if args.resolution.upper() == "D" else None
    with DuckDBStorage(archive_dir=archive_dir) as storage:
        fetcher = HistoricalBarFetcher()
        for symbol in symbols:
            df = fetcher.fetch_bars(symbol, resolution=args.resolution,
                                    range_from=start.isoformat(), range_to=end.isoformat())
            if not df.is_empty():
                storage.store_bars(df, symbol=symbol)
            print(f"{symbol}: {df.height} bars fetched for {start} .. {end}")
            for f in fetcher.failed_chunks:
                any_failed = True
                print(f"  FAILED {f['range_from']} .. {f['range_to']}: {f['reason']}")

    if any_failed:
        print("Backfill finished with failed chunks. Fix the cause (plan limit, login) and re-run.")
        return 1
    print("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
