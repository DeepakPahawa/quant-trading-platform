"""DuckDB and Parquet Storage Manager for Historical OHLCV and Market Data."""

from datetime import date, datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

from dotenv import load_dotenv
import duckdb
import polars as pl

from backend.app.data.historical import HISTORICAL_BAR_SCHEMA

load_dotenv()
logger = logging.getLogger(__name__)


def to_epoch_seconds(
    val: Optional[Union[str, date, datetime, int, float]],
    is_end_of_day: bool = False,
) -> Optional[int]:
    """Convert input date/datetime/string/epoch into integer epoch seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.isdigit():
            return int(val_str)
        try:
            # Handle YYYY-MM-DD or ISO strings
            if len(val_str) == 10:
                dt = datetime.strptime(val_str, "%Y-%m-%d")
            else:
                dt = datetime.fromisoformat(val_str)
        except ValueError as ex:
            raise ValueError(f"Unrecognized date/time format: {val}") from ex
    elif isinstance(val, datetime):
        dt = val
    elif isinstance(val, date):
        dt = datetime(val.year, val.month, val.day)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(val)}")

    if is_end_of_day and dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        dt = dt.replace(hour=23, minute=59, second=59)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp())


class DuckDBStorage:
    """
    High-performance storage engine using DuckDB and partitioned Parquet files.
    
    Structure:
      {archive_dir}/symbol={symbol}/year_month={YYYY-MM}/bars.parquet
    """

    DEFAULT_DB_REL_PATH = "backend/data/market_data.duckdb"
    DEFAULT_ARCHIVE_REL_PATH = "backend/data/parquet"

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        archive_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        env_db = os.getenv("DUCKDB_PATH", self.DEFAULT_DB_REL_PATH)
        self.db_path = Path(db_path) if db_path else Path(env_db)

        if archive_dir:
            self.archive_dir = Path(archive_dir)
        else:
            env_archive = os.getenv("PARQUET_ARCHIVE_PATH")
            if env_archive:
                self.archive_dir = Path(env_archive)
            else:
                self.archive_dir = self.db_path.parent / "parquet"

        # Ensure directories exist
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        if self.db_path != Path(":memory:") and str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._con: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        """Get or initialize DuckDB connection."""
        if self._con is None:
            self._con = duckdb.connect(database=str(self.db_path), read_only=False)
        return self._con

    def close(self) -> None:
        """Close open DuckDB connection."""
        if self._con is not None:
            try:
                self._con.close()
            except Exception as e:
                logger.debug("Error closing DuckDB connection: %s", e)
            finally:
                self._con = None

    def __enter__(self) -> "DuckDBStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def store_bars(self, df: pl.DataFrame, symbol: Optional[str] = None) -> int:
        """
        Store bars into the Parquet archive partitioned by symbol and year_month.
        
        Performs in-partition deduplication on timestamp and updates disk archives.
        
        Args:
            df: Typed Polars DataFrame containing OHLCV+OI columns.
            symbol: Symbol string (required if not present as a column in df).
            
        Returns:
            Number of rows in the input DataFrame.
        """
        if df.is_empty():
            logger.info("Empty DataFrame passed to store_bars. Skipping.")
            return 0

        # Ensure symbol column
        if "symbol" not in df.columns:
            if not symbol:
                raise ValueError("symbol must be specified in DataFrame or as an argument")
            df = df.with_columns(pl.lit(symbol).alias("symbol"))

        # Derive year_month if missing
        if "year_month" not in df.columns:
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="s")
                .dt.strftime("%Y-%m")
                .alias("year_month")
            )

        # Cast to standard types
        for col, dtype in HISTORICAL_BAR_SCHEMA.items():
            if col in df.columns and df[col].dtype != dtype:
                df = df.with_columns(pl.col(col).cast(dtype))

        # Group by symbol and year_month to write partition files
        partitions = df.select(["symbol", "year_month"]).unique().to_dicts()

        total_stored = 0
        for part in partitions:
            sym = part["symbol"]
            ym = part["year_month"]

            part_dir = self.archive_dir / f"symbol={sym}" / f"year_month={ym}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_file = part_dir / "bars.parquet"

            part_df = df.filter((pl.col("symbol") == sym) & (pl.col("year_month") == ym))

            # Select only bar columns for the partition file (DuckDB Hive partitioning supplies symbol and year_month)
            bar_cols = [c for c in HISTORICAL_BAR_SCHEMA.keys() if c in part_df.columns]
            part_bars = part_df.select(bar_cols)

            if part_file.exists():
                # Read existing partition, merge, and deduplicate
                try:
                    existing_df = self.con.sql(f"SELECT * FROM read_parquet('{part_file}')").pl()
                    combined_df = (
                        pl.concat([existing_df, part_bars], how="diagonal")
                        .unique(subset=["timestamp"], keep="last")
                        .sort("timestamp")
                    )
                except Exception as ex:
                    logger.warning("Failed reading existing partition %s: %s. Overwriting.", part_file, ex)
                    combined_df = part_bars.unique(subset=["timestamp"], keep="last").sort("timestamp")
            else:
                combined_df = part_bars.unique(subset=["timestamp"], keep="last").sort("timestamp")

            # Write merged partition directly via Polars
            combined_df.write_parquet(part_file, compression="zstd")
            total_stored += len(combined_df)

        logger.info("Stored %d bars across %d partition(s) for symbol '%s'", len(df), len(partitions), symbol or "multi")
        return len(df)

    def query_bars(
        self,
        symbol: str,
        range_from: Optional[Union[str, date, datetime, int, float]] = None,
        range_to: Optional[Union[str, date, datetime, int, float]] = None,
        resolution: str = "1",
    ) -> pl.DataFrame:
        """
        Query time-bounded bar data for a symbol from DuckDB/Parquet storage into Polars.
        
        Args:
            symbol: Product symbol (e.g. 'NSE:NIFTY50-INDEX')
            range_from: Optional start date/datetime/string/epoch
            range_to: Optional end date/datetime/string/epoch
            resolution: Bar resolution ('1', 'D', etc.)
            
        Returns:
            Typed Polars DataFrame sorted chronologically.
        """
        sym_dir = self.archive_dir / f"symbol={symbol}"

        if not sym_dir.exists():
            # Also check URL-encoded symbol directory if applicable
            url_encoded_dir = self.archive_dir / f"symbol={symbol.replace(':', '%3A')}"
            if url_encoded_dir.exists():
                sym_dir = url_encoded_dir
            else:
                return pl.DataFrame(schema=HISTORICAL_BAR_SCHEMA)

        parquet_files = list(sym_dir.glob("*/*.parquet"))
        if not parquet_files:
            return pl.DataFrame(schema=HISTORICAL_BAR_SCHEMA)

        start_ts = to_epoch_seconds(range_from, is_end_of_day=False)
        end_ts = to_epoch_seconds(range_to, is_end_of_day=True)

        # Build SQL query with DuckDB partition reader
        pattern = str(sym_dir / "*/*.parquet")
        where_clauses: List[str] = []

        if start_ts is not None:
            where_clauses.append(f"timestamp >= {start_ts}")
        if end_ts is not None:
            where_clauses.append(f"timestamp <= {end_ts}")

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT DISTINCT ON (timestamp)
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                open_interest
            FROM read_parquet('{pattern}', hive_partitioning=1)
            {where_str}
            ORDER BY timestamp ASC
        """

        try:
            rel = self.con.sql(sql)
            df = rel.pl()
            return df
        except Exception as ex:
            logger.error("Error querying bars for %s: %s", symbol, ex)
            return pl.DataFrame(schema=HISTORICAL_BAR_SCHEMA)

    def get_available_symbols(self) -> List[str]:
        """List all symbols currently stored in the archive."""
        if not self.archive_dir.exists():
            return []
        symbols: List[str] = []
        for p in self.archive_dir.glob("symbol=*"):
            if p.is_dir():
                raw_sym = p.name.split("symbol=", 1)[1]
                symbols.append(unquote(raw_sym))
        return sorted(list(set(symbols)))

    def get_date_range(self, symbol: str) -> Optional[Tuple[int, int]]:
        """Get (min_timestamp, max_timestamp) for a given symbol."""
        sym_dir = self.archive_dir / f"symbol={symbol}"
        if not sym_dir.exists() or not list(sym_dir.glob("*/*.parquet")):
            return None

        pattern = str(sym_dir / "*/*.parquet")
        sql = f"SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM read_parquet('{pattern}')"
        try:
            rows = self.con.sql(sql).fetchall()
            if rows and rows[0] and rows[0][0] is not None and rows[0][1] is not None:
                return int(rows[0][0]), int(rows[0][1])
            return None
        except Exception as ex:
            logger.warning("Failed getting date range for %s: %s", symbol, ex)
            return None
