"""Historical OHLCV and OI data fetcher with chunked pagination for FYERS API v3."""

from datetime import date, datetime, timedelta, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from fyers_apiv3 import fyersModel
import polars as pl

from backend.app.auth.fyers_auth import get_valid_fyers_client

logger = logging.getLogger(__name__)

# Standard typed schema for OHLCV + OI data
HISTORICAL_BAR_SCHEMA: Dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "open_interest": pl.Int64,
}


def normalize_resolution(resolution: Union[str, int]) -> str:
    """Normalize input resolution into FYERS API format ('1', '5', '15', 'D', etc.)."""
    res_str = str(resolution).strip().upper()
    if res_str in ("1", "1M", "1MIN", "MINUTE"):
        return "1"
    if res_str in ("D", "1D", "DAY", "DAILY"):
        return "D"
    if res_str in ("2", "3", "5", "10", "15", "20", "30", "60", "120", "240"):
        return res_str
    # Strip any trailing 'M' or 'MIN' (e.g. '5m' -> '5')
    if res_str.endswith("M") and res_str[:-1].isdigit():
        return res_str[:-1]
    return res_str


def parse_date(value: Union[str, date, datetime, int, float]) -> date:
    """Parse various date/datetime/epoch formats into a datetime.date object."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        # Epoch timestamp in seconds
        return datetime.fromtimestamp(value, tz=timezone.utc).date()
    if isinstance(value, str):
        value_str = value.strip()
        # Check if numeric string (epoch)
        if value_str.isdigit():
            return datetime.fromtimestamp(int(value_str), tz=timezone.utc).date()
        # Try YYYY-MM-DD
        try:
            return datetime.strptime(value_str[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
        # Try ISO format
        try:
            return datetime.fromisoformat(value_str).date()
        except ValueError as exc:
            raise ValueError(f"Unable to parse date string: {value}") from exc

    raise TypeError(f"Unsupported date type: {type(value)} for value {value}")


def split_date_range(start_date: date, end_date: date, max_days: int = 100) -> List[Tuple[date, date]]:
    """
    Split a date range into contiguous blocks of at most max_days.
    
    Prevents FYERS API rate-limit rejections and range-too-large errors.
    """
    if start_date > end_date:
        return []

    chunks: List[Tuple[date, date]] = []
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=max_days - 1), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return chunks


def parse_candles_to_polars(candles: Optional[List[List[Any]]]) -> pl.DataFrame:
    """
    Convert raw FYERS API candle payload into a strictly-typed Polars DataFrame.
    
    Columns: [timestamp, open, high, low, close, volume, open_interest]
    """
    if not candles:
        return pl.DataFrame(schema=HISTORICAL_BAR_SCHEMA)

    timestamps: List[int] = []
    opens: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    volumes: List[int] = []
    open_interests: List[int] = []

    for c in candles:
        if not c or len(c) < 5:
            continue
        try:
            ts = int(c[0])
            o = float(c[1])
            h = float(c[2])
            l = float(c[3])
            cl = float(c[4])
            v = int(c[5]) if len(c) > 5 and c[5] is not None else 0
            oi = int(c[6]) if len(c) > 6 and c[6] is not None else 0

            timestamps.append(ts)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(cl)
            volumes.append(v)
            open_interests.append(oi)
        except (ValueError, TypeError) as ex:
            logger.debug("Skipping malformed candle %s: %s", c, ex)
            continue

    if not timestamps:
        return pl.DataFrame(schema=HISTORICAL_BAR_SCHEMA)

    df = pl.DataFrame(
        {
            "timestamp": pl.Series(timestamps, dtype=pl.Int64),
            "open": pl.Series(opens, dtype=pl.Float64),
            "high": pl.Series(highs, dtype=pl.Float64),
            "low": pl.Series(lows, dtype=pl.Float64),
            "close": pl.Series(closes, dtype=pl.Float64),
            "volume": pl.Series(volumes, dtype=pl.Int64),
            "open_interest": pl.Series(open_interests, dtype=pl.Int64),
        }
    )

    # Deduplicate on timestamp and ensure ascending chronological order
    return df.unique(subset=["timestamp"], keep="last").sort("timestamp")


class HistoricalBarFetcher:
    """Fetches historical OHLCV + OI data from FYERS API v3 with automatic pagination."""

    MAX_CHUNK_DAYS = 100

    def __init__(self, fyers_client: Optional[fyersModel.FyersModel] = None) -> None:
        self._client = fyers_client

    @property
    def client(self) -> fyersModel.FyersModel:
        """Get or lazily initialize the authenticated FYERS API client."""
        if self._client is None:
            self._client = get_valid_fyers_client()
        return self._client

    def fetch_history_chunk(
        self,
        symbol: str,
        resolution: str,
        range_from: date,
        range_to: date,
        cont_flag: int = 1,
    ) -> List[List[Any]]:
        """Fetch a single contiguous chunk of historical data (<= 100 days)."""
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": range_from.strftime("%Y-%m-%d"),
            "range_to": range_to.strftime("%Y-%m-%d"),
            "cont_flag": str(cont_flag),
        }

        logger.debug(
            "Fetching FYERS history chunk: %s %s [%s to %s]",
            symbol, resolution, data["range_from"], data["range_to"]
        )

        response = self.client.history(data=data)

        if not isinstance(response, dict):
            logger.warning("Unexpected FYERS history response: %s", response)
            return []

        status = response.get("s")
        if status == "ok":
            return response.get("candles", [])

        if status == "no_data":
            logger.debug("No data returned for %s [%s to %s]", symbol, range_from, range_to)
            return []

        error_msg = response.get("message") or str(response)
        logger.warning(
            "FYERS history returned status '%s' for %s: %s",
            status, symbol, error_msg
        )
        return []

    def fetch_bars(
        self,
        symbol: str,
        resolution: Union[str, int] = "1",
        range_from: Optional[Union[str, date, datetime, int, float]] = None,
        range_to: Optional[Union[str, date, datetime, int, float]] = None,
        cont_flag: int = 1,
        inter_chunk_pause_sec: float = 0.05,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV + OI bars over an arbitrary date range with chunked 100-day pagination.
        
        Args:
            symbol: FYERS product symbol (e.g. 'NSE:NIFTY50-INDEX', 'NSE:SBIN-EQ')
            resolution: '1' (1 min), 'D' (daily), '5', '15', etc.
            range_from: Start date/datetime/string/epoch. Defaults to 5 days prior to range_to.
            range_to: End date/datetime/string/epoch. Defaults to current date.
            cont_flag: Flag for continuous data (1 for active futures continuity).
            inter_chunk_pause_sec: Throttle delay between chunk requests to respect API rate limits.
            
        Returns:
            Typed Polars DataFrame sorted by timestamp ascending.
        """
        norm_res = normalize_resolution(resolution)

        # Default end_date to today
        end_d = parse_date(range_to) if range_to is not None else date.today()

        # Default start_date to 5 days prior if not specified
        if range_from is not None:
            start_d = parse_date(range_from)
        else:
            start_d = end_d - timedelta(days=5)

        if start_d > end_d:
            logger.warning("start_date %s is after end_date %s. Returning empty DataFrame.", start_d, end_d)
            return pl.DataFrame(schema=HISTORICAL_BAR_SCHEMA)

        chunks = split_date_range(start_d, end_d, max_days=self.MAX_CHUNK_DAYS)
        logger.info(
            "Fetching %s bars for %s from %s to %s across %d chunk(s)",
            norm_res, symbol, start_d, end_d, len(chunks)
        )

        all_candles: List[List[Any]] = []
        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_candles = self.fetch_history_chunk(
                symbol=symbol,
                resolution=norm_res,
                range_from=chunk_start,
                range_to=chunk_end,
                cont_flag=cont_flag,
            )
            all_candles.extend(chunk_candles)

            # Throttle between consecutive chunks if multiple
            if idx < len(chunks) - 1 and inter_chunk_pause_sec > 0:
                time.sleep(inter_chunk_pause_sec)

        df = parse_candles_to_polars(all_candles)
        logger.info("Successfully fetched %d total bars for %s", len(df), symbol)
        return df

