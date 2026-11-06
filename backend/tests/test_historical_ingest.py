"""Tests for Historical OHLCV ingestion pipeline, DuckDB storage, and REST API."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import polars as pl
import pytest

from backend.app.data.historical import (
    HISTORICAL_BAR_SCHEMA,
    HistoricalBarFetcher,
    normalize_resolution,
    parse_candles_to_polars,
    split_date_range,
)
from backend.app.data.storage import DuckDBStorage, to_epoch_seconds
from backend.app.main import app
from backend.app.routers.market import get_storage


def generate_nifty_5day_1m_candles() -> List[List[Any]]:
    """Generate 5 days (Mon-Fri) of 1-minute OHLCV+OI bars for NSE (375 bars/day = 1875 bars)."""
    days = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
    ist = timezone(timedelta(hours=5, minutes=30))
    candles: List[List[Any]] = []
    base_price = 21500.0

    for day_idx, d_str in enumerate(days):
        day_start = datetime.strptime(d_str, "%Y-%m-%d").replace(hour=9, minute=15, tzinfo=ist)
        start_ts = int(day_start.timestamp())

        for m in range(375):
            ts = start_ts + m * 60
            o = round(base_price + (day_idx * 15.0) + (m * 0.1), 2)
            h = round(o + 2.5, 2)
            l = round(o - 2.0, 2)
            c = round(o + 1.0, 2)
            v = 1000 + m * 2
            oi = 50000 + (day_idx * 1000) + m * 10
            candles.append([ts, o, h, l, c, v, oi])

    return candles


@pytest.fixture
def temp_storage(tmp_path: Path):
    """Fixture providing an isolated DuckDBStorage instance in a temp directory."""
    db_file = tmp_path / "market_test.duckdb"
    archive_dir = tmp_path / "parquet"
    storage = DuckDBStorage(db_path=db_file, archive_dir=archive_dir)
    yield storage
    storage.close()


def test_date_range_pagination():
    """Verify multi-month requests are split into 100-day blocks."""
    # 5 days
    chunks = split_date_range(date(2024, 1, 1), date(2024, 1, 5), max_days=100)
    assert len(chunks) == 1
    assert chunks[0] == (date(2024, 1, 1), date(2024, 1, 5))

    # Exactly 100 days
    chunks = split_date_range(date(2024, 1, 1), date(2024, 4, 10), max_days=100)
    # Jan: 31, Feb: 29 (2024 leap year), Mar: 31, Apr: 10 = 101 days
    assert len(chunks) == 2
    assert chunks[0] == (date(2024, 1, 1), date(2024, 4, 9))
    assert chunks[1] == (date(2024, 4, 10), date(2024, 4, 10))

    # Multi-month (300 days)
    chunks_long = split_date_range(date(2024, 1, 1), date(2024, 10, 27), max_days=100)
    assert len(chunks_long) == 4
    for start, end in chunks_long:
        delta = (end - start).days + 1
        assert delta <= 100

    # Inverted date range
    assert split_date_range(date(2024, 2, 1), date(2024, 1, 1)) == []


def test_parse_candles_to_polars():
    """Verify conversion of raw FYERS candles into strictly typed Polars DataFrame."""
    raw = [
        [1704685500, 21500.0, 21510.0, 21490.0, 21505.0, 1000, 50000],
        [1704685560, 21505.0, 21515.0, 21495.0, 21510.0, 1200],  # 6 elements (no OI)
    ]
    df = parse_candles_to_polars(raw)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == list(HISTORICAL_BAR_SCHEMA.keys())
    assert df["open_interest"][1] == 0  # Defaulted to 0
    assert df["timestamp"].dtype == pl.Int64
    assert df["open"].dtype == pl.Float64
    assert df["volume"].dtype == pl.Int64

    # Empty payload
    empty_df = parse_candles_to_polars([])
    assert empty_df.is_empty()
    assert list(empty_df.columns) == list(HISTORICAL_BAR_SCHEMA.keys())


def test_historical_fetcher_pagination_mock():
    """Verify HistoricalBarFetcher queries in 100-day blocks and aggregates results."""
    mock_client = MagicMock()
    # 2 calls expected for a 150-day span
    mock_client.history.side_effect = [
        {
            "s": "ok",
            "candles": [[1704067200, 100.0, 105.0, 95.0, 102.0, 50, 100]],
        },
        {
            "s": "ok",
            "candles": [[1712707200, 102.0, 108.0, 101.0, 106.0, 60, 120]],
        },
    ]

    fetcher = HistoricalBarFetcher(fyers_client=mock_client)
    df = fetcher.fetch_bars(
        symbol="NSE:NIFTY50-INDEX",
        resolution="1",
        range_from="2024-01-01",
        range_to="2024-05-30",
        inter_chunk_pause_sec=0.0,
    )

    assert mock_client.history.call_count == 2
    assert len(df) == 2
    assert df["timestamp"].to_list() == [1704067200, 1712707200]


def test_duckdb_storage_ingest_and_query(temp_storage: DuckDBStorage):
    """
    Verification Protocol:
    Ingest 5 days of dummy Nifty 1m data into DuckDB, verify row counts, partition structure, and query back.
    """
    symbol = "NSE:NIFTY50-INDEX"
    candles = generate_nifty_5day_1m_candles()
    assert len(candles) == 1875  # 5 days * 375 minutes

    df = parse_candles_to_polars(candles)
    assert len(df) == 1875

    # 1. Ingest into DuckDB storage
    stored_count = temp_storage.store_bars(df, symbol=symbol)
    assert stored_count == 1875

    # 2. Verify Parquet partition folder layout
    expected_partition = temp_storage.archive_dir / f"symbol={symbol}" / "year_month=2024-01" / "bars.parquet"
    assert expected_partition.exists(), f"Partition file missing at {expected_partition}"

    # 3. Query back all 5 days
    result_df = temp_storage.query_bars(symbol=symbol)
    assert len(result_df) == 1875
    assert result_df["timestamp"][0] == candles[0][0]
    assert result_df["timestamp"][-1] == candles[-1][0]
    assert list(result_df.columns) == list(HISTORICAL_BAR_SCHEMA.keys())

    # 4. Verify time-bounded query (Day 2 and Day 3 = 2 * 375 = 750 bars)
    # Day 1 is 0..374; Day 2 is 375..749; Day 3 is 750..1124
    day2_start_ts = candles[375][0]
    day3_end_ts = candles[1124][0]
    bounded_df = temp_storage.query_bars(
        symbol=symbol,
        range_from=day2_start_ts,
        range_to=day3_end_ts,
    )
    assert len(bounded_df) == 750
    assert bounded_df["timestamp"][0] == day2_start_ts
    assert bounded_df["timestamp"][-1] == day3_end_ts

    # 5. Verify symbols and date range metadata
    symbols = temp_storage.get_available_symbols()
    assert symbol in symbols
    date_range = temp_storage.get_date_range(symbol)
    assert date_range == (candles[0][0], candles[-1][0])


def test_storage_overlap_deduplication(temp_storage: DuckDBStorage):
    """Verify that re-ingesting overlapping slices does not create duplicate rows."""
    symbol = "NSE:NIFTY50-INDEX"
    candles = generate_nifty_5day_1m_candles()
    df_all = parse_candles_to_polars(candles)

    # Ingest first 3 days (1125 bars)
    df_first3 = df_all.slice(0, 1125)
    temp_storage.store_bars(df_first3, symbol=symbol)
    assert len(temp_storage.query_bars(symbol=symbol)) == 1125

    # Ingest last 3 days (with 1 day overlap on day 3)
    df_last3 = df_all.slice(750, 1125)
    temp_storage.store_bars(df_last3, symbol=symbol)

    # Total should be exactly 1875 distinct rows
    final_df = temp_storage.query_bars(symbol=symbol)
    assert len(final_df) == 1875


def test_rest_history_endpoint(temp_storage: DuckDBStorage):
    """Verify GET /api/history/{symbol}?resolution=1 endpoint returns data from DuckDB."""
    symbol = "NSE:NIFTY50-INDEX"
    candles = generate_nifty_5day_1m_candles()
    df = parse_candles_to_polars(candles)
    temp_storage.store_bars(df, symbol=symbol)

    # Override FastAPI storage dependency with temp_storage
    app.dependency_overrides[get_storage] = lambda: temp_storage
    client = TestClient(app)

    try:
        # 1. Query full history
        resp = client.get(f"/api/history/{symbol}?resolution=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["symbol"] == symbol
        assert data["resolution"] == "1"
        assert data["count"] == 1875
        assert len(data["candles"]) == 1875
        assert data["candles"][0][0] == candles[0][0]

        # 2. Query time-bounded
        start_date = "2024-01-08"
        end_date = "2024-01-09"
        resp_bounded = client.get(
            f"/api/history/{symbol}?resolution=1&range_from={start_date}&range_to={end_date}"
        )
        assert resp_bounded.status_code == 200
        bounded_data = resp_bounded.json()
        # 2 trading days = 750 bars
        assert bounded_data["count"] == 750

        # 3. Query non-existent symbol
        resp_unknown = client.get("/api/history/NSE:UNKNOWN-EQ?resolution=1")
        assert resp_unknown.status_code == 200
        unknown_data = resp_unknown.json()
        assert unknown_data["count"] == 0
        assert unknown_data["candles"] == []

        # 4. Query /api/symbols
        resp_syms = client.get("/api/symbols")
        assert resp_syms.status_code == 200
        syms_data = resp_syms.json()
        assert symbol in syms_data["symbols"]
    finally:
        app.dependency_overrides.clear()
