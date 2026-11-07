"""Historical fetch: rate-limit retries and recorded (never silent) chunk failures. No live FYERS calls."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import date
from unittest.mock import MagicMock

from backend.app.data.historical import HistoricalBarFetcher

OK_CANDLES = [[1772422500, 1.0, 2.0, 0.5, 1.5, 100, 0]]


def make_fetcher(responses: list) -> HistoricalBarFetcher:
    client = MagicMock()
    client.history.side_effect = responses
    fetcher = HistoricalBarFetcher(fyers_client=client)
    fetcher.RATE_LIMIT_BACKOFF_SEC = 0.0  # keep the test fast
    return fetcher


def test_rate_limit_is_retried_then_succeeds():
    fetcher = make_fetcher([
        {"s": "error", "message": "You've reached the Standard plan limit of 5 requests/second"},
        {"s": "ok", "candles": OK_CANDLES},
    ])
    candles = fetcher.fetch_history_chunk("NSE:SBIN-EQ", "1", date(2026, 1, 1), date(2026, 1, 10))
    assert candles == OK_CANDLES
    assert fetcher.failed_chunks == []
    assert fetcher.client.history.call_count == 2


def test_rejected_chunk_is_recorded_not_silent():
    fetcher = make_fetcher([{"s": "error", "message": "Invalid input"}])
    candles = fetcher.fetch_history_chunk("NSE:SBIN-EQ", "1", date(2025, 1, 1), date(2025, 4, 10))
    assert candles == []
    assert len(fetcher.failed_chunks) == 1
    failure = fetcher.failed_chunks[0]
    assert failure["range_from"] == "2025-01-01"
    assert failure["range_to"] == "2025-04-10"
    assert failure["reason"] == "Invalid input"


def test_non_rate_limit_error_is_not_retried():
    fetcher = make_fetcher([{"s": "error", "message": "Invalid input"}])
    fetcher.fetch_history_chunk("NSE:SBIN-EQ", "1", date(2025, 1, 1), date(2025, 4, 10))
    assert fetcher.client.history.call_count == 1


def test_rate_limit_gives_up_after_retries_and_records_failure():
    fetcher = make_fetcher([{"s": "error", "message": "rate limit"}] * (HistoricalBarFetcher.RATE_LIMIT_RETRIES + 1))
    fetcher.fetch_history_chunk("NSE:SBIN-EQ", "1", date(2026, 1, 1), date(2026, 1, 10))
    assert fetcher.client.history.call_count == HistoricalBarFetcher.RATE_LIMIT_RETRIES + 1
    assert len(fetcher.failed_chunks) == 1
