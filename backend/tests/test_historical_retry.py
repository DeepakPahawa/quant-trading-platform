"""Historical fetch: rate-limit retries and recorded (never silent) chunk failures. No live FYERS calls."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import date
from unittest.mock import MagicMock

from backend.app.data.historical import HistoricalBarFetcher, RequestRateLimiter

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


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_limiter_allows_burst_under_cap_without_waiting():
    fc = FakeClock()
    limiter = RequestRateLimiter(max_calls=5, window_sec=60, clock=fc.clock, sleep=fc.sleep)
    for _ in range(5):
        limiter.acquire()
    assert fc.sleeps == []


def test_limiter_waits_for_window_when_cap_reached():
    fc = FakeClock()
    limiter = RequestRateLimiter(max_calls=5, window_sec=60, clock=fc.clock, sleep=fc.sleep)
    for _ in range(5):
        limiter.acquire()
    limiter.acquire()  # 6th request inside the same minute must wait
    assert len(fc.sleeps) == 1 and fc.sleeps[0] >= 60 - 0.0


def test_limiter_never_exceeds_cap_over_many_requests():
    fc = FakeClock()
    limiter = RequestRateLimiter(max_calls=45, window_sec=60, clock=fc.clock, sleep=fc.sleep)
    times = []
    for _ in range(200):
        limiter.acquire()
        times.append(fc.now)
    for i in range(len(times)):
        window = [t for t in times[: i + 1] if times[i] - t < 60]
        assert len(window) <= 45


def test_minute_limit_message_waits_full_window():
    fetcher = make_fetcher([
        {"s": "error", "message": "You've reached the Standard plan limit of 50 requests/minute."},
        {"s": "ok", "candles": OK_CANDLES},
    ])
    waits: list = []
    fetcher.RATE_LIMIT_BACKOFF_SEC = 0.0
    import backend.app.data.historical as hist
    original = hist.time.sleep
    hist.time.sleep = lambda s: waits.append(s)
    try:
        fetcher.fetch_history_chunk("NSE:SBIN-EQ", "D", date(2026, 1, 1), date(2026, 1, 10))
    finally:
        hist.time.sleep = original
    assert fetcher.failed_chunks == []
    assert any(w >= 60 for w in waits)


def test_daily_uses_year_long_chunks_and_intraday_stays_100_days():
    from backend.app.data.historical import split_date_range
    assert len(split_date_range(date(2022, 1, 1), date(2026, 11, 7), max_days=HistoricalBarFetcher.MAX_DAILY_CHUNK_DAYS)) == 5
    assert len(split_date_range(date(2022, 1, 1), date(2026, 11, 7), max_days=HistoricalBarFetcher.MAX_CHUNK_DAYS)) > 15
