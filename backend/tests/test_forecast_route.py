"""Forecast route tests against an isolated DuckDB/Parquet archive (no FYERS calls)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timezone

from fastapi.testclient import TestClient
import numpy as np
import polars as pl
import pytest

from backend.app.data.storage import DuckDBStorage
from backend.app.main import app
from backend.app.routers.market import get_storage

SYMBOL = "NSE:FORECASTTEST-EQ"


def make_bars(n: int) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.7, n))
    t0 = int(datetime(2026, 3, 2, 3, 45, tzinfo=timezone.utc).timestamp())
    return pl.DataFrame({
        "timestamp": np.arange(n, dtype=np.int64) * 60 + t0,
        "open": close, "high": close + 0.4, "low": close - 0.4, "close": close,
        "volume": rng.integers(800, 4000, n).astype(np.int64),
        "open_interest": np.zeros(n, dtype=np.int64),
    })


@pytest.fixture()
def client(tmp_path: Path):
    storage = DuckDBStorage(db_path=tmp_path / "t.duckdb", archive_dir=tmp_path / "parquet")
    app.dependency_overrides[get_storage] = lambda: storage
    yield TestClient(app), storage
    app.dependency_overrides.clear()
    storage.close()


def test_unknown_symbol_is_buffering(client):
    c, _ = client
    body = c.get("/api/forecast/NSE:NOSUCH-EQ").json()
    assert body["status"] == "BUFFERING"
    assert body["buffer_count"] == 0


def test_short_archive_is_buffering(client):
    c, storage = client
    storage.store_bars(make_bars(30), symbol=SYMBOL)
    body = c.get(f"/api/forecast/{SYMBOL}").json()
    assert body["status"] == "BUFFERING"
    assert body["buffer_count"] == 30


def test_enough_bars_returns_ordered_quantiles(client):
    c, storage = client
    storage.store_bars(make_bars(150), symbol=SYMBOL)
    resp = c.get(f"/api/forecast/{SYMBOL}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "READY"
    assert body["symbol"] == SYMBOL
    assert body["p10"] <= body["p50"] <= body["p90"]
    assert set(body["horizons"]) == {"t+5m", "t+15m"}
    for h in body["horizons"].values():
        assert h["p10"] <= h["p50"] <= h["p90"]
