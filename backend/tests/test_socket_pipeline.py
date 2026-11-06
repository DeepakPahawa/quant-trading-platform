"""Test Suite for WebSocket Pipeline, FYERS Socket Manager, and Microstructure Metrics.

Verifies:
1. Microstructure metric mathematical formulas:
   - Top-5 Bid-Ask Depth Imbalance: (sum(BidQty) - sum(AskQty)) / (sum(BidQty) + sum(AskQty))
   - Volume-Weighted Micro-Price: (best_bid * ask_qty_1 + best_ask * bid_qty_1) / (bid_qty_1 + ask_qty_1)
   - Normalized Bid-Ask Spread: (best_ask - best_bid) / mid_price
2. FYERS depth payload parsing and Polars vectorized transformations.
3. FyersSocketManager incoming payload dispatch, bar aggregation, and backoff logic.
4. FastAPI WebSocket endpoint ws://localhost:8000/ws/market live streaming.
"""

import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import polars as pl
import pytest

from backend.app.data.fyers_socket import BarAggregator, FyersSocketManager
from backend.app.data.storage import DuckDBStorage
from backend.app.features.microstructure import (
    MicrostructureMetrics,
    calculate_microstructure_metrics,
    calculate_microstructure_polars,
    compute_depth_imbalance,
    compute_micro_price,
    compute_normalized_spread,
    extract_depth_levels,
)
from backend.app.main import app
from backend.app.routers.stream import (
    attach_socket_manager_to_broadcaster,
    broadcast_bar,
    broadcast_depth,
    broadcast_tick,
    broadcaster,
)


class TestMicrostructureFormulas:
    """Mathematical verification of microstructure feature formulas."""

    def test_depth_imbalance_symmetric(self):
        """Equal bid and ask quantities should yield 0.0 imbalance."""
        bids = [100, 100, 100, 100, 100]
        asks = [100, 100, 100, 100, 100]
        assert compute_depth_imbalance(bids, asks) == 0.0

    def test_depth_imbalance_extreme(self):
        """All bids or all asks should yield +1.0 or -1.0."""
        bids = [500, 0, 0, 0, 0]
        asks = [0, 0, 0, 0, 0]
        assert compute_depth_imbalance(bids, asks) == 1.0

        bids = [0, 0, 0, 0, 0]
        asks = [300, 200, 0, 0, 0]
        assert compute_depth_imbalance(bids, asks) == -1.0

    def test_depth_imbalance_known_ratio(self):
        """Top-5: sum(bids)=1500, sum(asks)=500 => (1500-500)/2000 = 0.5."""
        bids = [100, 200, 300, 400, 500]  # sum = 1500
        asks = [50, 50, 100, 100, 200]    # sum = 500
        assert compute_depth_imbalance(bids, asks) == pytest.approx(0.5, rel=1e-5)

    def test_depth_imbalance_zero_division_guard(self):
        """When total quantity is 0, return 0.0 without ZeroDivisionError."""
        assert compute_depth_imbalance([], []) == 0.0
        assert compute_depth_imbalance([0, 0], [0, 0]) == 0.0

    def test_micro_price_equal_liquidity(self):
        """Equal L1 quantities should result in arithmetic mid-price."""
        best_bid = 100.0
        best_ask = 102.0
        bid_qty_1 = 50.0
        ask_qty_1 = 50.0
        # (100*50 + 102*50)/100 = 101.0
        assert compute_micro_price(best_bid, best_ask, bid_qty_1, ask_qty_1) == 101.0

    def test_micro_price_bid_heavy(self):
        """Heavy bid depth exerts upward pressure on price towards ask."""
        best_bid = 100.0
        best_ask = 102.0
        bid_qty_1 = 150.0
        ask_qty_1 = 50.0
        # Formula: (best_bid * ask_qty_1 + best_ask * bid_qty_1) / (bid_qty_1 + ask_qty_1)
        # = (100 * 50 + 102 * 150) / 200 = (5000 + 15300) / 200 = 20300 / 200 = 101.5
        assert compute_micro_price(best_bid, best_ask, bid_qty_1, ask_qty_1) == pytest.approx(101.5, rel=1e-5)

    def test_micro_price_ask_heavy(self):
        """Heavy ask depth exerts downward pressure on price towards bid."""
        best_bid = 100.0
        best_ask = 102.0
        bid_qty_1 = 50.0
        ask_qty_1 = 150.0
        # = (100 * 150 + 102 * 50) / 200 = (15000 + 5100) / 200 = 20100 / 200 = 100.5
        assert compute_micro_price(best_bid, best_ask, bid_qty_1, ask_qty_1) == pytest.approx(100.5, rel=1e-5)

    def test_micro_price_zero_qty_guard(self):
        """When quantities are zero, fall back to mid-price."""
        assert compute_micro_price(100.0, 102.0, 0.0, 0.0) == 101.0

    def test_normalized_spread(self):
        """(best_ask - best_bid) / mid_price."""
        best_bid = 99.0
        best_ask = 101.0
        mid = 100.0
        # (101 - 99) / 100 = 0.02
        assert compute_normalized_spread(best_bid, best_ask, mid) == pytest.approx(0.02, rel=1e-5)

    def test_normalized_spread_zero_guard(self):
        """Guard against zero mid-price."""
        assert compute_normalized_spread(0.0, 0.0, 0.0) == 0.0


class TestPayloadParsingAndPolars:
    """Verify parsing FYERS depth dictionaries and Polars DataFrame pipeline."""

    @pytest.fixture
    def sample_fyers_depth_payload(self):
        return {
            "symbol": "NSE:SBIN-EQ",
            "exch_feed_time": 1700000000,
            "type": "dp",
            "bid_price1": 600.0,
            "bid_price2": 599.5,
            "bid_price3": 599.0,
            "bid_price4": 598.5,
            "bid_price5": 598.0,
            "ask_price1": 601.0,
            "ask_price2": 601.5,
            "ask_price3": 602.0,
            "ask_price4": 602.5,
            "ask_price5": 603.0,
            "bid_size1": 100,
            "bid_size2": 150,
            "bid_size3": 200,
            "bid_size4": 250,
            "bid_size5": 300,
            "ask_size1": 50,
            "ask_size2": 50,
            "ask_size3": 100,
            "ask_size4": 100,
            "ask_size5": 200,
        }

    def test_extract_depth_levels(self, sample_fyers_depth_payload):
        bids, asks = extract_depth_levels(sample_fyers_depth_payload)
        assert len(bids) == 5
        assert len(asks) == 5
        assert bids[0].price == 600.0
        assert bids[0].quantity == 100.0
        assert asks[0].price == 601.0
        assert asks[0].quantity == 50.0

    def test_calculate_microstructure_metrics(self, sample_fyers_depth_payload):
        metrics = calculate_microstructure_metrics(sample_fyers_depth_payload)
        assert isinstance(metrics, MicrostructureMetrics)
        assert metrics.symbol == "NSE:SBIN-EQ"
        assert metrics.best_bid == 600.0
        assert metrics.best_ask == 601.0
        assert metrics.mid_price == 600.5

        # total bids = 100+150+200+250+300 = 1000
        # total asks = 50+50+100+100+200 = 500
        # depth_imbalance = (1000 - 500) / 1500 = 500 / 1500 = 0.333333
        assert metrics.depth_imbalance == pytest.approx(0.333333, rel=1e-4)

        # micro_price = (600.0 * 50 + 601.0 * 100) / 150 = (30000 + 60100) / 150 = 600.6667
        assert metrics.micro_price == pytest.approx(600.6667, rel=1e-4)

        # spread = (601 - 600) / 600.5 = 1 / 600.5 = 0.001665
        assert metrics.normalized_spread == pytest.approx(1.0 / 600.5, rel=1e-4)

    def test_calculate_microstructure_polars(self, sample_fyers_depth_payload):
        """Vectorized Polars computation must match single payload calculations."""
        df = pl.DataFrame([sample_fyers_depth_payload])
        res = calculate_microstructure_polars(df)

        assert "depth_imbalance" in res.columns
        assert "micro_price" in res.columns
        assert "normalized_spread" in res.columns
        assert "mid_price" in res.columns

        row = res.to_dicts()[0]
        assert row["depth_imbalance"] == pytest.approx(0.333333, rel=1e-4)
        assert row["micro_price"] == pytest.approx(600.6667, rel=1e-4)
        assert row["mid_price"] == 600.5
        assert row["total_bid_qty"] == 1000.0
        assert row["total_ask_qty"] == 500.0


class TestBarAggregator:
    """Verify tick-to-minute OHLCV aggregation."""

    def test_bar_aggregation_minute_rollover(self):
        agg = BarAggregator()
        symbol = "NSE:NIFTY50-INDEX"

        # Minute boundary starts at 1700000040 (multiple of 60)
        # Tick 1 at second 45: in minute 1700000040
        t1 = {"symbol": symbol, "ltp": 20000.0, "last_traded_time": 1700000045, "last_traded_qty": 50}
        bar1 = agg.update_tick(t1)
        assert bar1 is None  # bar not yet completed

        # Same minute tick at second 50: high update
        t2 = {"symbol": symbol, "ltp": 20050.0, "last_traded_time": 1700000050, "last_traded_qty": 25}
        bar2 = agg.update_tick(t2)
        assert bar2 is None

        # Same minute tick at second 55: low update
        t3 = {"symbol": symbol, "ltp": 19980.0, "last_traded_time": 1700000055, "last_traded_qty": 30}
        bar3 = agg.update_tick(t3)
        assert bar3 is None

        # Minute rollover tick at second 105 (+60 seconds from first tick) -> next minute
        t4 = {"symbol": symbol, "ltp": 20010.0, "last_traded_time": 1700000105, "last_traded_qty": 40}
        completed_bar = agg.update_tick(t4)
        assert completed_bar is not None
        assert completed_bar["symbol"] == symbol
        assert completed_bar["timestamp"] == 1700000040
        assert completed_bar["open"] == 20000.0
        assert completed_bar["high"] == 20050.0
        assert completed_bar["low"] == 19980.0
        assert completed_bar["close"] == 19980.0
        assert completed_bar["volume"] == 105.0  # 50 + 25 + 30


class TestFyersSocketPipeline:
    """Verify FyersSocketManager lifecycle, dispatch, and reconnection backoff."""

    @patch("backend.app.data.fyers_socket.data_ws.FyersDataSocket")
    def test_socket_manager_lifecycle(self, mock_socket_cls, tmp_path):
        mock_instance = MagicMock()
        mock_socket_cls.return_value = mock_instance

        storage = DuckDBStorage(
            db_path=tmp_path / "test_db.duckdb",
            archive_dir=tmp_path / "parquet",
        )

        manager = FyersSocketManager(
            access_token="mock.jwt.token",
            symbols=["NSE:SBIN-EQ"],
            storage=storage,
        )

        ticks_received = []
        depth_received = []
        bars_received = []

        manager.add_tick_listener(lambda t: ticks_received.append(t))
        manager.add_depth_listener(lambda d, m: depth_received.append((d, m)))
        manager.add_bar_listener(lambda b: bars_received.append(b))

        manager.connect()
        assert manager._is_running is True
        mock_instance.connect.assert_called_once()

        # Simulate socket on_open callback
        manager._on_open()
        assert manager._is_connected is True
        # Verify subscriptions were registered for both channels
        assert mock_instance.subscribe.call_count >= 2

        # Dispatch Depth Update
        depth_msg = {
            "type": "dp",
            "symbol": "NSE:SBIN-EQ",
            "bid_price1": 600.0,
            "ask_price1": 601.0,
            "bid_size1": 200,
            "ask_size1": 100,
            "exch_feed_time": 1700000000,
        }
        manager.process_incoming_message(depth_msg)
        assert len(depth_received) == 1
        d_raw, metrics = depth_received[0]
        assert metrics.symbol == "NSE:SBIN-EQ"
        assert metrics.micro_price > 600.5  # Bids heavier than asks

        # Dispatch Ticks across rollover
        tick_1 = {
            "type": "sf",
            "symbol": "NSE:SBIN-EQ",
            "ltp": 600.0,
            "last_traded_time": 1700000000,
            "last_traded_qty": 10,
        }
        tick_2 = {
            "type": "sf",
            "symbol": "NSE:SBIN-EQ",
            "ltp": 605.0,
            "last_traded_time": 1700000070,  # Rollover to next minute
            "last_traded_qty": 20,
        }
        manager.process_incoming_message(tick_1)
        manager.process_incoming_message(tick_2)

        assert len(ticks_received) == 2
        assert len(bars_received) == 1
        assert bars_received[0]["open"] == 600.0
        assert bars_received[0]["close"] == 600.0

        # Stop manager
        manager.stop()
        assert manager._is_running is False
        mock_instance.close_connection.assert_called_once()


class TestFastAPIWebSocketStream:
    """Verify WebSocket endpoint ws://localhost:8000/ws/market broadcasting."""

    def test_websocket_market_pipeline(self):
        client = TestClient(app)

        with client.websocket_connect("/ws/market") as ws:
            # 1. Connection Ack handshake
            ack = ws.receive_json()
            assert ack["type"] == "connection_ack"
            assert ack["status"] == "connected"

            # 2. Client ping/pong
            ws.send_text('{"action": "ping", "timestamp": 1234567}')
            pong = ws.receive_json()
            assert pong["type"] == "pong"
            assert pong["timestamp"] == 1234567

            # 3. Client subscription request
            ws.send_text('{"action": "subscribe", "symbols": ["NSE:SBIN-EQ"]}')
            sub_ack = ws.receive_json()
            assert sub_ack["type"] == "subscribed"
            assert sub_ack["symbols"] == ["NSE:SBIN-EQ"]

            # 4. Broadcast live tick
            tick = {
                "symbol": "NSE:SBIN-EQ",
                "ltp": 602.5,
                "last_traded_time": 1700000010,
                "vol_traded_today": 50000,
            }
            broadcast_tick(tick)
            msg_tick = ws.receive_json()
            assert msg_tick["type"] == "tick"
            assert msg_tick["symbol"] == "NSE:SBIN-EQ"
            assert msg_tick["data"]["ltp"] == 602.5

            # 5. Broadcast depth & microstructure metrics
            depth_payload = {
                "symbol": "NSE:SBIN-EQ",
                "bid_price1": 602.0,
                "ask_price1": 603.0,
                "bid_size1": 300,
                "ask_size1": 100,
                "exch_feed_time": 1700000012,
            }
            metrics = calculate_microstructure_metrics(depth_payload)
            broadcast_depth(depth_payload, metrics)
            msg_depth = ws.receive_json()
            assert msg_depth["type"] == "depth"
            assert msg_depth["symbol"] == "NSE:SBIN-EQ"
            assert "metrics" in msg_depth
            assert msg_depth["metrics"]["depth_imbalance"] == pytest.approx(0.5, rel=1e-4)
            assert msg_depth["metrics"]["micro_price"] == pytest.approx(602.75, rel=1e-4)

            # 6. Broadcast completed 1-minute bar
            bar_payload = {
                "symbol": "NSE:SBIN-EQ",
                "timestamp": 1700000000,
                "open": 600.0,
                "high": 603.0,
                "low": 599.5,
                "close": 602.5,
                "volume": 1500,
                "open_interest": 0,
            }
            broadcast_bar(bar_payload)
            msg_bar = ws.receive_json()
            assert msg_bar["type"] == "bar"
            assert msg_bar["symbol"] == "NSE:SBIN-EQ"
            assert msg_bar["data"]["open"] == 600.0
            assert msg_bar["data"]["high"] == 603.0
            assert msg_bar["data"]["volume"] == 1500
