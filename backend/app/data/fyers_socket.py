"""FYERS API v3 WebSocket Streaming Client with Automated Backoff and Microstructure Integration.

Features:
- Connects using fyers_apiv3.FyersWebsocket.data_ws.FyersDataSocket.
- Subscribes to SymbolUpdate (channel 11) and DepthUpdate (channel 12).
- Automated exponential backoff reconnection on socket disconnects/errors.
- Buffers incoming ticks in-memory and flushes to DuckDB every 5 seconds.
- Aggregates live ticks into real-time 1-minute OHLCV bars.
- Computes real-time microstructure metrics on depth updates.
"""

from collections import deque
from datetime import datetime, timezone
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from dotenv import load_dotenv
from fyers_apiv3.FyersWebsocket import data_ws
import polars as pl

from backend.app.auth.fyers_auth import FyersAuthManager
from backend.app.data.historical import HISTORICAL_BAR_SCHEMA
from backend.app.data.storage import DuckDBStorage
from backend.app.features.microstructure import (
    MicrostructureMetrics,
    calculate_microstructure_metrics,
)

load_dotenv()
logger = logging.getLogger(__name__)


class BarAggregator:
    """Aggregates high-frequency ticks into 1-minute OHLCV bars."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # symbol -> current bar dict
        self._current_bars: Dict[str, Dict[str, Any]] = {}
        # symbol -> last cumulative volume seen today (for deriving tick volume delta)
        self._last_cum_volume: Dict[str, float] = {}

    def update_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Feed a tick and return a completed 1-minute bar if a minute boundary was crossed."""
        symbol = tick.get("symbol")
        ltp = tick.get("ltp")
        if not symbol or ltp is None:
            return None

        # Extract timestamp, defaulting to current time
        raw_ts = tick.get("last_traded_time") or tick.get("exch_feed_time") or time.time()
        try:
            ts = int(raw_ts)
        except (ValueError, TypeError):
            ts = int(time.time())

        # Floor to 60-second minute boundary
        minute_ts = ts - (ts % 60)

        # Volume calculation
        cum_vol = tick.get("vol_traded_today")
        tick_qty = tick.get("last_traded_qty") or 0.0

        if cum_vol is not None:
            last_vol = self._last_cum_volume.get(symbol)
            if last_vol is not None and cum_vol >= last_vol:
                vol_delta = float(cum_vol - last_vol)
            else:
                vol_delta = float(tick_qty)
            self._last_cum_volume[symbol] = float(cum_vol)
        else:
            vol_delta = float(tick_qty)

        oi = tick.get("OI") or 0

        completed_bar: Optional[Dict[str, Any]] = None

        with self.lock:
            active = self._current_bars.get(symbol)
            if active is not None and active["timestamp"] != minute_ts:
                # The minute rolled over; current bar is completed!
                completed_bar = dict(active)
                # Initialize new bar
                self._current_bars[symbol] = {
                    "symbol": symbol,
                    "timestamp": minute_ts,
                    "open": float(ltp),
                    "high": float(ltp),
                    "low": float(ltp),
                    "close": float(ltp),
                    "volume": vol_delta,
                    "open_interest": oi,
                }
            elif active is None:
                # Initialize first bar
                self._current_bars[symbol] = {
                    "symbol": symbol,
                    "timestamp": minute_ts,
                    "open": float(ltp),
                    "high": float(ltp),
                    "low": float(ltp),
                    "close": float(ltp),
                    "volume": vol_delta,
                    "open_interest": oi,
                }
            else:
                # Update existing bar
                active["high"] = max(active["high"], float(ltp))
                active["low"] = min(active["low"], float(ltp))
                active["close"] = float(ltp)
                active["volume"] += vol_delta
                if oi:
                    active["open_interest"] = oi

        return completed_bar

    def flush_active_bars(self) -> List[Dict[str, Any]]:
        """Flush and return any currently open bars."""
        with self.lock:
            bars = list(self._current_bars.values())
            self._current_bars.clear()
            return bars


class FyersSocketManager:
    """Production-grade WebSocket manager for FYERS streaming API v3."""

    CHANNEL_SYMBOLS = 11
    CHANNEL_DEPTH = 12

    def __init__(
        self,
        access_token: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        storage: Optional[DuckDBStorage] = None,
        flush_interval_seconds: float = 5.0,
        base_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        max_reconnect_attempts: int = 20,
    ) -> None:
        self.access_token = access_token
        self.symbols: Set[str] = set(symbols or [])
        self.storage = storage or DuckDBStorage()
        self.flush_interval_seconds = flush_interval_seconds
        self.base_reconnect_delay = base_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        # Internal state
        self._is_running = False
        self._is_connected = False
        self._reconnect_attempts = 0
        self._ws: Optional[data_ws.FyersDataSocket] = None
        self._lock = threading.Lock()

        # Callbacks
        self._tick_listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._depth_listeners: List[Callable[[Dict[str, Any], MicrostructureMetrics], None]] = []
        self._bar_listeners: List[Callable[[Dict[str, Any]], None]] = []

        # Ticks and Bar Aggregation
        self.bar_aggregator = BarAggregator()
        self._completed_bars_buffer: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self._tick_buffer: Deque[Dict[str, Any]] = deque(maxlen=10000)

        # Background flush and reconnect threads
        self._flush_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _resolve_access_token(self) -> str:
        """Fetch or authenticate access token if not directly passed."""
        if self.access_token:
            return self.access_token

        auth_mgr = FyersAuthManager()
        cached = auth_mgr.load_cached_token()
        if cached:
            self.access_token = cached
            return cached

        token = auth_mgr.authenticate_headless()
        self.access_token = token
        return token

    def add_tick_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for incoming tick updates."""
        with self._lock:
            self._tick_listeners.append(listener)

    def add_depth_listener(
        self, listener: Callable[[Dict[str, Any], MicrostructureMetrics], None]
    ) -> None:
        """Register a callback for incoming depth updates and calculated metrics."""
        with self._lock:
            self._depth_listeners.append(listener)

    def add_bar_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for completed 1-minute OHLCV bars."""
        with self._lock:
            self._bar_listeners.append(listener)

    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to live symbols across SymbolUpdate and DepthUpdate channels."""
        new_symbols = set(symbols)
        with self._lock:
            self.symbols.update(new_symbols)

        if self._is_connected and self._ws is not None:
            sym_list = list(new_symbols)
            try:
                logger.info("Subscribing to SymbolUpdate for symbols: %s", sym_list)
                self._ws.subscribe(
                    symbols=sym_list,
                    data_type="SymbolUpdate",
                    channel=self.CHANNEL_SYMBOLS,
                )
                logger.info("Subscribing to DepthUpdate for symbols: %s", sym_list)
                self._ws.subscribe(
                    symbols=sym_list,
                    data_type="DepthUpdate",
                    channel=self.CHANNEL_DEPTH,
                )
            except Exception as e:
                logger.error("Failed to subscribe symbols on active socket: %s", e)

    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe symbols from the active socket."""
        rem_symbols = set(symbols)
        with self._lock:
            self.symbols.difference_update(rem_symbols)

        if self._is_connected and self._ws is not None:
            try:
                self._ws.unsubscribe(symbols=list(rem_symbols))
            except Exception as e:
                logger.error("Failed to unsubscribe symbols: %s", e)

    def connect(self) -> None:
        """Initialize connection to FYERS WebSocket."""
        if self._is_running:
            logger.warning("FyersSocketManager is already running.")
            return

        self._is_running = True
        self._stop_event.clear()
        token = self._resolve_access_token()

        # Start periodic DuckDB flush thread
        self._flush_thread = threading.Thread(target=self._periodic_flush_worker, daemon=True)
        self._flush_thread.start()

        self._create_and_connect_socket(token)

    def _create_and_connect_socket(self, token: str) -> None:
        """Instantiate FyersDataSocket and connect."""
        try:
            self._ws = data_ws.FyersDataSocket(
                access_token=token,
                reconnect=False,  # We manage custom backoff reconnection explicitly
                on_connect=self._on_open,
                on_close=self._on_close,
                on_error=self._on_error,
                on_message=self._on_message,
                write_to_file=False,
            )
            # FyersDataSocket.connect() starts background threads
            self._ws.connect()
        except Exception as e:
            logger.error("Error creating/connecting FyersDataSocket: %s", e)
            self._schedule_reconnect()

    def _on_open(self) -> None:
        """Socket connection opened callback."""
        logger.info("FYERS WebSocket connection successfully established.")
        self._is_connected = True
        self._reconnect_attempts = 0

        # Resubscribe to registered symbols
        with self._lock:
            active_symbols = list(self.symbols)

        if active_symbols and self._ws is not None:
            try:
                self._ws.subscribe(
                    symbols=active_symbols,
                    data_type="SymbolUpdate",
                    channel=self.CHANNEL_SYMBOLS,
                )
                self._ws.subscribe(
                    symbols=active_symbols,
                    data_type="DepthUpdate",
                    channel=self.CHANNEL_DEPTH,
                )
            except Exception as e:
                logger.error("Error subscribing on connect: %s", e)

    def _on_close(self, message: Any = None) -> None:
        """Socket connection closed callback."""
        logger.warning("FYERS WebSocket connection closed: %s", message)
        self._is_connected = False
        if self._is_running and not self._stop_event.is_set():
            self._schedule_reconnect()

    def _on_error(self, error: Any) -> None:
        """Socket error callback."""
        logger.error("FYERS WebSocket error encountered: %s", error)
        # Check if auth token expired error
        if isinstance(error, dict) and error.get("code") == -300:
            logger.warning("Received invalid/expired token code from FYERS socket.")
            self.access_token = None

        self._is_connected = False
        if self._is_running and not self._stop_event.is_set():
            self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Trigger automated exponential backoff reconnection."""
        with self._lock:
            if self._reconnect_thread and self._reconnect_thread.is_alive():
                return
            if not self._is_running or self._stop_event.is_set():
                return
            self._reconnect_attempts += 1
            if self._reconnect_attempts > self.max_reconnect_attempts:
                logger.error("Exceeded max reconnect attempts (%d). Aborting.", self.max_reconnect_attempts)
                return

            delay = min(
                self.max_reconnect_delay,
                self.base_reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
            )

        def _reconnect_task():
            logger.info("Attempting WebSocket reconnect #%d in %.2fs...", self._reconnect_attempts, delay)
            time.sleep(delay)
            if self._stop_event.is_set():
                return
            try:
                token = self._resolve_access_token()
                self._create_and_connect_socket(token)
            except Exception as ex:
                logger.error("Reconnect attempt failed: %s", ex)
                self._schedule_reconnect()

        self._reconnect_thread = threading.Thread(target=_reconnect_task, daemon=True)
        self._reconnect_thread.start()

    def _on_message(self, message: Dict[str, Any]) -> None:
        """Dispatcher for incoming WebSocket messages."""
        self.process_incoming_message(message)

    def process_incoming_message(self, message: Dict[str, Any]) -> None:
        """Process incoming raw or decoded message payload.
        
        Dispatches ticks, aggregates 1-minute bars, computes depth microstructure metrics.
        """
        if not isinstance(message, dict):
            return

        msg_type = message.get("type")

        # 1. Depth Update Message
        if msg_type == "dp" or "bid_price1" in message or "bids" in message:
            metrics = calculate_microstructure_metrics(message)

            with self._lock:
                listeners = list(self._depth_listeners)
            for listener in listeners:
                try:
                    listener(message, metrics)
                except Exception as e:
                    logger.error("Error in depth listener: %s", e)
            return

        # 2. Tick / Scrip Update Message
        if msg_type in ("sf", "if") or "ltp" in message:
            self._tick_buffer.append(message)

            # Check 1-minute bar aggregation
            completed_bar = self.bar_aggregator.update_tick(message)
            if completed_bar is not None:
                self._completed_bars_buffer.append(completed_bar)
                with self._lock:
                    bar_listeners = list(self._bar_listeners)
                for listener in bar_listeners:
                    try:
                        listener(completed_bar)
                    except Exception as e:
                        logger.error("Error in bar listener: %s", e)

            with self._lock:
                tick_listeners = list(self._tick_listeners)
            for listener in tick_listeners:
                try:
                    listener(message)
                except Exception as e:
                    logger.error("Error in tick listener: %s", e)

    def _periodic_flush_worker(self) -> None:
        """Periodic background task flushing completed bars and buffered data to DuckDB."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval_seconds)
            self.flush_buffers()

    def flush_buffers(self) -> int:
        """Flush queued completed bars to DuckDB/Parquet storage."""
        bars_to_flush: List[Dict[str, Any]] = []
        while self._completed_bars_buffer:
            try:
                bars_to_flush.append(self._completed_bars_buffer.popleft())
            except IndexError:
                break

        if not bars_to_flush:
            return 0

        # Convert to Polars DataFrame matching HISTORICAL_BAR_SCHEMA
        try:
            df = pl.DataFrame(bars_to_flush)
            # Ensure schema compatibility
            for col, dtype in HISTORICAL_BAR_SCHEMA.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(dtype))
                elif col == "open_interest":
                    df = df.with_columns(pl.lit(0).cast(dtype).alias("open_interest"))

            num_stored = self.storage.store_bars(df)
            logger.info("Successfully flushed %d completed 1-minute bars to DuckDB storage.", num_stored)
            return num_stored
        except Exception as e:
            logger.error("Failed flushing bars to DuckDB: %s", e)
            return 0

    def stop(self) -> None:
        """Stop socket and flush remaining buffers."""
        self._is_running = False
        self._is_connected = False
        self._stop_event.set()

        if self._ws is not None:
            try:
                self._ws.close_connection()
            except Exception as e:
                logger.debug("Error closing socket connection: %s", e)
            self._ws = None

        # Flush any open bars from aggregator
        open_bars = self.bar_aggregator.flush_active_bars()
        for b in open_bars:
            self._completed_bars_buffer.append(b)

        self.flush_buffers()
        logger.info("FyersSocketManager stopped.")

