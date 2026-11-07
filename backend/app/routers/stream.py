"""FastAPI WebSocket endpoint for broadcasting real-time ticks, bars, and microstructure metrics."""

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.app.data.fyers_socket import FyersSocketManager
from backend.app.features.microstructure import (
    MicrostructureMetrics,
    calculate_microstructure_metrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Market Streaming"])


class MarketStreamBroadcaster:
    """Manages active client WebSocket connections and broadcasts market events."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store running asyncio loop for threadsafe cross-thread broadcasts."""
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        """Accept incoming connection and register client."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("Client connected to /ws/market. Total active: %d", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove disconnected client."""
        self.active_connections.discard(websocket)
        logger.info("Client disconnected from /ws/market. Total active: %d", len(self.active_connections))

    async def broadcast_json(self, payload: Dict[str, Any]) -> None:
        """Asynchronously send JSON payload to all active clients."""
        if not self.active_connections:
            return

        dead_connections: List[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(payload)
            except Exception as e:
                logger.debug("Failed sending to client: %s. Marking for removal.", e)
                dead_connections.append(connection)

        for conn in dead_connections:
            self.active_connections.discard(conn)

    def dispatch_from_thread(self, payload: Dict[str, Any]) -> None:
        """Safely dispatch broadcast payload from background OS threads into asyncio event loop."""
        if not self.active_connections:
            return

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast_json(payload), self._loop)
        else:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast_json(payload))
            except RuntimeError:
                pass


# Global singleton broadcaster
broadcaster = MarketStreamBroadcaster()


def broadcast_tick(tick: Dict[str, Any]) -> None:
    """Publish tick update to WebSocket clients."""
    payload = {
        "type": "tick",
        "symbol": tick.get("symbol"),
        "timestamp": tick.get("last_traded_time") or tick.get("exch_feed_time"),
        "data": tick,
    }
    broadcaster.dispatch_from_thread(payload)


def broadcast_depth(depth: Dict[str, Any], metrics: MicrostructureMetrics) -> None:
    """Publish depth snapshot and microstructure metrics to WebSocket clients."""
    payload = {
        "type": "depth",
        "symbol": metrics.symbol or depth.get("symbol"),
        "timestamp": metrics.timestamp,
        "metrics": metrics.model_dump(),
        "data": depth,
    }
    broadcaster.dispatch_from_thread(payload)


def broadcast_bar(bar: Dict[str, Any]) -> None:
    """Publish completed 1-minute OHLCV bar to WebSocket clients and trigger TFT inference hook."""
    payload = {
        "type": "bar",
        "symbol": bar.get("symbol"),
        "timestamp": bar.get("timestamp"),
        "data": bar,
    }
    broadcaster.dispatch_from_thread(payload)

    # Real-time TFT multi-horizon inference hook
    try:
        from backend.app.models.tft_model import on_candle_completed
        on_candle_completed(bar, broadcaster_dispatch=broadcaster.dispatch_from_thread)
    except Exception as e:
        logger.error("Error executing real-time TFT inference hook: %s", e)


def attach_socket_manager_to_broadcaster(socket_manager: FyersSocketManager) -> None:
    """Wire FyersSocketManager callbacks directly to the WebSocket broadcaster."""
    socket_manager.add_tick_listener(broadcast_tick)
    socket_manager.add_depth_listener(broadcast_depth)
    socket_manager.add_bar_listener(broadcast_bar)


@router.websocket("/ws/market")
async def market_websocket_endpoint(websocket: WebSocket) -> None:
    """FastAPI WebSocket endpoint: ws://localhost:8000/ws/market.
    
    Broadcasts live ticks, completed 1-minute bars, and depth imbalance metrics.
    Clients can also query status or send ping/pong heartbeats.
    """
    loop = asyncio.get_running_loop()
    broadcaster.set_event_loop(loop)
    await broadcaster.connect(websocket)

    # Send initial welcome / handshake ack
    await websocket.send_json({
        "type": "connection_ack",
        "status": "connected",
        "channel": "/ws/market",
    })

    try:
        while True:
            # Await client messages (e.g. ping, subscriptions)
            data_text = await websocket.receive_text()
            try:
                req = json.loads(data_text)
                action = req.get("action")
                if action == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": req.get("timestamp")})
                elif action == "subscribe":
                    symbols = req.get("symbols", [])
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbols": symbols,
                    })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await broadcaster.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket client error: %s", e)
        await broadcaster.disconnect(websocket)

