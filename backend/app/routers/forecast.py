"""TFT forecast endpoint.

Uses archived 1-minute bars only. It never calls FYERS, so a page load cannot
trigger broker traffic, and it never trains (see `.claude/rules/tft-modeling.md`).
"""

import logging

from fastapi import APIRouter, Depends, Query

from backend.app.models.tft_model import (
    SEQUENCE_LENGTH,
    TFTForecastResponse,
    predict_realtime,
)
from backend.app.data.storage import DuckDBStorage
from backend.app.routers.market import get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Forecast"])

# Recent bars passed to inference. Extra history warms up RSI and the rolling volume mean.
LOOKBACK_BARS = 240


@router.get("/forecast/{symbol:path}", response_model=TFTForecastResponse)
def get_tft_forecast(
    symbol: str,
    lookback: int = Query(default=LOOKBACK_BARS, ge=SEQUENCE_LENGTH, le=2000, description="Archived 1-minute bars to use"),
    storage: DuckDBStorage = Depends(get_storage),
) -> TFTForecastResponse:
    """Quantile price forecasts (p10 / p50 / p90) for t+5m and t+15m from archived bars."""
    bars = storage.query_bars(symbol=symbol, resolution="1")
    if bars.is_empty():
        return TFTForecastResponse(
            status="BUFFERING",
            symbol=symbol,
            buffer_count=0,
            required_bars=SEQUENCE_LENGTH,
        )

    recent = bars.tail(lookback)
    if recent.height < SEQUENCE_LENGTH:
        return TFTForecastResponse(
            status="BUFFERING",
            symbol=symbol,
            buffer_count=recent.height,
            required_bars=SEQUENCE_LENGTH,
        )

    result = predict_realtime(recent, symbol=symbol)
    return TFTForecastResponse(**result)
