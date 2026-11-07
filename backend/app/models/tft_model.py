"""Temporal Fusion Transformer (TFT) Multi-Horizon Price and Return Forecasting.

Implements:
1. TFTPriceForecaster: PyTorch Forecasting Temporal Fusion Transformer configured for:
   - Input shape: [batch_size, 60_bars, n_features]
   - Multivariate continuous inputs:
     [log_return, volume_ratio, rsi_14, vwap_diff, depth_imbalance, micro_price_spread]
   - Time covariates: normalized minute-of-day (0 to 375) and day-of-week
   - Quantile predictions (p10, p50, p90) for horizons t+5m and t+15m
2. predict_realtime(feature_buffer): Real-time inference hook in torch.no_grad() mode:
   - Returns {"status": "BUFFERING"} if buffer has < 60 bars.
   - Computes multi-horizon quantile price forecasts and directional bias.
3. CandleBufferManager & on_candle_completed: Real-time hook for market WebSocket broadcast.
"""

from collections import defaultdict
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field
import torch
import torch.nn as nn

# Suppress PyTorch Lightning hyperparameter saving warnings during module instantiation
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

from backend.app.data.market_overview import prepare_tft_feature_tensor

logger = logging.getLogger(__name__)

# TFT Skill Specifications
SEQUENCE_LENGTH = 60
PREDICTION_HORIZONS = [5, 15]
MAX_PREDICTION_LENGTH = 15
QUANTILES = [0.1, 0.5, 0.9]

CONTINUOUS_FEATURES = [
    "log_return",
    "volume_ratio",
    "rsi_14",
    "vwap_diff",
    "depth_imbalance",
    "micro_price_spread",
]
TIME_COVARIATES = ["minute_of_day", "day_of_week"]
ALL_FEATURES = CONTINUOUS_FEATURES + TIME_COVARIATES


class HorizonForecast(BaseModel):
    """Quantile predictions for a single forecast horizon."""
    p10: float = Field(..., description="10th percentile forecasted price")
    p50: float = Field(..., description="50th percentile (median) forecasted price")
    p90: float = Field(..., description="90th percentile forecasted price")
    p10_return: float = Field(..., description="10th percentile expected return")
    p50_return: float = Field(..., description="Median expected return")
    p90_return: float = Field(..., description="90th percentile expected return")
    directional_bias: str = Field(..., description="BULLISH, BEARISH, or NEUTRAL")


class TFTForecastResponse(BaseModel):
    """Standardized response schema for TFT real-time inference."""
    status: str = Field(..., description="READY or BUFFERING")
    symbol: Optional[str] = Field(default=None, description="Trading product symbol")
    timestamp: Optional[int] = Field(default=None, description="Current epoch second timestamp")
    current_price: Optional[float] = Field(default=None, description="Reference close price")
    p10: Optional[float] = Field(default=None, description="t+15m p10 forecasted price")
    p50: Optional[float] = Field(default=None, description="t+15m p50 forecasted price")
    p90: Optional[float] = Field(default=None, description="t+15m p90 forecasted price")
    directional_bias: Optional[str] = Field(default=None, description="BULLISH, BEARISH, or NEUTRAL")
    horizons: Optional[Dict[str, HorizonForecast]] = Field(default=None, description="Per-horizon forecast details")
    buffer_count: Optional[int] = Field(default=None, description="Current bars in buffer if buffering")
    required_bars: Optional[int] = Field(default=SEQUENCE_LENGTH, description="Required bars for inference")


class TFTPriceForecaster(nn.Module):
    """Temporal Fusion Transformer configured for 60-bar multivariate price return forecasting."""

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        prediction_length: int = MAX_PREDICTION_LENGTH,
        hidden_size: int = 16,
        attention_head_size: int = 2,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.quantiles = QUANTILES

        # Initialize underlying PyTorch Forecasting TemporalFusionTransformer
        total_len = sequence_length + prediction_length
        dummy_df = pd.DataFrame({
            "time_idx": list(range(total_len)),
            "group": ["0"] * total_len,
            "log_return": [0.0] * total_len,
            "volume_ratio": [1.0] * total_len,
            "rsi_14": [50.0] * total_len,
            "vwap_diff": [0.0] * total_len,
            "depth_imbalance": [0.0] * total_len,
            "micro_price_spread": [0.0] * total_len,
            "minute_of_day": [float(i % 375) for i in range(total_len)],
            "day_of_week": [1.0] * total_len,
            "target": [0.0] * total_len,
        })

        self.dataset = TimeSeriesDataSet(
            dummy_df,
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            max_encoder_length=sequence_length,
            min_encoder_length=sequence_length,
            max_prediction_length=prediction_length,
            min_prediction_length=prediction_length,
            time_varying_known_reals=TIME_COVARIATES,
            time_varying_unknown_reals=CONTINUOUS_FEATURES,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tft = TemporalFusionTransformer.from_dataset(
                self.dataset,
                learning_rate=0.03,
                hidden_size=hidden_size,
                attention_head_size=attention_head_size,
                dropout=dropout,
                hidden_continuous_size=hidden_continuous_size,
                loss=QuantileLoss(self.quantiles),
            )

    def _prepare_input_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert a [batch_size, seq_len, n_features] tensor into the dictionary expected by TFT."""
        batch_size, seq_len, n_features = x.shape
        device = x.device

        if n_features == 6:
            # Append synthesized normalized time covariates: minute_of_day (0-375) and day_of_week
            minute_cov = torch.linspace(0, 375, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
            dow_cov = torch.ones(batch_size, seq_len, 1, device=device)
            encoder_cont = torch.cat([x, minute_cov, dow_cov], dim=-1)
        elif n_features == 8:
            encoder_cont = x
        elif n_features < 8:
            pad = torch.zeros(batch_size, seq_len, 8 - n_features, device=device)
            encoder_cont = torch.cat([x, pad], dim=-1)
        else:
            encoder_cont = x[:, :, :8]

        pred_len = self.prediction_length
        decoder_cont = torch.zeros(batch_size, pred_len, 8, device=device)

        # Propagate forward time covariates into decoder steps
        last_min = encoder_cont[:, -1:, 6:7]
        future_mins = ((last_min + torch.arange(1, pred_len + 1, device=device).unsqueeze(0).unsqueeze(-1)) % 375.0)
        decoder_cont[:, :, 6:7] = future_mins
        decoder_cont[:, :, 7:8] = encoder_cont[:, -1:, 7:8].repeat(1, pred_len, 1)

        return {
            "encoder_cat": torch.zeros(batch_size, seq_len, 0, dtype=torch.int64, device=device),
            "encoder_cont": encoder_cont.float(),
            "encoder_target": encoder_cont[:, :, 0].float(),
            "encoder_lengths": torch.full((batch_size,), seq_len, dtype=torch.int64, device=device),
            "decoder_cat": torch.zeros(batch_size, pred_len, 0, dtype=torch.int64, device=device),
            "decoder_cont": decoder_cont.float(),
            "decoder_target": torch.zeros(batch_size, pred_len, dtype=torch.float32, device=device),
            "decoder_lengths": torch.full((batch_size,), pred_len, dtype=torch.int64, device=device),
            "decoder_time_idx": torch.arange(seq_len, seq_len + pred_len, device=device).unsqueeze(0).repeat(batch_size, 1),
            "groups": torch.zeros(batch_size, 1, dtype=torch.int64, device=device),
            "target_scale": torch.tensor([[0.0, 1.0]], device=device).repeat(batch_size, 1),
        }

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward inference pass.
        
        Args:
            x: Input tensor [batch_size, 60, n_features] or raw TFT input dict.
            
        Returns:
            Quantile prediction tensor [batch_size, 15, 3] sorted such that
            p10 <= p50 <= p90 across all horizons.
        """
        if isinstance(x, torch.Tensor):
            x_dict = self._prepare_input_dict(x)
        else:
            x_dict = x

        raw_pred = self.tft(x_dict).prediction
        # Quantile sorting guarantees strictly non-crossing quantiles: p10 <= p50 <= p90
        sorted_pred, _ = torch.sort(raw_pred, dim=-1)
        return sorted_pred

    def predict_horizons(
        self,
        x: torch.Tensor,
        horizons: List[int] = PREDICTION_HORIZONS,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-horizon quantile predictions for specified forward steps (e.g., t+5m, t+15m)."""
        pred = self.forward(x)  # [batch_size, 15, 3]
        results = {}
        for h in horizons:
            # 1-based horizon step to 0-based index
            idx = min(max(h - 1, 0), self.prediction_length - 1)
            results[f"t+{h}m"] = pred[:, idx, :]
        return results


# Module-level model singleton
_global_tft_model: Optional[TFTPriceForecaster] = None
_model_lock = threading.Lock()


def get_tft_model() -> TFTPriceForecaster:
    """Retrieve or initialize the global singleton TFT model in eval mode."""
    global _global_tft_model
    if _global_tft_model is None:
        with _model_lock:
            if _global_tft_model is None:
                logger.info("Initializing Temporal Fusion Transformer model...")
                model = TFTPriceForecaster()
                model.eval()
                _global_tft_model = model
    return _global_tft_model


def prepare_buffer_tensor(
    buffer: Union[pl.DataFrame, List[Dict[str, Any]], torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[int]]:
    """Convert input buffer into a [1, 60, n_features] PyTorch tensor and extract latest price/timestamp.
    
    Returns:
        (tensor, current_price, latest_timestamp) or (None, None, None) if buffer < 60 bars.
    """
    if isinstance(buffer, torch.Tensor):
        t = buffer
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[1] < SEQUENCE_LENGTH:
            return None, None, None
        return t[:, -SEQUENCE_LENGTH:, :], None, None

    # Handle DataFrame or list of bar dicts
    if isinstance(buffer, list):
        if len(buffer) < SEQUENCE_LENGTH:
            return None, None, None
        df = pl.DataFrame(buffer)
    elif isinstance(buffer, pl.DataFrame):
        if len(buffer) < SEQUENCE_LENGTH:
            return None, None, None
        df = buffer
    else:
        return None, None, None

    # Compute features via Polars feature pipeline
    features_df = prepare_tft_feature_tensor(
        df,
        depth_imbalance=float(df["depth_imbalance"][-1]) if "depth_imbalance" in df.columns else 0.0,
        micro_price_spread=float(df["micro_price_spread"][-1]) if "micro_price_spread" in df.columns else 0.0,
        sequence_length=SEQUENCE_LENGTH,
    )

    if len(features_df) < SEQUENCE_LENGTH:
        return None, None, None

    cols = ALL_FEATURES
    # Verify required feature columns exist
    for c in cols:
        if c not in features_df.columns:
            features_df = features_df.with_columns(pl.lit(0.0).alias(c))

    np_mat = features_df.select(cols).fill_nan(0.0).fill_null(0.0).to_numpy()
    tensor = torch.tensor(np_mat, dtype=torch.float32).unsqueeze(0)

    current_price = float(df["close"][-1]) if "close" in df.columns else None
    latest_ts = int(df["timestamp"][-1]) if "timestamp" in df.columns else int(time.time())

    return tensor, current_price, latest_ts


def predict_realtime(
    feature_buffer: Union[pl.DataFrame, List[Dict[str, Any]], torch.Tensor],
    current_price: Optional[float] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute real-time TFT inference in torch.no_grad() mode.
    
    Args:
        feature_buffer: List of candle dicts, Polars DataFrame, or raw feature tensor.
        current_price: Reference price to convert forecasted log returns into price levels.
        symbol: Trading symbol identifier.
        
    Returns:
        If buffer has < 60 bars:
            {"status": "BUFFERING"}
        If buffer has >= 60 bars:
            {
                "status": "READY",
                "timestamp": int,
                "current_price": float,
                "p10": float,
                "p50": float,
                "p90": float,
                "directional_bias": "BULLISH" | "BEARISH" | "NEUTRAL",
                "horizons": {
                    "t+5m": {...},
                    "t+15m": {...}
                }
            }
    """
    tensor, inferred_price, inferred_ts = prepare_buffer_tensor(feature_buffer)
    if tensor is None or tensor.shape[1] < SEQUENCE_LENGTH:
        return {"status": "BUFFERING"}

    ref_price = current_price if current_price is not None else (inferred_price if inferred_price is not None else 100.0)
    ref_ts = inferred_ts if inferred_ts is not None else int(time.time())

    model = get_tft_model()
    model.eval()

    with torch.no_grad():
        preds = model(tensor)  # Shape [1, 15, 3]

    # Step 5: index 4; Step 15: index 14
    step_5_ret = preds[0, 4, :].cpu().numpy()
    step_15_ret = preds[0, 14, :].cpu().numpy()

    # Enforce strict quantile monotonicity
    p10_r5, p50_r5, p90_r5 = float(step_5_ret[0]), float(step_5_ret[1]), float(step_5_ret[2])
    p10_r15, p50_r15, p90_r15 = float(step_15_ret[0]), float(step_15_ret[1]), float(step_15_ret[2])

    # Convert log-returns to forecasted price levels: P_forecast = P_current * exp(r)
    p10_p5 = ref_price * np.exp(p10_r5)
    p50_p5 = ref_price * np.exp(p50_r5)
    p90_p5 = ref_price * np.exp(p90_r5)

    p10_p15 = ref_price * np.exp(p10_r15)
    p50_p15 = ref_price * np.exp(p50_r15)
    p90_p15 = ref_price * np.exp(p90_r15)

    # Determine directional bias from median expected return
    def derive_bias(med_ret: float) -> str:
        if med_ret > 0.0001:
            return "BULLISH"
        elif med_ret < -0.0001:
            return "BEARISH"
        return "NEUTRAL"

    bias_5m = derive_bias(p50_r5)
    bias_15m = derive_bias(p50_r15)

    return {
        "status": "READY",
        "symbol": symbol,
        "timestamp": ref_ts,
        "current_price": round(ref_price, 2),
        "p10": round(float(p10_p15), 2),
        "p50": round(float(p50_p15), 2),
        "p90": round(float(p90_p15), 2),
        "directional_bias": bias_15m,
        "p10_return": round(p10_r15, 6),
        "p50_return": round(p50_r15, 6),
        "p90_return": round(p90_r15, 6),
        "horizons": {
            "t+5m": {
                "p10": round(float(p10_p5), 2),
                "p50": round(float(p50_p5), 2),
                "p90": round(float(p90_p5), 2),
                "p10_return": round(p10_r5, 6),
                "p50_return": round(p50_r5, 6),
                "p90_return": round(p90_r5, 6),
                "directional_bias": bias_5m,
            },
            "t+15m": {
                "p10": round(float(p10_p15), 2),
                "p50": round(float(p50_p15), 2),
                "p90": round(float(p90_p15), 2),
                "p10_return": round(p10_r15, 6),
                "p50_return": round(p50_r15, 6),
                "p90_return": round(p90_r15, 6),
                "directional_bias": bias_15m,
            },
        },
    }


class CandleBufferManager:
    """Thread-safe rolling buffer of completed 1-minute candles per symbol."""

    def __init__(self, max_buffer_size: int = 120) -> None:
        self.max_buffer_size = max_buffer_size
        self._buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def append_bar(self, symbol: str, bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Append a bar to the symbol's buffer and prune to max_buffer_size."""
        with self._lock:
            buf = self._buffers[symbol]
            # Avoid duplicate bar timestamps
            if not buf or buf[-1].get("timestamp") != bar.get("timestamp"):
                buf.append(dict(bar))
                if len(buf) > self.max_buffer_size:
                    self._buffers[symbol] = buf[-self.max_buffer_size:]
            return list(self._buffers[symbol])

    def get_buffer(self, symbol: str) -> List[Dict[str, Any]]:
        """Retrieve a copy of the buffer for a symbol."""
        with self._lock:
            return list(self._buffers.get(symbol, []))

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear buffer for a specific symbol or all symbols."""
        with self._lock:
            if symbol:
                self._buffers.pop(symbol, None)
            else:
                self._buffers.clear()


# Global candle buffer manager instance
candle_buffer_manager = CandleBufferManager()


def on_candle_completed(
    bar: Dict[str, Any],
    broadcaster_dispatch: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Real-time inference hook executed on every completed 1-minute candle.
    
    Updates the rolling buffer, runs predict_realtime in torch.no_grad() mode,
    and broadcasts the result over the market WebSocket.
    """
    symbol = bar.get("symbol", "UNKNOWN")
    buffer = candle_buffer_manager.append_bar(symbol, bar)

    forecast = predict_realtime(
        feature_buffer=buffer,
        current_price=bar.get("close"),
        symbol=symbol,
    )

    if forecast.get("status") == "BUFFERING":
        payload = {
            "type": "tft_forecast",
            "symbol": symbol,
            "status": "BUFFERING",
            "bars_collected": len(buffer),
            "required_bars": SEQUENCE_LENGTH,
        }
    else:
        payload = {
            "type": "tft_forecast",
            "symbol": symbol,
            "status": "READY",
            "timestamp": forecast["timestamp"],
            "current_price": forecast["current_price"],
            "p10": forecast["p10"],
            "p50": forecast["p50"],
            "p90": forecast["p90"],
            "directional_bias": forecast["directional_bias"],
            "horizons": forecast.get("horizons"),
        }

    # Dispatch to WebSocket clients if broadcaster callback provided
    if broadcaster_dispatch is not None:
        broadcaster_dispatch(payload)
    else:
        try:
            from backend.app.routers.stream import broadcaster
            broadcaster.dispatch_from_thread(payload)
        except Exception as e:
            logger.debug("Broadcaster not available for TFT dispatch: %s", e)

    return payload

