"""Temporal Fusion Transformer (TFT) multi-horizon price forecasting.

Contract
--------
- Input: 60 one-minute bars, 8 columns in `ALL_FEATURES` order
  (6 continuous features followed by `minute_of_day` and `day_of_week`, taken
  from real IST session time).
- Output: quantile log-returns of shape [batch, 15, 3] for steps t+1 .. t+15
  bars, quantiles (p10, p50, p90). Step k is the return from the last observed
  bar to bar t+k. Index 4 is t+5m and index 14 is t+15m.
- Normalization (per-feature mean/std and the target scale) is stored as buffers,
  so it is part of the saved state_dict and inference matches training exactly.

Training lives in `backend/app/models/tft_training.py`. This module is inference only.
"""

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
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
from backend.app.features.tft_features import (
    ALL_FEATURES,
    CONTINUOUS_FEATURES,
    SESSION_LENGTH_MINUTES,
    TIME_COVARIATES,
)

logger = logging.getLogger(__name__)

# TFT architecture constants
SEQUENCE_LENGTH = 60
PREDICTION_HORIZONS = [5, 15]
MAX_PREDICTION_LENGTH = 15
QUANTILES = [0.1, 0.5, 0.9]

MODEL_FILENAME = "model.pt"
SIDECAR_FILENAME = "sidecar.json"
MODEL_DIR_ENV = "TFT_MODEL_DIR"


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
    """Temporal Fusion Transformer for 60-bar multivariate quantile return forecasting."""

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

        # The dataset only defines the variable layout the TFT is built from.
        # Its values are never used for training or inference.
        total_len = sequence_length + prediction_length
        dummy_df = pd.DataFrame({
            "time_idx": list(range(total_len)),
            "group": ["0"] * total_len,
            **{name: [0.0] * total_len for name in ALL_FEATURES},
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

        # pytorch-forecasting orders reals as known-first (time covariates), then unknown.
        # Map our canonical ALL_FEATURES order onto that layout by name, never by position.
        x_reals: List[str] = list(self.tft.hparams.x_reals)
        known_reals: List[str] = list(self.tft.hparams.time_varying_reals_decoder)
        if set(known_reals) != set(TIME_COVARIATES):
            raise RuntimeError(f"Unexpected known reals {known_reals}; expected {TIME_COVARIATES}")
        self._encoder_perm = torch.tensor([ALL_FEATURES.index(n) for n in x_reals], dtype=torch.long)
        self._minute_idx = ALL_FEATURES.index("minute_of_day")
        self._dow_idx = ALL_FEATURES.index("day_of_week")

        # Normalization state. Defaults are identity, so an untrained model is well-defined.
        self.register_buffer("input_mean", torch.zeros(len(ALL_FEATURES)))
        self.register_buffer("input_std", torch.ones(len(ALL_FEATURES)))
        self.register_buffer("target_scale", torch.tensor(1.0))

    # ------------------------------------------------------------------ inputs

    def _assemble(
        self,
        x: torch.Tensor,
        time_covariates: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return [B, L, 8] raw features in ALL_FEATURES order."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == len(ALL_FEATURES):
            return x.float()
        if x.shape[-1] == len(CONTINUOUS_FEATURES):
            if time_covariates is None:
                raise ValueError(
                    "Time covariates (minute_of_day, day_of_week) are required with 6-feature input. "
                    "Pass real IST time covariates; they are not synthesized."
                )
            if time_covariates.ndim == 2:
                time_covariates = time_covariates.unsqueeze(0)
            if time_covariates.shape[:2] != x.shape[:2] or time_covariates.shape[-1] != len(TIME_COVARIATES):
                raise ValueError(
                    f"time_covariates must have shape [B, L, {len(TIME_COVARIATES)}] matching x; "
                    f"got {tuple(time_covariates.shape)} for x {tuple(x.shape)}"
                )
            return torch.cat([x.float(), time_covariates.float()], dim=-1)
        raise ValueError(
            f"Expected last dimension {len(CONTINUOUS_FEATURES)} or {len(ALL_FEATURES)}, got {x.shape[-1]}"
        )

    def _future_time_covariates(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Known future time covariates for the decoder, in raw units.

        Bars are one minute apart, so minute_of_day advances by one per step, clamped
        to the session length. day_of_week stays at the last observed value.
        """
        batch_size = x_raw.shape[0]
        device = x_raw.device
        last_minute = x_raw[:, -1, self._minute_idx]  # [B]
        steps = torch.arange(1, self.prediction_length + 1, device=device, dtype=x_raw.dtype)  # [P]
        future_minute = torch.clamp(last_minute.unsqueeze(1) + steps.unsqueeze(0), 0.0, float(SESSION_LENGTH_MINUTES))
        future_dow = x_raw[:, -1, self._dow_idx].unsqueeze(1).expand(batch_size, self.prediction_length)
        future = torch.zeros(batch_size, self.prediction_length, len(ALL_FEATURES), device=device, dtype=x_raw.dtype)
        future[:, :, self._minute_idx] = future_minute
        future[:, :, self._dow_idx] = future_dow
        return future

    def _prepare_input_dict(self, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build the TFT input dict from raw [B, L, 8] features (normalizes internally)."""
        batch_size, seq_len, _ = x_raw.shape
        device = x_raw.device
        pred_len = self.prediction_length

        mean = self.input_mean.to(device)
        std = self.input_std.to(device)

        x_norm = (x_raw - mean) / std
        # pytorch-forecasting concatenates encoder and decoder along time, so both must carry
        # every real in x_reals order. Only the known (time) features are real in the decoder.
        # Unknown continuous features are unobserved in the future and are set to zero.
        future_norm = (self._future_time_covariates(x_raw) - mean) / std
        future_norm[:, :, : len(CONTINUOUS_FEATURES)] = 0.0

        encoder_cont = x_norm[:, :, self._encoder_perm]
        decoder_cont = future_norm[:, :, self._encoder_perm]

        return {
            "encoder_cat": torch.zeros(batch_size, seq_len, 0, dtype=torch.int64, device=device),
            "encoder_cont": encoder_cont.float(),
            "encoder_target": x_norm[:, :, ALL_FEATURES.index("log_return")].float(),
            "encoder_lengths": torch.full((batch_size,), seq_len, dtype=torch.int64, device=device),
            "decoder_cat": torch.zeros(batch_size, pred_len, 0, dtype=torch.int64, device=device),
            "decoder_cont": decoder_cont.float(),
            "decoder_target": torch.zeros(batch_size, pred_len, dtype=torch.float32, device=device),
            "decoder_lengths": torch.full((batch_size,), pred_len, dtype=torch.int64, device=device),
            "decoder_time_idx": torch.arange(seq_len, seq_len + pred_len, device=device).unsqueeze(0).repeat(batch_size, 1),
            "groups": torch.zeros(batch_size, 1, dtype=torch.int64, device=device),
            "target_scale": torch.tensor([[0.0, 1.0]], device=device).repeat(batch_size, 1),
        }

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        time_covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Quantile log-return forecasts in raw return units.

        Args:
            x: [B, 60, 8] features in ALL_FEATURES order, or [B, 60, 6] continuous
               features together with `time_covariates` [B, 60, 2].
            time_covariates: Real minute_of_day and day_of_week, used with 6-feature input.

        Returns:
            Tensor [B, 15, 3] of (p10, p50, p90) log-returns, sorted so p10 <= p50 <= p90.
        """
        if isinstance(x, dict):
            raw_pred = self.tft(x).prediction
            return torch.sort(raw_pred, dim=-1).values

        x_raw = self._assemble(x, time_covariates)
        raw_pred = self.tft(self._prepare_input_dict(x_raw)).prediction  # normalized target units
        pred = torch.sort(raw_pred, dim=-1).values
        return pred * self.target_scale.to(pred.device)

    def predict_horizons(
        self,
        x: torch.Tensor,
        horizons: List[int] = PREDICTION_HORIZONS,
        time_covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract quantile forecasts at horizons t+h minutes (h is 1-based bar count)."""
        pred = self.forward(x, time_covariates=time_covariates)  # [B, 15, 3]
        results: Dict[str, torch.Tensor] = {}
        for h in horizons:
            idx = min(max(h - 1, 0), self.prediction_length - 1)
            results[f"t+{h}m"] = pred[:, idx, :]
        return results

    # ------------------------------------------------------------- persistence

    def set_normalization(self, mean: np.ndarray, std: np.ndarray, target_scale: float) -> None:
        """Set input normalization (per ALL_FEATURES column) and the target scale."""
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(np.where(std > 0, std, 1.0), dtype=torch.float32))
        self.target_scale.fill_(float(target_scale) if target_scale > 0 else 1.0)

    def save_run(self, run_dir: Union[str, Path], sidecar: Dict[str, Any]) -> None:
        """Write model weights and a JSON sidecar describing the run."""
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), run_path / MODEL_FILENAME)
        payload = {
            **sidecar,
            "features": ALL_FEATURES,
            "sequence_length": self.sequence_length,
            "prediction_length": self.prediction_length,
            "quantiles": self.quantiles,
            "horizons": PREDICTION_HORIZONS,
        }
        (run_path / SIDECAR_FILENAME).write_text(json.dumps(payload, indent=2, default=str))

    @classmethod
    def load_run(cls, run_dir: Union[str, Path]) -> "TFTPriceForecaster":
        """Load a run written by `save_run`. Refuses to load if the feature schema differs."""
        run_path = Path(run_dir)
        sidecar = json.loads((run_path / SIDECAR_FILENAME).read_text())
        if sidecar.get("features") != ALL_FEATURES:
            raise ValueError(
                f"Checkpoint feature schema {sidecar.get('features')} does not match {ALL_FEATURES}"
            )
        if sidecar.get("sequence_length") != SEQUENCE_LENGTH:
            raise ValueError(f"Checkpoint sequence_length {sidecar.get('sequence_length')} != {SEQUENCE_LENGTH}")
        if sidecar.get("prediction_length") != MAX_PREDICTION_LENGTH:
            raise ValueError("Checkpoint prediction_length does not match the running code")

        model = cls()
        state = torch.load(run_path / MODEL_FILENAME, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


# Module-level model singleton
_global_tft_model: Optional[TFTPriceForecaster] = None
_model_lock = threading.Lock()


def get_tft_model() -> TFTPriceForecaster:
    """Return the singleton model in eval mode.

    Loads the trained run at $TFT_MODEL_DIR when set. Otherwise falls back to untrained
    weights and logs a warning, so forecasts from an untrained model are never silent.
    """
    global _global_tft_model
    if _global_tft_model is None:
        with _model_lock:
            if _global_tft_model is None:
                run_dir = os.getenv(MODEL_DIR_ENV)
                if run_dir:
                    logger.info("Loading trained TFT run from %s", run_dir)
                    model = TFTPriceForecaster.load_run(run_dir)
                else:
                    logger.warning("%s not set: serving UNTRAINED TFT weights", MODEL_DIR_ENV)
                    model = TFTPriceForecaster()
                    model.eval()
                _global_tft_model = model
    return _global_tft_model


def prepare_buffer_tensor(
    buffer: Union[pl.DataFrame, List[Dict[str, Any]], torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[int]]:
    """Convert a bar buffer into a [1, 60, 8] tensor, plus the latest close and timestamp.

    Returns:
        (tensor, current_price, latest_timestamp), or (None, None, None) if fewer than 60 bars.
    """
    if isinstance(buffer, torch.Tensor):
        t = buffer
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[1] < SEQUENCE_LENGTH:
            return None, None, None
        return t[:, -SEQUENCE_LENGTH:, :], None, None

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

    features_df = prepare_tft_feature_tensor(df, sequence_length=SEQUENCE_LENGTH)
    if len(features_df) < SEQUENCE_LENGTH:
        return None, None, None

    np_mat = features_df.select(ALL_FEATURES).to_numpy()
    tensor = torch.tensor(np_mat, dtype=torch.float32).unsqueeze(0)

    current_price = float(df["close"][-1]) if "close" in df.columns else None
    latest_ts = int(df["timestamp"][-1]) if "timestamp" in df.columns else int(time.time())

    return tensor, current_price, latest_ts


def predict_realtime(
    feature_buffer: Union[pl.DataFrame, List[Dict[str, Any]], torch.Tensor],
    current_price: Optional[float] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Run TFT inference under torch.no_grad().

    Returns {"status": "BUFFERING"} when fewer than 60 bars are available. Otherwise
    returns a READY payload with per-horizon price quantiles.
    """
    tensor, inferred_price, inferred_ts = prepare_buffer_tensor(feature_buffer)
    if tensor is None or tensor.shape[1] < SEQUENCE_LENGTH:
        return {"status": "BUFFERING"}

    ref_price = current_price if current_price is not None else (inferred_price if inferred_price is not None else 100.0)
    ref_ts = inferred_ts if inferred_ts is not None else int(time.time())

    model = get_tft_model()
    model.eval()

    with torch.no_grad():
        preds = model(tensor)  # [1, 15, 3] log-returns

    step_5_ret = preds[0, 4, :].cpu().numpy()
    step_15_ret = preds[0, 14, :].cpu().numpy()

    p10_r5, p50_r5, p90_r5 = float(step_5_ret[0]), float(step_5_ret[1]), float(step_5_ret[2])
    p10_r15, p50_r15, p90_r15 = float(step_15_ret[0]), float(step_15_ret[1]), float(step_15_ret[2])

    # P_forecast = P_current * exp(r)
    p10_p5 = ref_price * np.exp(p10_r5)
    p50_p5 = ref_price * np.exp(p50_r5)
    p90_p5 = ref_price * np.exp(p90_r5)

    p10_p15 = ref_price * np.exp(p10_r15)
    p50_p15 = ref_price * np.exp(p50_r15)
    p90_p15 = ref_price * np.exp(p90_r15)

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
    """Real-time inference hook run on every completed 1-minute candle.

    Updates the rolling buffer, runs predict_realtime, and broadcasts the result.
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

    if broadcaster_dispatch is not None:
        broadcaster_dispatch(payload)
    else:
        try:
            from backend.app.routers.stream import broadcaster
            broadcaster.dispatch_from_thread(payload)
        except Exception as e:
            logger.debug("Broadcaster not available for TFT dispatch: %s", e)

    return payload
