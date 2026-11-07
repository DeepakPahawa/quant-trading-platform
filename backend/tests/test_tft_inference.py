"""Verification Suite for Temporal Fusion Transformer (TFT) Multi-Horizon Price Forecasting.

Verifies:
1. TFTPriceForecaster forward pass with [1, 60, n_features] input tensors.
2. Quantile ordering monotonicity: p10 <= p50 <= p90 across all horizons (including t+5m and t+15m).
3. Real-time inference hook:
   - Emits {"status": "BUFFERING"} when buffer has < 60 bars.
   - Emits { timestamp, current_price, p10, p50, p90, directional_bias } when buffer has >= 60 bars.
4. WebSocket integration of on_candle_completed hook.
"""

import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Any, Dict, List
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import numpy as np
import polars as pl
import pytest
import torch

from backend.app.models.tft_model import (
    SEQUENCE_LENGTH,
    TFTPriceForecaster,
    candle_buffer_manager,
    get_tft_model,
    on_candle_completed,
    predict_realtime,
)


def real_time_covariates(batch_size: int, seq_len: int) -> torch.Tensor:
    """Ascending IST minute_of_day in [0, 375] and a fixed day_of_week, shaped [B, L, 2]."""
    minutes = torch.linspace(0.0, 375.0, seq_len).unsqueeze(0).repeat(batch_size, 1)
    dow = torch.full((batch_size, seq_len), 2.0)
    return torch.stack([minutes, dow], dim=-1)


def test_minute_of_day_lands_in_its_own_slot(shared_model: TFTPriceForecaster):
    """Regression: reals are ordered known-first by pytorch-forecasting. Columns must map by name."""
    x_reals = list(shared_model.tft.hparams.x_reals)
    assert x_reals[:2] == ["minute_of_day", "day_of_week"]
    perm = shared_model._encoder_perm
    # encoder column j must come from ALL_FEATURES[perm[j]] whose name is x_reals[j]
    from backend.app.features.tft_features import ALL_FEATURES
    for j, name in enumerate(x_reals):
        assert ALL_FEATURES[int(perm[j])] == name


def test_six_feature_input_without_time_is_rejected(shared_model: TFTPriceForecaster):
    """Time covariates must be real. Synthesized values are not allowed."""
    x = torch.randn(1, 60, 6)
    with pytest.raises(ValueError, match="Time covariates"):
        shared_model(x)


def generate_synthetic_candles(count: int, base_price: float = 2500.0) -> List[Dict[str, Any]]:
    """Helper to generate realistic 1-minute OHLCV candles."""
    now = int(time.time()) - (count * 60)
    bars = []
    price = base_price
    for i in range(count):
        delta = float(np.sin(i / 5.0) * 2.0 + np.random.normal(0, 0.5))
        open_p = price
        close_p = price + delta
        high_p = max(open_p, close_p) + abs(float(np.random.normal(0, 0.3)))
        low_p = min(open_p, close_p) - abs(float(np.random.normal(0, 0.3)))
        vol = int(5000 + np.random.randint(100, 2000))

        bars.append({
            "symbol": "NSE:RELIANCE-EQ",
            "timestamp": now + (i * 60),
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": vol,
            "open_interest": 0,
            "depth_imbalance": 0.15,
            "micro_price_spread": 0.05,
        })
        price = close_p
    return bars


@pytest.fixture(scope="module")
def shared_model() -> TFTPriceForecaster:
    """Shared TFT model fixture in eval mode."""
    m = get_tft_model()
    m.eval()
    return m


class TestTFTInference:
    """Test suite for TFT multi-horizon inference and quantile properties."""

    def test_tensor_forward_pass_continuous_features(self, shared_model: TFTPriceForecaster):
        """Pass [1, 60, 6] tensor and assert valid quantile shapes (p10 <= p50 <= p90)."""
        batch_size = 1
        seq_len = 60
        n_features = 6

        # Continuous inputs: [log_return, volume_ratio, rsi_14, vwap_diff, depth_imbalance, micro_price_spread]
        x = torch.randn(batch_size, seq_len, n_features, dtype=torch.float32)
        time_cov = real_time_covariates(batch_size, seq_len)

        with torch.no_grad():
            out = shared_model(x, time_covariates=time_cov)

        # Expected shape: [batch_size, 15_horizons, 3_quantiles]
        assert out.shape == (batch_size, 15, 3), f"Expected shape (1, 15, 3), got {out.shape}"

        # Assert valid quantile ordering across all prediction horizons: p10 <= p50 <= p90
        p10 = out[..., 0]
        p50 = out[..., 1]
        p90 = out[..., 2]

        assert (p10 <= p50).all(), "Violation: p10 must be <= p50 across all horizons"
        assert (p50 <= p90).all(), "Violation: p50 must be <= p90 across all horizons"

    def test_tensor_forward_pass_with_time_covariates(self, shared_model: TFTPriceForecaster):
        """Pass [1, 60, 8] tensor including time covariates and assert quantile ordering."""
        batch_size = 1
        seq_len = 60
        n_features = 8

        # 6 continuous features + normalized minute_of_day (0-375) + day_of_week
        x = torch.randn(batch_size, seq_len, n_features, dtype=torch.float32)
        # Ensure minute_of_day is within valid range [0, 375]
        x[:, :, 6] = torch.clamp(x[:, :, 6] * 100.0, 0.0, 375.0)

        with torch.no_grad():
            out = shared_model(x)

        assert out.shape == (batch_size, 15, 3)

        p10 = out[..., 0]
        p50 = out[..., 1]
        p90 = out[..., 2]

        assert (p10 <= p50).all()
        assert (p50 <= p90).all()

    def test_multi_horizon_predictions_t5_and_t15(self, shared_model: TFTPriceForecaster):
        """Verify multi-horizon quantile predictions for forward steps t+5m and t+15m."""
        x = torch.randn(1, 60, 6, dtype=torch.float32)

        with torch.no_grad():
            horizons = shared_model.predict_horizons(
                x, horizons=[5, 15], time_covariates=real_time_covariates(1, 60)
            )

        assert "t+5m" in horizons
        assert "t+15m" in horizons

        # Shapes must be [1, 3] for each horizon
        assert horizons["t+5m"].shape == (1, 3)
        assert horizons["t+15m"].shape == (1, 3)

        t5_p10, t5_p50, t5_p90 = horizons["t+5m"][0, 0], horizons["t+5m"][0, 1], horizons["t+5m"][0, 2]
        t15_p10, t15_p50, t15_p90 = horizons["t+15m"][0, 0], horizons["t+15m"][0, 1], horizons["t+15m"][0, 2]

        assert t5_p10 <= t5_p50 <= t5_p90, f"t+5m quantiles out of order: {t5_p10}, {t5_p50}, {t5_p90}"
        assert t15_p10 <= t15_p50 <= t15_p90, f"t+15m quantiles out of order: {t15_p10}, {t15_p50}, {t15_p90}"

    def test_realtime_inference_buffering_state(self):
        """If buffer has fewer than 60 bars, predict_realtime must return {'status': 'BUFFERING'}."""
        bars_short = generate_synthetic_candles(count=45)
        result = predict_realtime(bars_short)
        assert result == {"status": "BUFFERING"}

        # Empty buffer test
        assert predict_realtime([]) == {"status": "BUFFERING"}

        # Tensor with < 60 bars
        t_short = torch.randn(1, 40, 6)
        assert predict_realtime(t_short) == {"status": "BUFFERING"}

    def test_realtime_inference_ready_state(self):
        """When buffer has >= 60 bars, predict_realtime returns complete payload with valid quantiles."""
        candles = generate_synthetic_candles(count=70, base_price=2600.0)
        result = predict_realtime(candles, current_price=2600.0, symbol="NSE:RELIANCE-EQ")

        assert result["status"] == "READY"
        assert result["symbol"] == "NSE:RELIANCE-EQ"
        assert result["current_price"] == 2600.0
        assert "timestamp" in result
        assert "p10" in result
        assert "p50" in result
        assert "p90" in result
        assert "directional_bias" in result
        assert result["directional_bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]

        # Validate quantile ordering for forecasted prices
        assert result["p10"] <= result["p50"] <= result["p90"]

        # Validate horizons structure
        horizons = result["horizons"]
        assert "t+5m" in horizons
        assert "t+15m" in horizons

        for h in ["t+5m", "t+15m"]:
            h_data = horizons[h]
            assert h_data["p10"] <= h_data["p50"] <= h_data["p90"]
            assert h_data["directional_bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]

    def test_websocket_broadcast_hook_buffering_and_ready(self):
        """Verify on_candle_completed hook emits BUFFERING initially and full broadcast when 60 bars reached."""
        candle_buffer_manager.clear("NSE:TCS-EQ")
        broadcast_mock = MagicMock()

        candles = generate_synthetic_candles(count=60, base_price=3500.0)

        # Bar 1 (< 60 bars)
        bar_1 = candles[0]
        bar_1["symbol"] = "NSE:TCS-EQ"
        res_1 = on_candle_completed(bar_1, broadcaster_dispatch=broadcast_mock)
        assert res_1["status"] == "BUFFERING"
        broadcast_mock.assert_called_with(res_1)

        # Append remaining bars up to 60
        for b in candles[1:59]:
            b["symbol"] = "NSE:TCS-EQ"
            candle_buffer_manager.append_bar("NSE:TCS-EQ", b)

        # 60th bar completes sequence!
        bar_60 = candles[59]
        bar_60["symbol"] = "NSE:TCS-EQ"
        broadcast_mock.reset_mock()
        res_60 = on_candle_completed(bar_60, broadcaster_dispatch=broadcast_mock)

        assert res_60["status"] == "READY"
        assert res_60["symbol"] == "NSE:TCS-EQ"
        assert res_60["p10"] <= res_60["p50"] <= res_60["p90"]
        assert res_60["directional_bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert "timestamp" in res_60
        assert "current_price" in res_60
        broadcast_mock.assert_called_with(res_60)


def run_verification_protocol() -> None:
    """Standalone entry point verifying the TFT inference protocol and exiting with code 0."""
    print("=== TFT Inference Verification Protocol ===")
    model = get_tft_model()
    model.eval()

    # Pass [1, 60, n_features] tensor
    n_features = 6
    x = torch.randn(1, SEQUENCE_LENGTH, n_features, dtype=torch.float32)
    with torch.no_grad():
        output = model(x)

    print(f"1. Input tensor shape: {list(x.shape)}")
    print(f"2. Output prediction shape: {list(output.shape)}")
    assert output.shape == (1, 15, 3), f"Unexpected shape: {output.shape}"

    p10 = output[0, :, 0]
    p50 = output[0, :, 1]
    p90 = output[0, :, 2]

    # Assert valid quantile shapes: p10 <= p50 <= p90
    assert (p10 <= p50).all(), "Quantile violation: p10 > p50"
    assert (p50 <= p90).all(), "Quantile violation: p50 > p90"
    print("3. Quantile ordering monotonicity (p10 <= p50 <= p90): PASS")

    # Multi-horizon extraction: t+5m (index 4) and t+15m (index 14)
    p10_5m, p50_5m, p90_5m = float(p10[4]), float(p50[4]), float(p90[4])
    p10_15m, p50_15m, p90_15m = float(p10[14]), float(p50[14]), float(p90[14])
    print(f"4. Horizon t+5m  Quantiles: [p10={p10_5m:.6f}, p50={p50_5m:.6f}, p90={p90_5m:.6f}]")
    print(f"5. Horizon t+15m Quantiles: [p10={p10_15m:.6f}, p50={p50_15m:.6f}, p90={p90_15m:.6f}]")

    # Buffering check
    buffering_resp = predict_realtime([])
    assert buffering_resp == {"status": "BUFFERING"}
    print("6. Buffer < 60 bars test: {'status': 'BUFFERING'} - PASS")

    # Real-time inference with 60 bars
    candles = generate_synthetic_candles(60, base_price=2500.0)
    ready_resp = predict_realtime(candles, current_price=2500.0, symbol="NSE:NIFTY50-INDEX")
    assert ready_resp["status"] == "READY"
    assert ready_resp["p10"] <= ready_resp["p50"] <= ready_resp["p90"]
    print(f"7. Real-time inference ready test: Price {ready_resp['current_price']} -> "
          f"p10={ready_resp['p10']}, p50={ready_resp['p50']}, p90={ready_resp['p90']}, "
          f"bias={ready_resp['directional_bias']} - PASS")

    print("\nAll TFT verification checks successfully passed! Exiting cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    run_verification_protocol()
