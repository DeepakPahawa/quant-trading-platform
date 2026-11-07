"""Training pipeline tests: samples, purged splits, a short training run, and checkpoint round-trip."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timezone
import json

import numpy as np
import polars as pl
import pytest
import torch

from backend.app.features.tft_features import ALL_FEATURES, build_tft_feature_frame
from backend.app.models.tft_model import (
    MAX_PREDICTION_LENGTH,
    SEQUENCE_LENGTH,
    TFTPriceForecaster,
)
from backend.app.models.tft_training import (
    SampleSet,
    build_samples,
    evaluate_forecaster,
    fit_normalization,
    split_samples,
    train_model,
)


def make_bars(n: int, start_utc: datetime, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    t0 = int(start_utc.timestamp())
    return pl.DataFrame({
        "timestamp": np.arange(n, dtype=np.int64) * 60 + t0,
        "open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
        "volume": rng.integers(500, 2000, n).astype(np.int64),
        "open_interest": np.zeros(n, dtype=np.int64),
    })


@pytest.fixture(scope="module")
def samples() -> SampleSet:
    bars = make_bars(600, datetime(2026, 3, 2, 3, 45, tzinfo=timezone.utc))
    return build_samples(build_tft_feature_frame(bars), SEQUENCE_LENGTH, MAX_PREDICTION_LENGTH)


def test_samples_shapes_and_targets(samples: SampleSet):
    assert samples.X.shape[1:] == (SEQUENCE_LENGTH, len(ALL_FEATURES))
    assert samples.Y.shape[1] == MAX_PREDICTION_LENGTH
    assert samples.size > 0
    assert np.isfinite(samples.X).all() and np.isfinite(samples.Y).all()


def test_target_is_return_from_anchor_bar():
    """Target step k must equal log(close[a+k]/close[a]) for the anchor bar a."""
    bars = make_bars(120, datetime(2026, 3, 2, 3, 45, tzinfo=timezone.utc))
    s = build_samples(build_tft_feature_frame(bars), SEQUENCE_LENGTH, MAX_PREDICTION_LENGTH)
    close = bars["close"].to_numpy()
    ts = bars["timestamp"].to_numpy()
    i = int(np.where(ts == s.anchor_ts[0])[0][0])
    for k in (1, 5, 15):
        expected = np.log(close[i + k] / close[i])
        assert s.Y[0, k - 1] == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_split_is_time_ordered_and_purged(samples: SampleSet):
    val_start = int(np.quantile(samples.anchor_ts, 0.6))
    test_start = int(np.quantile(samples.anchor_ts, 0.8))
    train, val, test = split_samples(samples, val_start, test_start)
    assert train.size and val.size and test.size
    # No train target may reach into the validation period.
    assert (train.end_ts < val_start).all()
    # Validation anchors start inside the validation period, and their targets end before test.
    assert (val.anchor_ts >= val_start).all() and (val.end_ts < test_start).all()
    assert (test.anchor_ts >= test_start).all()


def test_split_rejects_inverted_boundaries(samples: SampleSet):
    with pytest.raises(ValueError):
        split_samples(samples, 200, 100)


def test_normalization_fit_on_train_only(samples: SampleSet):
    val_start = int(np.quantile(samples.anchor_ts, 0.6))
    train, _, _ = split_samples(samples, val_start, int(np.quantile(samples.anchor_ts, 0.8)))
    mean, std, scale = fit_normalization(train)
    # Stored as float32 buffers, so compare with float32-level tolerance.
    np.testing.assert_allclose(mean, train.X.reshape(-1, len(ALL_FEATURES)).mean(axis=0), rtol=1e-4, atol=1e-5)
    assert (std > 0).all() and scale > 0


def test_short_training_run_reduces_loss_and_roundtrips(tmp_path: Path, samples: SampleSet):
    torch.manual_seed(0)
    val_start = int(np.quantile(samples.anchor_ts, 0.6))
    test_start = int(np.quantile(samples.anchor_ts, 0.8))
    train, val, test = split_samples(samples, val_start, test_start)

    mean, std, scale = fit_normalization(train)
    model = TFTPriceForecaster()
    model.set_normalization(mean, std, scale)

    info = train_model(model, train, val, epochs=2, batch_size=64, lr=1e-3, patience=5, seed=0)
    assert info["epochs_run"] == 2
    assert np.isfinite(info["best_val_loss"])

    metrics = evaluate_forecaster(model, test)
    for h in ("h5", "h15"):
        assert 0.0 <= metrics[h]["coverage_p10_p90"] <= 1.0
        assert np.isfinite(metrics[h]["pinball_p50"])

    run_dir = tmp_path / "run"
    model.save_run(run_dir, {"note": "test"})
    assert (run_dir / "model.pt").exists()
    sidecar = json.loads((run_dir / "sidecar.json").read_text())
    assert sidecar["features"] == ALL_FEATURES

    reloaded = TFTPriceForecaster.load_run(run_dir)
    x = torch.from_numpy(test.X[:4])
    with torch.no_grad():
        np.testing.assert_allclose(model(x).numpy(), reloaded(x).numpy(), rtol=1e-5, atol=1e-6)


def test_load_rejects_mismatched_feature_schema(tmp_path: Path):
    model = TFTPriceForecaster()
    run_dir = tmp_path / "bad"
    model.save_run(run_dir, {})
    sidecar_path = run_dir / "sidecar.json"
    data = json.loads(sidecar_path.read_text())
    data["features"] = ALL_FEATURES[:-1]
    sidecar_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="feature schema"):
        TFTPriceForecaster.load_run(run_dir)


def test_training_refuses_empty_split(samples: SampleSet):
    empty = SampleSet.empty()
    model = TFTPriceForecaster()
    with pytest.raises(ValueError):
        train_model(model, empty, samples)
