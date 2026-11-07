"""Training utilities for the TFT forecaster.

Pipeline: bars -> `build_tft_feature_frame` -> `build_samples` (windows + forward targets)
-> `split_samples` (time-ordered with purging) -> `fit_normalization` -> `train_model`
-> `evaluate_forecaster`.

Targets are cumulative log-returns from the last observed bar: for anchor bar `a`,
step k (1..15) is log(close[a+k] / close[a]). This matches inference, where output
index k-1 is the return to t+k bars.

Training is offline only. Nothing here runs inside an API request.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from numpy.lib.stride_tricks import sliding_window_view
from pytorch_forecasting.metrics import QuantileLoss

from backend.app.features.tft_features import ALL_FEATURES
from backend.app.models.tft_model import (
    MAX_PREDICTION_LENGTH,
    QUANTILES,
    SEQUENCE_LENGTH,
    TFTPriceForecaster,
)

logger = logging.getLogger(__name__)


@dataclass
class SampleSet:
    """Windowed samples for one or more symbols."""
    X: np.ndarray            # [N, SEQUENCE_LENGTH, len(ALL_FEATURES)] raw features
    Y: np.ndarray            # [N, MAX_PREDICTION_LENGTH] cumulative forward log-returns
    anchor_ts: np.ndarray    # [N] timestamp of the last observed bar
    end_ts: np.ndarray       # [N] timestamp of the last target bar

    @property
    def size(self) -> int:
        return int(self.X.shape[0])

    @staticmethod
    def empty() -> "SampleSet":
        return SampleSet(
            X=np.zeros((0, SEQUENCE_LENGTH, len(ALL_FEATURES)), dtype=np.float32),
            Y=np.zeros((0, MAX_PREDICTION_LENGTH), dtype=np.float32),
            anchor_ts=np.zeros((0,), dtype=np.int64),
            end_ts=np.zeros((0,), dtype=np.int64),
        )

    @staticmethod
    def concat(parts: List["SampleSet"]) -> "SampleSet":
        parts = [p for p in parts if p.size > 0]
        if not parts:
            return SampleSet.empty()
        return SampleSet(
            X=np.concatenate([p.X for p in parts]),
            Y=np.concatenate([p.Y for p in parts]),
            anchor_ts=np.concatenate([p.anchor_ts for p in parts]),
            end_ts=np.concatenate([p.end_ts for p in parts]),
        )


def build_samples(
    feature_frame: pl.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    prediction_length: int = MAX_PREDICTION_LENGTH,
) -> SampleSet:
    """Turn one symbol's feature frame into (window, forward-return) samples.

    `feature_frame` must come from `build_tft_feature_frame`, sorted by timestamp, for one symbol.
    Windows that contain any non-finite feature, or whose targets are not finite, are dropped.
    """
    n = feature_frame.height
    if n < sequence_length + prediction_length:
        return SampleSet.empty()

    feats = feature_frame.select(ALL_FEATURES).to_numpy().astype(np.float64)
    close = feature_frame["close"].to_numpy().astype(np.float64)
    ts = feature_frame["timestamp"].to_numpy().astype(np.int64)

    # Anchor a has a full encoder window ending at a and a full target window a+1..a+P.
    anchors = np.arange(sequence_length - 1, n - prediction_length)
    count = anchors.shape[0]
    windows = sliding_window_view(feats, (sequence_length, feats.shape[1]))[:, 0]
    X = windows[:count]

    Y = np.stack(
        [np.log(close[anchors + k] / close[anchors]) for k in range(1, prediction_length + 1)],
        axis=1,
    )

    valid = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(Y).all(axis=1)
    return SampleSet(
        X=X[valid].astype(np.float32),
        Y=Y[valid].astype(np.float32),
        anchor_ts=ts[anchors][valid],
        end_ts=ts[anchors + prediction_length][valid],
    )


def split_samples(
    samples: SampleSet,
    val_start_ts: int,
    test_start_ts: int,
) -> Tuple[SampleSet, SampleSet, SampleSet]:
    """Time-ordered split with purging.

    train: targets end before validation starts (so no label overlaps validation).
    val:   anchors in [val_start, test_start) whose targets end before test starts.
    test:  anchors at or after test_start.
    """
    if not (val_start_ts < test_start_ts):
        raise ValueError("val_start must be earlier than test_start")

    train_mask = samples.end_ts < val_start_ts
    val_mask = (samples.anchor_ts >= val_start_ts) & (samples.end_ts < test_start_ts)
    test_mask = samples.anchor_ts >= test_start_ts

    def take(mask: np.ndarray) -> SampleSet:
        return SampleSet(
            X=samples.X[mask], Y=samples.Y[mask],
            anchor_ts=samples.anchor_ts[mask], end_ts=samples.end_ts[mask],
        )

    return take(train_mask), take(val_mask), take(test_mask)


def fit_normalization(train: SampleSet) -> Tuple[np.ndarray, np.ndarray, float]:
    """Mean/std per feature and target scale, fit on the training split only."""
    flat = train.X.reshape(-1, len(ALL_FEATURES)).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    target_scale = float(train.Y.std())
    return mean, std, target_scale if target_scale > 1e-12 else 1.0


def _pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def train_model(
    model: TFTPriceForecaster,
    train: SampleSet,
    val: SampleSet,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train with quantile loss on normalized targets and early stopping on validation loss.

    The best validation weights are restored before returning.
    """
    if train.size == 0:
        raise ValueError("Training split is empty")
    if val.size == 0:
        raise ValueError("Validation split is empty")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    loss_fn = QuantileLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scale = float(model.target_scale.item())

    X_tr = torch.from_numpy(train.X)
    Y_tr = torch.from_numpy(train.Y) / scale
    X_va = torch.from_numpy(val.X)
    Y_va = torch.from_numpy(val.Y) / scale

    def batch_loss(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        pred_norm = model(xb) / model.target_scale  # [B, P, Q] in normalized units
        return loss_fn.loss(pred_norm, yb).mean()

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(X_tr.shape[0])
        train_losses: List[float] = []
        for start in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[start:start + batch_size])
            loss = batch_loss(X_tr[idx], Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(batch_loss(X_va, Y_va).item())

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        logger.info("epoch %d train=%.5f val=%.5f", epoch, history[-1]["train_loss"], val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_val_loss": best_val, "epochs_run": len(history)}


def evaluate_forecaster(model: TFTPriceForecaster, samples: SampleSet) -> Dict[str, Any]:
    """Held-out metrics in return units at horizons t+5 and t+15 bars.

    - pinball loss per quantile
    - empirical coverage of the p10..p90 band (target about 0.80)
    - directional accuracy of the median sign
    - MAE of the median versus the zero-return baseline (no change)
    """
    if samples.size == 0:
        raise ValueError("Evaluation split is empty")

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(samples.X)).numpy().astype(np.float64)  # [N, P, 3]

    results: Dict[str, Any] = {"n_samples": samples.size}
    for h in (5, 15):
        idx = h - 1
        y = samples.Y[:, idx].astype(np.float64)
        p10, p50, p90 = pred[:, idx, 0], pred[:, idx, 1], pred[:, idx, 2]
        nonzero = y != 0.0
        results[f"h{h}"] = {
            "pinball_p10": _pinball(y, p10, QUANTILES[0]),
            "pinball_p50": _pinball(y, p50, QUANTILES[1]),
            "pinball_p90": _pinball(y, p90, QUANTILES[2]),
            "coverage_p10_p90": float(np.mean((y >= p10) & (y <= p90))),
            "directional_accuracy": float(np.mean(np.sign(p50[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else None,
            "mae_p50": float(np.mean(np.abs(y - p50))),
            "mae_zero_baseline": float(np.mean(np.abs(y))),
        }
    return results
