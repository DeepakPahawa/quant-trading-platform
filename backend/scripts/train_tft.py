"""Train the TFT price forecaster on locally archived 1-minute bars.

Run from the repo root, for example:

    .venv/bin/python -m backend.scripts.train_tft \
        --symbols NSE:SBIN-EQ,NSE:RELIANCE-EQ \
        --start 2025-01-01 --end 2026-06-30 \
        --val-start 2026-01-01 --test-start 2026-04-01

Reads bars from DuckDB/Parquet only. It never calls the FYERS API. Sync the bars first
with /api/history?sync=true. Writes a run directory with model.pt and sidecar.json,
then serve it by setting TFT_MODEL_DIR to that directory.
"""

import argparse
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import List, Optional
from zoneinfo import ZoneInfo

import torch

from backend.app.data.storage import DuckDBStorage
from backend.app.features.tft_features import build_tft_feature_frame
from backend.app.models.tft_model import MAX_PREDICTION_LENGTH, SEQUENCE_LENGTH, TFTPriceForecaster
from backend.app.models.tft_training import (
    SampleSet,
    build_samples,
    evaluate_forecaster,
    fit_normalization,
    split_samples,
    train_model,
)

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_OUT_ROOT = Path("backend/data/models/tft")

logger = logging.getLogger("train_tft")


def ist_midnight_epoch(value: str) -> int:
    """YYYY-MM-DD interpreted as IST midnight, returned as epoch seconds."""
    d = date.fromisoformat(value)
    return int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp())


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the TFT forecaster from archived bars.")
    p.add_argument("--symbols", required=True, help="Comma-separated, e.g. NSE:SBIN-EQ,NSE:RELIANCE-EQ")
    p.add_argument("--resolution", default="1", help="Bar resolution; the model expects 1-minute bars")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (IST), inclusive")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (IST), inclusive")
    p.add_argument("--val-start", required=True, help="YYYY-MM-DD (IST). Validation begins here.")
    p.add_argument("--test-start", required=True, help="YYYY-MM-DD (IST). Test begins here, used once.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help="Run directory. Defaults to backend/data/models/tft/<UTC timestamp>")
    return p.parse_args(argv)


def load_samples(storage: DuckDBStorage, symbols: List[str], resolution: str,
                 start: Optional[str], end: Optional[str]) -> SampleSet:
    parts: List[SampleSet] = []
    for symbol in symbols:
        bars = storage.query_bars(symbol=symbol, range_from=start, range_to=end, resolution=resolution)
        if bars.is_empty():
            logger.warning("No archived bars for %s at resolution %s; skipping", symbol, resolution)
            continue
        features = build_tft_feature_frame(bars)  # historical bars carry no depth: constants are 0
        samples = build_samples(features, SEQUENCE_LENGTH, MAX_PREDICTION_LENGTH)
        logger.info("%s: %d bars -> %d samples", symbol, bars.height, samples.size)
        parts.append(samples)
    return SampleSet.concat(parts)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols given")
        return 2

    storage = DuckDBStorage()
    try:
        samples = load_samples(storage, symbols, args.resolution, args.start, args.end)
    finally:
        storage.close()

    if samples.size == 0:
        logger.error("No training samples. Sync bars for these symbols and date range first.")
        return 1

    train, val, test = split_samples(
        samples,
        val_start_ts=ist_midnight_epoch(args.val_start),
        test_start_ts=ist_midnight_epoch(args.test_start),
    )
    logger.info("Split sizes: train=%d val=%d test=%d", train.size, val.size, test.size)
    if train.size == 0 or val.size == 0 or test.size == 0:
        logger.error("Every split must be non-empty. Adjust --val-start / --test-start.")
        return 1

    mean, std, target_scale = fit_normalization(train)
    model = TFTPriceForecaster()
    model.set_normalization(mean, std, target_scale)

    fit_info = train_model(
        model, train, val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience, seed=args.seed,
    )
    metrics = evaluate_forecaster(model, test)

    created = datetime.now(timezone.utc)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / created.strftime("%Y%m%dT%H%M%SZ")
    sidecar = {
        "created_at_utc": created.isoformat(),
        "git_commit": git_commit(),
        "symbols": symbols,
        "resolution": args.resolution,
        "data_range": {"start": args.start, "end": args.end},
        "split": {
            "val_start": args.val_start,
            "test_start": args.test_start,
            "train_n": train.size,
            "val_n": val.size,
            "test_n": test.size,
        },
        "seed": args.seed,
        "training": {
            "epochs_run": fit_info["epochs_run"],
            "best_val_loss": fit_info["best_val_loss"],
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "history": fit_info["history"],
        },
        "normalization": {"target_scale": target_scale},
        "test_metrics": metrics,
        "notes": [
            "depth_imbalance and micro_price_spread are constant 0.0 for historical training: "
            "depth is not archived. They carry no signal in this model.",
            "Targets are cumulative log-returns from the last observed bar.",
        ],
    }
    model.save_run(out_dir, sidecar)
    print(json.dumps({"run_dir": str(out_dir), "test_metrics": metrics}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
