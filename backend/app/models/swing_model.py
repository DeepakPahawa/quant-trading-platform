"""Walk-forward ranker for swing trades.

A gradient-boosted regressor is trained to predict each stock's excess-return percentile.
At every test date it scores the universe, and the top-N scores are the suggestions.

Walk-forward discipline:
- Each fold trains only on rows whose label is fully known before the fold starts
  (exit_date < test_start). This purges the overlap between training labels and the test period.
- Test periods are non-overlapping and strictly after training.
"""

from datetime import date
from typing import List, Optional

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from backend.app.features.daily_features import FEATURE_COLUMNS

RANDOM_SEED = 42

MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    l2_regularization=1.0,
    random_state=RANDOM_SEED,
)


def fit_ranker(train: pl.DataFrame) -> HistGradientBoostingRegressor:
    """Fit on rows that have a target. Raises if nothing is trainable."""
    labelled = train.filter(pl.col("target").is_not_null())
    if labelled.height == 0:
        raise ValueError("No labelled training rows")
    all_null = [c for c in FEATURE_COLUMNS if labelled[c].null_count() == labelled.height]
    if all_null:
        raise ValueError(f"Feature(s) entirely null in training data: {all_null}. Need more history.")
    X = labelled.select(FEATURE_COLUMNS).to_numpy()
    y = labelled["target"].to_numpy()
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def score(model: HistGradientBoostingRegressor, rows: pl.DataFrame) -> np.ndarray:
    return model.predict(rows.select(FEATURE_COLUMNS).to_numpy())


def walk_forward_predictions(
    dataset: pl.DataFrame,
    test_starts: List[date],
) -> pl.DataFrame:
    """Out-of-sample scores for every test date.

    Fold i tests on dates in [test_starts[i], test_starts[i+1]) and trains on labels
    that matured before test_starts[i]. Returns one row per (symbol, date) in a test period.
    """
    starts = sorted(test_starts)
    parts: List[pl.DataFrame] = []
    for i, start in enumerate(starts):
        end: Optional[date] = starts[i + 1] if i + 1 < len(starts) else None
        train = dataset.filter(pl.col("exit_date") < start)
        test_filter = pl.col("date") >= start
        if end is not None:
            test_filter = test_filter & (pl.col("date") < end)
        test = dataset.filter(test_filter).filter(pl.col("label_raw").is_not_null())
        if train.height == 0 or test.height == 0:
            continue

        model = fit_ranker(train)
        scores = score(model, test)
        parts.append(
            test.select(["symbol", "date", "label_raw", "bench_label", "label_excess"])
            .with_columns(pl.Series("score", scores), pl.lit(start).alias("fold_start"))
        )

    if not parts:
        raise ValueError("No walk-forward folds produced predictions. Check dates and data length.")
    return pl.concat(parts).sort(["date", "score"], descending=[False, True])
