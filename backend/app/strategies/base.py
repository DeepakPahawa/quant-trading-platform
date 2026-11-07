"""Strategy contract: a strategy turns daily bars into per-day entry/exit signals.

A strategy knows nothing about portfolios, costs or data sources. The backtest engine
(`backend/app/backtest/strategy_engine.py`) consumes the signal frame, so any strategy runs on
any bars (Nifty 50, Nifty 500, or a larger set later).

Signal semantics (all evaluated at the CLOSE of `date`, using data up to and including it):
  entry_signal  buy at the NEXT day's open (if a slot is free)
  exit_signal   sell at the NEXT day's open (only meaningful while holding)
  priority      higher is preferred when more entry signals than free slots
  atr           ATR in price units, used by the engine for an ATR stop (`stop_atr_mult`)
"""

from abc import ABC, abstractmethod
import dataclasses
from typing import Any, ClassVar, Dict, Optional

import polars as pl

SIGNAL_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close",
    "entry_signal", "exit_signal", "priority", "atr",
]


class Strategy(ABC):
    """Subclasses are frozen dataclasses whose fields are the tunable parameters."""

    name: ClassVar[str]
    description: ClassVar[str]
    stop_atr_mult: Optional[float]  # stop = entry fill - mult * ATR(signal day). None = no stop
    max_hold_days: Optional[int]    # time stop in bars held (entry bar counts as 1). None = no limit

    @abstractmethod
    def prepare(self, bars: pl.DataFrame, bench: pl.DataFrame) -> pl.DataFrame:
        """Return a frame with `SIGNAL_COLUMNS`.

        bars:  symbol, date, open, high, low, close, volume (daily, one row per symbol and date).
        bench: date, close for the benchmark index (Nifty 50).
        """

    def params(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[call-overload]


def sorted_bars(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.sort(["symbol", "date"])


def finalize_signals(
    df: pl.DataFrame, entry: pl.Expr, exit_: pl.Expr, priority: pl.Expr
) -> pl.DataFrame:
    """Attach the signal columns. Null (warm-up) conditions count as False. `df` must carry `atr`."""
    return df.with_columns(
        entry.fill_null(False).alias("entry_signal"),
        exit_.fill_null(False).alias("exit_signal"),
        priority.fill_null(0.0).alias("priority"),
    ).select(SIGNAL_COLUMNS)


def validate_signal_frame(df: pl.DataFrame) -> None:
    missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Signal frame is missing columns: {missing}")
    if df.select(pl.struct(["symbol", "date"]).is_duplicated().any()).item():
        raise ValueError("Signal frame has duplicate (symbol, date) rows")
