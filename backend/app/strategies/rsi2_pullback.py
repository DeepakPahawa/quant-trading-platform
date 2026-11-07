"""Strategy 1: RSI(2) pullback in an uptrend, with a Nifty regime filter.

Entry (signal at close): RSI(2) < 10, close > 200-day SMA, and Nifty 50 close > its 200-day SMA.
Exit (signal at close): close > 5-day SMA. Execution is at the next open for both.
No price stop by default (the original system relies on the SMA exit). `max_hold_days` and
`stop_atr_mult` are optional safety nets, off by default so the rule matches its definition.
"""

from dataclasses import dataclass
from typing import Optional

import polars as pl

from backend.app.strategies.base import Strategy, finalize_signals, sorted_bars
from backend.app.strategies.indicators import add_regime, atr, sma, wilder_rsi


@dataclass(frozen=True)
class Rsi2Pullback(Strategy):
    name = "rsi2_pullback"
    description = "Buy RSI(2)<10 above the 200d SMA while Nifty is above its 200d SMA; exit on close above the 5d SMA."

    rsi_period: int = 2
    rsi_entry_below: float = 10.0
    trend_sma: int = 200
    exit_sma: int = 5
    use_regime_filter: bool = True
    regime_sma: int = 200
    stop_atr_mult: Optional[float] = None
    max_hold_days: Optional[int] = None

    def prepare(self, bars: pl.DataFrame, bench: pl.DataFrame) -> pl.DataFrame:
        df = sorted_bars(bars)
        df = add_regime(df, bench, self.regime_sma) if self.use_regime_filter else df.with_columns(
            pl.lit(True).alias("regime_ok")
        )
        df = df.with_columns(
            wilder_rsi(self.rsi_period).alias("rsi"),
            sma("close", self.trend_sma).alias("trend"),
            sma("close", self.exit_sma).alias("exit_ma"),
            atr(14).alias("atr"),
        )
        entry = (
            (pl.col("rsi") < self.rsi_entry_below)
            & (pl.col("close") > pl.col("trend"))
            & pl.col("regime_ok")
        )
        exit_ = pl.col("close") > pl.col("exit_ma")
        return finalize_signals(df, entry, exit_, priority=-pl.col("rsi"))  # deepest oversold first
