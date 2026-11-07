"""Strategy 2: 20-day breakout momentum with volume confirmation and relative strength.

Entry (signal at close): close above the prior 20-day high, volume above `volume_mult` x the prior
20-day average, and the stock's 20-day return beats Nifty 50's 20-day return.
Stop: entry fill minus 2 x ATR(14), checked intraday by the engine (gap-through exits at the open).
Exit (signal at close): close below the prior 10-day low. Execution is at the next open.
"""

from dataclasses import dataclass
from typing import Optional

import polars as pl

from backend.app.strategies.base import Strategy, finalize_signals, sorted_bars
from backend.app.strategies.indicators import (
    add_bench_return,
    add_regime,
    atr,
    pct_return,
    prior_avg_volume,
    prior_high,
    prior_low,
)


@dataclass(frozen=True)
class BreakoutMomentum(Strategy):
    name = "breakout_momentum"
    description = "Buy a 20d high on 1.5x volume with positive 20d strength vs Nifty; 2xATR stop, exit on a 10d low."

    breakout_window: int = 20
    exit_window: int = 10
    volume_window: int = 20
    volume_mult: float = 1.5
    rs_window: int = 20
    use_regime_filter: bool = False
    regime_sma: int = 200
    stop_atr_mult: Optional[float] = 2.0
    max_hold_days: Optional[int] = None

    def prepare(self, bars: pl.DataFrame, bench: pl.DataFrame) -> pl.DataFrame:
        df = sorted_bars(bars)
        df = add_bench_return(df, bench, self.rs_window, "bench_ret")
        df = add_regime(df, bench, self.regime_sma) if self.use_regime_filter else df.with_columns(
            pl.lit(True).alias("regime_ok")
        )
        df = df.with_columns(
            prior_high(self.breakout_window).alias("hi_prior"),
            prior_low(self.exit_window).alias("lo_prior"),
            prior_avg_volume(self.volume_window).alias("vol_prior"),
            (pct_return(self.rs_window) - pl.col("bench_ret")).alias("rel_ret"),
            atr(14).alias("atr"),
        )
        entry = (
            (pl.col("close") > pl.col("hi_prior"))
            & (pl.col("volume") > self.volume_mult * pl.col("vol_prior"))
            & (pl.col("rel_ret") > 0)
            & pl.col("regime_ok")
        )
        exit_ = pl.col("close") < pl.col("lo_prior")
        return finalize_signals(df, entry, exit_, priority=pl.col("rel_ret"))  # strongest vs Nifty first
