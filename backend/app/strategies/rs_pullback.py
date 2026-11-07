"""Strategy 3: relative-strength pullback.

Leaders: each day, rank every stock in the supplied universe by 3-6 month momentum (the mean of
the 63-day and 126-day returns) and keep the top `top_fraction` (20%). The ranking is
cross-sectional and uses only same-day values, so run it on Nifty 500 for the intended behaviour.
On a smaller universe the same rule picks the top 20% of that universe.

Entry (signal at close): a leader whose short-term RSI(3) has dropped below `pullback_rsi_below`
while the close is still above its 50-day SMA (a dip inside an intact uptrend).
Exit (signal at close): RSI(3) back above `exit_rsi_above`, or the stock falls out of the top
`exit_keep_fraction` (40%) by momentum. Execution is at the next open.
Stop and time stop are safety nets: 3 x ATR and 15 bars. They are choices of this implementation, not
part of the original idea, so tune or disable them (None) when testing.
"""

from dataclasses import dataclass
from typing import Optional

import polars as pl

from backend.app.strategies.base import Strategy, finalize_signals, sorted_bars
from backend.app.strategies.indicators import add_regime, atr, pct_return, sma, wilder_rsi


@dataclass(frozen=True)
class RsPullback(Strategy):
    name = "rs_pullback"
    description = "Top 20% by 3-6 month momentum; buy a short RSI(3) pullback above the 50d SMA."

    mom_short_window: int = 63
    mom_long_window: int = 126
    top_fraction: float = 0.20
    exit_keep_fraction: float = 0.40
    pullback_rsi_period: int = 3
    pullback_rsi_below: float = 30.0
    exit_rsi_above: float = 60.0
    trend_sma: int = 50
    use_regime_filter: bool = False
    regime_sma: int = 200
    stop_atr_mult: Optional[float] = 3.0
    max_hold_days: Optional[int] = 15

    def prepare(self, bars: pl.DataFrame, bench: pl.DataFrame) -> pl.DataFrame:
        df = sorted_bars(bars)
        df = add_regime(df, bench, self.regime_sma) if self.use_regime_filter else df.with_columns(
            pl.lit(True).alias("regime_ok")
        )
        df = df.with_columns(
            ((pct_return(self.mom_short_window) + pct_return(self.mom_long_window)) / 2.0).alias("mom"),
            wilder_rsi(self.pullback_rsi_period).alias("rsi"),
            sma("close", self.trend_sma).alias("trend"),
            atr(14).alias("atr"),
        )
        # Same-day percentile among stocks that have a momentum value (1.0 = strongest).
        df = df.with_columns(
            (pl.col("mom").rank("average").over("date") / pl.col("mom").count().over("date")).alias("mom_rank")
        )
        entry = (
            (pl.col("mom_rank") >= 1.0 - self.top_fraction)
            & (pl.col("rsi") < self.pullback_rsi_below)
            & (pl.col("close") > pl.col("trend"))
            & pl.col("regime_ok")
        )
        exit_ = (pl.col("rsi") > self.exit_rsi_above) | (pl.col("mom_rank") < 1.0 - self.exit_keep_fraction)
        return finalize_signals(df, entry, exit_, priority=pl.col("mom"))  # strongest leaders first
