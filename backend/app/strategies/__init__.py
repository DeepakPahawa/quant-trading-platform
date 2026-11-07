"""Rule-based daily swing strategies. Each is independent and data-agnostic (see base.py)."""

from typing import Any, Dict, Type

from backend.app.strategies.base import SIGNAL_COLUMNS, Strategy
from backend.app.strategies.breakout_momentum import BreakoutMomentum
from backend.app.strategies.rs_pullback import RsPullback
from backend.app.strategies.rsi2_pullback import Rsi2Pullback

STRATEGIES: Dict[str, Type[Strategy]] = {
    cls.name: cls for cls in (Rsi2Pullback, BreakoutMomentum, RsPullback)
}


def create_strategy(name: str, **params: Any) -> Strategy:
    """Build a strategy by name with optional parameter overrides."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)  # type: ignore[call-arg]


__all__ = ["STRATEGIES", "SIGNAL_COLUMNS", "Strategy", "create_strategy",
           "Rsi2Pullback", "BreakoutMomentum", "RsPullback"]
