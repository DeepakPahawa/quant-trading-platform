"""Event-driven backtesting engine for Indian equities and options.

Features:
- Sequential 1-minute candle processing simulating execution order queues.
- Realistic execution slippage (default 0.05%) and bid-ask spread adjustments.
- Stop-Loss (SL) and Take-Profit (TP) intra-bar evaluation.
- Mandatory intraday 15:15 IST square-off enforcement.
- Integrated Indian statutory trading taxes (STT, Exchange turnover fee, SEBI, Stamp Duty, GST).
- High-precision performance metrics: Net CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown (MDD),
  Profit Factor, Win Rate, and Total Deductions Breakdown.
"""

from collections import deque
from datetime import datetime, time, timedelta, timezone
from enum import Enum
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from backend.app.backtest.taxes import (
    FLAT_BROKERAGE_PER_ORDER,
    InstrumentType,
    OrderCharges,
    OrderSide,
    TotalDeductionsBreakdown,
    TradeChargesSummary,
    aggregate_deductions,
    calculate_order_charges,
    calculate_trade_charges,
)

logger = logging.getLogger(__name__)

# IST timezone: UTC+5:30
IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
MANDATORY_SQUARE_OFF_TIME = time(15, 15)  # 15:15 IST


class OrderType(str, Enum):
    """Execution order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderQueueItem(BaseModel):
    """Queued order waiting for execution."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    tag: str = "STRATEGY"
    created_at: int = Field(default=0, description="Epoch seconds timestamp of submission")


class TradeRecord(BaseModel):
    """Completed round-trip trade record."""
    trade_id: int
    symbol: str
    side: str = Field(..., description="'LONG' or 'SHORT'")
    quantity: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    total_charges: float
    exit_reason: str = Field(..., description="'TAKE_PROFIT', 'STOP_LOSS', 'INTRADAY_SQUARE_OFF', 'SIGNAL', 'SESSION_END'")
    charges: TradeChargesSummary


class EquityPoint(BaseModel):
    """Point in time equity curve tracking."""
    timestamp: int
    equity: float
    cash: float
    position: int
    close_price: float
    drawdown_pct: float


class BacktestConfig(BaseModel):
    """Configuration settings for the event backtest engine."""
    symbol: str = "NSE:NIFTY50-INDEX"
    initial_capital: float = Field(default=100_000.0, gt=0, description="Starting cash capital in INR")
    slippage_pct: float = Field(default=0.0005, ge=0.0, description="Execution slippage (0.05% = 0.0005)")
    spread: float = Field(default=0.0, ge=0.0, description="Half-spread / liquidity gap")
    brokerage_flat: float = Field(default=FLAT_BROKERAGE_PER_ORDER, ge=0.0, description="Brokerage flat ₹ per order")
    instrument_type: InstrumentType = InstrumentType.EQUITY
    mandatory_square_off: bool = Field(default=True, description="Enforce 15:15 IST mandatory square-off")
    square_off_hour: int = Field(default=15, description="Intraday square-off hour (IST)")
    square_off_minute: int = Field(default=15, description="Intraday square-off minute (IST)")
    risk_free_rate: float = Field(default=0.06, description="Annual risk-free rate (6% = 0.06)")
    default_quantity: int = Field(default=1, gt=0, description="Default lot/quantity per trade")
    take_profit_pct: Optional[float] = Field(default=None, description="Take profit percentage (e.g. 0.02 for 2%)")
    stop_loss_pct: Optional[float] = Field(default=None, description="Stop loss percentage (e.g. 0.01 for 1%)")


class BacktestResult(BaseModel):
    """Comprehensive backtest execution results and metrics."""
    symbol: str
    initial_capital: float
    final_capital: float
    gross_profit: float
    net_profit: float
    net_cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    deductions_breakdown: TotalDeductionsBreakdown
    trades: List[TradeRecord]
    equity_curve: List[EquityPoint]


def is_past_square_off_time(
    epoch_ts: int,
    cutoff_hour: int = 15,
    cutoff_minute: int = 15,
) -> bool:
    """Check whether timestamp is at or past intraday square-off cutoff in IST."""
    dt_ist = datetime.fromtimestamp(epoch_ts, tz=IST_TIMEZONE)
    cutoff = time(cutoff_hour, cutoff_minute)
    return dt_ist.time() >= cutoff


def calculate_fill_price(
    base_price: float,
    side: OrderSide,
    slippage_pct: float = 0.0005,
    spread: float = 0.0,
) -> float:
    """Calculate conservative execution fill price applying spread and slippage.

    Buys fill at: Ask + (Spread * 0.5) + Slippage
    Sells fill at: Bid - (Spread * 0.5) - Slippage
    """
    if side == OrderSide.BUY:
        return float(base_price * (1.0 + slippage_pct) + (spread * 0.5))
    else:
        return float(base_price * (1.0 - slippage_pct) - (spread * 0.5))


class EventBacktestEngine:
    """Sequential 1-minute event-driven backtesting engine with Indian statutory deductions."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self) -> None:
        """Reset internal simulator state."""
        self.cash: float = float(self.config.initial_capital)
        self.position: int = 0  # +qty for LONG, -qty for SHORT, 0 for FLAT
        self.entry_price: float = 0.0
        self.entry_time: int = 0
        self.entry_order_charges: Optional[OrderCharges] = None
        self.active_sl_price: Optional[float] = None
        self.active_tp_price: Optional[float] = None

        self.order_queue: deque[OrderQueueItem] = deque()
        self.trades: List[TradeRecord] = []
        self.order_charges_history: List[OrderCharges] = []
        self.equity_curve: List[EquityPoint] = []
        self.peak_equity: float = float(self.config.initial_capital)
        self.max_drawdown: float = 0.0
        self.trade_counter: int = 0

    def submit_order(
        self,
        side: Union[OrderSide, str],
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        tag: str = "STRATEGY",
        timestamp: int = 0,
    ) -> str:
        """Add an order to the execution queue."""
        if isinstance(side, OrderSide):
            order_side = side
        else:
            s = str(side).lower().strip()
            order_side = OrderSide.BUY if s in ("buy", "b", "long", "1") else OrderSide.SELL
        order_id = f"ORD_{len(self.order_queue) + len(self.trades) + 1}_{timestamp}"
        order = OrderQueueItem(
            order_id=order_id,
            symbol=self.config.symbol,
            side=order_side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_loss_pct=stop_loss_pct if stop_loss_pct is not None else self.config.stop_loss_pct,
            take_profit_pct=take_profit_pct if take_profit_pct is not None else self.config.take_profit_pct,
            tag=tag,
            created_at=timestamp,
        )
        self.order_queue.append(order)
        return order_id

    def _close_position(
        self,
        exit_price: float,
        exit_time: int,
        reason: str,
    ) -> None:
        """Close active open position and calculate statutory round-trip trade costs."""
        if self.position == 0 or self.entry_order_charges is None:
            return

        is_long = self.position > 0
        qty = abs(self.position)
        exit_side = OrderSide.SELL if is_long else OrderSide.BUY

        # Calculate exit leg statutory deductions
        exit_charges = calculate_order_charges(
            side=exit_side,
            quantity=qty,
            price=exit_price,
            instrument_type=self.config.instrument_type,
            brokerage_flat=self.config.brokerage_flat,
        )
        self.order_charges_history.append(exit_charges)

        # Gross & Net PnL calculation
        if is_long:
            gross_pnl = (exit_price - self.entry_price) * qty
            # Realized cash return: release entry capital + gross profit - exit charges
            self.cash += (exit_price * qty) - exit_charges.total_charges
        else:
            # Short position: gross pnl = (entry - exit) * qty
            gross_pnl = (self.entry_price - exit_price) * qty
            # Realized cash return: return collateral/margin + gross profit - exit charges
            self.cash += gross_pnl - exit_charges.total_charges

        total_trade_charges = self.entry_order_charges.total_charges + exit_charges.total_charges
        net_pnl = gross_pnl - total_trade_charges

        summary = TradeChargesSummary(
            instrument_type=self.config.instrument_type,
            quantity=qty,
            entry_side=OrderSide.BUY if is_long else OrderSide.SELL,
            entry_price=self.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            entry_charges=self.entry_order_charges,
            exit_charges=exit_charges,
            total_charges=total_trade_charges,
        )

        self.trade_counter += 1
        trade = TradeRecord(
            trade_id=self.trade_counter,
            symbol=self.config.symbol,
            side="LONG" if is_long else "SHORT",
            quantity=qty,
            entry_time=self.entry_time,
            exit_time=exit_time,
            entry_price=self.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_charges=total_trade_charges,
            exit_reason=reason,
            charges=summary,
        )
        self.trades.append(trade)

        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = 0
        self.entry_order_charges = None
        self.active_sl_price = None
        self.active_tp_price = None

    def _open_position(
        self,
        side: OrderSide,
        quantity: int,
        fill_price: float,
        timestamp: int,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> None:
        """Open a new position, deduct entry fees, and establish TP/SL levels."""
        entry_charges = calculate_order_charges(
            side=side,
            quantity=quantity,
            price=fill_price,
            instrument_type=self.config.instrument_type,
            brokerage_flat=self.config.brokerage_flat,
        )
        self.order_charges_history.append(entry_charges)

        if side == OrderSide.BUY:
            self.position = quantity
            # Cash outlay: capital cost + taxes
            self.cash -= (fill_price * quantity) + entry_charges.total_charges
        else:
            self.position = -quantity
            # For short, entry taxes debited
            self.cash -= entry_charges.total_charges

        self.entry_price = fill_price
        self.entry_time = timestamp
        self.entry_order_charges = entry_charges

        # Set TP / SL price thresholds
        sl_pct = stop_loss_pct if stop_loss_pct is not None else self.config.stop_loss_pct
        tp_pct = take_profit_pct if take_profit_pct is not None else self.config.take_profit_pct

        if side == OrderSide.BUY:
            self.active_sl_price = fill_price * (1.0 - sl_pct) if sl_pct is not None else None
            self.active_tp_price = fill_price * (1.0 + tp_pct) if tp_pct is not None else None
        else:
            self.active_sl_price = fill_price * (1.0 + sl_pct) if sl_pct is not None else None
            self.active_tp_price = fill_price * (1.0 - tp_pct) if tp_pct is not None else None

    def _process_queued_orders(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_timestamp: int,
        is_square_off_time: bool,
    ) -> None:
        """Evaluate and execute pending orders in queue against incoming 1-minute candle."""
        orders_to_requeue: deque[OrderQueueItem] = deque()

        while self.order_queue:
            order = self.order_queue.popleft()

            # Reject new entry orders during or after intraday square-off window
            if is_square_off_time:
                # Only allow closing orders
                is_closing = (self.position > 0 and order.side == OrderSide.SELL) or (
                    self.position < 0 and order.side == OrderSide.BUY
                )
                if not is_closing:
                    continue

            # Determine if order conditions are satisfied
            should_fill = False
            base_price = bar_open

            if order.order_type == OrderType.MARKET:
                should_fill = True
                base_price = bar_open
            elif order.order_type == OrderType.LIMIT and order.limit_price is not None:
                if order.side == OrderSide.BUY and bar_low <= order.limit_price:
                    should_fill = True
                    base_price = min(bar_open, order.limit_price)
                elif order.side == OrderSide.SELL and bar_high >= order.limit_price:
                    should_fill = True
                    base_price = max(bar_open, order.limit_price)
                else:
                    orders_to_requeue.append(order)
                    continue
            elif order.order_type == OrderType.STOP and order.stop_price is not None:
                if order.side == OrderSide.BUY and bar_high >= order.stop_price:
                    should_fill = True
                    base_price = max(bar_open, order.stop_price)
                elif order.side == OrderSide.SELL and bar_low <= order.stop_price:
                    should_fill = True
                    base_price = min(bar_open, order.stop_price)
                else:
                    orders_to_requeue.append(order)
                    continue

            if should_fill:
                fill_price = calculate_fill_price(
                    base_price=base_price,
                    side=order.side,
                    slippage_pct=self.config.slippage_pct,
                    spread=self.config.spread,
                )

                # Check if this order closes an existing position or opens a new one
                if self.position > 0:
                    if order.side == OrderSide.SELL:
                        self._close_position(
                            exit_price=fill_price,
                            exit_time=bar_timestamp,
                            reason=f"SIGNAL_CLOSE_{order.tag}",
                        )
                elif self.position < 0:
                    if order.side == OrderSide.BUY:
                        self._close_position(
                            exit_price=fill_price,
                            exit_time=bar_timestamp,
                            reason=f"SIGNAL_CLOSE_{order.tag}",
                        )
                else:
                    # Flat -> Open new position
                    self._open_position(
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        timestamp=bar_timestamp,
                        stop_loss_pct=order.stop_loss_pct,
                        take_profit_pct=order.take_profit_pct,
                    )

        # Restore unfilled pending limit/stop orders
        self.order_queue = orders_to_requeue

    def run(
        self,
        candles: Union[pl.DataFrame, List[Dict[str, Any]]],
        signals: Optional[Union[pl.DataFrame, List[Dict[str, Any]]]] = None,
        strategy_fn: Optional[Callable[[Dict[str, Any], "EventBacktestEngine"], None]] = None,
    ) -> BacktestResult:
        """Execute event-driven backtesting sequentially across 1-minute bars.

        Args:
            candles: Polars DataFrame or list of candle records containing
                     timestamp, open, high, low, close, volume.
            signals: Optional precomputed signals with 'timestamp' and 'signal' ('BUY', 'SELL', 'FLAT').
            strategy_fn: Optional callback invoked on each bar: `strategy_fn(bar, engine)`.

        Returns:
            BacktestResult with net CAGR, Sharpe, Sortino, MDD, and statutory deduction breakdown.
        """
        self.reset()

        if isinstance(candles, list):
            df = pl.DataFrame(candles)
        else:
            df = candles

        if len(df) == 0:
            raise ValueError("Candles dataset cannot be empty for backtesting.")

        # Ensure sorted chronologically
        df = df.sort("timestamp")
        records = df.to_dicts()

        # Build signal lookup if signals provided
        signal_map: Dict[int, str] = {}
        if signals is not None:
            sig_df = pl.DataFrame(signals) if isinstance(signals, list) else signals
            for row in sig_df.to_dicts():
                sig_map_val = str(row.get("signal", "")).upper()
                signal_map[int(row["timestamp"])] = sig_map_val

        for i, bar in enumerate(records):
            ts = int(bar["timestamp"])
            b_open = float(bar["open"])
            b_high = float(bar["high"])
            b_low = float(bar["low"])
            b_close = float(bar["close"])

            is_sq_off = False
            if self.config.mandatory_square_off:
                is_sq_off = is_past_square_off_time(
                    epoch_ts=ts,
                    cutoff_hour=self.config.square_off_hour,
                    cutoff_minute=self.config.square_off_minute,
                )

            # -----------------------------------------------------------------
            # 1. MANDATORY INTRADAY SQUARE-OFF CHECK (15:15 IST)
            # -----------------------------------------------------------------
            if is_sq_off and self.position != 0:
                is_long = self.position > 0
                sq_side = OrderSide.SELL if is_long else OrderSide.BUY
                sq_price = calculate_fill_price(
                    base_price=b_open,
                    side=sq_side,
                    slippage_pct=self.config.slippage_pct,
                    spread=self.config.spread,
                )
                self._close_position(
                    exit_price=sq_price,
                    exit_time=ts,
                    reason="INTRADAY_SQUARE_OFF",
                )
                # Clear pending entry orders for the session
                self.order_queue.clear()

            # -----------------------------------------------------------------
            # 2. PROCESS QUEUED ORDERS (At Bar Open)
            # -----------------------------------------------------------------
            self._process_queued_orders(
                bar_open=b_open,
                bar_high=b_high,
                bar_low=b_low,
                bar_timestamp=ts,
                is_square_off_time=is_sq_off,
            )

            # -----------------------------------------------------------------
            # 3. INTRA-BAR STOP-LOSS & TAKE-PROFIT EVALUATION (Bar High/Low)
            # -----------------------------------------------------------------
            if self.position != 0:
                if self.position > 0:  # LONG
                    sl_hit = (self.active_sl_price is not None) and (b_low <= self.active_sl_price)
                    tp_hit = (self.active_tp_price is not None) and (b_high >= self.active_tp_price)

                    # If both hit in same candle, prioritize Stop-Loss conservatively
                    if sl_hit:
                        exec_price = calculate_fill_price(
                            base_price=min(b_open, self.active_sl_price),  # type: ignore
                            side=OrderSide.SELL,
                            slippage_pct=self.config.slippage_pct,
                            spread=self.config.spread,
                        )
                        self._close_position(exit_price=exec_price, exit_time=ts, reason="STOP_LOSS")
                    elif tp_hit:
                        exec_price = calculate_fill_price(
                            base_price=max(b_open, self.active_tp_price),  # type: ignore
                            side=OrderSide.SELL,
                            slippage_pct=self.config.slippage_pct,
                            spread=self.config.spread,
                        )
                        self._close_position(exit_price=exec_price, exit_time=ts, reason="TAKE_PROFIT")

                elif self.position < 0:  # SHORT
                    sl_hit = (self.active_sl_price is not None) and (b_high >= self.active_sl_price)
                    tp_hit = (self.active_tp_price is not None) and (b_low <= self.active_tp_price)

                    if sl_hit:
                        exec_price = calculate_fill_price(
                            base_price=max(b_open, self.active_sl_price),  # type: ignore
                            side=OrderSide.BUY,
                            slippage_pct=self.config.slippage_pct,
                            spread=self.config.spread,
                        )
                        self._close_position(exit_price=exec_price, exit_time=ts, reason="STOP_LOSS")
                    elif tp_hit:
                        exec_price = calculate_fill_price(
                            base_price=min(b_open, self.active_tp_price),  # type: ignore
                            side=OrderSide.BUY,
                            slippage_pct=self.config.slippage_pct,
                            spread=self.config.spread,
                        )
                        self._close_position(exit_price=exec_price, exit_time=ts, reason="TAKE_PROFIT")

            # -----------------------------------------------------------------
            # 4. DISPATCH SIGNALS OR STRATEGY CALLBACK (For Next Bar)
            # -----------------------------------------------------------------
            if not is_sq_off:
                # Check precomputed signals
                if ts in signal_map:
                    sig = signal_map[ts]
                    if sig in ("BUY", "LONG", "1"):
                        if self.position < 0:
                            self.submit_order(OrderSide.BUY, abs(self.position), tag="REVERSE_EXIT", timestamp=ts)
                        if self.position <= 0:
                            self.submit_order(OrderSide.BUY, self.config.default_quantity, tag="ENTRY_BUY", timestamp=ts)
                    elif sig in ("SELL", "SHORT", "-1"):
                        if self.position > 0:
                            self.submit_order(OrderSide.SELL, abs(self.position), tag="EXIT_SELL", timestamp=ts)
                        elif self.position == 0:
                            self.submit_order(OrderSide.SELL, self.config.default_quantity, tag="ENTRY_SHORT", timestamp=ts)
                    elif sig in ("FLAT", "CLOSE", "EXIT", "0"):
                        if self.position > 0:
                            self.submit_order(OrderSide.SELL, abs(self.position), tag="CLOSE_FLAT", timestamp=ts)
                        elif self.position < 0:
                            self.submit_order(OrderSide.BUY, abs(self.position), tag="CLOSE_FLAT", timestamp=ts)

                # Invoke custom strategy hook if provided
                if strategy_fn is not None:
                    strategy_fn(bar, self)

            # -----------------------------------------------------------------
            # 5. MARK-TO-MARKET VALUATION & EQUITY CURVE
            # -----------------------------------------------------------------
            if self.position > 0:
                current_equity = self.cash + (b_close * self.position)
            elif self.position < 0:
                # Short: unrealized gross pnl = (entry - close) * qty
                unrealized = (self.entry_price - b_close) * abs(self.position)
                current_equity = self.cash + unrealized
            else:
                current_equity = self.cash

            self.peak_equity = max(self.peak_equity, current_equity)
            drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
            self.max_drawdown = max(self.max_drawdown, drawdown_pct)

            self.equity_curve.append(
                EquityPoint(
                    timestamp=ts,
                    equity=current_equity,
                    cash=self.cash,
                    position=self.position,
                    close_price=b_close,
                    drawdown_pct=drawdown_pct,
                )
            )

        # ---------------------------------------------------------------------
        # 6. END OF SESSION: CLOSE ANY REMAINING OPEN POSITIONS
        # ---------------------------------------------------------------------
        if self.position != 0 and len(records) > 0:
            last_bar = records[-1]
            last_ts = int(last_bar["timestamp"])
            last_close = float(last_bar["close"])
            is_long = self.position > 0
            exit_side = OrderSide.SELL if is_long else OrderSide.BUY
            exit_price = calculate_fill_price(
                base_price=last_close,
                side=exit_side,
                slippage_pct=self.config.slippage_pct,
                spread=self.config.spread,
            )
            self._close_position(exit_price=exit_price, exit_time=last_ts, reason="SESSION_END")

        # ---------------------------------------------------------------------
        # 7. PERFORMANCE METRICS COMPUTATION
        # ---------------------------------------------------------------------
        final_capital = self.cash
        gross_profit = sum(t.gross_pnl for t in self.trades)
        net_profit = sum(t.net_pnl for t in self.trades)
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.net_pnl < 0)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        # Profit Factor
        gross_wins = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        gross_losses = sum(abs(t.gross_pnl) for t in self.trades if t.gross_pnl < 0)
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

        # Statutory Deductions Breakdown
        deductions_breakdown = aggregate_deductions(self.order_charges_history)

        # Net CAGR
        start_ts = int(records[0]["timestamp"])
        end_ts = int(records[-1]["timestamp"])
        duration_seconds = max(end_ts - start_ts, 1)
        duration_years = duration_seconds / (365.25 * 86400.0)

        if final_capital > 0 and self.config.initial_capital > 0:
            if duration_years >= (1.0 / 365.25):
                net_cagr = (final_capital / self.config.initial_capital) ** (1.0 / duration_years) - 1.0
            else:
                # Less than a day duration: net return rate
                net_cagr = (final_capital - self.config.initial_capital) / self.config.initial_capital
        else:
            net_cagr = -1.0

        # Sharpe and Sortino Ratios
        sharpe_ratio, sortino_ratio = self._calculate_risk_adjusted_ratios()

        return BacktestResult(
            symbol=self.config.symbol,
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            gross_profit=gross_profit,
            net_profit=net_profit,
            net_cagr=net_cagr,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=self.max_drawdown,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            deductions_breakdown=deductions_breakdown,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

    def _calculate_risk_adjusted_ratios(self) -> Tuple[float, float]:
        """Calculate annualized Sharpe and Sortino Ratios using daily or bar periodic returns."""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0

        # Aggregate equity curve by calendar date (IST) to obtain daily returns
        daily_equity: Dict[str, float] = {}
        for pt in self.equity_curve:
            d_str = datetime.fromtimestamp(pt.timestamp, tz=IST_TIMEZONE).strftime("%Y-%m-%d")
            daily_equity[d_str] = pt.equity

        daily_vals = list(daily_equity.values())

        if len(daily_vals) >= 3:
            # Multi-day returns annualized with 252 trading days
            arr = np.array(daily_vals, dtype=np.float64)
            returns = np.diff(arr) / arr[:-1]
            ann_factor = 252.0
            daily_rf = self.config.risk_free_rate / 252.0
        else:
            # Intraday 1-minute bars: annualization factor = 252 days * 375 mins = 94,500
            eq_arr = np.array([pt.equity for pt in self.equity_curve], dtype=np.float64)
            returns = np.diff(eq_arr) / eq_arr[:-1]
            ann_factor = 94500.0
            daily_rf = self.config.risk_free_rate / 94500.0

        if len(returns) == 0:
            return 0.0, 0.0

        excess_returns = returns - daily_rf
        mean_excess = float(np.mean(excess_returns))
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else float(np.std(returns))

        sharpe = (math.sqrt(ann_factor) * (mean_excess / std_ret)) if std_ret > 1e-8 else 0.0

        # Downside deviation for Sortino
        downside_diff = np.minimum(0.0, excess_returns)
        downside_variance = float(np.mean(downside_diff ** 2))
        downside_std = math.sqrt(downside_variance)

        sortino = (math.sqrt(ann_factor) * (mean_excess / downside_std)) if downside_std > 1e-8 else 0.0

        # Handle inf/nan gracefully
        sharpe = 0.0 if math.isnan(sharpe) or math.isinf(sharpe) else sharpe
        sortino = 0.0 if math.isnan(sortino) or math.isinf(sortino) else sortino

        return sharpe, sortino
