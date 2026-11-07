"""FastAPI Router for Event-Driven Backtesting and Statutory Indian Trading Taxes."""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, status
import polars as pl
from pydantic import BaseModel, Field

from backend.app.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    EventBacktestEngine,
)
from backend.app.backtest.strategies import build_strategy_signals
from backend.app.backtest.taxes import (
    FLAT_BROKERAGE_PER_ORDER,
    InstrumentType,
    OrderCharges,
    OrderSide,
    TotalDeductionsBreakdown,
    TradeChargesSummary,
    calculate_order_charges,
    calculate_trade_charges,
)
from backend.app.data.storage import DuckDBStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["Backtesting Engine"])

_storage_instance: Optional[DuckDBStorage] = None


def get_storage() -> DuckDBStorage:
    """Singleton getter for historical storage."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = DuckDBStorage()
    return _storage_instance


class CandleInput(BaseModel):
    """Schema for individual OHLCV input candle."""
    timestamp: int = Field(..., description="Epoch seconds timestamp")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: int = Field(default=0, description="Volume traded")
    open_interest: Optional[int] = Field(default=0, description="Open interest")


class SignalInput(BaseModel):
    """Schema for custom trade signal."""
    timestamp: int = Field(..., description="Epoch seconds timestamp")
    signal: str = Field(..., description="Signal directive ('BUY', 'SELL', 'FLAT')")


class BacktestRunRequest(BaseModel):
    """Request payload for running event-driven backtesting."""
    symbol: str = Field(default="NSE:NIFTY50-INDEX", description="Trading instrument symbol")
    initial_capital: float = Field(default=100_000.0, gt=0, description="Starting capital in INR")
    slippage_pct: float = Field(default=0.0005, ge=0.0, description="Execution slippage (0.05% = 0.0005)")
    spread: float = Field(default=0.0, ge=0.0, description="Half-spread in INR")
    brokerage_flat: float = Field(default=FLAT_BROKERAGE_PER_ORDER, ge=0.0, description="Flat brokerage ₹/order")
    instrument_type: str = Field(default="equity", description="'equity' or 'options'")
    mandatory_square_off: bool = Field(default=True, description="Enforce 15:15 IST intraday square-off")
    square_off_hour: int = Field(default=15, description="Cutoff hour in IST")
    square_off_minute: int = Field(default=15, description="Cutoff minute in IST")
    risk_free_rate: float = Field(default=0.06, description="Annual risk-free rate for Sharpe/Sortino")
    default_quantity: int = Field(default=1, gt=0, description="Quantity / Lot size per trade")
    take_profit_pct: Optional[float] = Field(default=None, description="Take-profit percentage (e.g. 0.02)")
    stop_loss_pct: Optional[float] = Field(default=None, description="Stop-loss percentage (e.g. 0.01)")
    strategy: str = Field(default="sma_cross", description="Strategy: 'sma_cross', 'rsi_momentum', 'vwap_bands', 'custom'")
    strategy_params: Optional[Dict[str, Any]] = Field(default=None, description="Strategy hyperparameters")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    candles: Optional[List[CandleInput]] = Field(default=None, description="Explicit candle sequence")
    signals: Optional[List[SignalInput]] = Field(default=None, description="Explicit signals if strategy='custom'")


class BacktestRunResponse(BaseModel):
    """Response payload for backtest execution."""
    status: str = Field(default="ok", description="Status code")
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
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]


class TaxEstimateRequest(BaseModel):
    """Request schema for estimating statutory Indian trade taxes."""
    buy_price: float = Field(..., gt=0, description="Execution buy price")
    sell_price: float = Field(..., gt=0, description="Execution sell price")
    quantity: int = Field(default=1, gt=0, description="Traded quantity")
    instrument_type: str = Field(default="equity", description="'equity' or 'options'")
    is_short: bool = Field(default=False, description="Whether short trade (sell first)")
    brokerage_flat: float = Field(default=FLAT_BROKERAGE_PER_ORDER, ge=0.0, description="Brokerage per order")


@router.post("/run", response_model=BacktestRunResponse)
def run_backtest(request: BacktestRunRequest) -> BacktestRunResponse:
    """Execute event-driven 1-minute backtesting incorporating full Indian statutory charges.

    Simulates:
    - Sequential 1-minute candle loops and order queues.
    - 0.05% slippage and spread adjustments.
    - Stop-Loss and Take-Profit intra-candle checks.
    - Mandatory 15:15 IST intraday square-off.
    - Itemized statutory deductions: STT, Exchange fees, SEBI charges, Stamp duty, GST.
    - Comprehensive performance metrics: Net CAGR, Sharpe, Sortino, MDD, Profit Factor.
    """
    # 1. Load or validate candles
    if request.candles and len(request.candles) > 0:
        candle_dicts = [c.model_dump() for c in request.candles]
        df_candles = pl.DataFrame(candle_dicts)
    else:
        # Query from DuckDB storage
        storage = get_storage()
        df_candles = storage.query_bars(
            symbol=request.symbol,
            range_from=request.start_date,
            range_to=request.end_date,
            resolution="1",
        )

    if df_candles.is_empty():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No 1-minute candle data available for symbol '{request.symbol}' "
                f"in range [{request.start_date} to {request.end_date}]. "
                f"Please provide candles in the request payload or ingest historical bars first."
            ),
        )

    # 2. Build configuration
    inst_type = InstrumentType.OPTIONS if "opt" in request.instrument_type.lower() else InstrumentType.EQUITY
    config = BacktestConfig(
        symbol=request.symbol,
        initial_capital=request.initial_capital,
        slippage_pct=request.slippage_pct,
        spread=request.spread,
        brokerage_flat=request.brokerage_flat,
        instrument_type=inst_type,
        mandatory_square_off=request.mandatory_square_off,
        square_off_hour=request.square_off_hour,
        square_off_minute=request.square_off_minute,
        risk_free_rate=request.risk_free_rate,
        default_quantity=request.default_quantity,
        take_profit_pct=request.take_profit_pct,
        stop_loss_pct=request.stop_loss_pct,
    )

    # 3. Determine trade signals
    signals_df: Optional[pl.DataFrame] = None
    if request.signals and len(request.signals) > 0:
        sig_dicts = [s.model_dump() for s in request.signals]
        signals_df = pl.DataFrame(sig_dicts)
    elif request.strategy.lower() != "custom":
        signals_df = build_strategy_signals(
            strategy_name=request.strategy,
            df=df_candles,
            params=request.strategy_params,
        )

    # 4. Run event-driven simulation
    engine = EventBacktestEngine(config=config)
    result: BacktestResult = engine.run(candles=df_candles, signals=signals_df)

    # 5. Build and return structured response
    return BacktestRunResponse(
        status="ok",
        symbol=result.symbol,
        initial_capital=result.initial_capital,
        final_capital=result.final_capital,
        gross_profit=result.gross_profit,
        net_profit=result.net_profit,
        net_cagr=result.net_cagr,
        sharpe_ratio=result.sharpe_ratio,
        sortino_ratio=result.sortino_ratio,
        max_drawdown=result.max_drawdown,
        profit_factor=result.profit_factor,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate=result.win_rate,
        deductions_breakdown=result.deductions_breakdown,
        trades=[t.model_dump() for t in result.trades],
        equity_curve=[e.model_dump() for e in result.equity_curve],
    )


@router.post("/taxes/estimate", response_model=TradeChargesSummary)
def estimate_trade_taxes(request: TaxEstimateRequest) -> TradeChargesSummary:
    """Calculate exact statutory Indian charges breakdown for a round-trip trade."""
    inst_type = InstrumentType.OPTIONS if "opt" in request.instrument_type.lower() else InstrumentType.EQUITY
    summary = calculate_trade_charges(
        buy_price=request.buy_price,
        sell_price=request.sell_price,
        quantity=request.quantity,
        instrument_type=inst_type,
        is_short=request.is_short,
        brokerage_flat=request.brokerage_flat,
    )
    return summary

