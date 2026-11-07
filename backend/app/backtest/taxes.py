"""Indian Statutory Trading Taxes and Charges Calculator.

Implements exact statutory deduction schedules for Indian equity and options markets:
- Brokerage: Flat ₹20 per executed order.
- STT (Securities Transaction Tax):
    - Intraday Equity: 0.025% on Sell turnover. (0% on Buy).
    - Options: 0.125% on Sell turnover / premium. (0% on Buy).
- Exchange Turnover Fees (NSE):
    - Equity: 0.00297% on turnover (Buy & Sell).
    - Options: 0.03503% on premium turnover (Buy & Sell).
- SEBI Charges:
    - ₹10 per crore (0.0001% = 0.000001 * turnover) on Buy & Sell.
- Stamp Duty:
    - Intraday Equity: 0.003% on Buy turnover. (0% on Sell).
    - Options: 0.002% on Buy turnover. (0% on Sell).
- GST:
    - 18% on (Brokerage + Exchange Turnover Fees + SEBI Charges).
"""

from enum import Enum
from typing import Iterable, List, Union
from pydantic import BaseModel, Field


class InstrumentType(str, Enum):
    """Supported trading instrument types."""
    EQUITY = "equity"
    OPTIONS = "options"


class OrderSide(str, Enum):
    """Order transaction side."""
    BUY = "buy"
    SELL = "sell"


# Statutory rate constants
FLAT_BROKERAGE_PER_ORDER: float = 20.0  # ₹20 flat per order

# Securities Transaction Tax (STT)
STT_EQUITY_INTRADAY_SELL_RATE: float = 0.00025  # 0.025%
STT_OPTIONS_SELL_RATE: float = 0.00125  # 0.125%

# Exchange Turnover Charges (NSE)
EXCHANGE_FEE_EQUITY_RATE: float = 0.0000297  # 0.00297%
EXCHANGE_FEE_OPTIONS_RATE: float = 0.0003503  # 0.03503%

# SEBI Turnover Fee (₹10 per crore)
SEBI_TURNOVER_RATE: float = 10.0 / 10_000_000.0  # 0.000001 (0.0001%)

# Stamp Duty
STAMP_DUTY_EQUITY_BUY_RATE: float = 0.00003  # 0.003%
STAMP_DUTY_OPTIONS_BUY_RATE: float = 0.00002  # 0.002%

# Goods and Services Tax (GST)
GST_RATE: float = 0.18  # 18% on Brokerage + Exchange Fees + SEBI Charges


class OrderCharges(BaseModel):
    """Itemized deductions for a single order leg."""
    side: OrderSide = Field(..., description="Order side ('buy' or 'sell')")
    price: float = Field(..., description="Executed fill price")
    quantity: int = Field(..., description="Quantity executed")
    turnover: float = Field(..., description="Order turnover value (price * quantity)")
    brokerage: float = Field(default=FLAT_BROKERAGE_PER_ORDER, description="Brokerage fee")
    stt: float = Field(default=0.0, description="Securities Transaction Tax")
    exchange_fee: float = Field(default=0.0, description="Exchange turnover charge")
    sebi_charges: float = Field(default=0.0, description="SEBI regulatory fee")
    stamp_duty: float = Field(default=0.0, description="Stamp duty on buy orders")
    gst: float = Field(default=0.0, description="GST (18% on Brokerage + Exchange Fee + SEBI Charges)")
    total_charges: float = Field(..., description="Total statutory + brokerage deductions")


class TotalDeductionsBreakdown(BaseModel):
    """Aggregated statutory tax and fee breakdown."""
    total_turnover: float = Field(default=0.0, description="Total trading turnover")
    total_brokerage: float = Field(default=0.0, description="Total brokerage paid")
    total_stt: float = Field(default=0.0, description="Total Securities Transaction Tax")
    total_exchange_fees: float = Field(default=0.0, description="Total Exchange Turnover Charges")
    total_sebi_charges: float = Field(default=0.0, description="Total SEBI regulatory charges")
    total_stamp_duty: float = Field(default=0.0, description="Total Stamp Duty paid")
    total_gst: float = Field(default=0.0, description="Total GST paid")
    grand_total: float = Field(default=0.0, description="Grand total of all deductions")


class TradeChargesSummary(BaseModel):
    """Charges breakdown for a complete round-trip trade (entry and exit)."""
    instrument_type: InstrumentType
    quantity: int
    entry_side: OrderSide
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    entry_charges: OrderCharges
    exit_charges: OrderCharges
    total_charges: float


def _parse_instrument(instrument: Union[InstrumentType, str]) -> InstrumentType:
    """Normalize string or enum to InstrumentType."""
    if isinstance(instrument, InstrumentType):
        return instrument
    val = str(instrument).strip().lower()
    if "opt" in val:
        return InstrumentType.OPTIONS
    return InstrumentType.EQUITY


def _parse_side(side: Union[OrderSide, str]) -> OrderSide:
    """Normalize string or enum to OrderSide."""
    if isinstance(side, OrderSide):
        return side
    val = str(side).strip().lower()
    if val in ("buy", "b", "long"):
        return OrderSide.BUY
    if val in ("sell", "s", "short"):
        return OrderSide.SELL
    raise ValueError(f"Invalid order side: {side}")


def calculate_order_charges(
    side: Union[OrderSide, str],
    quantity: int,
    price: float,
    instrument_type: Union[InstrumentType, str] = InstrumentType.EQUITY,
    brokerage_flat: float = FLAT_BROKERAGE_PER_ORDER,
) -> OrderCharges:
    """Calculate statutory Indian trading charges and brokerage for a single order leg.

    Args:
        side: Order side ('buy' or 'sell')
        quantity: Executed quantity (> 0)
        price: Executed price (> 0)
        instrument_type: 'equity' or 'options'
        brokerage_flat: Flat brokerage per order in ₹ (default ₹20)

    Returns:
        OrderCharges with itemized statutory costs and total deductions.
    """
    if quantity <= 0:
        raise ValueError(f"Quantity must be positive, got {quantity}")
    if price < 0:
        raise ValueError(f"Price cannot be negative, got {price}")

    inst = _parse_instrument(instrument_type)
    order_side = _parse_side(side)
    turnover = float(price * quantity)

    brokerage = float(brokerage_flat)

    # 1. Securities Transaction Tax (STT)
    if inst == InstrumentType.EQUITY:
        # Intraday equity: 0.025% on sell, 0% on buy
        stt = turnover * STT_EQUITY_INTRADAY_SELL_RATE if order_side == OrderSide.SELL else 0.0
    else:
        # Options: 0.125% on sell turnover (premium), 0% on buy
        stt = turnover * STT_OPTIONS_SELL_RATE if order_side == OrderSide.SELL else 0.0

    # 2. Exchange Turnover Fees
    if inst == InstrumentType.EQUITY:
        exchange_fee = turnover * EXCHANGE_FEE_EQUITY_RATE
    else:
        exchange_fee = turnover * EXCHANGE_FEE_OPTIONS_RATE

    # 3. SEBI Regulatory Turnover Charges (₹10 per crore = 0.0001%)
    sebi_charges = turnover * SEBI_TURNOVER_RATE

    # 4. Stamp Duty (Buy orders only)
    if order_side == OrderSide.BUY:
        if inst == InstrumentType.EQUITY:
            stamp_duty = turnover * STAMP_DUTY_EQUITY_BUY_RATE
        else:
            stamp_duty = turnover * STAMP_DUTY_OPTIONS_BUY_RATE
    else:
        stamp_duty = 0.0

    # 5. GST (18% on Brokerage + Exchange Fees + SEBI Charges)
    gst = (brokerage + exchange_fee + sebi_charges) * GST_RATE

    # 6. Total Charges
    total_charges = brokerage + stt + exchange_fee + sebi_charges + stamp_duty + gst

    return OrderCharges(
        side=order_side,
        price=price,
        quantity=quantity,
        turnover=turnover,
        brokerage=brokerage,
        stt=stt,
        exchange_fee=exchange_fee,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        gst=gst,
        total_charges=total_charges,
    )


def calculate_trade_charges(
    buy_price: float,
    sell_price: float,
    quantity: int,
    instrument_type: Union[InstrumentType, str] = InstrumentType.EQUITY,
    is_short: bool = False,
    brokerage_flat: float = FLAT_BROKERAGE_PER_ORDER,
) -> TradeChargesSummary:
    """Calculate complete round-trip trade costs (both entry and exit legs).

    Args:
        buy_price: Executed price of buy leg
        sell_price: Executed price of sell leg
        quantity: Trade quantity
        instrument_type: 'equity' or 'options'
        is_short: True if trade started with Sell (short) and exited with Buy
        brokerage_flat: Flat brokerage per order (default ₹20)

    Returns:
        TradeChargesSummary with entry/exit breakdowns, gross PnL, net PnL, and total deductions.
    """
    inst = _parse_instrument(instrument_type)

    buy_charges = calculate_order_charges(
        side=OrderSide.BUY,
        quantity=quantity,
        price=buy_price,
        instrument_type=inst,
        brokerage_flat=brokerage_flat,
    )
    sell_charges = calculate_order_charges(
        side=OrderSide.SELL,
        quantity=quantity,
        price=sell_price,
        instrument_type=inst,
        brokerage_flat=brokerage_flat,
    )

    if is_short:
        entry_side = OrderSide.SELL
        entry_charges = sell_charges
        exit_charges = buy_charges
        # Short gross pnl: (sell - buy) * qty
        gross_pnl = (sell_price - buy_price) * quantity
    else:
        entry_side = OrderSide.BUY
        entry_charges = buy_charges
        exit_charges = sell_charges
        # Long gross pnl: (sell - buy) * qty
        gross_pnl = (sell_price - buy_price) * quantity

    total_charges = entry_charges.total_charges + exit_charges.total_charges
    net_pnl = gross_pnl - total_charges

    return TradeChargesSummary(
        instrument_type=inst,
        quantity=quantity,
        entry_side=entry_side,
        entry_price=sell_price if is_short else buy_price,
        exit_price=buy_price if is_short else sell_price,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        entry_charges=entry_charges,
        exit_charges=exit_charges,
        total_charges=total_charges,
    )


def aggregate_deductions(charges_list: Iterable[OrderCharges]) -> TotalDeductionsBreakdown:
    """Aggregate individual order charges into a total deductions breakdown."""
    breakdown = TotalDeductionsBreakdown()
    for ch in charges_list:
        breakdown.total_turnover += ch.turnover
        breakdown.total_brokerage += ch.brokerage
        breakdown.total_stt += ch.stt
        breakdown.total_exchange_fees += ch.exchange_fee
        breakdown.total_sebi_charges += ch.sebi_charges
        breakdown.total_stamp_duty += ch.stamp_duty
        breakdown.total_gst += ch.gst
        breakdown.grand_total += ch.total_charges

    return breakdown

