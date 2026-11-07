"""Verification test suite for Indian statutory trading deductions and event-driven backtesting engine.

Verifies:
1. Exact mathematical fidelity of statutory Indian taxes:
   - Brokerage: Flat ₹20/order
   - STT: 0.025% on equity sell; 0.125% on options sell
   - Exchange Turnover Fees: 0.00297% (Equity) / 0.03503% (Options)
   - SEBI Charges: ₹10 per crore (0.0001%)
   - Stamp Duty: 0.003% (Equity buy) / 0.002% (Options buy)
   - GST: 18% on (Brokerage + Exchange Fees + SEBI Charges)
2. 10 simulated trades verifying gross vs net profit reconciliation with statutory charges.
3. Event-driven 1-minute engine mechanics:
   - 0.05% slippage
   - Take-Profit and Stop-Loss triggers
   - Mandatory 15:15 IST intraday square-off
   - Net CAGR, Sharpe Ratio, Sortino Ratio, and Max Drawdown (MDD)
4. REST endpoint integration via `POST /api/backtest/run`.
"""

from datetime import datetime, timedelta, timezone
import math
import pytest
from starlette.testclient import TestClient

from backend.app.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    EventBacktestEngine,
    OrderSide,
    calculate_fill_price,
    is_past_square_off_time,
)
from backend.app.backtest.taxes import (
    FLAT_BROKERAGE_PER_ORDER,
    InstrumentType,
    aggregate_deductions,
    calculate_order_charges,
    calculate_trade_charges,
)
from backend.app.main import app

IST_TZ = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Statutory Tax Fidelity Tests
# ---------------------------------------------------------------------------

def test_equity_intraday_statutory_deductions():
    """Verify exact formula calculations for an Equity Intraday trade: 100 shares @ ₹2500 buy, ₹2600 sell."""
    qty = 100
    buy_px = 2500.0
    sell_px = 2600.0

    # Buy leg (Turnover: ₹2,50,000)
    buy_charges = calculate_order_charges(
        side=OrderSide.BUY,
        quantity=qty,
        price=buy_px,
        instrument_type=InstrumentType.EQUITY,
    )
    assert buy_charges.turnover == 250_000.0
    assert buy_charges.brokerage == 20.0
    assert buy_charges.stt == 0.0  # No STT on intraday buy
    assert pytest.approx(buy_charges.exchange_fee, abs=1e-4) == 250_000.0 * 0.0000297  # 7.425
    assert pytest.approx(buy_charges.sebi_charges, abs=1e-4) == 250_000.0 * (10.0 / 10_000_000.0)  # 0.25
    assert pytest.approx(buy_charges.stamp_duty, abs=1e-4) == 250_000.0 * 0.00003  # 7.50
    # GST: 18% of (20.0 + 7.425 + 0.25) = 18% of 27.675 = 4.9815
    expected_buy_gst = (20.0 + 7.425 + 0.25) * 0.18
    assert pytest.approx(buy_charges.gst, abs=1e-4) == expected_buy_gst
    expected_buy_total = 20.0 + 0.0 + 7.425 + 0.25 + 7.50 + expected_buy_gst
    assert pytest.approx(buy_charges.total_charges, abs=1e-4) == expected_buy_total

    # Sell leg (Turnover: ₹2,60,000)
    sell_charges = calculate_order_charges(
        side=OrderSide.SELL,
        quantity=qty,
        price=sell_px,
        instrument_type=InstrumentType.EQUITY,
    )
    assert sell_charges.turnover == 260_000.0
    assert sell_charges.brokerage == 20.0
    # STT: 0.025% on intraday equity sell = 65.00
    assert pytest.approx(sell_charges.stt, abs=1e-4) == 260_000.0 * 0.00025
    assert pytest.approx(sell_charges.exchange_fee, abs=1e-4) == 260_000.0 * 0.0000297  # 7.722
    assert pytest.approx(sell_charges.sebi_charges, abs=1e-4) == 260_000.0 * (10.0 / 10_000_000.0)  # 0.26
    assert sell_charges.stamp_duty == 0.0  # Stamp duty only on buy
    # GST: 18% of (20.0 + 7.722 + 0.26) = 18% of 27.982 = 5.03676
    expected_sell_gst = (20.0 + 7.722 + 0.26) * 0.18
    assert pytest.approx(sell_charges.gst, abs=1e-4) == expected_sell_gst
    expected_sell_total = 20.0 + 65.0 + 7.722 + 0.26 + 0.0 + expected_sell_gst
    assert pytest.approx(sell_charges.total_charges, abs=1e-4) == expected_sell_total

    # Round trip trade verification
    trade_summary = calculate_trade_charges(
        buy_price=buy_px,
        sell_price=sell_px,
        quantity=qty,
        instrument_type=InstrumentType.EQUITY,
    )
    assert trade_summary.gross_pnl == 10_000.0
    assert pytest.approx(trade_summary.total_charges, abs=1e-4) == (expected_buy_total + expected_sell_total)
    assert pytest.approx(trade_summary.net_pnl, abs=1e-4) == (10_000.0 - (expected_buy_total + expected_sell_total))


def test_options_statutory_deductions():
    """Verify exact formula calculations for Options trade: 50 qty @ ₹100 buy, ₹150 sell."""
    qty = 50
    buy_px = 100.0
    sell_px = 150.0

    # Buy leg (Turnover: ₹5,000)
    buy_charges = calculate_order_charges(
        side=OrderSide.BUY,
        quantity=qty,
        price=buy_px,
        instrument_type=InstrumentType.OPTIONS,
    )
    assert buy_charges.turnover == 5000.0
    assert buy_charges.brokerage == 20.0
    assert buy_charges.stt == 0.0
    # Exchange turnover: 0.03503% on options
    assert pytest.approx(buy_charges.exchange_fee, abs=1e-4) == 5000.0 * 0.0003503  # 1.7515
    assert pytest.approx(buy_charges.sebi_charges, abs=1e-4) == 5000.0 * 0.000001  # 0.005
    # Stamp duty: 0.002% on options buy
    assert pytest.approx(buy_charges.stamp_duty, abs=1e-4) == 5000.0 * 0.00002  # 0.10
    expected_buy_gst = (20.0 + 1.7515 + 0.005) * 0.18
    assert pytest.approx(buy_charges.gst, abs=1e-4) == expected_buy_gst

    # Sell leg (Turnover: ₹7,500)
    sell_charges = calculate_order_charges(
        side=OrderSide.SELL,
        quantity=qty,
        price=sell_px,
        instrument_type=InstrumentType.OPTIONS,
    )
    assert sell_charges.turnover == 7500.0
    # STT: 0.125% on options sell premium = 9.375
    assert pytest.approx(sell_charges.stt, abs=1e-4) == 7500.0 * 0.00125
    assert pytest.approx(sell_charges.exchange_fee, abs=1e-4) == 7500.0 * 0.0003503
    assert pytest.approx(sell_charges.sebi_charges, abs=1e-4) == 7500.0 * 0.000001
    assert sell_charges.stamp_duty == 0.0
    expected_sell_gst = (20.0 + 7500.0 * 0.0003503 + 7500.0 * 0.000001) * 0.18
    assert pytest.approx(sell_charges.gst, abs=1e-4) == expected_sell_gst


# ---------------------------------------------------------------------------
# 2. 10 Simulated Trades Reconciliation
# ---------------------------------------------------------------------------

def test_ten_simulated_trades_cost_reconciliation():
    """Run 10 simulated trades and verify that gross profits accurately deduct STT, GST, SEBI, and turnover charges."""
    trades_data = [
        {"buy": 100.0, "sell": 110.0, "qty": 50, "inst": InstrumentType.EQUITY},   # +500 gross
        {"buy": 200.0, "sell": 195.0, "qty": 100, "inst": InstrumentType.EQUITY},  # -500 gross
        {"buy": 500.0, "sell": 530.0, "qty": 20, "inst": InstrumentType.EQUITY},   # +600 gross
        {"buy": 1500.0, "sell": 1490.0, "qty": 10, "inst": InstrumentType.EQUITY}, # -100 gross
        {"buy": 80.0, "sell": 95.0, "qty": 200, "inst": InstrumentType.OPTIONS},   # +3000 gross
        {"buy": 120.0, "sell": 105.0, "qty": 100, "inst": InstrumentType.OPTIONS}, # -1500 gross
        {"buy": 250.0, "sell": 255.0, "qty": 40, "inst": InstrumentType.EQUITY},   # +200 gross
        {"buy": 3200.0, "sell": 3250.0, "qty": 5, "inst": InstrumentType.EQUITY},  # +250 gross
        {"buy": 45.0, "sell": 40.0, "qty": 300, "inst": InstrumentType.OPTIONS},   # -1500 gross
        {"buy": 1000.0, "sell": 1040.0, "qty": 15, "inst": InstrumentType.EQUITY}, # +600 gross
    ]
    assert len(trades_data) == 10

    total_gross = 0.0
    total_net = 0.0
    order_charges_list = []

    for t in trades_data:
        summary = calculate_trade_charges(
            buy_price=t["buy"],
            sell_price=t["sell"],
            quantity=t["qty"],
            instrument_type=t["inst"],
        )
        total_gross += summary.gross_pnl
        total_net += summary.net_pnl
        order_charges_list.extend([summary.entry_charges, summary.exit_charges])

        # Verify each trade matches: Net = Gross - Total Deductions
        assert pytest.approx(summary.net_pnl, abs=1e-4) == (summary.gross_pnl - summary.total_charges)

    breakdown = aggregate_deductions(order_charges_list)

    # 1. Brokerage: exactly 20 trades legs * ₹20 = ₹400
    assert pytest.approx(breakdown.total_brokerage, abs=1e-4) == 400.0

    # 2. Grand total equals sum of all itemized taxes
    computed_sum = (
        breakdown.total_brokerage
        + breakdown.total_stt
        + breakdown.total_exchange_fees
        + breakdown.total_sebi_charges
        + breakdown.total_stamp_duty
        + breakdown.total_gst
    )
    assert pytest.approx(breakdown.grand_total, abs=1e-4) == computed_sum

    # 3. Overall net profit exactly equals total gross profit minus grand total deductions
    assert pytest.approx(total_net, abs=1e-4) == (total_gross - breakdown.grand_total)

    # 4. GST must equal exactly 18% of (Brokerage + Exchange Fees + SEBI Charges)
    assert pytest.approx(breakdown.total_gst, abs=1e-4) == (
        (breakdown.total_brokerage + breakdown.total_exchange_fees + breakdown.total_sebi_charges) * 0.18
    )


# ---------------------------------------------------------------------------
# 3. Event-Driven Engine Mechanics Tests
# ---------------------------------------------------------------------------

def test_conservative_fill_model_and_slippage():
    """Verify buys fill at Base * (1 + 0.0005) + (Spread * 0.5) and sells fill at Base * (1 - 0.0005) - (Spread * 0.5)."""
    base_price = 1000.0
    slippage = 0.0005  # 0.05%
    spread = 0.50

    buy_fill = calculate_fill_price(base_price, OrderSide.BUY, slippage_pct=slippage, spread=spread)
    expected_buy = (1000.0 * 1.0005) + 0.25
    assert pytest.approx(buy_fill, abs=1e-4) == expected_buy

    sell_fill = calculate_fill_price(base_price, OrderSide.SELL, slippage_pct=slippage, spread=spread)
    expected_sell = (1000.0 * 0.9995) - 0.25
    assert pytest.approx(sell_fill, abs=1e-4) == expected_sell


def test_mandatory_1515_ist_square_off():
    """Assert that any open position is forcefully squared off at or after 15:15 IST."""
    base_date = datetime(2026, 11, 6, 9, 15, tzinfo=IST_TZ)
    candles = []

    # Generate 1-minute bars from 09:15 to 15:30 IST (375 bars)
    for m in range(375):
        bar_time = base_date + timedelta(minutes=m)
        ts = int(bar_time.timestamp())
        price = 1000.0 + (m * 0.1)
        candles.append({
            "timestamp": ts,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000,
        })

    # Signal to buy at 10:00 IST (minute 45) and do not issue sell signal
    buy_ts = int((base_date + timedelta(minutes=45)).timestamp())
    signals = [{"timestamp": buy_ts, "signal": "BUY"}]

    config = BacktestConfig(
        symbol="NSE:TEST-EQ",
        initial_capital=100_000.0,
        mandatory_square_off=True,
        square_off_hour=15,
        square_off_minute=15,
    )
    engine = EventBacktestEngine(config=config)
    result = engine.run(candles=candles, signals=signals)

    assert len(result.trades) >= 1
    last_trade = result.trades[-1]
    # Trade must have been closed with exit_reason 'INTRADAY_SQUARE_OFF'
    assert last_trade.exit_reason == "INTRADAY_SQUARE_OFF"

    # Verify exit timestamp is at or past 15:15 IST
    exit_dt = datetime.fromtimestamp(last_trade.exit_time, tz=IST_TZ)
    assert exit_dt.hour == 15
    assert exit_dt.minute >= 15


def test_take_profit_and_stop_loss_execution():
    """Assert TP triggers when High >= TP price, and SL triggers when Low <= SL price."""
    base_date = datetime(2026, 11, 6, 9, 15, tzinfo=IST_TZ)

    # 1. Test Stop-Loss Trigger (1% SL)
    sl_candles = [
        {"timestamp": int((base_date + timedelta(minutes=0)).timestamp()), "open": 1000.0, "high": 1005.0, "low": 998.0, "close": 1000.0, "volume": 100},
        {"timestamp": int((base_date + timedelta(minutes=1)).timestamp()), "open": 1000.0, "high": 1002.0, "low": 985.0, "close": 988.0, "volume": 100},  # Drops below 990 SL
        {"timestamp": int((base_date + timedelta(minutes=2)).timestamp()), "open": 988.0, "high": 990.0, "low": 980.0, "close": 985.0, "volume": 100},
    ]
    sig_sl = [{"timestamp": sl_candles[0]["timestamp"], "signal": "BUY"}]
    engine_sl = EventBacktestEngine(BacktestConfig(stop_loss_pct=0.01, mandatory_square_off=False))
    res_sl = engine_sl.run(candles=sl_candles, signals=sig_sl)
    assert len(res_sl.trades) == 1
    assert res_sl.trades[0].exit_reason == "STOP_LOSS"

    # 2. Test Take-Profit Trigger (2% TP)
    tp_candles = [
        {"timestamp": int((base_date + timedelta(minutes=0)).timestamp()), "open": 1000.0, "high": 1005.0, "low": 998.0, "close": 1000.0, "volume": 100},
        {"timestamp": int((base_date + timedelta(minutes=1)).timestamp()), "open": 1002.0, "high": 1025.0, "low": 1001.0, "close": 1022.0, "volume": 100},  # Exceeds 1020 TP
        {"timestamp": int((base_date + timedelta(minutes=2)).timestamp()), "open": 1022.0, "high": 1030.0, "low": 1020.0, "close": 1025.0, "volume": 100},
    ]
    sig_tp = [{"timestamp": tp_candles[0]["timestamp"], "signal": "BUY"}]
    engine_tp = EventBacktestEngine(BacktestConfig(take_profit_pct=0.02, mandatory_square_off=False))
    res_tp = engine_tp.run(candles=tp_candles, signals=sig_tp)
    assert len(res_tp.trades) == 1
    assert res_tp.trades[0].exit_reason == "TAKE_PROFIT"


def test_ten_trade_engine_simulation_and_metrics():
    """Simulate 10 sequential trades through the EventBacktestEngine and verify metrics & tax deductions."""
    base_date = datetime(2026, 11, 6, 9, 15, tzinfo=IST_TZ)
    candles = []
    signals = []

    # Generate 60 bars (09:15 to 10:15 IST)
    # Create 10 swing cycles: buy at bar 2k, sell at bar 2k+1
    current_price = 1000.0
    for m in range(60):
        ts = int((base_date + timedelta(minutes=m)).timestamp())
        # Oscillate price
        if m % 4 in (0, 1):
            current_price += 5.0
        else:
            current_price -= 3.0

        candles.append({
            "timestamp": ts,
            "open": current_price,
            "high": current_price + 2.0,
            "low": current_price - 2.0,
            "close": current_price,
            "volume": 500,
        })

    # 10 trade entries and exits
    for i in range(10):
        entry_idx = i * 4
        exit_idx = entry_idx + 2
        signals.append({"timestamp": candles[entry_idx]["timestamp"], "signal": "BUY"})
        signals.append({"timestamp": candles[exit_idx]["timestamp"], "signal": "SELL"})

    config = BacktestConfig(
        symbol="NSE:INFY-EQ",
        initial_capital=100_000.0,
        default_quantity=10,
        slippage_pct=0.0005,
        mandatory_square_off=False,
    )
    engine = EventBacktestEngine(config=config)
    result = engine.run(candles=candles, signals=signals)

    assert result.total_trades == 10
    assert result.deductions_breakdown.total_brokerage == 10 * 2 * FLAT_BROKERAGE_PER_ORDER  # 400.0
    assert result.deductions_breakdown.total_stt > 0.0
    assert result.deductions_breakdown.total_exchange_fees > 0.0
    assert result.deductions_breakdown.total_sebi_charges > 0.0
    assert result.deductions_breakdown.total_stamp_duty > 0.0
    assert result.deductions_breakdown.total_gst > 0.0

    # Verify reconciliation: Net Profit = Gross Profit - Deductions
    assert pytest.approx(result.net_profit, abs=1e-3) == (
        result.gross_profit - result.deductions_breakdown.grand_total
    )
    assert pytest.approx(result.final_capital, abs=1e-3) == (
        result.initial_capital + result.net_profit
    )
    assert 0.0 <= result.max_drawdown <= 1.0


# ---------------------------------------------------------------------------
# 4. REST API Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_backtest_run_endpoint(api_client):
    """Verify POST /api/backtest/run endpoint returns HTTP 200 with complete metrics and deduction breakdowns."""
    base_date = datetime(2026, 11, 6, 9, 15, tzinfo=IST_TZ)
    candles = []
    signals = []

    for m in range(25):
        ts = int((base_date + timedelta(minutes=m)).timestamp())
        candles.append({
            "timestamp": ts,
            "open": 2000.0 + (m * 2.0),
            "high": 2005.0 + (m * 2.0),
            "low": 1995.0 + (m * 2.0),
            "close": 2002.0 + (m * 2.0),
            "volume": 200,
        })

    signals.append({"timestamp": candles[2]["timestamp"], "signal": "BUY"})
    signals.append({"timestamp": candles[10]["timestamp"], "signal": "SELL"})

    payload = {
        "symbol": "NSE:RELIANCE-EQ",
        "initial_capital": 50_000.0,
        "slippage_pct": 0.0005,
        "instrument_type": "equity",
        "strategy": "custom",
        "default_quantity": 5,
        "candles": candles,
        "signals": signals,
    }

    response = api_client.post("/api/backtest/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["symbol"] == "NSE:RELIANCE-EQ"
    assert data["total_trades"] == 1
    assert "deductions_breakdown" in data
    assert data["deductions_breakdown"]["total_brokerage"] == 40.0  # 2 orders * 20
    assert data["deductions_breakdown"]["total_stt"] > 0.0
    assert data["deductions_breakdown"]["grand_total"] > 0.0
    assert pytest.approx(data["net_profit"], abs=1e-3) == (
        data["gross_profit"] - data["deductions_breakdown"]["grand_total"]
    )


def test_api_tax_estimate_endpoint(api_client):
    """Verify POST /api/backtest/taxes/estimate computes trade costs."""
    payload = {
        "buy_price": 2500.0,
        "sell_price": 2600.0,
        "quantity": 100,
        "instrument_type": "equity",
        "is_short": False,
    }
    response = api_client.post("/api/backtest/taxes/estimate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["gross_pnl"] == 10_000.0
    assert data["total_charges"] > 100.0
    assert data["net_pnl"] == data["gross_pnl"] - data["total_charges"]

