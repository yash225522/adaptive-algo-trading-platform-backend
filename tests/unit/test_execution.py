"""Unit tests for the Broker execution abstraction, PaperBroker, and safety rules."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from adaptive_trading.execution.angelone_broker import AngelOneBroker
from adaptive_trading.execution.broker import Broker
from adaptive_trading.execution.cli import cmd_account, cmd_status, cmd_submit
from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.execution.exceptions import (
    LiveTradingNotEnabledError,
)
from adaptive_trading.execution.models import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    ProductType,
)
from adaptive_trading.execution.paper_broker import (
    PaperBroker,
    StaticMarketDataProvider,
)
from adaptive_trading.execution.service import ExecutionService
from adaptive_trading.portfolio.models import PortfolioState, PositionSide
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.strategy.models import SignalAction, TradingSignal

UTC_TZ = timezone.utc


def test_broker_interface_conformance() -> None:
    """Verify PaperBroker and AngelOneBroker adhere to Broker abstract interface."""
    assert issubclass(PaperBroker, Broker)
    assert issubclass(AngelOneBroker, Broker)

    paper = PaperBroker()
    assert isinstance(paper, Broker)
    angel = AngelOneBroker()
    assert isinstance(angel, Broker)


def test_order_request_validation() -> None:
    """Test validation errors for invalid OrderRequest instances."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    with pytest.raises(ValueError):
        OrderRequest(
            symbol="",
            exchange="NSE",
            side=OrderSide.BUY,
            quantity=10,
            timestamp=ts,
        )

    with pytest.raises(ValueError):
        OrderRequest(
            symbol="NIFTY",
            exchange="NSE",
            side=OrderSide.BUY,
            quantity=-5,
            timestamp=ts,
        )

    # Missing timezone on timestamp
    with pytest.raises(ValueError, match="timezone-aware"):
        OrderRequest(
            symbol="NIFTY",
            exchange="NSE",
            side=OrderSide.BUY,
            quantity=1,
            timestamp=datetime(2026, 1, 2, 9, 15),
        )


def test_paper_market_buy_order_execution() -> None:
    """Test standard market BUY order execution in PaperBroker."""
    provider = StaticMarketDataProvider({"NIFTY": 20000.0})
    config = ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0
    )
    broker = PaperBroker(config=config, market_data_provider=provider)

    req = OrderRequest(
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=2.0,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )
    order = broker.submit_order(req)

    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 2.0
    assert order.average_fill_price == 20000.0

    account = broker.get_account()
    assert account.cash == 60000.0  # 100,000 - 40,000
    assert "NIFTY" in account.positions
    pos = account.positions["NIFTY"]
    assert pos.side == PositionSide.LONG
    assert pos.quantity == 2.0
    assert pos.average_entry_price == 20000.0


def test_paper_market_sell_order_execution() -> None:
    """Test opening a SHORT position via market SELL in PaperBroker."""
    provider = StaticMarketDataProvider({"BANKNIFTY": 45000.0})
    config = ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0
    )
    broker = PaperBroker(config=config, market_data_provider=provider)

    req = OrderRequest(
        symbol="BANKNIFTY",
        exchange="NSE",
        side=OrderSide.SELL,
        quantity=1.0,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )
    order = broker.submit_order(req)

    assert order.status == OrderStatus.FILLED
    assert order.average_fill_price == 45000.0

    account = broker.get_account()
    pos = account.positions["BANKNIFTY"]
    assert pos.side == PositionSide.SHORT
    assert pos.quantity == 1.0


def test_paper_position_closing_and_realized_pnl() -> None:
    """Test full cycle: open LONG, price increases, close LONG, verify realized P&L."""
    provider = StaticMarketDataProvider({"NIFTY": 100.0})
    config = ExecutionConfig(
        initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0
    )
    broker = PaperBroker(config=config, market_data_provider=provider)

    # 1. Buy 10 @ 100 (Cost = 1000, Cash = 9000)
    req_buy = OrderRequest(
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=10.0,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )
    broker.submit_order(req_buy)

    # 2. Market price moves to 110
    provider.set_price("NIFTY", 110.0)

    # 3. Sell 10 @ 110 (Revenue = 1100, Profit = +100)
    req_sell = OrderRequest(
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.SELL,
        quantity=10.0,
        timestamp=datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ),
    )
    broker.submit_order(req_sell)

    account = broker.get_account()
    assert account.cash == 10100.0
    assert account.realized_pnl == 100.0
    assert account.positions["NIFTY"].side == PositionSide.FLAT
    assert account.positions["NIFTY"].quantity == 0.0


def test_paper_position_reversal() -> None:
    """Test reversing LONG into SHORT with excess sell quantity."""
    provider = StaticMarketDataProvider({"NIFTY": 100.0})
    config = ExecutionConfig(
        initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0
    )
    broker = PaperBroker(config=config, market_data_provider=provider)

    # 1. Buy 10 @ 100 -> LONG 10
    broker.submit_order(
        OrderRequest(
            symbol="NIFTY",
            exchange="NSE",
            side=OrderSide.BUY,
            quantity=10.0,
            timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
        )
    )

    # 2. Sell 20 @ 105 -> Closes 10 LONG (+50 PnL) and opens 10 SHORT @ 105
    provider.set_price("NIFTY", 105.0)
    broker.submit_order(
        OrderRequest(
            symbol="NIFTY",
            exchange="NSE",
            side=OrderSide.SELL,
            quantity=20.0,
            timestamp=datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ),
        )
    )

    account = broker.get_account()
    pos = account.positions["NIFTY"]
    assert pos.side == PositionSide.SHORT
    assert pos.quantity == 10.0
    assert pos.average_entry_price == 105.0
    assert account.realized_pnl == 50.0


def test_paper_insufficient_funds_rejection() -> None:
    """Test order rejection when account cash is insufficient."""
    provider = StaticMarketDataProvider({"NIFTY": 25000.0})
    config = ExecutionConfig(
        initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0
    )
    broker = PaperBroker(config=config, market_data_provider=provider)

    # Attempt to buy 1 unit worth 25,000 with only 10,000 cash
    req = OrderRequest(
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=1.0,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )
    order = broker.submit_order(req)

    assert order.status == OrderStatus.REJECTED
    assert "Insufficient funds" in (order.rejection_reason or "")
    assert broker.get_account().cash == 10_000.0


def test_paper_commission_and_slippage() -> None:
    """Verify configured commission and slippage calculations."""
    # Market price = 100.0
    # Slippage = 100 bps (1.0%) -> Buy price = 101.0
    # Commission = 10 bps (0.1%) -> 101.0 * 10 * 0.001 = 1.01
    provider = StaticMarketDataProvider({"NIFTY": 100.0})
    config = ExecutionConfig(
        initial_cash=10_000.0,
        commission_bps=10.0,  # 0.1%
        slippage_bps=100.0,  # 1.0%
    )
    broker = PaperBroker(config=config, market_data_provider=provider)

    req = OrderRequest(
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=10.0,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )
    order = broker.submit_order(req)

    assert order.status == OrderStatus.FILLED
    assert order.average_fill_price == 101.0

    fill = broker.fills[0]
    assert fill.price == 101.0
    assert fill.commission == 1.01
    assert fill.slippage == 1.0


def test_paper_idempotency_duplicate_client_order_id() -> None:
    """Verify duplicate submission with same client_order_id returns existing order."""
    provider = StaticMarketDataProvider({"NIFTY": 100.0})
    broker = PaperBroker(market_data_provider=provider)

    req1 = OrderRequest(
        client_order_id="unique-client-id-123",
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=1.0,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )
    order1 = broker.submit_order(req1)

    # Submit second request with same client_order_id
    req2 = OrderRequest(
        client_order_id="unique-client-id-123",
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=1.0,
        timestamp=datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ),
    )
    order2 = broker.submit_order(req2)

    assert order1.order_id == order2.order_id
    assert len(broker.fills) == 1


def test_execution_mode_safety_live_disabled() -> None:
    """Verify AngelOneBroker fails closed and blocks live orders."""
    angel = AngelOneBroker()
    req = OrderRequest(
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=1.0,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )

    with pytest.raises(LiveTradingNotEnabledError, match="disabled"):
        angel.submit_order(req)

    with pytest.raises(LiveTradingNotEnabledError):
        angel.cancel_order("order-1")

    with pytest.raises(LiveTradingNotEnabledError):
        angel.get_positions()

    with pytest.raises(LiveTradingNotEnabledError):
        angel.get_account()


def test_angel_one_broker_payload_formatting() -> None:
    """Verify AngelOneBroker builds correct SmartAPI payload structure."""
    angel = AngelOneBroker()
    req = OrderRequest(
        symbol="NIFTY",
        exchange="NSE",
        side=OrderSide.BUY,
        quantity=5.0,
        order_type=OrderType.MARKET,
        product_type=ProductType.INTRADAY,
        timestamp=datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
    )
    payload = angel.build_order_payload(req, symbol_token="99926000")

    assert payload["variety"] == "NORMAL"
    assert payload["tradingsymbol"] == "NIFTY"
    assert payload["symboltoken"] == "99926000"
    assert payload["transactiontype"] == "BUY"
    assert payload["exchange"] == "NSE"
    assert payload["ordertype"] == "MARKET"
    assert payload["producttype"] == "INTRADAY"
    assert payload["quantity"] == "5"


def test_execution_service_end_to_end_flow() -> None:
    """Test full pipeline: Signal -> RiskEngine -> ExecutionService -> PaperBroker."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    portfolio_state = PortfolioState(
        timestamp=ts,
        cash=100_000.0,
        equity=100_000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        portfolio_exposure=0.0,
        peak_equity=100_000.0,
        daily_start_equity=100_000.0,
        current_date=ts.date(),
        positions={},
    )

    # 1. Generate TradingSignal
    signal = TradingSignal(
        signal_id="sig-001",
        timestamp=ts,
        symbol="NIFTY",
        action=SignalAction.LONG,
        confidence=0.85,
        strategy_name="prob",
        strategy_version="v1",
        reason="Strong long",
    )

    # 2. RiskEngine evaluates and approves
    risk_engine = RiskEngine(
        config=RiskConfig(fixed_quantity=5.0, max_position_pct=0.20)
    )
    decision = risk_engine.evaluate(
        signal=signal,
        portfolio_state=portfolio_state,
        current_price=2000.0,
    )
    assert decision.approved is True
    assert decision.approved_quantity == 5.0

    # 3. ExecutionService submits to PaperBroker
    provider = StaticMarketDataProvider({"NIFTY": 2000.0})
    broker = PaperBroker(
        config=ExecutionConfig(initial_cash=100_000.0),
        market_data_provider=provider,
    )
    service = ExecutionService(broker=broker)

    order = service.execute_risk_decision(decision=decision, current_price=2000.0)
    assert order is not None
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 5.0

    account = broker.get_account()
    assert account.positions["NIFTY"].quantity == 5.0


def test_execution_cli_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test CLI commands status, account, and submit."""
    state_file = tmp_path / "account_state.json"
    monkeypatch.setattr("adaptive_trading.execution.cli.PAPER_STATE_FILE", state_file)

    cfg = ExecutionConfig()
    assert cmd_status(cfg) == 0
    assert cmd_account(cfg) == 0

    # Submit a paper order
    res = cmd_submit(
        symbol="NIFTY",
        side="BUY",
        quantity=1.0,
        price=25000.0,
        exchange="NSE",
        order_type="MARKET",
        config=cfg,
    )
    assert res == 0
