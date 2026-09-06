"""Unit tests for core domain models and data contracts."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain import (
    Candle,
    FeatureVector,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    Prediction,
    RiskDecision,
    Signal,
    SignalDirection,
    Trade,
)

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
UTC_TZ = timezone.utc


# ============================================================================
# 1. Market Candle Tests
# ============================================================================


def test_valid_candle() -> None:
    """Test that a valid Candle is accepted."""
    candle = Candle(
        timestamp=datetime(2026, 8, 30, 9, 15, tzinfo=KOLKATA_TZ),
        symbol="NIFTY",
        timeframe=TimeFrame.FIVE_MINUTES,
        open=24000.0,
        high=24050.0,
        low=23980.0,
        close=24020.0,
        volume=15000.0,
        open_interest=50000.0,
    )
    assert candle.symbol == "NIFTY"
    assert candle.high == 24050.0
    assert candle.open_interest == 50000.0


def test_candle_naive_timestamp_rejected() -> None:
    """Test that a naive timestamp without timezone is rejected."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        Candle(
            timestamp=datetime(2026, 8, 30, 9, 15),  # naive
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=24000.0,
            high=24050.0,
            low=23980.0,
            close=24020.0,
            volume=15000.0,
        )


@pytest.mark.parametrize(
    ("open_p", "high_p", "low_p", "close_p"),
    [
        (24100.0, 24050.0, 23980.0, 24000.0),  # high < open
        (24000.0, 24050.0, 23980.0, 24100.0),  # high < close
        (24000.0, 23950.0, 23980.0, 24000.0),  # high < low
        (23900.0, 24050.0, 23950.0, 24000.0),  # low > open
        (24000.0, 24050.0, 24030.0, 24020.0),  # low > close
    ],
)
def test_candle_invalid_ohlc_relationships_rejected(
    open_p: float, high_p: float, low_p: float, close_p: float
) -> None:
    """Test that physically impossible OHLC relationships are rejected."""
    with pytest.raises(ValidationError):
        Candle(
            timestamp=datetime(2026, 8, 30, 9, 15, tzinfo=KOLKATA_TZ),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=100.0,
        )


def test_candle_negative_prices_rejected() -> None:
    """Test that zero or negative prices are rejected."""
    with pytest.raises(ValidationError):
        Candle(
            timestamp=datetime(2026, 8, 30, 9, 15, tzinfo=KOLKATA_TZ),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=-100.0,
            high=24050.0,
            low=23980.0,
            close=24020.0,
            volume=100.0,
        )


def test_candle_negative_volume_rejected() -> None:
    """Test that negative volume is rejected."""
    with pytest.raises(ValidationError):
        Candle(
            timestamp=datetime(2026, 8, 30, 9, 15, tzinfo=KOLKATA_TZ),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=24000.0,
            high=24050.0,
            low=23980.0,
            close=24020.0,
            volume=-10.0,
        )


# ============================================================================
# 2. Feature Vector Tests
# ============================================================================


def test_valid_feature_vector() -> None:
    """Test that a valid FeatureVector is accepted."""
    features = {
        "return_5m": 0.0025,
        "volatility": 0.012,
        "volume_ratio": 1.45,
        "vwap_distance": -0.001,
        "oi_change": 2500.0,
    }
    fv = FeatureVector(
        timestamp=datetime(2026, 8, 30, 9, 20, tzinfo=UTC_TZ),
        symbol="NIFTY",
        features=features,
        feature_version="v1.0.0",
    )
    assert fv.features["return_5m"] == 0.0025
    assert fv.feature_version == "v1.0.0"


def test_feature_vector_naive_timestamp_rejected() -> None:
    """Test that FeatureVector rejects naive timestamp."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        FeatureVector(
            timestamp=datetime(2026, 8, 30, 9, 20),
            symbol="NIFTY",
            features={"volatility": 0.01},
            feature_version="v1.0.0",
        )


# ============================================================================
# 3. Prediction Tests
# ============================================================================


def test_valid_prediction() -> None:
    """Test that a valid Prediction model is accepted."""
    pred = Prediction(
        timestamp=datetime(2026, 8, 30, 9, 20, tzinfo=UTC_TZ),
        symbol="NIFTY",
        model_version="lgbm-v2.1",
        probability_up=0.65,
        probability_down=0.35,
        expected_return=0.0045,
    )
    assert pred.probability_up == 0.65
    assert pred.probability_down == 0.35
    assert pred.expected_return == 0.0045


def test_prediction_probability_out_of_bounds_rejected() -> None:
    """Test that probabilities outside [0, 1] are rejected."""
    with pytest.raises(ValidationError):
        Prediction(
            timestamp=datetime(2026, 8, 30, 9, 20, tzinfo=UTC_TZ),
            symbol="NIFTY",
            model_version="lgbm-v1",
            probability_up=1.2,
            probability_down=-0.2,
            expected_return=0.01,
        )


def test_prediction_probabilities_sum_mismatch_rejected() -> None:
    """Test that probabilities not summing to ~1.0 are rejected."""
    with pytest.raises(ValidationError, match="Probabilities must sum to ~1.0"):
        Prediction(
            timestamp=datetime(2026, 8, 30, 9, 20, tzinfo=UTC_TZ),
            symbol="NIFTY",
            model_version="lgbm-v1",
            probability_up=0.70,
            probability_down=0.70,
            expected_return=0.01,
        )


# ============================================================================
# 4. Trading Signal Tests
# ============================================================================


def test_valid_signal() -> None:
    """Test valid trading signals with different directions."""
    buy_sig = Signal(
        signal_id="sig-001",
        timestamp=datetime(2026, 8, 30, 9, 25, tzinfo=KOLKATA_TZ),
        symbol="NIFTY",
        direction=SignalDirection.BUY,
        quantity=50.0,
        entry_price=24000.0,
        stop_loss=23950.0,
        take_profit=24150.0,
        prediction_reference="pred-123",
    )
    assert buy_sig.direction == SignalDirection.BUY
    assert buy_sig.quantity == 50.0

    flat_sig = Signal(
        signal_id="sig-002",
        timestamp=datetime(2026, 8, 30, 9, 25, tzinfo=KOLKATA_TZ),
        symbol="NIFTY",
        direction=SignalDirection.FLAT,
        quantity=0.0,
    )
    assert flat_sig.direction == SignalDirection.FLAT


def test_signal_invalid_quantity_for_buy_rejected() -> None:
    """Test that zero or negative quantity is rejected for BUY/SELL signals."""
    with pytest.raises(ValidationError, match="Quantity must be positive"):
        Signal(
            signal_id="sig-003",
            timestamp=datetime(2026, 8, 30, 9, 25, tzinfo=KOLKATA_TZ),
            symbol="NIFTY",
            direction=SignalDirection.BUY,
            quantity=0.0,
        )


# ============================================================================
# 5. Risk Decision Tests
# ============================================================================


def test_valid_risk_decision() -> None:
    """Test approved and rejected risk decisions."""
    approved_decision = RiskDecision(
        timestamp=datetime(2026, 8, 30, 9, 26, tzinfo=KOLKATA_TZ),
        signal_id="sig-001",
        approved=True,
        risk_limit_version="risk-v1",
    )
    assert approved_decision.approved is True

    rejected_decision = RiskDecision(
        timestamp=datetime(2026, 8, 30, 9, 26, tzinfo=KOLKATA_TZ),
        signal_id="sig-001",
        approved=False,
        reason="Daily max drawdown limit reached",
        risk_limit_version="risk-v1",
    )
    assert rejected_decision.approved is False
    assert rejected_decision.reason == "Daily max drawdown limit reached"


# ============================================================================
# 6. Order Tests
# ============================================================================


def test_valid_market_order() -> None:
    """Test valid market order without explicit price."""
    order = Order(
        order_id="ord-001",
        timestamp=datetime(2026, 8, 30, 9, 27, tzinfo=KOLKATA_TZ),
        symbol="NIFTY",
        side=OrderSide.BUY,
        quantity=50.0,
        order_type=OrderType.MARKET,
        signal_id="sig-001",
    )
    assert order.order_type == OrderType.MARKET
    assert order.price is None


def test_valid_limit_order() -> None:
    """Test valid limit order with price."""
    order = Order(
        order_id="ord-002",
        timestamp=datetime(2026, 8, 30, 9, 27, tzinfo=KOLKATA_TZ),
        symbol="NIFTY",
        side=OrderSide.SELL,
        quantity=25.0,
        order_type=OrderType.LIMIT,
        price=24100.0,
    )
    assert order.order_type == OrderType.LIMIT
    assert order.price == 24100.0


def test_limit_order_missing_price_rejected() -> None:
    """Test that a limit order without a price is rejected."""
    with pytest.raises(ValidationError, match="Price is required for LIMIT orders"):
        Order(
            order_id="ord-003",
            timestamp=datetime(2026, 8, 30, 9, 27, tzinfo=KOLKATA_TZ),
            symbol="NIFTY",
            side=OrderSide.BUY,
            quantity=50.0,
            order_type=OrderType.LIMIT,
        )


# ============================================================================
# 7. Fill Tests
# ============================================================================


def test_valid_fill() -> None:
    """Test valid execution fill record."""
    fill = Fill(
        fill_id="fill-001",
        order_id="ord-001",
        timestamp=datetime(2026, 8, 30, 9, 27, 2, tzinfo=KOLKATA_TZ),
        symbol="NIFTY",
        side=OrderSide.BUY,
        quantity=50.0,
        fill_price=24002.5,
        fees=20.5,
        slippage=2.5,
    )
    assert fill.fill_price == 24002.5
    assert fill.fees == 20.5


def test_fill_negative_price_or_fees_rejected() -> None:
    """Test that negative fill price or negative fees are rejected."""
    with pytest.raises(ValidationError):
        Fill(
            fill_id="fill-002",
            order_id="ord-001",
            timestamp=datetime(2026, 8, 30, 9, 27, 2, tzinfo=KOLKATA_TZ),
            symbol="NIFTY",
            side=OrderSide.BUY,
            quantity=50.0,
            fill_price=-100.0,
        )


# ============================================================================
# 8. Position & Trade Tests
# ============================================================================


def test_valid_position() -> None:
    """Test valid portfolio position representation."""
    pos = Position(
        symbol="NIFTY",
        quantity=50.0,
        average_price=24000.0,
        unrealized_pnl=1250.0,
        realized_pnl=5000.0,
    )
    assert pos.symbol == "NIFTY"
    assert pos.unrealized_pnl == 1250.0


def test_valid_trade() -> None:
    """Test valid completed trade record."""
    trade = Trade(
        trade_id="tr-001",
        symbol="NIFTY",
        entry_timestamp=datetime(2026, 8, 30, 9, 30, tzinfo=KOLKATA_TZ),
        exit_timestamp=datetime(2026, 8, 30, 10, 15, tzinfo=KOLKATA_TZ),
        entry_price=24000.0,
        exit_price=24150.0,
        quantity=50.0,
        gross_pnl=7500.0,
        fees=45.0,
        slippage=5.0,
        net_pnl=7450.0,
        model_version="lgbm-v2.1",
        strategy_version="strat-trend-v1",
    )
    assert trade.net_pnl == 7450.0
    assert trade.strategy_version == "strat-trend-v1"


def test_trade_exit_before_entry_rejected() -> None:
    """Test that exit timestamp preceding entry timestamp is rejected."""
    with pytest.raises(ValidationError, match="cannot precede Entry timestamp"):
        Trade(
            trade_id="tr-002",
            symbol="NIFTY",
            entry_timestamp=datetime(2026, 8, 30, 10, 30, tzinfo=KOLKATA_TZ),
            exit_timestamp=datetime(2026, 8, 30, 9, 30, tzinfo=KOLKATA_TZ),  # earlier
            entry_price=24000.0,
            exit_price=24150.0,
            quantity=50.0,
            gross_pnl=7500.0,
            net_pnl=7450.0,
        )


# ============================================================================
# 9. Serialization / Deserialization Roundtrip Tests
# ============================================================================


def test_json_roundtrip_serialization() -> None:
    """Test that domain models cleanly serialize and deserialize to/from JSON."""
    candle = Candle(
        timestamp=datetime(2026, 8, 30, 9, 15, tzinfo=UTC_TZ),
        symbol="NIFTY",
        timeframe=TimeFrame.FIVE_MINUTES,
        open=24000.0,
        high=24050.0,
        low=23980.0,
        close=24020.0,
        volume=15000.0,
    )
    json_data = candle.model_dump_json()
    loaded_candle = Candle.model_validate_json(json_data)
    assert loaded_candle == candle

    prediction = Prediction(
        timestamp=datetime(2026, 8, 30, 9, 20, tzinfo=UTC_TZ),
        symbol="NIFTY",
        model_version="v1",
        probability_up=0.6,
        probability_down=0.4,
        expected_return=0.005,
    )
    assert Prediction.model_validate_json(prediction.model_dump_json()) == prediction
