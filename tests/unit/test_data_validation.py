"""Unit tests for MarketDataValidator."""

from datetime import datetime, timezone

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.validators.market_data import MarketDataValidator

UTC_TZ = timezone.utc


def test_validate_valid_record() -> None:
    """Test validating a normal valid record."""
    validator = MarketDataValidator()
    record = {
        "timestamp": "2026-01-02T09:15:00+05:30",
        "symbol": "NIFTY",
        "timeframe": "5m",
        "open": "26000.0",
        "high": "26050.0",
        "low": "25980.0",
        "close": "26030.0",
        "volume": "1000.0",
        "open_interest": "50000.0",
    }
    candle, errors = validator.validate_record(2, record)

    assert errors == []
    assert candle is not None
    assert candle.symbol == "NIFTY"
    assert candle.timeframe == TimeFrame.FIVE_MINUTES
    assert candle.open == 26000.0


def test_validate_naive_timestamp_rejected() -> None:
    """Test that naive timestamp is rejected without default timezone."""
    validator = MarketDataValidator()
    record = {
        "timestamp": "2026-01-02T09:15:00",  # naive
        "symbol": "NIFTY",
        "timeframe": "5m",
        "open": "26000.0",
        "high": "26050.0",
        "low": "25980.0",
        "close": "26030.0",
        "volume": "1000.0",
    }
    candle, errors = validator.validate_record(3, record)

    assert candle is None
    assert len(errors) == 1
    assert errors[0].field == "timestamp"
    assert "naive" in errors[0].problem


def test_validate_naive_timestamp_with_default_timezone() -> None:
    """Test that default timezone handles naive timestamp when configured."""
    validator = MarketDataValidator(default_timezone="Asia/Kolkata")
    record = {
        "timestamp": "2026-01-02T09:15:00",
        "symbol": "NIFTY",
        "timeframe": "5m",
        "open": "26000.0",
        "high": "26050.0",
        "low": "25980.0",
        "close": "26030.0",
        "volume": "1000.0",
    }
    candle, errors = validator.validate_record(4, record)

    assert errors == []
    assert candle is not None
    assert candle.timestamp.tzinfo is not None


def test_validate_negative_price_rejected() -> None:
    """Test that negative prices are rejected and reported in errors."""
    validator = MarketDataValidator()
    record = {
        "timestamp": "2026-01-02T09:15:00+05:30",
        "symbol": "NIFTY",
        "timeframe": "5m",
        "open": "26000.0",
        "high": "26050.0",
        "low": "25980.0",
        "close": "-26030.0",  # negative
        "volume": "1000.0",
    }
    candle, errors = validator.validate_record(5, record)

    assert candle is None
    assert any(e.field == "close" for e in errors)


def test_validate_invalid_ohlc_relationship() -> None:
    """Test that high < low or high < open is rejected."""
    validator = MarketDataValidator()
    record = {
        "timestamp": "2026-01-02T09:15:00+05:30",
        "symbol": "NIFTY",
        "timeframe": "5m",
        "open": "26100.0",
        "high": "26050.0",  # high < open
        "low": "25980.0",
        "close": "26030.0",
        "volume": "1000.0",
    }
    candle, errors = validator.validate_record(6, record)

    assert candle is None
    assert any(e.field == "high" for e in errors)


def test_validate_negative_volume_rejected() -> None:
    """Test that negative volume is rejected."""
    validator = MarketDataValidator()
    record = {
        "timestamp": "2026-01-02T09:15:00+05:30",
        "symbol": "NIFTY",
        "timeframe": "5m",
        "open": "26000.0",
        "high": "26050.0",
        "low": "25980.0",
        "close": "26030.0",
        "volume": "-10.0",
    }
    candle, errors = validator.validate_record(7, record)

    assert candle is None
    assert any(e.field == "volume" for e in errors)


def test_validate_batch_chronological_sorting() -> None:
    """Test that batch validation sorts candles chronologically."""
    validator = MarketDataValidator()
    records = [
        (
            2,
            {
                "timestamp": "2026-01-02T09:20:00Z",
                "symbol": "NIFTY",
                "timeframe": "5m",
                "open": "26030.0",
                "high": "26070.0",
                "low": "26020.0",
                "close": "26050.0",
                "volume": "1000",
            },
        ),
        (
            3,
            {
                "timestamp": "2026-01-02T09:15:00Z",  # Earlier timestamp
                "symbol": "NIFTY",
                "timeframe": "5m",
                "open": "26000.0",
                "high": "26050.0",
                "low": "25980.0",
                "close": "26030.0",
                "volume": "1000",
            },
        ),
    ]
    candles, errors = validator.validate_batch(records)

    assert errors == []
    assert len(candles) == 2
    assert candles[0].timestamp == datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    assert candles[1].timestamp == datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ)
