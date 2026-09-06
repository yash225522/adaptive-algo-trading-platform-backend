"""Unit tests for MarketDataNormalizer."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.models import QualitySeverity
from adaptive_trading.data.normalizers.market_data import (
    MarketDataNormalizer,
)

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
UTC_TZ = timezone.utc


def test_normalize_valid_record() -> None:
    """Test normalizing a well-formed raw record dictionary."""
    normalizer = MarketDataNormalizer()
    raw = {
        "timestamp": "2026-01-02T09:15:00+05:30",
        "symbol": " nifty ",
        "timeframe": " 5m ",
        "open": "26000.5",
        "high": " 26050.0 ",
        "low": "25980.0",
        "close": "26030.0",
        "volume": "1000",
        "open_interest": "50000",
    }
    normalized, issues = normalizer.normalize_record(2, raw)

    assert issues == []
    assert normalized is not None
    assert normalized["symbol"] == "NIFTY"
    assert normalized["timeframe"] == TimeFrame.FIVE_MINUTES
    assert normalized["timestamp"] == datetime(2026, 1, 2, 9, 15, tzinfo=KOLKATA_TZ)
    assert normalized["open"] == 26000.5
    assert normalized["volume"] == 1000.0
    assert normalized["open_interest"] == 50000.0


def test_normalize_naive_timestamp_with_default_timezone() -> None:
    """Test assigning default timezone to naive timestamp string."""
    normalizer = MarketDataNormalizer(default_timezone="Asia/Kolkata")
    raw = {
        "timestamp": "2026-01-02 09:15:00",
        "symbol": "NIFTY",
        "timeframe": "5m",
        "open": "26000.0",
        "high": "26050.0",
        "low": "25980.0",
        "close": "26030.0",
        "volume": "1000",
    }
    normalized, issues = normalizer.normalize_record(3, raw)

    assert issues == []
    assert normalized is not None
    assert normalized["timestamp"].tzinfo is not None


def test_normalize_missing_or_invalid_fields() -> None:
    """Test error reporting on invalid symbol, timeframe, or unparseable numbers."""
    normalizer = MarketDataNormalizer()
    raw = {
        "timestamp": "invalid_ts",
        "symbol": "",
        "timeframe": "invalid_tf",
        "open": "not_a_number",
        "high": "26050.0",
        "low": "25980.0",
        "close": "26030.0",
        "volume": "1000",
    }
    normalized, issues = normalizer.normalize_record(4, raw)

    assert normalized is None
    assert len(issues) >= 4
    assert all(i.severity == QualitySeverity.ERROR for i in issues)


def test_normalize_batch() -> None:
    """Test normalizing a list of raw records."""
    normalizer = MarketDataNormalizer()
    batch = [
        (
            2,
            {
                "timestamp": "2026-01-02T09:15:00Z",
                "symbol": "banknifty",
                "timeframe": "15m",
                "open": "50000",
                "high": "50100",
                "low": "49950",
                "close": "50050",
                "volume": "2500",
            },
        ),
        (
            3,
            {
                "timestamp": "2026-01-02T09:30:00Z",
                "symbol": "nifty",
                "timeframe": "5m",
                "open": "26000",
                "high": "26050",
                "low": "25980",
                "close": "26030",
                "volume": "1000",
            },
        ),
    ]
    normalized_records, issues = normalizer.normalize_batch(batch)

    assert issues == []
    assert len(normalized_records) == 2
    assert normalized_records[0][1]["symbol"] == "BANKNIFTY"
    assert normalized_records[1][1]["symbol"] == "NIFTY"
