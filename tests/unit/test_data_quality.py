"""Unit tests for MarketDataQualityChecker and DataQualityReport."""

from datetime import datetime, timezone

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.models import QualitySeverity
from adaptive_trading.data.quality.checker import MarketDataQualityChecker

UTC_TZ = timezone.utc


def test_quality_checker_valid_candle() -> None:
    """Test that a valid record passes quality check without errors."""
    checker = MarketDataQualityChecker()
    record = {
        "timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
        "symbol": "NIFTY",
        "timeframe": TimeFrame.FIVE_MINUTES,
        "open": 26000.0,
        "high": 26050.0,
        "low": 25980.0,
        "close": 26030.0,
        "volume": 1500.0,
    }
    candle, issues = checker.check_record(2, record)

    assert candle is not None
    assert issues == []


def test_quality_checker_fatal_errors() -> None:
    """Test fatal error detection for negative price and invalid OHLC bounds."""
    checker = MarketDataQualityChecker()

    # High < Open
    rec1 = {
        "timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
        "symbol": "NIFTY",
        "timeframe": TimeFrame.FIVE_MINUTES,
        "open": 26100.0,
        "high": 26050.0,  # invalid
        "low": 25980.0,
        "close": 26030.0,
        "volume": 1000.0,
    }
    candle, issues = checker.check_record(2, rec1)
    assert candle is None
    assert any(
        i.issue_type == "INVALID_OHLC_HIGH" and i.severity == QualitySeverity.ERROR
        for i in issues
    )

    # Negative price
    rec2 = {
        "timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
        "symbol": "NIFTY",
        "timeframe": TimeFrame.FIVE_MINUTES,
        "open": 26000.0,
        "high": 26050.0,
        "low": -25980.0,  # negative
        "close": 26030.0,
        "volume": 1000.0,
    }
    candle, issues = checker.check_record(3, rec2)
    assert candle is None
    assert any(
        i.issue_type == "NON_POSITIVE_PRICE" and i.severity == QualitySeverity.ERROR
        for i in issues
    )


def test_missing_candle_detection() -> None:
    """Test that missing intraday candles are identified as warnings."""
    checker = MarketDataQualityChecker()
    # 5m timeframe: jump from 09:15 -> 09:30 (missing 09:20 and 09:25 = 2 candles)
    batch = [
        (
            2,
            {
                "timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 26000.0,
                "high": 26050.0,
                "low": 25980.0,
                "close": 26030.0,
                "volume": 1000.0,
            },
        ),
        (
            3,
            {
                "timestamp": datetime(2026, 1, 2, 9, 30, tzinfo=UTC_TZ),
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 26030.0,
                "high": 26070.0,
                "low": 26020.0,
                "close": 26050.0,
                "volume": 1200.0,
            },
        ),
    ]
    candles, report = checker.check_batch(batch)

    assert len(candles) == 2
    assert report.missing_candle_count == 2
    assert any(i.issue_type == "MISSING_CANDLES" for i in report.issues)


def test_session_transition_not_flagged_as_missing() -> None:
    """Test that overnight day transitions are not flagged as missing candles."""
    checker = MarketDataQualityChecker()
    batch = [
        (
            2,
            {
                "timestamp": datetime(2026, 1, 2, 15, 25, tzinfo=UTC_TZ),
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 26000.0,
                "high": 26050.0,
                "low": 25980.0,
                "close": 26030.0,
                "volume": 1000.0,
            },
        ),
        (
            3,
            {
                "timestamp": datetime(
                    2026, 1, 5, 9, 15, tzinfo=UTC_TZ
                ),  # next session (Monday)
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 26030.0,
                "high": 26070.0,
                "low": 26020.0,
                "close": 26050.0,
                "volume": 1200.0,
            },
        ),
    ]
    candles, report = checker.check_batch(batch)

    assert len(candles) == 2
    assert report.missing_candle_count == 0


def test_suspicious_price_jump_warning() -> None:
    """Test that price jump exceeding threshold generates a warning."""
    checker = MarketDataQualityChecker(price_jump_threshold_pct=5.0)
    batch = [
        (
            2,
            {
                "timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 20000.0,
                "high": 20050.0,
                "low": 19980.0,
                "close": 20000.0,
                "volume": 1000.0,
            },
        ),
        (
            3,
            {
                "timestamp": datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ),
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 22000.0,
                "high": 22100.0,
                "low": 21950.0,
                "close": 22000.0,  # 10% jump > 5% threshold
                "volume": 1000.0,
            },
        ),
    ]
    candles, report = checker.check_batch(batch)

    assert len(candles) == 2
    assert any(i.issue_type == "SUSPICIOUS_PRICE_JUMP" for i in report.issues)


def test_zero_volume_warning() -> None:
    """Test that zero volume candle produces a warning."""
    checker = MarketDataQualityChecker(warn_on_zero_volume=True)
    batch = [
        (
            2,
            {
                "timestamp": datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ),
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 26000.0,
                "high": 26050.0,
                "low": 25980.0,
                "close": 26030.0,
                "volume": 0.0,  # zero volume
            },
        ),
    ]
    candles, report = checker.check_batch(batch)

    assert len(candles) == 1
    assert any(i.issue_type == "ZERO_VOLUME" for i in report.issues)


def test_stale_flat_ohlc_warning() -> None:
    """Test that consecutive flat OHLC bars trigger a stale price warning."""
    checker = MarketDataQualityChecker(max_stale_bars=3)
    batch = [
        (
            i + 2,
            {
                "timestamp": datetime(2026, 1, 2, 9, 15 + i * 5, tzinfo=UTC_TZ),
                "symbol": "NIFTY",
                "timeframe": TimeFrame.FIVE_MINUTES,
                "open": 26000.0,
                "high": 26000.0,
                "low": 26000.0,
                "close": 26000.0,
                "volume": 100.0,
            },
        )
        for i in range(4)
    ]
    candles, report = checker.check_batch(batch)

    assert len(candles) == 4
    assert any(i.issue_type == "STALE_PRICES" for i in report.issues)
