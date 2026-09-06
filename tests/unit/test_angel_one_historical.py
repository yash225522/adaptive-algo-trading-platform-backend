"""Unit and integration tests for Angel One historical data downloader."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.integrations.angel_one.cli import run_historical_smoke_test
from adaptive_trading.integrations.angel_one.client import AngelOneClient
from adaptive_trading.integrations.angel_one.config import AngelOneSettings
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneHistoricalDataError,
)
from adaptive_trading.integrations.angel_one.historical import (
    AngelOneHistoricalService,
    format_datetime_for_smartapi,
)
from adaptive_trading.integrations.angel_one.models import (
    ANGELONE_TO_TIMEFRAME,
    TIMEFRAME_TO_ANGELONE,
    AngelOneInterval,
    HistoricalDataRequest,
    HistoricalDataResult,
)

INDIA_TZ = ZoneInfo("Asia/Kolkata")
UTC_TZ = timezone.utc


def test_format_datetime_for_smartapi() -> None:
    """Verify conversion of timezone-aware datetimes to SmartAPI format."""
    # 1. Datetime in IST
    dt_ist = datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ)
    assert format_datetime_for_smartapi(dt_ist) == "2026-01-02 09:15"

    # 2. Datetime in UTC (03:45 UTC = 09:15 IST)
    dt_utc = datetime(2026, 1, 2, 3, 45, tzinfo=UTC_TZ)
    assert format_datetime_for_smartapi(dt_utc) == "2026-01-02 09:15"


def test_timeframe_interval_bidirectional_mappings() -> None:
    """Verify bidirectional mappings between TimeFrame and AngelOneInterval."""
    assert TIMEFRAME_TO_ANGELONE[TimeFrame.FIVE_MINUTES] == AngelOneInterval.FIVE_MINUTE
    assert ANGELONE_TO_TIMEFRAME[AngelOneInterval.FIVE_MINUTE] == TimeFrame.FIVE_MINUTES
    assert TIMEFRAME_TO_ANGELONE[TimeFrame.ONE_HOUR] == AngelOneInterval.ONE_HOUR
    assert ANGELONE_TO_TIMEFRAME[AngelOneInterval.ONE_DAY] == TimeFrame.ONE_DAY


def test_historical_request_validation() -> None:
    """Verify HistoricalDataRequest validates chronological bounds."""
    from_dt = datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ)
    to_dt = datetime(2026, 1, 2, 15, 30, tzinfo=INDIA_TZ)

    req = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        symbol_token="99926000",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=from_dt,
        to_datetime=to_dt,
    )
    assert req.symbol == "NIFTY"
    assert req.symbol_token == "99926000"

    # Inverted timestamps rejected
    with pytest.raises(ValueError, match="cannot be after to_datetime"):
        HistoricalDataRequest(
            exchange="NSE",
            symbol="NIFTY",
            symbol_token="99926000",
            interval=AngelOneInterval.FIVE_MINUTE,
            from_datetime=to_dt,
            to_datetime=from_dt,
        )


def test_historical_service_request_construction_and_mapping() -> None:
    """Verify SmartAPI parameter payload and response candle mapping."""
    mock_client = MagicMock(spec=AngelOneClient)
    mock_client.is_authenticated.return_value = True
    mock_smart_connect = MagicMock()
    mock_client.smart_connect = mock_smart_connect

    mock_smart_connect.getCandleData.return_value = {
        "status": True,
        "message": "SUCCESS",
        "errorcode": "",
        "data": [
            [
                "2026-01-02T09:15:00+05:30",
                26000.0,
                26050.0,
                25980.0,
                26030.0,
                120000,
            ],
            [
                "2026-01-02T09:20:00+05:30",
                26030.0,
                26070.0,
                26020.0,
                26050.0,
                95000,
            ],
        ],
    }

    service = AngelOneHistoricalService(client=mock_client)
    request = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        symbol_token="99926000",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
    )

    result = service.get_candles(request)

    # Verify SmartAPI call parameters
    mock_smart_connect.getCandleData.assert_called_once_with(
        {
            "exchange": "NSE",
            "symboltoken": "99926000",
            "interval": "FIVE_MINUTE",
            "fromdate": "2026-01-02 09:15",
            "todate": "2026-01-02 09:20",
        }
    )

    assert result.candle_count == 2
    assert isinstance(result.candles[0], Candle)
    assert result.candles[0].open == 26000.0
    assert result.candles[0].close == 26030.0
    assert result.candles[0].volume == 120000.0
    assert result.candles[0].open_interest is None
    assert result.candles[1].close == 26050.0


def test_historical_service_with_open_interest() -> None:
    """Verify open interest extraction when 7 elements are present."""
    mock_client = MagicMock(spec=AngelOneClient)
    mock_client.is_authenticated.return_value = True
    mock_smart_connect = MagicMock()
    mock_client.smart_connect = mock_smart_connect

    mock_smart_connect.getCandleData.return_value = {
        "status": True,
        "data": [
            [
                "2026-01-02T09:15:00+05:30",
                26000.0,
                26050.0,
                25980.0,
                26030.0,
                120000,
                1500000.0,
            ],
        ],
    }

    service = AngelOneHistoricalService(client=mock_client)
    request = HistoricalDataRequest(
        exchange="NFO",
        symbol="NIFTY_FUT",
        symbol_token="12345",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
    )

    result = service.get_candles(request)
    assert result.candle_count == 1
    assert result.candles[0].open_interest == 1500000.0


def test_historical_service_empty_data_response() -> None:
    """Verify empty candle payload returns clean empty result."""
    mock_client = MagicMock(spec=AngelOneClient)
    mock_client.is_authenticated.return_value = True
    mock_smart_connect = MagicMock()
    mock_client.smart_connect = mock_smart_connect

    mock_smart_connect.getCandleData.return_value = {
        "status": True,
        "message": "SUCCESS",
        "data": [],
    }

    service = AngelOneHistoricalService(client=mock_client)
    request = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        symbol_token="99926000",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
    )

    result = service.get_candles(request)
    assert result.candle_count == 0
    assert result.candles == []


def test_historical_service_api_failure_raises_error() -> None:
    """Verify rejected SmartAPI request raises AngelOneHistoricalDataError."""
    mock_client = MagicMock(spec=AngelOneClient)
    mock_client.is_authenticated.return_value = True
    mock_smart_connect = MagicMock()
    mock_client.smart_connect = mock_smart_connect

    mock_smart_connect.getCandleData.return_value = {
        "status": False,
        "message": "Invalid symbol token",
        "errorcode": "AB2001",
        "data": None,
    }

    service = AngelOneHistoricalService(client=mock_client)
    request = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        symbol_token="INVALID_TOKEN",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
    )

    with pytest.raises(AngelOneHistoricalDataError, match="Invalid symbol token"):
        service.get_candles(request)


def test_historical_service_network_exception_wrapped() -> None:
    """Verify transport errors are wrapped in AngelOneHistoricalDataError."""
    mock_client = MagicMock(spec=AngelOneClient)
    mock_client.is_authenticated.return_value = True
    mock_smart_connect = MagicMock()
    mock_client.smart_connect = mock_smart_connect

    mock_smart_connect.getCandleData.side_effect = TimeoutError("Request timed out")

    service = AngelOneHistoricalService(client=mock_client)
    request = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        symbol_token="99926000",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
    )

    with pytest.raises(AngelOneHistoricalDataError, match="Request timed out"):
        service.get_candles(request)


def test_historical_service_data_quality_integration() -> None:
    """Verify duplicate candles and data quality issues are detected."""
    mock_client = MagicMock(spec=AngelOneClient)
    mock_client.is_authenticated.return_value = True
    mock_smart_connect = MagicMock()
    mock_client.smart_connect = mock_smart_connect

    # Contains 1 duplicate row and 1 zero-volume row
    mock_smart_connect.getCandleData.return_value = {
        "status": True,
        "data": [
            ["2026-01-02T09:15:00+05:30", 26000.0, 26050.0, 25980.0, 26030.0, 120000],
            ["2026-01-02T09:15:00+05:30", 26000.0, 26050.0, 25980.0, 26030.0, 120000],
            ["2026-01-02T09:20:00+05:30", 26030.0, 26070.0, 26020.0, 26050.0, 0.0],
        ],
    }

    service = AngelOneHistoricalService(client=mock_client)
    request = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        symbol_token="99926000",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
    )

    result = service.get_candles(request)
    assert result.quality_report is not None
    assert result.quality_report.duplicate_count >= 1
    assert len(result.quality_report.issues) >= 1


def test_client_get_historical_candles_integration() -> None:
    """Verify AngelOneClient.get_historical_candles delegates properly."""
    settings = AngelOneSettings(
        api_key="key",
        client_code="code",
        password="pass",
        totp_secret="JBSWY3DPEHPK3PXP",
    )
    client = AngelOneClient(settings=settings)

    mock_result = HistoricalDataResult(
        symbol="NIFTY",
        symbol_token="99926000",
        exchange="NSE",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
        candles=[],
    )

    with patch.object(
        AngelOneHistoricalService, "get_candles", return_value=mock_result
    ) as mock_get:
        req = HistoricalDataRequest(
            exchange="NSE",
            symbol="NIFTY",
            symbol_token="99926000",
            interval=AngelOneInterval.FIVE_MINUTE,
            from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
            to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
        )
        res = client.get_historical_candles(req)
        assert res == mock_result
        mock_get.assert_called_once_with(req)


def test_historical_cli_subcommand() -> None:
    """Verify CLI historical subcommand executes and returns 0."""
    mock_result = HistoricalDataResult(
        symbol="NIFTY",
        symbol_token="99926000",
        exchange="NSE",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
        candles=[],
    )

    with patch.object(
        AngelOneClient, "get_historical_candles", return_value=mock_result
    ):
        exit_code = run_historical_smoke_test(
            exchange="NSE",
            symbol="NIFTY",
            symbol_token="99926000",
            interval_str="FIVE_MINUTE",
            from_str="2026-01-02 09:15",
            to_str="2026-01-02 09:20",
        )
        assert exit_code == 0


@pytest.mark.skipif(
    os.getenv("RUN_ANGELONE_INTEGRATION_TESTS") != "true",
    reason="Requires RUN_ANGELONE_INTEGRATION_TESTS=true and valid .env",
)
def test_real_angel_one_historical_integration() -> None:
    """Optional real integration test against Angel One API."""
    client = AngelOneClient()
    client.authenticate()

    # Request 15 minutes of historical candles
    request = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        symbol_token="99926000",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 30, tzinfo=INDIA_TZ),
    )

    result = client.get_historical_candles(request)
    assert result.symbol == "NIFTY"
    assert isinstance(result.candles, list)
