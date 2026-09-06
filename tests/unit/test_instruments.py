"""Unit tests for Instrument Master, repository, importer, and resolver."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from adaptive_trading.instruments.cli import run_refresh, run_resolve
from adaptive_trading.instruments.exceptions import (
    InstrumentNotFoundError,
    InstrumentResolutionError,
)
from adaptive_trading.instruments.importer import AngelOneInstrumentImporter
from adaptive_trading.instruments.models import (
    Instrument,
    InstrumentQuery,
    InstrumentType,
    OptionType,
)
from adaptive_trading.instruments.repository import InstrumentRepository
from adaptive_trading.instruments.resolver import InstrumentResolver
from adaptive_trading.integrations.angel_one.historical import (
    AngelOneHistoricalService,
)
from adaptive_trading.integrations.angel_one.models import (
    AngelOneInterval,
    HistoricalDataRequest,
)

INDIA_TZ = ZoneInfo("Asia/Kolkata")


SAMPLE_INSTRUMENT_DATA: list[dict[str, object]] = [
    {
        "token": "TEST_IDX_001",
        "symbol": "Nifty 50",
        "name": "NIFTY",
        "expiry": "",
        "strike": "0.000000",
        "lotsize": "1",
        "instrumenttype": "AMXIDX",
        "exch_seg": "NSE",
        "tick_size": "0.000000",
    },
    {
        "token": "TEST_EQ_001",
        "symbol": "SBIN-EQ",
        "name": "SBIN",
        "expiry": "",
        "strike": "-1.000000",
        "lotsize": "1",
        "instrumenttype": "",
        "exch_seg": "NSE",
        "tick_size": "5.000000",
    },
    {
        "token": "TEST_FUT_001",
        "symbol": "NIFTY29JAN26FUT",
        "name": "NIFTY",
        "expiry": "29JAN2026",
        "strike": "-1.000000",
        "lotsize": "50",
        "instrumenttype": "FUTIDX",
        "exch_seg": "NFO",
        "tick_size": "5.000000",
    },
    {
        "token": "TEST_OPT_CE_001",
        "symbol": "NIFTY29JAN2626000CE",
        "name": "NIFTY",
        "expiry": "29JAN2026",
        "strike": "2600000.000000",
        "lotsize": "50",
        "instrumenttype": "OPTIDX",
        "exch_seg": "NFO",
        "tick_size": "5.000000",
    },
    {
        "token": "TEST_OPT_PE_001",
        "symbol": "NIFTY29JAN2626000PE",
        "name": "NIFTY",
        "expiry": "29JAN2026",
        "strike": "2600000.000000",
        "lotsize": "50",
        "instrumenttype": "OPTIDX",
        "exch_seg": "NFO",
        "tick_size": "5.000000",
    },
    {
        "token": "TEST_EQ_002",
        "symbol": "RELIANCE-EQ",
        "name": "RELIANCE",
        "expiry": "",
        "strike": "-1.000000",
        "lotsize": "1",
        "instrumenttype": "EQ",
        "exch_seg": "NSE",
        "tick_size": "5.000000",
    },
]


def test_instrument_domain_model() -> None:
    """Test valid Instrument domain model instantiation."""
    inst = Instrument(
        symbol_token="TEST_TOKEN_001",
        exchange="NSE",
        trading_symbol="Nifty 50",
        symbol="NIFTY",
        name="NIFTY",
        instrument_type=InstrumentType.INDEX,
        lot_size=1,
        tick_size=0.05,
    )
    assert inst.symbol_token == "TEST_TOKEN_001"
    assert inst.exchange == "NSE"
    assert inst.instrument_type == InstrumentType.INDEX
    assert inst.expiry is None
    assert inst.strike is None


def test_importer_item_parsing() -> None:
    """Test parsing of different asset classes from Angel One JSON items."""
    importer = AngelOneInstrumentImporter()

    # 1. Index
    idx = importer.parse_item(SAMPLE_INSTRUMENT_DATA[0])
    assert idx is not None
    assert idx.instrument_type == InstrumentType.INDEX
    assert idx.trading_symbol == "Nifty 50"

    # 2. Equity
    eq = importer.parse_item(SAMPLE_INSTRUMENT_DATA[1])
    assert eq is not None
    assert eq.instrument_type == InstrumentType.EQUITY
    assert eq.trading_symbol == "SBIN-EQ"

    # 3. Futures
    fut = importer.parse_item(SAMPLE_INSTRUMENT_DATA[2])
    assert fut is not None
    assert fut.instrument_type == InstrumentType.FUTURES
    assert fut.expiry == date(2026, 1, 29)

    # 4. Options
    opt_ce = importer.parse_item(SAMPLE_INSTRUMENT_DATA[3])
    assert opt_ce is not None
    assert opt_ce.instrument_type == InstrumentType.OPTIONS
    assert opt_ce.option_type == OptionType.CE
    assert opt_ce.strike == 26000.0

    opt_pe = importer.parse_item(SAMPLE_INSTRUMENT_DATA[4])
    assert opt_pe is not None
    assert opt_pe.option_type == OptionType.PE


def test_importer_malformed_item_rejected() -> None:
    """Test that records missing mandatory fields are rejected."""
    importer = AngelOneInstrumentImporter()
    assert importer.parse_item({"token": "", "exch_seg": "NSE"}) is None
    assert importer.parse_item({"token": "123", "exch_seg": ""}) is None


def test_repository_upsert_and_idempotency(test_session: Session) -> None:
    """Test bulk upsert and verify no duplicates on repeated import."""
    importer = AngelOneInstrumentImporter()
    repo = InstrumentRepository()

    # First import
    stats1 = importer.import_instruments(
        source=SAMPLE_INSTRUMENT_DATA, session=test_session
    )
    assert stats1.records_inserted == 6
    assert stats1.records_updated == 0
    assert repo.count(test_session) == 6

    # Second import with identical data -> 0 new records
    stats2 = importer.import_instruments(
        source=SAMPLE_INSTRUMENT_DATA, session=test_session
    )
    assert stats2.records_inserted == 0
    assert stats2.records_unchanged == 6
    assert repo.count(test_session) == 6


def test_resolver_exact_matches(test_session: Session) -> None:
    """Test resolving exact index and equity symbols."""
    importer = AngelOneInstrumentImporter()
    importer.import_instruments(source=SAMPLE_INSTRUMENT_DATA, session=test_session)

    resolver = InstrumentResolver()

    # Resolve NIFTY index
    nifty = resolver.resolve("NIFTY", exchange="NSE", session=test_session)
    assert nifty.symbol_token == "TEST_IDX_001"
    assert nifty.trading_symbol == "Nifty 50"

    # Resolve SBIN equity
    sbin = resolver.resolve("SBIN", exchange="NSE", session=test_session)
    assert sbin.symbol_token == "TEST_EQ_001"
    assert sbin.instrument_type == InstrumentType.EQUITY


def test_resolver_structured_derivatives(test_session: Session) -> None:
    """Test resolving structured derivative contracts."""
    importer = AngelOneInstrumentImporter()
    importer.import_instruments(source=SAMPLE_INSTRUMENT_DATA, session=test_session)

    resolver = InstrumentResolver()

    # Resolve NIFTY CE option
    query_ce = InstrumentQuery(
        symbol="NIFTY",
        exchange="NFO",
        instrument_type=InstrumentType.OPTIONS,
        expiry=date(2026, 1, 29),
        strike=26000.0,
        option_type=OptionType.CE,
    )
    opt = resolver.resolve(query_ce, session=test_session)
    assert opt.symbol_token == "TEST_OPT_CE_001"


def test_resolver_not_found(test_session: Session) -> None:
    """Test InstrumentNotFoundError is raised when no instrument matches."""
    resolver = InstrumentResolver()
    with pytest.raises(InstrumentNotFoundError, match="No instrument found"):
        resolver.resolve("NON_EXISTENT_TICKER", exchange="NSE", session=test_session)


def test_resolver_ambiguous_matches(test_session: Session) -> None:
    """Test InstrumentResolutionError is raised on ambiguous derivative queries."""
    importer = AngelOneInstrumentImporter()
    importer.import_instruments(source=SAMPLE_INSTRUMENT_DATA, session=test_session)

    resolver = InstrumentResolver()
    query_ambiguous = InstrumentQuery(symbol="NIFTY", exchange="NFO")

    with pytest.raises(InstrumentResolutionError, match="Ambiguous query"):
        resolver.resolve(query_ambiguous, session=test_session)


def test_historical_service_with_resolver_integration() -> None:
    """Test HistoricalService resolves symbol token automatically when omitted."""
    mock_resolver = MagicMock(spec=InstrumentResolver)
    mock_resolved_inst = Instrument(
        symbol_token="TEST_RESOLVED_TOKEN",
        exchange="NSE",
        trading_symbol="Nifty 50",
        symbol="NIFTY",
        instrument_type=InstrumentType.INDEX,
    )
    mock_resolver.resolve.return_value = mock_resolved_inst

    mock_client = MagicMock()
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
            ]
        ],
    }

    service = AngelOneHistoricalService(
        client=mock_client,
        resolver=mock_resolver,
    )

    # Request without explicit symbol_token
    request = HistoricalDataRequest(
        exchange="NSE",
        symbol="NIFTY",
        interval=AngelOneInterval.FIVE_MINUTE,
        from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=INDIA_TZ),
        to_datetime=datetime(2026, 1, 2, 9, 20, tzinfo=INDIA_TZ),
    )

    result = service.get_candles(request)
    mock_resolver.resolve.assert_called_once_with(query="NIFTY", exchange="NSE")
    assert result.symbol_token == "TEST_RESOLVED_TOKEN"
    assert result.candle_count == 1


def test_cli_refresh_and_resolve(test_session: Session) -> None:
    """Test CLI commands for refresh and resolve."""
    from collections.abc import Generator
    from contextlib import contextmanager

    @contextmanager
    def mock_session_ctx() -> Generator[Session, None, None]:
        yield test_session

    with (
        patch(
            "adaptive_trading.instruments.importer.get_db_session",
            mock_session_ctx,
        ),
        patch(
            "adaptive_trading.instruments.resolver.get_db_session",
            mock_session_ctx,
        ),
        patch(
            "adaptive_trading.instruments.importer.AngelOneInstrumentImporter.fetch_raw_data",
            return_value=SAMPLE_INSTRUMENT_DATA,
        ),
    ):
        # 1. Test refresh command
        exit_code_ref = run_refresh(source=None)
        assert exit_code_ref == 0

        # 2. Test resolve command
        exit_code_res = run_resolve(
            symbol="NIFTY",
            exchange="NSE",
            inst_type_str="INDEX",
            expiry_str=None,
            strike=None,
            opt_type_str=None,
        )
        assert exit_code_res == 0
