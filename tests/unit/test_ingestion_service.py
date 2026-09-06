"""Unit and integration tests for MarketDataIngestionService and CLI."""

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, StaticPool, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from adaptive_trading.data.cli import main as cli_main
from adaptive_trading.data.services.ingestion import (
    MarketDataIngestionService,
)
from adaptive_trading.database.base import Base
from adaptive_trading.database.models import MarketCandleModel


@pytest.fixture
def db_engine() -> Generator[Engine, None, None]:
    """Create an isolated in-memory SQLite engine for ingestion testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    """Create a database session bound to the test engine."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = factory()
    yield session
    session.close()


def test_end_to_end_ingestion(db_session: Session) -> None:
    """Test full pipeline ingestion of the sample dataset."""
    sample_file = Path("data/sample/nifty_5m_sample.csv")
    assert sample_file.is_file()

    service = MarketDataIngestionService()
    stats = service.ingest_csv_file(sample_file, session=db_session)

    # 9 rows in sample: 7 unique valid + 1 batch duplicate + 1 invalid
    assert stats.rows_read == 9
    assert stats.rows_valid == 8
    assert stats.rows_inserted == 7
    assert stats.rows_skipped == 1
    assert stats.rows_rejected == 1
    assert len(stats.errors) == 1
    assert stats.errors[0].row_number == 10
    assert stats.errors[0].field == "close"
    assert stats.warnings == 1
    assert stats.quality_report is not None
    assert stats.quality_report.duplicate_count == 1

    # Verify rows in database
    db_count = db_session.scalar(select(func.count(MarketCandleModel.id)))
    assert db_count == 7


def test_idempotent_ingestion(db_session: Session) -> None:
    """Test that re-ingesting the exact same file skips all existing candles."""
    sample_file = Path("data/sample/nifty_5m_sample.csv")
    service = MarketDataIngestionService()

    # First ingestion run
    stats_1 = service.ingest_csv_file(sample_file, session=db_session)
    assert stats_1.rows_inserted == 7

    count_after_run_1 = db_session.scalar(select(func.count(MarketCandleModel.id)))

    # Second ingestion run on same database
    stats_2 = service.ingest_csv_file(sample_file, session=db_session)
    assert stats_2.rows_inserted == 0
    assert stats_2.rows_skipped == 8  # 1 batch duplicate + 7 existing rows

    count_after_run_2 = db_session.scalar(select(func.count(MarketCandleModel.id)))

    assert count_after_run_1 == count_after_run_2 == 7


def test_cli_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running ingestion via CLI entrypoint."""
    csv_file = tmp_path / "cli_test.csv"
    csv_file.write_text(
        "timestamp,symbol,timeframe,open,high,low,close,volume\n"
        "2026-01-02T09:15:00+05:30,NIFTY,5m,26000.0,26050.0,25980.0,26030.0,1000\n",
        encoding="utf-8",
    )

    test_args = ["adaptive_trading.data.cli", "--file", str(csv_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch.object(MarketDataIngestionService, "ingest_csv_file") as mock_ingest:
        from adaptive_trading.data.models import IngestionStats

        mock_ingest.return_value = IngestionStats(
            source_file=str(csv_file),
            rows_read=1,
            rows_valid=1,
            rows_inserted=1,
        )
        # Should execute and print summary cleanly without exiting non-zero
        cli_main()
        assert mock_ingest.called
