"""Unit and integration tests for PostgreSQL database layer, models, and migrations."""

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import Engine, StaticPool, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from adaptive_trading.database.base import Base
from adaptive_trading.database.models import (
    MarketCandleModel,
    ModelMetadataModel,
    SystemRunModel,
)
from adaptive_trading.database.session import (
    create_db_engine,
    get_db_session,
    get_session_factory,
)
from alembic import command

UTC_TZ = timezone.utc


@pytest.fixture
def test_engine() -> Generator[Engine, None, None]:
    """Provide an in-memory SQLite engine with tables created for testing."""
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
def test_session(test_engine: Engine) -> Generator[Session, None, None]:
    """Provide a clean session bound to the test engine."""
    session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)
    session = session_factory()
    yield session
    session.close()


# ============================================================================
# 1. Engine & Session Management Tests
# ============================================================================


def test_engine_creation_from_settings() -> None:
    """Test creating a database engine using default application settings."""
    engine = create_db_engine()
    assert engine is not None
    assert "postgresql+psycopg://" in str(engine.url)


def test_session_context_manager_commit(test_engine: Engine) -> None:
    """Test that get_db_session commits on successful exit."""
    candle_ts = datetime(2026, 8, 30, 9, 15, tzinfo=UTC_TZ)

    with get_db_session(engine=test_engine) as session:
        candle = MarketCandleModel(
            timestamp=candle_ts,
            symbol="NIFTY",
            timeframe="5m",
            open=24000.0,
            high=24050.0,
            low=23980.0,
            close=24020.0,
            volume=5000.0,
        )
        session.add(candle)

    # Verify persisted in a new session
    factory = get_session_factory(engine=test_engine)
    with factory() as verify_session:
        saved = verify_session.scalar(
            select(MarketCandleModel).where(
                MarketCandleModel.symbol == "NIFTY",
                MarketCandleModel.timeframe == "5m",
            )
        )
        assert saved is not None
        assert saved.open == 24000.0


def test_session_context_manager_rollback(test_engine: Engine) -> None:
    """Test that get_db_session rolls back uncommitted changes on exception."""
    candle_ts = datetime(2026, 8, 30, 9, 20, tzinfo=UTC_TZ)

    with pytest.raises(RuntimeError, match="Simulated error"):
        with get_db_session(engine=test_engine) as session:
            candle = MarketCandleModel(
                timestamp=candle_ts,
                symbol="BANKNIFTY",
                timeframe="5m",
                open=51000.0,
                high=51100.0,
                low=50950.0,
                close=51050.0,
                volume=3000.0,
            )
            session.add(candle)
            raise RuntimeError("Simulated error")

    # Verify row was NOT persisted
    factory = get_session_factory(engine=test_engine)
    with factory() as verify_session:
        saved = verify_session.scalar(
            select(MarketCandleModel).where(MarketCandleModel.symbol == "BANKNIFTY")
        )
        assert saved is None


# ============================================================================
# 2. Market Candle Model Tests
# ============================================================================


def test_market_candle_insert_and_retrieve(test_session: Session) -> None:
    """Test inserting and retrieving a valid market candle."""
    candle_ts = datetime(2026, 8, 30, 9, 30, tzinfo=UTC_TZ)
    candle = MarketCandleModel(
        timestamp=candle_ts,
        symbol="NIFTY",
        timeframe="5m",
        open=24100.0,
        high=24150.0,
        low=24080.0,
        close=24120.0,
        volume=12500.0,
        open_interest=45000.0,
    )
    test_session.add(candle)
    test_session.commit()

    retrieved = test_session.scalar(
        select(MarketCandleModel).where(MarketCandleModel.id == candle.id)
    )
    assert retrieved is not None
    assert retrieved.symbol == "NIFTY"
    assert retrieved.timeframe == "5m"
    assert retrieved.open == 24100.0
    assert retrieved.open_interest == 45000.0
    assert retrieved.created_at is not None


def test_duplicate_market_candle_rejected(test_session: Session) -> None:
    """Test that inserting duplicate (ts, symbol, timeframe) is rejected."""
    candle_ts = datetime(2026, 8, 30, 9, 35, tzinfo=UTC_TZ)
    candle1 = MarketCandleModel(
        timestamp=candle_ts,
        symbol="NIFTY",
        timeframe="5m",
        open=24100.0,
        high=24150.0,
        low=24080.0,
        close=24120.0,
        volume=1000.0,
    )
    candle2 = MarketCandleModel(
        timestamp=candle_ts,
        symbol="NIFTY",
        timeframe="5m",
        open=24110.0,
        high=24160.0,
        low=24090.0,
        close=24130.0,
        volume=2000.0,
    )
    test_session.add(candle1)
    test_session.commit()

    test_session.add(candle2)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()


# ============================================================================
# 3. Model Metadata & System Runs Tests
# ============================================================================


def test_models_metadata_insert_and_uniqueness(test_session: Session) -> None:
    """Test model metadata table insertion and model_version uniqueness."""
    m1 = ModelMetadataModel(
        model_version="lgbm_nifty_v1.0",
        model_type="LightGBMClassifier",
        status="active",
    )
    test_session.add(m1)
    test_session.commit()

    m2 = ModelMetadataModel(
        model_version="lgbm_nifty_v1.0",  # Duplicate version
        model_type="LightGBMClassifier",
        status="inactive",
    )
    test_session.add(m2)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()


def test_system_runs_lifecycle(test_session: Session) -> None:
    """Test tracking application run lifecycle in runs table."""
    start_time = datetime(2026, 8, 30, 10, 0, tzinfo=UTC_TZ)
    run = SystemRunModel(
        run_type="historical_ingestion",
        status="running",
        started_at=start_time,
    )
    test_session.add(run)
    test_session.commit()

    assert run.id is not None
    assert run.completed_at is None

    # Complete the run
    run.status = "completed"
    run.completed_at = datetime(2026, 8, 30, 10, 5, tzinfo=UTC_TZ)
    test_session.commit()

    updated = test_session.scalar(
        select(SystemRunModel).where(SystemRunModel.id == run.id)
    )
    assert updated is not None
    assert updated.status == "completed"
    assert updated.completed_at is not None


# ============================================================================
# 4. Alembic Migration Execution Tests
# ============================================================================


def test_alembic_migration_upgrade_and_downgrade(tmp_path: Path) -> None:
    """Test that Alembic migration 0001 applies and rolls back successfully."""
    test_db_path = tmp_path / "test_migration.db"
    db_url = f"sqlite:///{test_db_path.as_posix()}"

    alembic_cfg = Config(str(Path("alembic.ini").resolve()))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Upgrade to head
    command.upgrade(alembic_cfg, "head")

    # Verify tables created
    engine = create_engine(db_url)
    with engine.connect() as conn:
        tables = engine.dialect.get_table_names(conn)
        assert "market_candles" in tables
        assert "models" in tables
        assert "runs" in tables
        assert "instruments" in tables
        assert "alembic_version" in tables

    # Downgrade to base
    command.downgrade(alembic_cfg, "base")

    with engine.connect() as conn:
        tables_after_downgrade = engine.dialect.get_table_names(conn)
        assert "market_candles" not in tables_after_downgrade
        assert "models" not in tables_after_downgrade
        assert "runs" not in tables_after_downgrade
        assert "instruments" not in tables_after_downgrade

    engine.dispose()
