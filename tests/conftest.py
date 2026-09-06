"""Global pytest fixtures for unit and integration test suites."""

from collections.abc import Generator

import pytest
from sqlalchemy import Engine, StaticPool, create_engine
from sqlalchemy.orm import Session, sessionmaker

from adaptive_trading.database.base import Base


@pytest.fixture
def test_engine() -> Generator[Engine, None, None]:
    """Provide an in-memory SQLite engine with all domain tables created."""
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
    """Provide a clean SQLAlchemy session bound to the test engine."""
    session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)
    session = session_factory()
    yield session
    session.close()
