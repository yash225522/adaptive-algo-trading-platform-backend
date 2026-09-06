"""Database engine and session management."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from adaptive_trading.common.config import get_settings


def create_db_engine(database_url: str | None = None, echo: bool = False) -> Engine:
    """Create a SQLAlchemy Engine using application configuration."""
    url = database_url or get_settings().database_url
    connect_args = {}
    if "postgres" in url:
        connect_args["connect_timeout"] = 2
    return create_engine(url, echo=echo, pool_pre_ping=True, connect_args=connect_args)


def get_session_factory(
    engine: Engine | None = None,
) -> sessionmaker[Session]:
    """Create a sessionmaker factory bound to an engine."""
    eng = engine or create_db_engine()
    return sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)


@contextmanager
def get_db_session(
    engine: Engine | None = None,
) -> Generator[Session, None, None]:
    """Context manager providing a transactional database session."""
    factory = get_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
