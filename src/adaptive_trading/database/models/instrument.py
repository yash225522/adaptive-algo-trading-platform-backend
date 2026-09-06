"""Instrument master database models."""

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from adaptive_trading.database.base import Base


class InstrumentModel(Base):
    """Database persistence model for Angel One instruments."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol_token: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(16), nullable=False)
    trading_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    instrument_type: Mapped[str] = mapped_column(String(16), nullable=False)
    expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
    strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    option_type: Mapped[str | None] = mapped_column(String(8), nullable=True)
    lot_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    tick_size: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "symbol_token",
            name="uq_instruments_exchange_token",
        ),
        Index("ix_instruments_exch_sym", "exchange", "symbol"),
        Index("ix_instruments_exch_tsym", "exchange", "trading_symbol"),
        Index("ix_instruments_lookup", "exchange", "symbol", "instrument_type"),
    )
