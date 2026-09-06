"""Market data database models."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from adaptive_trading.database.base import Base


class MarketCandleModel(Base):
    """Database persistence model for market OHLCV candles."""

    __tablename__ = "market_candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "timestamp",
            "symbol",
            "timeframe",
            name="uq_market_candles_ts_sym_tf",
        ),
        Index("ix_market_candles_sym_tf_ts", "symbol", "timeframe", "timestamp"),
        Index("ix_market_candles_sym_ts", "symbol", "timestamp"),
    )
