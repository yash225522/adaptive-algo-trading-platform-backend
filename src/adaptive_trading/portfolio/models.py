"""Authoritative portfolio and position domain models."""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PositionSide(StrEnum):
    """Side of an open market position."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Position(BaseModel):
    """Represents an active or closed position in a specific asset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Market ticker symbol")
    side: PositionSide = Field(
        default=PositionSide.FLAT, description="Position side (LONG/SHORT/FLAT)"
    )
    quantity: float = Field(default=0.0, ge=0.0, description="Position size")
    entry_price: float = Field(
        default=0.0, ge=0.0, description="Average entry fill price"
    )
    entry_timestamp: datetime | None = Field(
        default=None, description="Timezone-aware entry timestamp"
    )
    unrealized_pnl: float = Field(
        default=0.0, description="Unrealized mark-to-market P&L"
    )
    realized_pnl: float = Field(default=0.0, description="Accumulated realized P&L")
    market_value: float = Field(
        default=0.0, description="Current market value of the position"
    )


class PortfolioState(BaseModel):
    """Authoritative snapshot of complete portfolio state at a specific time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(
        description="Timezone-aware timestamp of state evaluation"
    )
    cash: float = Field(description="Available cash balance")
    equity: float = Field(description="Total portfolio equity (cash + positions)")
    realized_pnl: float = Field(description="Cumulative realized P&L")
    unrealized_pnl: float = Field(description="Total open unrealized P&L")
    portfolio_exposure: float = Field(
        description="Total gross portfolio exposure (long + short market values)"
    )
    peak_equity: float = Field(description="All-time high peak equity observed")
    daily_start_equity: float = Field(
        description="Equity value at the beginning of the current calendar day"
    )
    current_date: date | None = Field(
        default=None, description="Current calendar date for daily tracking"
    )
    positions: dict[str, Position] = Field(
        default_factory=dict, description="Active and closed symbol positions"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("PortfolioState timestamp must be timezone-aware")
        return v
