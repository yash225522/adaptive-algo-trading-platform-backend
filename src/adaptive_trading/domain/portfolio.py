"""Portfolio position and completed trade domain models."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Position(BaseModel):
    """Represents a current open/closed position in an asset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Market ticker symbol")
    quantity: float = Field(
        description="Position size (+ for long, - for short, 0 for closed)"
    )
    average_price: float = Field(
        ge=0, description="Volume-weighted average entry price"
    )
    unrealized_pnl: float = Field(
        default=0.0, description="Unrealized profit/loss based on market price"
    )
    realized_pnl: float = Field(
        default=0.0, description="Accumulated realized profit/loss"
    )


class Trade(BaseModel):
    """Represents a completed round-trip trade record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_id: str = Field(min_length=1, description="Unique identifier for the trade")
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    entry_timestamp: datetime = Field(
        description="Timezone-aware timestamp of trade entry"
    )
    exit_timestamp: datetime = Field(
        description="Timezone-aware timestamp of trade exit"
    )
    entry_price: float = Field(gt=0, description="Average entry price")
    exit_price: float = Field(gt=0, description="Average exit price")
    quantity: float = Field(gt=0, description="Traded quantity")
    gross_pnl: float = Field(description="Gross profit/loss before fees and slippage")
    fees: float = Field(
        default=0.0, ge=0, description="Total transaction fees incurred"
    )
    slippage: float = Field(default=0.0, description="Total slippage cost incurred")
    net_pnl: float = Field(description="Net profit/loss after fees and slippage")
    model_version: str | None = Field(
        default=None, description="Optional ML model version reference"
    )
    strategy_version: str | None = Field(
        default=None, description="Optional strategy version reference"
    )

    @field_validator("entry_timestamp", "exit_timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Trade timestamps must be timezone-aware")
        return v

    @model_validator(mode="after")
    def validate_chronological_order(self) -> Self:
        """Ensure trade exit occurs after or at trade entry."""
        if self.exit_timestamp < self.entry_timestamp:
            raise ValueError(
                f"Exit timestamp ({self.exit_timestamp}) cannot precede "
                f"Entry timestamp ({self.entry_timestamp})"
            )
        return self
