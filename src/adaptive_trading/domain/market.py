"""Market data domain models."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adaptive_trading.common.config import TimeFrame


class Candle(BaseModel):
    """Represents a single validated OHLCV market candle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(
        description="Timezone-aware timestamp for the start of the candle interval"
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    timeframe: TimeFrame = Field(description="Candle time interval")
    open: float = Field(gt=0, description="Opening price")
    high: float = Field(gt=0, description="Highest price during interval")
    low: float = Field(gt=0, description="Lowest price during interval")
    close: float = Field(gt=0, description="Closing price")
    volume: float = Field(ge=0, description="Trading volume during interval")
    open_interest: float | None = Field(
        default=None, ge=0, description="Optional open interest"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Candle timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> Self:
        """Validate consistency across OHLC price values."""
        if self.high < self.open:
            raise ValueError(
                f"High price ({self.high}) cannot be lower than Open ({self.open})"
            )
        if self.high < self.close:
            raise ValueError(
                f"High price ({self.high}) cannot be lower than Close ({self.close})"
            )
        if self.high < self.low:
            raise ValueError(
                f"High price ({self.high}) cannot be lower than Low ({self.low})"
            )
        if self.low > self.open:
            raise ValueError(
                f"Low price ({self.low}) cannot be higher than Open ({self.open})"
            )
        if self.low > self.close:
            raise ValueError(
                f"Low price ({self.low}) cannot be higher than Close ({self.close})"
            )
        return self
