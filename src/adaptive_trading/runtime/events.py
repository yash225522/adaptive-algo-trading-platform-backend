"""Event definitions, market event models, and auditable event records."""

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle


class EventType(StrEnum):
    """Enumeration of event types processed and recorded by the runtime."""

    MARKET_DATA = "MARKET_DATA"
    PREDICTION = "PREDICTION"
    SIGNAL = "SIGNAL"
    RISK_DECISION = "RISK_DECISION"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    ERROR = "ERROR"


class MarketEvent(BaseModel):
    """Standardized market data event wrapping a market candle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(description="Timezone-aware bar close timestamp")
    symbol: str = Field(min_length=1, description="Asset ticker symbol")
    timeframe: str = Field(default="5m", description="Candle interval timeframe")
    open: float = Field(gt=0, description="Bar open price")
    high: float = Field(gt=0, description="Bar high price")
    low: float = Field(gt=0, description="Bar low price")
    close: float = Field(gt=0, description="Bar close price")
    volume: float = Field(ge=0, description="Bar volume")
    open_interest: float | None = Field(
        default=None, ge=0, description="Optional open interest"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure market event timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("MarketEvent timestamp must be timezone-aware")
        return v

    @classmethod
    def from_candle(cls, candle: Candle) -> Self:
        """Create a MarketEvent instance from a domain Candle."""
        tf_str = (
            candle.timeframe.value
            if hasattr(candle.timeframe, "value")
            else str(candle.timeframe)
        )
        return cls(
            timestamp=candle.timestamp,
            symbol=candle.symbol,
            timeframe=tf_str,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            open_interest=candle.open_interest,
        )

    def to_candle(self) -> Candle:
        """Convert MarketEvent back to domain Candle model."""
        valid_tfs = [e.value for e in TimeFrame]
        tf_enum = (
            TimeFrame(self.timeframe)
            if self.timeframe in valid_tfs
            else TimeFrame.FIVE_MINUTES
        )
        return Candle(
            timestamp=self.timestamp,
            symbol=self.symbol,
            timeframe=tf_enum,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            open_interest=self.open_interest,
        )


class EventRecord(BaseModel):
    """Auditable, serializable log entry capturing a single lifecycle event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique event record identifier",
    )
    correlation_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Correlation ID linking events spawned from the same bar",
    )
    event_type: EventType = Field(description="Type classification of the event")
    timestamp: datetime = Field(description="Logical trading timestamp of the event")
    symbol: str = Field(default="", description="Relevant market symbol")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Structured event payload data"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Wall-clock creation timestamp",
    )

    @field_validator("timestamp", "created_at")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("EventRecord timestamps must be timezone-aware")
        return v
