"""Domain and request/response data contracts for Angel One historical data."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.models import DataQualityReport
from adaptive_trading.domain.market import Candle


class AngelOneInterval(StrEnum):
    """Supported historical candle intervals in Angel One SmartAPI."""

    ONE_MINUTE = "ONE_MINUTE"
    THREE_MINUTE = "THREE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    TEN_MINUTE = "TEN_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    ONE_DAY = "ONE_DAY"


# Bidirectional mapping between domain TimeFrame and SmartAPI interval
TIMEFRAME_TO_ANGELONE: dict[TimeFrame, AngelOneInterval] = {
    TimeFrame.ONE_MINUTE: AngelOneInterval.ONE_MINUTE,
    TimeFrame.THREE_MINUTES: AngelOneInterval.THREE_MINUTE,
    TimeFrame.FIVE_MINUTES: AngelOneInterval.FIVE_MINUTE,
    TimeFrame.FIFTEEN_MINUTES: AngelOneInterval.FIFTEEN_MINUTE,
    TimeFrame.THIRTY_MINUTES: AngelOneInterval.THIRTY_MINUTE,
    TimeFrame.ONE_HOUR: AngelOneInterval.ONE_HOUR,
    TimeFrame.ONE_DAY: AngelOneInterval.ONE_DAY,
}

ANGELONE_TO_TIMEFRAME: dict[AngelOneInterval, TimeFrame] = {
    AngelOneInterval.ONE_MINUTE: TimeFrame.ONE_MINUTE,
    AngelOneInterval.THREE_MINUTE: TimeFrame.THREE_MINUTES,
    AngelOneInterval.FIVE_MINUTE: TimeFrame.FIVE_MINUTES,
    AngelOneInterval.FIFTEEN_MINUTE: TimeFrame.FIFTEEN_MINUTES,
    AngelOneInterval.THIRTY_MINUTE: TimeFrame.THIRTY_MINUTES,
    AngelOneInterval.ONE_HOUR: TimeFrame.ONE_HOUR,
    AngelOneInterval.ONE_DAY: TimeFrame.ONE_DAY,
}


class HistoricalDataRequest(BaseModel):
    """Request parameters for downloading historical candles from SmartAPI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str = Field(
        default="NSE",
        min_length=1,
        description="Exchange segment (e.g. NSE, BSE, NFO)",
    )
    symbol: str = Field(
        min_length=1,
        description="Human-readable ticker symbol (e.g. NIFTY, SBIN)",
    )
    symbol_token: str | None = Field(
        default=None,
        description="Optional symbol token (resolved automatically if omitted)",
    )
    interval: AngelOneInterval = Field(
        description="SmartAPI candle aggregation interval"
    )
    from_datetime: datetime = Field(
        description="Timezone-aware start timestamp of historical period"
    )
    to_datetime: datetime = Field(
        description="Timezone-aware end timestamp of historical period"
    )

    @model_validator(mode="after")
    def validate_request_bounds(self) -> Self:
        """Ensure timestamps are timezone-aware and chronological."""
        if (
            self.from_datetime.tzinfo is None
            or self.from_datetime.tzinfo.utcoffset(self.from_datetime) is None
        ):
            raise ValueError("from_datetime must be timezone-aware")

        if (
            self.to_datetime.tzinfo is None
            or self.to_datetime.tzinfo.utcoffset(self.to_datetime) is None
        ):
            raise ValueError("to_datetime must be timezone-aware")

        if self.from_datetime > self.to_datetime:
            raise ValueError(
                f"from_datetime ({self.from_datetime}) cannot be after "
                f"to_datetime ({self.to_datetime})"
            )
        return self


class HistoricalDataResult(BaseModel):
    """Encapsulates historical candles and accompanying data quality report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    symbol_token: str
    exchange: str
    interval: AngelOneInterval
    from_datetime: datetime
    to_datetime: datetime
    candles: list[Candle]
    quality_report: DataQualityReport | None = None

    @property
    def candle_count(self) -> int:
        """Return number of valid candles returned."""
        return len(self.candles)
