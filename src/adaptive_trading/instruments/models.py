"""Domain models and data contracts for instruments and resolution queries."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class InstrumentType(StrEnum):
    """Broad asset and instrument classifications."""

    EQUITY = "EQUITY"
    INDEX = "INDEX"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    OTHER = "OTHER"


class OptionType(StrEnum):
    """Option contract style (Call vs Put)."""

    CE = "CE"
    PE = "PE"


class Instrument(BaseModel):
    """Standardized immutable instrument domain representation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol_token: str = Field(
        min_length=1, description="Angel One instrument identifier token"
    )
    exchange: str = Field(
        min_length=1, description="Exchange segment (e.g. NSE, BSE, NFO)"
    )
    trading_symbol: str = Field(
        min_length=1, description="Exchange trading symbol (e.g. NIFTY26JANFUT)"
    )
    symbol: str = Field(
        min_length=1, description="Normalized underlying ticker symbol (e.g. NIFTY)"
    )
    name: str = Field(
        default="", description="Full descriptive security company/index name"
    )
    instrument_type: InstrumentType = Field(
        description="Instrument category (EQUITY, INDEX, FUTURES, OPTIONS)"
    )
    expiry: date | None = Field(
        default=None, description="Contract expiration date (for derivatives)"
    )
    strike: float | None = Field(
        default=None, description="Option strike price (if applicable)"
    )
    option_type: OptionType | None = Field(
        default=None, description="Option type (CE or PE, if applicable)"
    )
    lot_size: int = Field(default=1, gt=0, description="Minimum trading lot quantity")
    tick_size: float = Field(
        default=0.05, gt=0, description="Minimum price movement increment"
    )


class InstrumentQuery(BaseModel):
    """Specification for querying and resolving an instrument."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Underlying or ticker symbol")
    exchange: str = Field(default="NSE", min_length=1, description="Exchange segment")
    instrument_type: InstrumentType | None = Field(
        default=None, description="Optional instrument category filter"
    )
    expiry: date | None = Field(
        default=None, description="Optional expiration date filter"
    )
    strike: float | None = Field(
        default=None, description="Optional strike price filter"
    )
    option_type: OptionType | None = Field(
        default=None, description="Optional option type filter (CE/PE)"
    )


class ImportStats(BaseModel):
    """Summary statistics for instrument master synchronization."""

    records_read: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_unchanged: int = 0
    records_rejected: int = 0
