"""Configuration definitions for broker execution and paper trading."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.execution.exceptions import ExecutionModeError


class ExecutionMode(StrEnum):
    """Supported execution environments."""

    PAPER = "PAPER"
    LIVE = "LIVE"


class ExecutionConfig(BaseModel):
    """Configuration governing broker selection, fees, and initial capital."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ExecutionMode = Field(
        default=ExecutionMode.PAPER,
        description="Active execution environment mode (PAPER/LIVE)",
    )
    initial_cash: float = Field(
        default=100_000.0,
        gt=0.0,
        description="Starting cash balance for paper trading account",
    )
    commission_bps: float = Field(
        default=3.0,
        ge=0.0,
        description="Commission rate in basis points (1 bp = 0.01%)",
    )
    slippage_bps: float = Field(
        default=5.0,
        ge=0.0,
        description="Simulated slippage rate in basis points",
    )
    default_exchange: str = Field(
        default="NSE",
        min_length=1,
        description="Default target market exchange",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v: str | ExecutionMode) -> ExecutionMode:
        """Ensure execution mode is uppercase valid enum value."""
        if isinstance(v, str):
            try:
                return ExecutionMode(v.upper())
            except ValueError as exc:
                raise ExecutionModeError(
                    f"Unsupported execution mode '{v}'. Must be PAPER or LIVE."
                ) from exc
        return v
