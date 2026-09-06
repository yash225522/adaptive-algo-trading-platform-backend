"""Runtime configuration governing replay modes, safety policies, and warm-up."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.execution.exceptions import LiveTradingNotEnabledError
from adaptive_trading.runtime.exceptions import RuntimeConfigError


class RuntimeMode(StrEnum):
    """Execution modes supported by the runtime orchestrator."""

    REPLAY = "REPLAY"
    PAPER = "PAPER"
    LIVE = "LIVE"


class RuntimeConfig(BaseModel):
    """Configuration parameters for the paper trading event loop and replay runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: RuntimeMode = Field(
        default=RuntimeMode.REPLAY,
        description="Operating mode (REPLAY, PAPER, LIVE)",
    )
    fail_fast: bool = Field(
        default=True,
        description="If True, abort runtime immediately on unexpected errors",
    )
    deduplicate_events: bool = Field(
        default=True,
        description="If True, drop duplicate (symbol, timestamp) market events",
    )
    sort_events: bool = Field(
        default=True,
        description="If True, sort incoming events chronologically before processing",
    )
    checkpoint_enabled: bool = Field(
        default=True,
        description="If True, persist periodic checkpoints of runtime state",
    )
    warmup_period: int = Field(
        default=20,
        ge=1,
        description="Number of initial candles required before generating features",
    )
    initial_cash: float = Field(
        default=100_000.0,
        gt=0.0,
        description="Starting simulation cash balance",
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
        description="Target market exchange",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v: str | RuntimeMode) -> RuntimeMode:
        """Validate runtime mode and enforce safety constraints."""
        if isinstance(v, str):
            try:
                v = RuntimeMode(v.upper())
            except ValueError as exc:
                raise RuntimeConfigError(
                    f"Unsupported runtime mode '{v}'. Must be REPLAY, PAPER, or LIVE."
                ) from exc

        if v == RuntimeMode.LIVE:
            raise LiveTradingNotEnabledError(
                "LIVE trading mode is strictly disabled in Step 18. "
                "Only REPLAY and PAPER modes are permitted."
            )
        return v
