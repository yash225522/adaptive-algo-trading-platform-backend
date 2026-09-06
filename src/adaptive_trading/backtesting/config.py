"""Configuration parameters for historical backtesting simulations."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.backtesting.exceptions import BacktestConfigError


class BacktestConfig(BaseModel):
    """Configuration governing capital, costs, sizing, and execution convention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_capital: float = Field(
        default=100_000.0,
        gt=0.0,
        description="Starting cash capital for the simulation",
    )
    commission_bps: float = Field(
        default=3.0,
        ge=0.0,
        description="Commission rate in basis points (1 bp = 0.01% = 0.0001)",
    )
    slippage_bps: float = Field(
        default=5.0,
        ge=0.0,
        description="Slippage rate in basis points (1 bp = 0.01% = 0.0001)",
    )
    fixed_quantity: float = Field(
        default=1.0,
        gt=0.0,
        description="Fixed traded quantity for each signal execution",
    )
    execution_timing: str = Field(
        default="NEXT_OPEN",
        description="Execution price timing convention ('NEXT_OPEN')",
    )

    @field_validator("execution_timing")
    @classmethod
    def validate_execution_timing(cls, v: str) -> str:
        """Validate execution timing convention."""
        allowed = {"NEXT_OPEN"}
        if v.upper() not in allowed:
            raise BacktestConfigError(
                f"Unsupported execution_timing '{v}'. Allowed: {allowed}"
            )
        return v.upper()
