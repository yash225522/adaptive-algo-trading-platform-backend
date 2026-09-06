"""Configuration models and validation rules for the risk management engine."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.risk.exceptions import RiskConfigError


class RiskConfig(BaseModel):
    """Configuration defining risk thresholds, position sizing, and rule toggles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_position_value: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum allowed currency value for a single position",
    )
    max_position_pct: float | None = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="Maximum position value as fraction of portfolio equity",
    )
    max_portfolio_exposure_pct: float | None = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gross portfolio exposure as multiple of equity",
    )
    max_daily_loss_pct: float | None = Field(
        default=0.02,
        gt=0.0,
        le=1.0,
        description="Maximum allowed daily loss as fraction of start equity",
    )
    max_drawdown_pct: float | None = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="Maximum allowed drawdown from peak equity",
    )
    max_open_positions: int = Field(
        default=5,
        ge=1,
        description="Maximum number of concurrently open non-flat positions",
    )
    fixed_quantity: float = Field(
        default=1.0,
        gt=0.0,
        description="Default baseline quantity determined by position sizer",
    )
    allow_quantity_reduction: bool = Field(
        default=True,
        description="If True, cap position size to maximum allowed",
    )
    enable_daily_loss_check: bool = Field(
        default=True,
        description="Toggle daily loss threshold check",
    )
    enable_drawdown_check: bool = Field(
        default=True,
        description="Toggle maximum drawdown threshold check",
    )
    enable_open_position_check: bool = Field(
        default=True,
        description="Toggle maximum open positions count check",
    )
    enable_position_limit_check: bool = Field(
        default=True,
        description="Toggle individual position size/value check",
    )
    enable_exposure_limit_check: bool = Field(
        default=True,
        description="Toggle total gross portfolio exposure check",
    )

    @field_validator("max_open_positions")
    @classmethod
    def validate_open_positions(cls, v: int) -> int:
        if v < 1:
            raise RiskConfigError("max_open_positions must be at least 1")
        return v

    @field_validator("fixed_quantity")
    @classmethod
    def validate_fixed_quantity(cls, v: float) -> float:
        if v <= 0:
            raise RiskConfigError("fixed_quantity must be positive")
        return v
