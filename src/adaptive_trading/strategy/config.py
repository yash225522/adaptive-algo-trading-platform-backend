"""Configuration parameters and threshold validators for trading strategies."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_trading.strategy.exceptions import StrategyConfigError


class StrategyConfig(BaseModel):
    """Configuration governing probability thresholds and strategy metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_name: str = Field(
        default="probability_threshold",
        min_length=1,
        description="Name identifier for the strategy",
    )
    strategy_version: str = Field(
        default="v1",
        min_length=1,
        description="Version identifier for the strategy rules",
    )
    long_probability_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum probability_up required to trigger a LONG signal",
    )
    short_probability_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Maximum probability_up required to trigger a SHORT signal",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "StrategyConfig":
        """Enforce strict invariant: 0.0 <= short_threshold < long_threshold <= 1.0."""
        if self.short_probability_threshold >= self.long_probability_threshold:
            raise StrategyConfigError(
                f"Invalid strategy threshold configuration: "
                f"short_probability_threshold ({self.short_probability_threshold}) "
                f"must be strictly less than long_probability_threshold "
                f"({self.long_probability_threshold})"
            )
        return self
