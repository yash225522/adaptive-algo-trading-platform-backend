"""Configuration models for walk-forward validation experiments."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WindowType(StrEnum):
    """Walk-forward window evolution strategy."""

    EXPANDING = "EXPANDING"
    ROLLING = "ROLLING"


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward cross-validation splitting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_train_size: float = Field(
        default=0.50,
        gt=0.0,
        description=(
            "Initial training window size as fraction of dataset (if < 1.0) "
            "or sample count (if >= 1.0)"
        ),
    )
    validation_size: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "Validation window size as fraction of dataset (if < 1.0) "
            "or sample count (if >= 1.0)"
        ),
    )
    step_size: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Step increment between folds (defaults to validation_size if None)"
        ),
    )
    gap: int = Field(
        default=0,
        ge=0,
        description=("Temporal purge gap between train end and validation start"),
    )
    window_type: WindowType = Field(
        default=WindowType.EXPANDING,
        description="Whether training window expands across time or rolls",
    )
    min_train_samples: int = Field(
        default=20,
        gt=0,
        description="Minimum number of samples required in any training fold",
    )

    @model_validator(mode="after")
    def validate_sizes(self) -> "WalkForwardConfig":
        """Ensure initial train and validation sizes are logically consistent."""
        if self.initial_train_size < 1.0 and self.validation_size < 1.0:
            if self.initial_train_size + self.validation_size > 1.0:
                raise ValueError(
                    f"initial_train_size ({self.initial_train_size}) + "
                    f"validation_size ({self.validation_size}) cannot exceed 1.0"
                )
        return self
