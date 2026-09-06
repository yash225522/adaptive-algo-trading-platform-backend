"""Machine learning feature vectors and prediction data contracts."""

import math
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FeatureVector(BaseModel):
    """Structured collection of numerical ML features calculated at a timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(
        description="Timezone-aware timestamp when features were calculated"
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    features: dict[str, float] = Field(
        description="Mapping of feature names to numerical values"
    )
    feature_version: str = Field(
        min_length=1, description="Identifier for the feature set version"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("FeatureVector timestamp must be timezone-aware")
        return v


class Prediction(BaseModel):
    """Output contract for ML model inference predictions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(
        description="Timezone-aware timestamp when prediction was generated"
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    model_version: str = Field(
        min_length=1,
        description="Identifier of the model generating the prediction",
    )
    probability_up: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability of upward movement in [0.0, 1.0]",
    )
    probability_down: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability of downward movement in [0.0, 1.0]",
    )
    expected_return: float = Field(
        description="Predicted expected return for the target horizon"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Prediction timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def validate_probability_sum(self) -> Self:
        """Validate that probabilities sum to approximately 1.0."""
        total_prob = self.probability_up + self.probability_down
        if not math.isclose(total_prob, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"Probabilities must sum to ~1.0 (got up={self.probability_up}, "
                f"down={self.probability_down}, sum={total_prob})"
            )
        return self
