"""Machine learning target, label, and training example domain models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DirectionLabel(StrEnum):
    """Classification label for forward price movement direction."""

    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class Target(BaseModel):
    """Supervised learning target calculated from future market outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(
        description="Timezone-aware timestamp at prediction time T"
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    horizon: int = Field(gt=0, description="Forward prediction horizon in candle bars")
    future_return: float = Field(
        description="Forward fractional return over the horizon"
    )
    classification_label: DirectionLabel = Field(
        description="Direction classification label (UP, DOWN, NEUTRAL)"
    )
    target_version: str = Field(
        min_length=1, description="Identifier for the target definition version"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Target timestamp must be timezone-aware")
        return v


class TrainingExample(BaseModel):
    """Supervised training example aligning a FeatureVector with its future Target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(description="Timezone-aware prediction timestamp")
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    features: dict[str, float] = Field(
        description="Input feature dictionary calculated using data <= T"
    )
    feature_version: str = Field(description="Feature set version identifier")
    target: Target = Field(
        description="Future market target outcome for training supervision"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure the timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("TrainingExample timestamp must be timezone-aware")
        return v
