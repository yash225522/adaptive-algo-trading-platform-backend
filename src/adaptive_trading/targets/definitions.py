"""Configuration and versioning definitions for ML target generation."""

from pydantic import BaseModel, ConfigDict, Field

TARGET_VERSION = "v1"


class TargetConfig(BaseModel):
    """Configuration governing forward target calculation and classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(
        default=3,
        gt=0,
        description="Forward horizon in candle intervals (default: 3 bars)",
    )
    positive_threshold: float = Field(
        default=0.0,
        description="Minimum fractional return threshold to classify as UP",
    )
    negative_threshold: float | None = Field(
        default=None,
        description="Maximum fractional return threshold for DOWN (if 3-class)",
    )
    use_neutral_class: bool = Field(
        default=False,
        description="Whether to generate 3-class or binary labels",
    )
    target_version: str = Field(
        default=TARGET_VERSION,
        description="Target definition version identifier",
    )
