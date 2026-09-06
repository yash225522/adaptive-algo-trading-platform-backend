"""Configuration specifications for ML dataset splitting, training, and modeling."""

from pydantic import BaseModel, ConfigDict, Field


class MLConfig(BaseModel):
    """Governs train/test splitting, preprocessing, and model hyperparameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Time-based splitting
    train_ratio: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Fraction of chronological observations used for training",
    )

    C: float = Field(
        default=1.0,
        gt=0.0,
        description="Inverse regularization strength parameter",
    )
    max_iter: int = Field(
        default=1000,
        gt=0,
        description="Maximum solver iterations",
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    solver: str = Field(
        default="lbfgs",
        description="Algorithm to use in the optimization problem",
    )

    # Label encoding mapping
    positive_class: str = Field(
        default="UP",
        description="Target class label treated as binary 1",
    )
    negative_class: str = Field(
        default="DOWN",
        description="Target class label treated as binary 0",
    )
