"""Walk-forward window containers and sample boundary definitions."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class WalkForwardWindow(BaseModel):
    """Defines exact sample indices and timestamps for a walk-forward window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_id: str = Field(description="Unique window identifier (e.g. 'window_1')")
    window_index: int = Field(gt=0, description="1-indexed sequence number")
    train_indices: list[int] = Field(
        default_factory=list, description="Row indices for the training split"
    )
    validation_indices: list[int] = Field(
        default_factory=list,
        description="Optional row indices for the validation split",
    )
    test_indices: list[int] = Field(
        default_factory=list,
        description="Row indices for the out-of-sample test split",
    )
    train_start: datetime | None = Field(
        default=None, description="Timestamp of first training observation"
    )
    train_end: datetime | None = Field(
        default=None, description="Timestamp of last training observation"
    )
    validation_start: datetime | None = Field(
        default=None, description="Timestamp of first validation observation"
    )
    validation_end: datetime | None = Field(
        default=None, description="Timestamp of last validation observation"
    )
    test_start: datetime | None = Field(
        default=None, description="Timestamp of first test observation"
    )
    test_end: datetime | None = Field(
        default=None, description="Timestamp of last test observation"
    )
    dataset_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 cryptographic fingerprint of the parent dataset",
    )

    @property
    def train_samples(self) -> int:
        """Count of observations in training partition."""
        return len(self.train_indices)

    @property
    def validation_samples(self) -> int:
        """Count of observations in validation partition."""
        return len(self.validation_indices)

    @property
    def test_samples(self) -> int:
        """Count of observations in out-of-sample test partition."""
        return len(self.test_indices)

    @property
    def total_samples(self) -> int:
        """Total active sample count across all partitions in this window."""
        return (
            self.train_samples + self.validation_samples + self.test_samples
        )

