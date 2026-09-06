"""Deterministic time-series walk-forward dataset splitter."""

import logging
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.ml.validation.config import WalkForwardConfig, WindowType

logger = logging.getLogger(__name__)


class WalkForwardSplit(BaseModel):
    """Container defining sample indices and temporal boundaries for a fold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_index: int = Field(gt=0, description="1-indexed fold identifier")
    train_indices: list[int] = Field(description="Row indices for training fold")
    validation_indices: list[int] = Field(description="Row indices for validation fold")
    train_start: datetime | None = None
    train_end: datetime | None = None
    validation_start: datetime | None = None
    validation_end: datetime | None = None

    @property
    def train_samples(self) -> int:
        """Count of training observations in this fold."""
        return len(self.train_indices)

    @property
    def validation_samples(self) -> int:
        """Count of validation observations in this fold."""
        return len(self.validation_indices)


class WalkForwardSplitter:
    """Generates chronological expanding or rolling train/validation splits."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def split(self, df: pd.DataFrame) -> list[WalkForwardSplit]:
        """Generate walk-forward splits from a tabular dataset.

        Args:
            df: Input DataFrame containing historical features and targets.

        Returns:
            list[WalkForwardSplit]: List of generated cross-validation splits.

        Raises:
            ValueError: If dataset is too small or parameters produce zero splits.
        """
        n_samples = len(df)
        if n_samples < self.config.min_train_samples:
            raise ValueError(
                f"Dataset size ({n_samples}) is smaller than minimum required "
                f"train samples ({self.config.min_train_samples})"
            )

        # Resolve initial train size
        if self.config.initial_train_size < 1.0:
            init_train_len = int(n_samples * self.config.initial_train_size)
        else:
            init_train_len = int(self.config.initial_train_size)

        # Resolve validation window size
        if self.config.validation_size < 1.0:
            val_len = int(n_samples * self.config.validation_size)
        else:
            val_len = int(self.config.validation_size)

        # Resolve step size
        if self.config.step_size is None:
            step_len = val_len
        elif self.config.step_size < 1.0:
            step_len = int(n_samples * self.config.step_size)
        else:
            step_len = int(self.config.step_size)

        val_len = max(1, val_len)
        step_len = max(1, step_len)
        gap_len = self.config.gap

        if init_train_len < self.config.min_train_samples:
            raise ValueError(
                f"Initial train length ({init_train_len}) is less than "
                f"min_train_samples ({self.config.min_train_samples})"
            )

        splits: list[WalkForwardSplit] = []
        current_train_end = init_train_len
        fold_idx = 1

        has_timestamp = "timestamp" in df.columns

        while True:
            val_start = current_train_end + gap_len
            val_end = val_start + val_len

            if val_start >= n_samples or val_end > n_samples:
                break

            if self.config.window_type == WindowType.EXPANDING:
                train_start = 0
            else:
                train_start = max(0, current_train_end - init_train_len)

            train_indices = list(range(train_start, current_train_end))
            val_indices = list(range(val_start, val_end))

            # Temporal boundary extraction
            t_start: datetime | None = None
            t_end: datetime | None = None
            v_start: datetime | None = None
            v_end: datetime | None = None

            if has_timestamp:
                t_start = df["timestamp"].iloc[train_start]
                t_end = df["timestamp"].iloc[current_train_end - 1]
                v_start = df["timestamp"].iloc[val_start]
                v_end = df["timestamp"].iloc[val_end - 1]

            splits.append(
                WalkForwardSplit(
                    fold_index=fold_idx,
                    train_indices=train_indices,
                    validation_indices=val_indices,
                    train_start=t_start,
                    train_end=t_end,
                    validation_start=v_start,
                    validation_end=v_end,
                )
            )

            current_train_end += step_len
            fold_idx += 1

        if not splits:
            raise ValueError(
                f"No walk-forward splits could be created for dataset with "
                f"{n_samples} samples. Try reducing initial_train_size."
            )

        logger.info(
            "Generated %d walk-forward splits (%s window)",
            len(splits),
            self.config.window_type.value,
        )
        return splits
