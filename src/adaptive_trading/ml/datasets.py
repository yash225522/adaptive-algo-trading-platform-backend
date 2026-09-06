"""Chronological time-series dataset splitter and validation."""

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetSplitError(Exception):
    """Raised when time-series dataset splitting fails validation."""


@dataclass(frozen=True)
class DatasetSplit:
    """Holds chronologically partitioned train and test datasets with metadata."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_sample_count: int
    test_sample_count: int


class ChronologicalSplitter:
    """Splits time-series datasets strictly on a chronological boundary."""

    def __init__(self, train_ratio: float = 0.80) -> None:
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(
                f"train_ratio must be between 0.0 and 1.0, got {train_ratio}"
            )
        self.train_ratio = train_ratio

    def split(self, df: pd.DataFrame) -> DatasetSplit:
        """Split DataFrame into chronological train and test subsets.

        Args:
            df: DataFrame containing at least a 'timestamp' column.

        Returns:
            DatasetSplit: Container with train/test DataFrames and time boundaries.
        """
        if df.empty:
            raise DatasetSplitError("Cannot split an empty DataFrame")

        if "timestamp" not in df.columns:
            raise DatasetSplitError(
                "DataFrame must contain 'timestamp' column for chronological splitting"
            )

        # Guarantee strict chronological order
        sorted_df = df.sort_values(by="timestamp").copy().reset_index(drop=True)
        total_samples = len(sorted_df)

        split_idx = int(total_samples * self.train_ratio)

        # Enforce minimum sample counts
        if split_idx < 2 or (total_samples - split_idx) < 1:
            raise DatasetSplitError(
                f"Insufficient samples ({total_samples}) for ratio {self.train_ratio}"
            )

        train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

        train_start = pd.to_datetime(train_df["timestamp"].iloc[0]).to_pydatetime()
        train_end = pd.to_datetime(train_df["timestamp"].iloc[-1]).to_pydatetime()
        test_start = pd.to_datetime(test_df["timestamp"].iloc[0]).to_pydatetime()
        test_end = pd.to_datetime(test_df["timestamp"].iloc[-1]).to_pydatetime()

        # Invariant check: strictly chronological non-overlapping
        if train_end >= test_start:
            raise DatasetSplitError(
                f"Chronological overlap detected: train_end ({train_end}) >= "
                f"test_start ({test_start})"
            )

        logger.info(
            "Chronological split complete: train=%d (%s to %s), test=%d (%s to %s)",
            len(train_df),
            train_start,
            train_end,
            len(test_df),
            test_start,
            test_end,
        )

        return DatasetSplit(
            train_df=train_df,
            test_df=test_df,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_sample_count=len(train_df),
            test_sample_count=len(test_df),
        )
