"""Time-series walk-forward dataset splitter with purge and embargo support."""

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.evaluation.config import EvaluationConfig, WindowType
from adaptive_trading.evaluation.exceptions import (
    InsufficientDataError,
    TemporalOrderError,
)
from adaptive_trading.evaluation.windows import WalkForwardWindow
from adaptive_trading.experiments.fingerprint import (
    compute_dataset_fingerprint,
)

logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """Generates strictly chronological rolling or expanding walk-forward splits."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()

    def split(
        self,
        data: Sequence[Candle] | pd.DataFrame | Sequence[dict[str, Any]],
        dataset_fingerprint: str | None = None,
    ) -> list[WalkForwardWindow]:
        """Generate chronological walk-forward windows.

        Args:
            data: Chronologically ordered market candles, DataFrame, or records.
            dataset_fingerprint: Optional SHA-256 fingerprint; computed if None.

        Returns:
            list[WalkForwardWindow]: Chronologically isolated evaluation windows.

        Raises:
            InsufficientDataError: If sample count is inadequate for window parameters.
            TemporalOrderError: If timestamps are non-monotonic or inverted.
        """
        n_samples = len(data)
        if n_samples == 0:
            raise InsufficientDataError("Cannot split an empty dataset")

        # Resolve fingerprint if not provided
        if dataset_fingerprint:
            fp = dataset_fingerprint
        elif isinstance(data, pd.DataFrame):
            fp = compute_dataset_fingerprint(data.to_dict(orient="records"))
        else:
            fp = compute_dataset_fingerprint(data)

        # 1. Resolve integer partition lengths
        if isinstance(self.config.train_window, float):
            train_len = int(n_samples * self.config.train_window)
        else:
            train_len = int(self.config.train_window)

        if self.config.validation_window is not None:
            if isinstance(self.config.validation_window, float):
                val_len = int(n_samples * self.config.validation_window)
            else:
                val_len = int(self.config.validation_window)
        else:
            val_len = 0

        if isinstance(self.config.test_window, float):
            test_len = int(n_samples * self.config.test_window)
        else:
            test_len = int(self.config.test_window)

        if self.config.step_size is not None:
            if isinstance(self.config.step_size, float):
                step_len = int(n_samples * self.config.step_size)
            else:
                step_len = int(self.config.step_size)
        else:
            step_len = test_len

        val_len = max(0, val_len)
        test_len = max(1, test_len)
        step_len = max(1, step_len)
        purge_len = self.config.purge_period

        # 2. Validate data adequacy
        min_required = (
            train_len
            + (val_len + purge_len if val_len > 0 else 0)
            + purge_len
            + test_len
        )

        if n_samples < min_required:
            raise InsufficientDataError(
                f"Dataset size ({n_samples}) is smaller than minimum required "
                f"samples ({min_required}) for configured windows: train={train_len}, "
                f"val={val_len}, test={test_len}, purge={purge_len}"
            )

        if train_len < self.config.min_train_samples:
            raise InsufficientDataError(
                f"Train window length ({train_len}) is smaller than configured "
                f"min_train_samples threshold ({self.config.min_train_samples})"
            )

        # Extract timestamps for boundary assertions
        timestamps = self._extract_timestamps(data)

        # 3. Generate walk-forward windows
        windows: list[WalkForwardWindow] = []
        current_train_end = train_len
        window_idx = 1

        while True:
            # Training partition
            if self.config.window_type == WindowType.EXPANDING:
                train_start = 0
            else:
                train_start = max(0, current_train_end - train_len)

            train_indices = list(range(train_start, current_train_end))

            # Validation partition (optional)
            if val_len > 0:
                val_start = current_train_end + purge_len
                val_end = val_start + val_len
                val_indices = list(range(val_start, val_end))
                test_start = val_end + purge_len
            else:
                val_start = current_train_end
                val_end = current_train_end
                val_indices = []
                test_start = current_train_end + purge_len

            test_end = test_start + test_len
            if test_end > n_samples:
                break

            test_indices = list(range(test_start, test_end))

            # Extract temporal boundaries
            t_start: datetime | None = None
            t_end: datetime | None = None
            v_start: datetime | None = None
            v_end: datetime | None = None
            te_start: datetime | None = None
            te_end: datetime | None = None

            if timestamps:
                t_start = timestamps[train_start]
                t_end = timestamps[current_train_end - 1]
                if val_indices:
                    v_start = timestamps[val_start]
                    v_end = timestamps[val_end - 1]
                te_start = timestamps[test_start]
                te_end = timestamps[test_end - 1]

                # Temporal integrity checks
                if t_start and t_end and t_start > t_end:
                    raise TemporalOrderError(
                        f"Train window inverted: start ({t_start}) > end ({t_end})"
                    )
                if val_indices and t_end and v_start and t_end >= v_start:
                    raise TemporalOrderError(
                        f"Train/Val overlap in window {window_idx}: "
                        f"train_end ({t_end}) >= val_start ({v_start})"
                    )
                if val_indices and v_end and te_start and v_end >= te_start:
                    raise TemporalOrderError(
                        f"Val/Test overlap in window {window_idx}: "
                        f"val_end ({v_end}) >= test_start ({te_start})"
                    )
                if not val_indices and t_end and te_start and t_end >= te_start:
                    raise TemporalOrderError(
                        f"Train/Test overlap in window {window_idx}: "
                        f"train_end ({t_end}) >= test_start ({te_start})"
                    )

            windows.append(
                WalkForwardWindow(
                    window_id=f"window_{window_idx}",
                    window_index=window_idx,
                    train_indices=train_indices,
                    validation_indices=val_indices,
                    test_indices=test_indices,
                    train_start=t_start,
                    train_end=t_end,
                    validation_start=v_start,
                    validation_end=v_end,
                    test_start=te_start,
                    test_end=te_end,
                    dataset_fingerprint=fp,
                )
            )

            current_train_end += step_len
            window_idx += 1

        if not windows:
            raise InsufficientDataError(
                f"No walk-forward windows could be constructed from "
                f"{n_samples} samples. Try decreasing train_window, test_window, "
                "or step_size."
            )

        logger.info(
            "Generated %d walk-forward evaluation windows (%s)",
            len(windows),
            self.config.window_type.value,
        )
        return windows

    def _extract_timestamps(
        self, data: Sequence[Candle] | pd.DataFrame | Sequence[dict[str, Any]]
    ) -> list[datetime]:
        """Extract chronological datetime instances from dataset rows."""
        if isinstance(data, pd.DataFrame):
            if "timestamp" in data.columns:
                return [pd.to_datetime(t).to_pydatetime() for t in data["timestamp"]]
            return []
        if data and isinstance(data[0], Candle):
            return [c.timestamp for c in data if isinstance(c, Candle)]
        if data and isinstance(data[0], dict) and "timestamp" in data[0]:
            return [
                pd.to_datetime(r["timestamp"]).to_pydatetime()
                for r in data
                if isinstance(r, dict) and "timestamp" in r
            ]
        return []
