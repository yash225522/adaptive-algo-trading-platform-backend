"""Preprocessing and feature scaling pipeline for ML models."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        "timestamp",
        "symbol",
        "future_return",
        "label",
        "target_version",
        "feature_version",
    }
)


class PreprocessingError(Exception):
    """Raised when data preprocessing or scaling fails."""


class MLPreprocessor:
    """Extracts features, encodes labels, and fits scalers only on training data."""

    def __init__(
        self,
        positive_class: str = "UP",
        negative_class: str = "DOWN",
    ) -> None:
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self._is_fitted: bool = False

    def extract_features_and_labels(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, npt.NDArray[np.int_]]:
        """Separate input feature columns from target label column.

        Args:
            df: Input tabular DataFrame.

        Returns:
            tuple[pd.DataFrame, npt.NDArray[np.int_]]: (X_df, y_array)
        """
        if df.empty:
            raise PreprocessingError("Cannot extract features from empty DataFrame")

        if "label" not in df.columns:
            raise PreprocessingError("DataFrame missing required 'label' target column")

        # Exclude all target and non-feature columns
        feature_cols = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
        if not feature_cols:
            raise PreprocessingError("No valid feature columns found in DataFrame")

        X_df = df[feature_cols].copy()

        # Check for any remaining non-numeric columns
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise PreprocessingError(
                f"Non-numeric feature columns detected: {non_numeric}"
            )

        # Encode binary classification label (UP -> 1, DOWN -> 0)
        labels = df["label"].astype(str)
        y_list: list[int] = []
        for val in labels:
            if val == self.positive_class:
                y_list.append(1)
            elif val == self.negative_class:
                y_list.append(0)
            else:
                raise PreprocessingError(
                    f"Unexpected label '{val}'; expected '{self.positive_class}' "
                    f"or '{self.negative_class}'"
                )

        y = np.array(y_list, dtype=int)
        return X_df, y

    def fit_transform(
        self, train_df: pd.DataFrame
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
        """Fit scaler on training data only and return scaled X and y.

        Args:
            train_df: Training partition DataFrame.

        Returns:
            tuple: (scaled_X_train, y_train)
        """
        X_df, y_train = self.extract_features_and_labels(train_df)
        self.feature_names = list(X_df.columns)

        self.scaler = StandardScaler()
        X_scaled: npt.NDArray[np.float64] = self.scaler.fit_transform(X_df.to_numpy())
        self._is_fitted = True

        logger.debug(
            "Fitted StandardScaler on %d training rows with %d features",
            len(train_df),
            len(self.feature_names),
        )
        return X_scaled, y_train

    def transform(
        self, test_df: pd.DataFrame
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
        """Transform test data using the existing fitted scaler without refitting.

        Args:
            test_df: Test partition DataFrame.

        Returns:
            tuple: (scaled_X_test, y_test)
        """
        if not self._is_fitted or self.scaler is None:
            raise PreprocessingError(
                "Preprocessor has not been fitted yet; call fit_transform first"
            )

        X_df, y_test = self.extract_features_and_labels(test_df)

        # Verify feature columns match exactly
        if list(X_df.columns) != self.feature_names:
            raise PreprocessingError(
                f"Feature mismatch between train and test: {list(X_df.columns)}"
            )

        X_scaled: npt.NDArray[np.float64] = self.scaler.transform(X_df.to_numpy())
        return X_scaled, y_test
