"""Supervised ML model abstraction and Logistic Regression baseline."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from adaptive_trading.ml.config import MLConfig


class BaseMLModel(ABC):
    """Abstract base class establishing uniform model fitting and inference."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model architecture."""

    @abstractmethod
    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
    ) -> Self:
        """Fit model to training feature matrix and label vector."""

    @abstractmethod
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        """Predict discrete class labels."""

    @abstractmethod
    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict class probability distribution."""


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression baseline classifier for binary directional movement."""

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()
        self.estimator = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            solver=self.config.solver,
        )

    @property
    def model_name(self) -> str:
        return "logistic_regression"

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
    ) -> Self:
        """Fit logistic regression model.

        Args:
            X: Scaled feature matrix of shape (n_samples, n_features).
            y: Binary target vector of shape (n_samples,).

        Returns:
            Self: Fitted model instance.
        """
        self.estimator.fit(X, y)
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        """Generate binary class predictions.

        Args:
            X: Scaled feature matrix.

        Returns:
            npt.NDArray[np.int_]: 1D array of class predictions (0 or 1).
        """
        preds: npt.NDArray[np.int_] = self.estimator.predict(X)
        return preds

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            X: Scaled feature matrix.

        Returns:
            npt.NDArray[np.float64]: Probability array of shape (n_samples, 2).
        """
        probas: npt.NDArray[np.float64] = self.estimator.predict_proba(X)
        return probas
