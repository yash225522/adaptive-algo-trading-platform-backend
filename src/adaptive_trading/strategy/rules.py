"""Strategy abstractions and threshold-based rule implementations."""

import logging
from abc import ABC, abstractmethod

from adaptive_trading.strategy.config import StrategyConfig
from adaptive_trading.strategy.exceptions import InvalidPredictionError
from adaptive_trading.strategy.models import (
    SignalAction,
    StrategyPrediction,
    TradingSignal,
)

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract interface for all algorithmic trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the strategy."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version identifier of the strategy."""

    @abstractmethod
    def generate_signal(self, prediction: StrategyPrediction) -> TradingSignal:
        """Evaluate a single ML prediction and produce a TradingSignal.

        Args:
            prediction: Standardized prediction input contract.

        Returns:
            TradingSignal: Resulting LONG, SHORT, or NO_TRADE signal.
        """


class ProbabilityStrategy(BaseStrategy):
    """Baseline strategy converting directional probability into discrete signals."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    @property
    def name(self) -> str:
        return self.config.strategy_name

    @property
    def version(self) -> str:
        return self.config.strategy_version

    def generate_signal(self, prediction: StrategyPrediction) -> TradingSignal:
        """Generate a trading signal based on calibrated probability thresholds.

        Rules:
            - probability_up >= long_probability_threshold -> LONG
            - probability_up <= short_probability_threshold -> SHORT
            - Otherwise -> NO_TRADE

        Confidence:
            - LONG: probability_up
            - SHORT: 1.0 - probability_up
            - NO_TRADE: 0.0 (neutral / uncommitted)
        """
        if not isinstance(prediction, StrategyPrediction):
            raise InvalidPredictionError(
                f"Expected StrategyPrediction instance, got {type(prediction).__name__}"
            )

        prob = prediction.probability_up
        long_thresh = self.config.long_probability_threshold
        short_thresh = self.config.short_probability_threshold

        if prob >= long_thresh:
            action = SignalAction.LONG
            confidence = prob
            reason = (
                f"LONG: probability_up ({prob:.4f}) >= "
                f"long_threshold ({long_thresh:.4f})"
            )
        elif prob <= short_thresh:
            action = SignalAction.SHORT
            confidence = 1.0 - prob
            reason = (
                f"SHORT: probability_up ({prob:.4f}) <= "
                f"short_threshold ({short_thresh:.4f})"
            )
        else:
            action = SignalAction.NO_TRADE
            confidence = 0.0
            reason = (
                f"NO_TRADE: probability_up ({prob:.4f}) inside neutral zone "
                f"({short_thresh:.4f}, {long_thresh:.4f})"
            )

        return TradingSignal(
            timestamp=prediction.timestamp,
            symbol=prediction.symbol,
            action=action,
            confidence=round(confidence, 6),
            strategy_name=self.name,
            strategy_version=self.version,
            reason=reason,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
        )
