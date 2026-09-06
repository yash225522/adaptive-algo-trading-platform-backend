"""Strategy engine orchestrating single-item and batch trading signal generation."""

import logging
from collections.abc import Sequence
from datetime import datetime

from adaptive_trading.strategy.exceptions import (
    DuplicateSignalError,
    InvalidPredictionError,
)
from adaptive_trading.strategy.models import (
    SignalAction,
    SignalStatistics,
    StrategyPrediction,
    TradingSignal,
)
from adaptive_trading.strategy.rules import BaseStrategy, ProbabilityStrategy

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Evaluates ML predictions through strategy rules to produce actionable signals."""

    def __init__(self, strategy: BaseStrategy | None = None) -> None:
        self.strategy = strategy or ProbabilityStrategy()

    def generate_signal(self, prediction: StrategyPrediction) -> TradingSignal:
        """Evaluate a single prediction through strategy rules.

        Args:
            prediction: Prediction object with probabilities and metadata.

        Returns:
            TradingSignal: Output signal decision.
        """
        return self.strategy.generate_signal(prediction)

    def generate_signals(
        self,
        predictions: Sequence[StrategyPrediction],
        allow_unsorted: bool = True,
    ) -> list[TradingSignal]:
        """Process a sequence of predictions in batch, preserving chronological order.

        Args:
            predictions: Collection of StrategyPrediction inputs.
            allow_unsorted: If True, sorts predictions chronologically.

        Returns:
            list[TradingSignal]: Chronologically ordered sequence of TradingSignals.

        Raises:
            InvalidPredictionError: If batch is empty or elements are invalid.
            DuplicateSignalError: If duplicate (timestamp, symbol) pairs are found.
        """
        if not predictions:
            return []

        # Validate elements
        for pred in predictions:
            if not isinstance(pred, StrategyPrediction):
                raise InvalidPredictionError(
                    f"Invalid prediction in batch: {type(pred).__name__}"
                )

        # Sort chronologically by timestamp
        if allow_unsorted:
            ordered_preds = sorted(predictions, key=lambda p: p.timestamp)
        else:
            ordered_preds = list(predictions)

        # Detect duplicate timestamps for same symbol
        seen_keys: set[tuple[datetime, str]] = set()
        signals: list[TradingSignal] = []

        for pred in ordered_preds:
            key = (pred.timestamp, pred.symbol)
            if key in seen_keys:
                raise DuplicateSignalError(
                    f"Duplicate prediction timestamp detected for {pred.symbol} "
                    f"at {pred.timestamp.isoformat()}"
                )
            seen_keys.add(key)

            signal = self.generate_signal(pred)
            signals.append(signal)

        logger.info(
            "StrategyEngine [%s %s] generated %d signals from batch",
            self.strategy.name,
            self.strategy.version,
            len(signals),
        )
        return signals

    def calculate_statistics(
        self, signals: Sequence[TradingSignal]
    ) -> SignalStatistics:
        """Calculate summary distribution metrics for a batch of generated signals.

        Args:
            signals: Sequence of generated TradingSignals.

        Returns:
            SignalStatistics: Distribution statistics across signal actions.
        """
        total = len(signals)
        if total == 0:
            return SignalStatistics(
                total_predictions=0,
                long_signals=0,
                short_signals=0,
                no_trade_signals=0,
                long_percentage=0.0,
                short_percentage=0.0,
                no_trade_percentage=0.0,
            )

        long_count = sum(1 for s in signals if s.action == SignalAction.LONG)
        short_count = sum(1 for s in signals if s.action == SignalAction.SHORT)
        no_trade_count = sum(1 for s in signals if s.action == SignalAction.NO_TRADE)

        return SignalStatistics(
            total_predictions=total,
            long_signals=long_count,
            short_signals=short_count,
            no_trade_signals=no_trade_count,
            long_percentage=round((long_count / total) * 100.0, 2),
            short_percentage=round((short_count / total) * 100.0, 2),
            no_trade_percentage=round((no_trade_count / total) * 100.0, 2),
        )
