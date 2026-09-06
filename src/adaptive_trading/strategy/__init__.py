"""Strategy engine package converting ML inference predictions into trading signals."""

from adaptive_trading.strategy.config import StrategyConfig
from adaptive_trading.strategy.engine import StrategyEngine
from adaptive_trading.strategy.exceptions import (
    DuplicateSignalError,
    InvalidPredictionError,
    StrategyConfigError,
    StrategyError,
)
from adaptive_trading.strategy.models import (
    SignalAction,
    SignalStatistics,
    StrategyPrediction,
    TradingSignal,
)
from adaptive_trading.strategy.rules import BaseStrategy, ProbabilityStrategy

__all__ = [
    "BaseStrategy",
    "DuplicateSignalError",
    "InvalidPredictionError",
    "ProbabilityStrategy",
    "SignalAction",
    "SignalStatistics",
    "StrategyConfig",
    "StrategyConfigError",
    "StrategyEngine",
    "StrategyError",
    "StrategyPrediction",
    "TradingSignal",
]
