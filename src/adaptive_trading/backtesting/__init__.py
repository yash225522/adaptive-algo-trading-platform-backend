"""Historical backtesting engine and performance metrics evaluation."""

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.engine import BacktestEngine
from adaptive_trading.backtesting.exceptions import (
    BacktestConfigError,
    BacktestError,
    ExecutionError,
    InsufficientDataError,
)
from adaptive_trading.backtesting.execution import SimulatedExecutionHandler
from adaptive_trading.backtesting.metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from adaptive_trading.backtesting.models import (
    BacktestFill,
    BacktestOrder,
    BacktestPosition,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
    PositionSide,
)
from adaptive_trading.backtesting.portfolio import PortfolioTracker

__all__ = [
    "BacktestConfig",
    "BacktestConfigError",
    "BacktestEngine",
    "BacktestError",
    "BacktestFill",
    "BacktestOrder",
    "BacktestPosition",
    "BacktestResult",
    "BacktestTrade",
    "EquityPoint",
    "ExecutionError",
    "InsufficientDataError",
    "PerformanceMetrics",
    "PortfolioTracker",
    "PositionSide",
    "SimulatedExecutionHandler",
    "calculate_performance_metrics",
]
