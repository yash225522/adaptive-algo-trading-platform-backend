"""Position sizing algorithms and interface definitions."""

from abc import ABC, abstractmethod

from adaptive_trading.portfolio.models import PortfolioState
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.strategy.models import TradingSignal


class BasePositionSizer(ABC):
    """Abstract base class for all position sizing algorithms."""

    @abstractmethod
    def calculate_quantity(
        self,
        signal: TradingSignal,
        portfolio_state: PortfolioState,
        current_price: float,
        config: RiskConfig,
    ) -> float:
        """Calculate requested position quantity before risk limit checks.

        Args:
            signal: Incoming strategy TradingSignal.
            portfolio_state: Current authoritative portfolio state.
            current_price: Current market price of the asset.
            config: Active RiskConfig.

        Returns:
            float: Sized quantity (>= 0.0).
        """


class FixedPositionSizer(BasePositionSizer):
    """Baseline position sizer returning configured fixed quantity."""

    def calculate_quantity(
        self,
        signal: TradingSignal,
        portfolio_state: PortfolioState,
        current_price: float,
        config: RiskConfig,
    ) -> float:
        """Return fixed quantity from configuration."""
        return float(config.fixed_quantity)
