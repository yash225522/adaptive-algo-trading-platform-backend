"""Market condition scenario analysis and transaction cost sensitivity evaluation."""

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.engine import BacktestEngine
from adaptive_trading.domain.market import Candle
from adaptive_trading.evaluation.config import EvaluationConfig
from adaptive_trading.strategy.models import TradingSignal

logger = logging.getLogger(__name__)


class MarketScenarioType(StrEnum):
    """Categorical classification of market regime and price environment."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    SIDEWAYS = "SIDEWAYS"


class MarketScenario(BaseModel):
    """Defined chronological market scenario partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(description="Scenario identifier")
    scenario_type: MarketScenarioType = Field(
        description="Regime classification"
    )
    start: datetime | None = None
    end: datetime | None = None
    description: str = Field(
        default="", description="Descriptive context of the scenario"
    )
    candle_count: int = Field(
        default=0, ge=0, description="Total candles in scenario"
    )


class ScenarioEvaluationResult(BaseModel):
    """Strategy performance outcome within a specific market regime scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    scenario_type: MarketScenarioType
    candle_count: int = Field(ge=0)
    net_pnl: float
    total_return_pct: float
    total_trades: int = Field(ge=0)
    win_rate: float
    max_drawdown_pct: float


class CostSensitivityResult(BaseModel):
    """Backtest performance metrics under varied commission and slippage costs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost_scenario: str = Field(
        description="Scenario name (e.g. 'LOW_COST', 'NORMAL_COST', 'HIGH_COST')"
    )
    commission_bps: float = Field(ge=0.0, description="Commission in basis points")
    slippage_bps: float = Field(ge=0.0, description="Slippage in basis points")
    total_trades: int = Field(ge=0)
    net_pnl: float
    total_return_pct: float
    max_drawdown_pct: float


class ScenarioEvaluator:
    """Evaluates strategy performance across market regimes and execution cost tiers."""

    def classify_market_regime(
        self, candles: Sequence[Candle]
    ) -> MarketScenarioType:
        """Classify a sequence of candles into a deterministic market scenario."""
        if len(candles) < 2:
            return MarketScenarioType.SIDEWAYS

        start_open = float(candles[0].open)
        end_close = float(candles[-1].close)
        ret = (
            (end_close - start_open) / start_open
            if start_open > 0
            else 0.0
        )

        closes = np.array([float(c.close) for c in candles], dtype=np.float64)
        pct_changes = np.diff(closes) / closes[:-1]
        vol = float(np.std(pct_changes)) if len(pct_changes) > 1 else 0.0

        if vol > 0.02:
            return MarketScenarioType.HIGH_VOLATILITY
        if ret > 0.01:
            return MarketScenarioType.UPTREND
        if ret < -0.01:
            return MarketScenarioType.DOWNTREND
        if vol < 0.002:
            return MarketScenarioType.LOW_VOLATILITY
        return MarketScenarioType.SIDEWAYS

    def evaluate_cost_sensitivity(
        self,
        candles: Sequence[Candle],
        signals: Sequence[TradingSignal],
        base_config: EvaluationConfig,
    ) -> list[CostSensitivityResult]:
        """Run the strategy under Low, Normal, and High transaction cost assumptions."""
        if not candles or not signals:
            return []

        cost_tiers = [
            ("LOW_COST", 0.0, 0.0),
            (
                "NORMAL_COST",
                base_config.commission_bps,
                base_config.slippage_bps,
            ),
            (
                "HIGH_COST",
                max(10.0, base_config.commission_bps * 3),
                max(15.0, base_config.slippage_bps * 3),
            ),
        ]

        results: list[CostSensitivityResult] = []

        for name, comm, slip in cost_tiers:
            bt_cfg = BacktestConfig(
                initial_capital=base_config.initial_capital,
                commission_bps=comm,
                slippage_bps=slip,
                fixed_quantity=base_config.fixed_quantity,
            )
            engine = BacktestEngine(config=bt_cfg)
            try:
                bt_res = engine.run(
                    candles=candles,
                    signals=signals,
                    save_artifacts=False,
                )
                m = bt_res.metrics
                results.append(
                    CostSensitivityResult(
                        cost_scenario=name,
                        commission_bps=comm,
                        slippage_bps=slip,
                        total_trades=m.total_trades,
                        net_pnl=m.net_pnl,
                        total_return_pct=m.total_return_pct,
                        max_drawdown_pct=m.max_drawdown_pct,
                    )
                )
            except Exception as exc:
                logger.debug(
                    "Cost sensitivity run failed for %s: %s", name, exc
                )

        return results

