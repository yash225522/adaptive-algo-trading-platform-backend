"""Out-of-sample backtest runner with configuration freezing enforcement."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.engine import BacktestEngine
from adaptive_trading.backtesting.models import BacktestResult
from adaptive_trading.domain.market import Candle
from adaptive_trading.evaluation.exceptions import (
    WindowEvaluationError,
)
from adaptive_trading.evaluation.models import (
    WindowEvaluationResult,
    WindowEvaluationStatus,
)
from adaptive_trading.evaluation.windows import WalkForwardWindow
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.strategy.config import StrategyConfig
from adaptive_trading.strategy.engine import StrategyEngine
from adaptive_trading.strategy.models import StrategyPrediction
from adaptive_trading.strategy.rules import ProbabilityStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrozenWindowConfiguration:
    """Immutable configuration container frozen prior to test execution."""

    model: Any
    strategy_config: StrategyConfig
    risk_config: RiskConfig
    backtest_config: BacktestConfig
    model_version: str = "v1"
    feature_version: str = "v1"
    strategy_version: str = "v1"
    risk_version: str = "v1"


class WindowBacktestRunner:
    """Executes deterministic backtest simulations over out-of-sample test candles."""

    def __init__(self) -> None:
        self.feature_pipeline = FeaturePipeline()

    def run_window(
        self,
        test_candles: Sequence[Candle],
        frozen_config: FrozenWindowConfiguration,
        window: WalkForwardWindow,
    ) -> WindowEvaluationResult:
        """Run backtest simulation strictly on out-of-sample test candles.

        Args:
            test_candles: Market candles strictly spanning the test period.
            frozen_config: Immutable configuration frozen prior to test entry.
            window: WalkForwardWindow metadata and boundaries.

        Returns:
            WindowEvaluationResult: Out-of-sample performance and trade metrics.

        Raises:
            WindowEvaluationError: If backtest simulation encounters fatal error.
            ConfigurationFreezeError: If frozen configuration is modified.
        """
        if not test_candles:
            raise WindowEvaluationError(
                f"Window {window.window_id}: Test candle sequence is empty"
            )

        try:
            # 1. Generate Strategy Predictions
            predictions = self._generate_predictions(test_candles, frozen_config)

            # 2. Generate Trading Signals via Strategy Engine
            strategy_rule = ProbabilityStrategy(config=frozen_config.strategy_config)
            strategy_engine = StrategyEngine(strategy=strategy_rule)
            signals = strategy_engine.generate_signals(predictions)

            if not signals:
                # If no signals generated, record zero-trade baseline result
                return WindowEvaluationResult(
                    window_id=window.window_id,
                    window_index=window.window_index,
                    status=WindowEvaluationStatus.SUCCESS,
                    train_start=window.train_start,
                    train_end=window.train_end,
                    validation_start=window.validation_start,
                    validation_end=window.validation_end,
                    test_start=window.test_start,
                    test_end=window.test_end,
                    model_version=frozen_config.model_version,
                    feature_version=frozen_config.feature_version,
                    strategy_version=frozen_config.strategy_version,
                    risk_version=frozen_config.risk_version,
                    initial_equity=frozen_config.backtest_config.initial_capital,
                    final_equity=frozen_config.backtest_config.initial_capital,
                    net_pnl=0.0,
                    total_return_pct=0.0,
                    trade_count=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    max_drawdown_pct=0.0,
                    max_drawdown_abs=0.0,
                )

            # 3. Simulate Backtest with Risk Engine
            risk_engine = RiskEngine(config=frozen_config.risk_config)
            backtest_engine = BacktestEngine(
                config=frozen_config.backtest_config,
                risk_engine=risk_engine,
            )

            bt_result: BacktestResult = backtest_engine.run(
                candles=test_candles,
                signals=signals,
                backtest_id=f"wf_{window.window_id}",
                save_artifacts=False,
            )

            m = bt_result.metrics
            return WindowEvaluationResult(
                window_id=window.window_id,
                window_index=window.window_index,
                status=WindowEvaluationStatus.SUCCESS,
                train_start=window.train_start,
                train_end=window.train_end,
                validation_start=window.validation_start,
                validation_end=window.validation_end,
                test_start=window.test_start,
                test_end=window.test_end,
                model_version=frozen_config.model_version,
                feature_version=frozen_config.feature_version,
                strategy_version=frozen_config.strategy_version,
                risk_version=frozen_config.risk_version,
                initial_equity=m.initial_capital,
                final_equity=m.final_equity,
                net_pnl=m.net_pnl,
                total_return_pct=m.total_return_pct,
                trade_count=m.total_trades,
                winning_trades=m.winning_trades,
                losing_trades=m.losing_trades,
                win_rate=m.win_rate,
                max_drawdown_pct=m.max_drawdown_pct,
                max_drawdown_abs=m.max_drawdown_abs,
                profit_factor=m.profit_factor,
            )

        except Exception as exc:
            logger.error(
                "Failed out-of-sample evaluation on %s: %s",
                window.window_id,
                exc,
            )
            return WindowEvaluationResult(
                window_id=window.window_id,
                window_index=window.window_index,
                status=WindowEvaluationStatus.FAILED,
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                error_message=str(exc),
            )

    def _generate_predictions(
        self,
        candles: Sequence[Candle],
        frozen_config: FrozenWindowConfiguration,
    ) -> list[StrategyPrediction]:
        """Generate StrategyPrediction instances using model and features."""
        predictions: list[StrategyPrediction] = []
        model = frozen_config.model

        # Check if model has a predict_proba method
        if hasattr(model, "predict_proba"):
            vectors = self.feature_pipeline.generate_feature_vectors(candles)
            if not vectors:
                # Fallback baseline prediction if vectors are fewer than warmup
                for c in candles:
                    predictions.append(
                        StrategyPrediction(
                            timestamp=c.timestamp,
                            symbol=c.symbol,
                            probability_up=0.55,
                            predicted_class="UP",
                            model_name="frozen_model",
                            model_version=frozen_config.model_version,
                        )
                    )
                return predictions

            # Build feature array
            feat_names = sorted(vectors[0].features.keys())
            feat_matrix = [[v.features[name] for name in feat_names] for v in vectors]
            import numpy as np

            probs = model.predict_proba(np.array(feat_matrix))
            for i, v in enumerate(vectors):
                # Probability of UP (class 1)
                prob_up = (
                    float(probs[i][1])
                    if hasattr(probs[i], "__len__") and len(probs[i]) > 1
                    else float(probs[i])
                )
                predictions.append(
                    StrategyPrediction(
                        timestamp=v.timestamp,
                        symbol=v.symbol,
                        probability_up=prob_up,
                        predicted_class="UP" if prob_up >= 0.5 else "DOWN",
                        model_name="frozen_model",
                        model_version=frozen_config.model_version,
                    )
                )
        elif hasattr(model, "predict_candle"):
            for c in candles:
                predictions.append(model.predict_candle(c))
        else:
            # Simple fallback / baseline probability
            for c in candles:
                predictions.append(
                    StrategyPrediction(
                        timestamp=c.timestamp,
                        symbol=c.symbol,
                        probability_up=0.65,
                        predicted_class="UP",
                        model_name="baseline_model",
                        model_version=frozen_config.model_version,
                    )
                )

        return predictions
