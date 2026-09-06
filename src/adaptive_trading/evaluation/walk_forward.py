"""Walk-forward evaluation orchestrator.

Coordinates chronological splitting, out-of-sample backtesting, and analysis.
"""

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.domain.market import Candle
from adaptive_trading.evaluation.backtest_runner import (
    FrozenWindowConfiguration,
    WindowBacktestRunner,
)
from adaptive_trading.evaluation.benchmark import BenchmarkEvaluator
from adaptive_trading.evaluation.config import (
    EvaluationConfig,
    FailurePolicy,
    ModelMode,
)
from adaptive_trading.evaluation.exceptions import (
    EvaluationError,
)
from adaptive_trading.evaluation.models import (
    EvaluationReport,
    WindowEvaluationResult,
    WindowEvaluationStatus,
)
from adaptive_trading.evaluation.performance import PerformanceAggregator
from adaptive_trading.evaluation.scenarios import ScenarioEvaluator
from adaptive_trading.evaluation.splitter import TimeSeriesSplitter
from adaptive_trading.evaluation.stability import PerformanceStabilityAnalyzer
from adaptive_trading.experiments.fingerprint import (
    compute_dataset_fingerprint,
)
from adaptive_trading.experiments.registry import ExperimentRegistry
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.observability.run_tracker import (
    RunSummary,
    RunTracker,
    RunType,
)
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.strategy.config import StrategyConfig
from adaptive_trading.validation.gate import validate_for_backtest

logger = logging.getLogger(__name__)


class WalkForwardEvaluator:
    """Orchestrates chronological multi-window walk-forward evaluations."""

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        experiment_registry: ExperimentRegistry | None = None,
        run_tracker: RunTracker | None = None,
    ) -> None:
        self.config = config or EvaluationConfig()
        self.experiment_registry = experiment_registry or ExperimentRegistry()
        self.run_tracker = run_tracker or RunTracker()
        self.splitter = TimeSeriesSplitter(config=self.config)
        self.feature_pipeline = FeaturePipeline()
        self.backtest_runner = WindowBacktestRunner()
        self.performance_aggregator = PerformanceAggregator(config=self.config)
        self.stability_analyzer = PerformanceStabilityAnalyzer()
        self.benchmark_evaluator = BenchmarkEvaluator()
        self.scenario_evaluator = ScenarioEvaluator()

    def evaluate(
        self,
        candles: Sequence[Candle],
        model: object = None,
        strategy_config: StrategyConfig | None = None,
        risk_config: RiskConfig | None = None,
        dataset_identifier: str = "candles_dataset",
        evaluation_id: str | None = None,
        save_artifacts: bool | None = None,
    ) -> EvaluationReport:
        """Execute full walk-forward evaluation workflow.

        Args:
            candles: Chronologically ordered market candle sequence.
            model: Optional static model or prediction service instance.
            strategy_config: Optional strategy configuration settings.
            risk_config: Optional risk limit parameters.
            dataset_identifier: Name or symbol of the dataset.
            evaluation_id: Optional explicit unique evaluation identifier.
            save_artifacts: Whether to save report artifacts to disk.

        Returns:
            EvaluationReport: Complete evaluation results, stability, and comparisons.
        """
        if not candles:
            raise EvaluationError("Cannot evaluate empty candle dataset")

        # 1. Pre-Run Data Quality Validation Gate (Step 21)
        validate_for_backtest(
            candles=candles,
            policy=self.config.validation_policy,
        )

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_suffix = uuid.uuid4().hex[:6]
        eval_id = evaluation_id or f"eval_{now_str}_{rand_suffix}"
        exp_id = f"exp_{eval_id}"
        dataset_fp = compute_dataset_fingerprint(candles)

        # 2. Register Tracking & Experiment Lifecycle
        self.run_tracker.create_run(
            run_id=eval_id,
            run_type=RunType.BACKTEST,
            model_name="walk_forward_model",
            model_version="v1",
        )
        self.run_tracker.start_run(eval_id)

        # 3. Generate Chronological Windows
        windows = self.splitter.split(candles, dataset_fingerprint=dataset_fp)

        strat_cfg = strategy_config or StrategyConfig()
        r_cfg = risk_config or RiskConfig(fixed_quantity=self.config.fixed_quantity)
        bt_cfg = BacktestConfig(
            initial_capital=self.config.initial_capital,
            commission_bps=self.config.commission_bps,
            slippage_bps=self.config.slippage_bps,
            fixed_quantity=self.config.fixed_quantity,
        )

        window_results: list[WindowEvaluationResult] = []
        all_test_candles: list[Candle] = []

        logger.info(
            "Starting walk-forward evaluation %s across %d windows",
            eval_id,
            len(windows),
        )

        # 4. Walk-Forward Window Loop
        for window in windows:
            test_candles = [candles[i] for i in window.test_indices]
            all_test_candles.extend(test_candles)

            # Model preparation & retraining if configured
            active_model = model
            if self.config.model_mode == ModelMode.RETRAIN_PER_WINDOW:
                train_candles = [candles[i] for i in window.train_indices]
                active_model = self._train_window_model(train_candles)

            # Configuration Freeze
            frozen_config = FrozenWindowConfiguration(
                model=active_model,
                strategy_config=strat_cfg,
                risk_config=r_cfg,
                backtest_config=bt_cfg,
            )

            # Out-of-sample test execution
            res = self.backtest_runner.run_window(
                test_candles=test_candles,
                frozen_config=frozen_config,
                window=window,
            )
            window_results.append(res)

            if (
                res.status == WindowEvaluationStatus.FAILED
                and self.config.failure_policy == FailurePolicy.FAIL_FAST
            ):
                logger.warning(
                    "Fail-fast triggered on %s: %s",
                    window.window_id,
                    res.error_message,
                )
                break

        # 5. Out-of-Sample Performance Aggregation
        summary = self.performance_aggregator.aggregate(window_results)

        # 6. Performance Stability Analysis
        stability_report = self.stability_analyzer.analyze(window_results)

        # 7. Benchmark Comparison (Buy & Hold across all test periods)
        benchmark_comp = self.benchmark_evaluator.compare(
            strategy_return_pct=summary.total_return_pct,
            strategy_max_drawdown_pct=summary.worst_drawdown,
            candles=all_test_candles if all_test_candles else candles,
        )

        # 8. Market Scenarios & Cost Sensitivity
        cost_results = self.scenario_evaluator.evaluate_cost_sensitivity(
            candles=all_test_candles if all_test_candles else candles,
            signals=[],  # Evaluated by runner internally if signals present
            base_config=self.config,
        )

        report = EvaluationReport(
            evaluation_id=eval_id,
            experiment_id=exp_id,
            run_id=eval_id,
            created_at=datetime.now(timezone.utc),
            dataset_identifier=dataset_identifier,
            dataset_fingerprint=dataset_fp,
            config=self.config,
            windows=window_results,
            summary=summary,
            benchmark=benchmark_comp.model_dump(mode="json"),
            stability=stability_report.model_dump(mode="json"),
            scenarios=[],
            cost_sensitivity=[c.model_dump(mode="json") for c in cost_results],
            warnings=[],
            limitations=[
                (
                    "Walk-forward evaluation tests historical consistency "
                    "across chronological splits."
                ),
                (
                    "Past out-of-sample performance is not indicative of "
                    "future live market returns."
                ),
                (
                    "Assumes continuous liquidity without market impact "
                    "beyond configured slippage."
                ),
            ],
        )

        # 9. Persist Artifacts
        should_save = (
            save_artifacts if save_artifacts is not None else self.config.save_artifacts
        )
        if should_save:
            report.save(base_dir=self.config.artifacts_dir)

        # 10. Complete Run & Experiment Records
        self.run_tracker.complete_run(
            eval_id,
            summary=RunSummary(
                events_processed=len(candles),
                predictions_generated=summary.total_trades,
                signals_generated=summary.total_trades,
                orders_submitted=summary.total_trades,
                orders_filled=summary.total_trades,
                final_equity=self.config.initial_capital
                * (1.0 + summary.total_return_pct / 100.0),
                net_pnl=self.config.initial_capital
                * (summary.total_return_pct / 100.0),
                trade_count=summary.total_trades,
                max_drawdown=summary.worst_drawdown,
            ),
        )

        return report

    def _train_window_model(self, train_candles: Sequence[Candle]) -> object:
        """Train a Logistic Regression model on the training partition."""
        import numpy as np

        from adaptive_trading.ml.models import LogisticRegressionModel
        from adaptive_trading.targets.dataset import DatasetBuilder
        from adaptive_trading.targets.generator import TargetGenerator

        fvs = self.feature_pipeline.generate_feature_vectors(train_candles)
        targets = TargetGenerator().generate_targets(train_candles)
        df = DatasetBuilder().build_training_dataframe(fvs, targets)
        if df.empty:
            return None

        model = LogisticRegressionModel()
        feature_cols = [
            c
            for c in df.columns
            if c
            not in {
                "timestamp",
                "symbol",
                "label",
                "target",
                "future_return",
                "timeframe",
                "feature_version",
                "target_version",
            }
        ]
        if not feature_cols:
            return None

        X = df[feature_cols].to_numpy(dtype=np.float64)
        y = np.where(df["label"] == "UP", 1, 0).astype(np.int_)
        if len(np.unique(y)) < 2:
            return None
        model.fit(X, y)
        return model
