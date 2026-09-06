"""Orchestration runner connecting experiments with runtime replay and backtesting."""

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from adaptive_trading.domain.market import Candle
from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.experiments.models import (
    Experiment,
    ExperimentManifest,
    ExperimentResult,
    ExperimentType,
)
from adaptive_trading.experiments.registry import ExperimentRegistry
from adaptive_trading.experiments.versioning import (
    build_dataset_version,
    build_execution_version,
    build_experiment_manifest,
    build_experiment_result,
    build_feature_version,
    build_model_version,
    build_risk_version,
    build_runtime_version,
    build_strategy_version,
    capture_environment_metadata,
    verify_compatibility,
)
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.runtime.config import RuntimeConfig
from adaptive_trading.runtime.event_loop import EventLoop
from adaptive_trading.runtime.runner import HistoricalDataProvider, ReplayRunner
from adaptive_trading.strategy.config import StrategyConfig

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Executes experiments end-to-end, recording manifests and performance results."""

    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()

    def run_paper_replay_experiment(
        self,
        candles: Sequence[Candle],
        name: str = "paper_replay_exp",
        description: str = "Deterministic paper trading replay experiment",
        prediction_service: Any = None,
        runtime_config: RuntimeConfig | None = None,
        execution_config: ExecutionConfig | None = None,
        risk_config: RiskConfig | None = None,
        strategy_config: StrategyConfig | None = None,
        tags: list[str] | None = None,
    ) -> tuple[Experiment, ExperimentManifest, ExperimentResult]:
        """Run paper trading replay as a tracked, reproducible experiment."""
        r_cfg = runtime_config or RuntimeConfig()
        e_cfg = execution_config or ExecutionConfig()
        rk_cfg = risk_config or RiskConfig()
        s_cfg = strategy_config or StrategyConfig()

        # 1. Create experiment record
        exp = self.registry.create_experiment(
            name=name,
            experiment_type=ExperimentType.PAPER_REPLAY,
            description=description,
        )

        # 2. Build complete reproducible manifest
        manifest = build_experiment_manifest(
            experiment_id=exp.experiment_id,
            dataset=build_dataset_version(
                candles=candles,
                name=f"{name}_dataset",
            ),
            features=build_feature_version(),
            model=build_model_version(),
            strategy=build_strategy_version(
                config=s_cfg,
            ),
            risk=build_risk_version(
                config=rk_cfg,
            ),
            execution=build_execution_version(
                config=e_cfg,
            ),
            runtime=build_runtime_version(
                config=r_cfg,
            ),
            environment=capture_environment_metadata(),
        )

        # 3. Check version compatibility
        compat_issues = verify_compatibility(manifest)
        if compat_issues:
            logger.warning(
                "Experiment %s compatibility issues: %s",
                exp.experiment_id,
                compat_issues,
            )

        # 4. Start experiment
        self.registry.start_experiment(exp.experiment_id)

        try:
            # 5. Execute simulation replay
            event_loop = EventLoop(
                config=r_cfg,
                prediction_service=prediction_service,
            )
            runner = ReplayRunner(
                event_loop=event_loop,
                config=r_cfg,
            )
            provider = HistoricalDataProvider(candles=candles)
            runner.run(provider=provider, run_id=exp.experiment_id)

            # 6. Extract summary & build ExperimentResult
            last_ts = candles[-1].timestamp if candles else datetime.now(timezone.utc)
            last_pstate = event_loop.portfolio_manager.mark_to_market(
                timestamp=last_ts,
                current_prices=event_loop.state.last_market_price,
            )
            summary = {
                "initial_capital": event_loop.portfolio_manager.initial_capital,
                "final_equity": last_pstate.equity,
                "realized_pnl": last_pstate.realized_pnl,
                "unrealized_pnl": last_pstate.unrealized_pnl,
                "total_events": event_loop.state.stats.market_events_processed,
                "signals_generated": event_loop.state.stats.signals_generated,
                "orders_submitted": event_loop.state.stats.orders_submitted,
                "orders_filled": event_loop.state.stats.orders_filled,
                "orders_rejected": event_loop.state.stats.orders_rejected,
            }
            result = build_experiment_result(
                experiment_id=exp.experiment_id,
                result_data=summary,
            )

            # 7. Complete experiment & store manifest
            manifest = manifest.model_copy(update={"result": result})
            completed_exp = self.registry.complete_experiment(
                experiment_id=exp.experiment_id,
                result=result,
                manifest=manifest,
            )

            return completed_exp, manifest, result

        except Exception as exc:
            self.registry.fail_experiment(exp.experiment_id, error=exc)
            raise
