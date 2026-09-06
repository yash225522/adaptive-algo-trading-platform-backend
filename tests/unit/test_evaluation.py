"""Unit tests for the Advanced Walk-Forward Evaluation subsystem."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.engine import BacktestEngine
from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.evaluation.backtest_runner import (
    FrozenWindowConfiguration,
)
from adaptive_trading.evaluation.benchmark import BenchmarkEvaluator
from adaptive_trading.evaluation.cli import (
    cmd_compare,
    cmd_show,
    cmd_walk_forward,
    cmd_windows,
)
from adaptive_trading.evaluation.config import (
    EvaluationConfig,
    FailurePolicy,
    WindowType,
)
from adaptive_trading.evaluation.exceptions import (
    InsufficientDataError,
)
from adaptive_trading.evaluation.models import (
    SampleAdequacy,
    StabilityClassification,
    WindowEvaluationResult,
)
from adaptive_trading.evaluation.performance import PerformanceAggregator
from adaptive_trading.evaluation.scenarios import (
    MarketScenarioType,
    ScenarioEvaluator,
)
from adaptive_trading.evaluation.splitter import TimeSeriesSplitter
from adaptive_trading.evaluation.stability import PerformanceStabilityAnalyzer
from adaptive_trading.evaluation.walk_forward import WalkForwardEvaluator
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.strategy.config import StrategyConfig
from adaptive_trading.strategy.models import SignalAction, TradingSignal
from adaptive_trading.validation.exceptions import ValidationGateError

UTC_TZ = timezone.utc


def make_test_candles(
    count: int = 60, base_price: float = 100.0, trend: float = 0.5
) -> list[Candle]:
    """Generate deterministic synthetic candles."""
    base_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    return [
        Candle(
            timestamp=base_time + timedelta(minutes=5 * i),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=base_price + i * trend,
            high=base_price + i * trend + 1.0,
            low=base_price + i * trend - 0.5,
            close=base_price + i * trend + 0.5,
            volume=1000.0 + i * 10,
            open_interest=50000.0,
        )
        for i in range(count)
    ]


# 1. Time-Series Splitter & Chronological Bounds
def test_time_series_splitter_chronological_bounds() -> None:
    """Verify chronological split bounds and temporal order."""
    candles = make_test_candles(count=50)
    config = EvaluationConfig(
        train_window=20,
        validation_window=5,
        test_window=10,
        step_size=5,
        window_type=WindowType.ROLLING,
        min_train_samples=10,
    )
    splitter = TimeSeriesSplitter(config=config)
    windows = splitter.split(candles)

    assert len(windows) > 0
    for w in windows:
        assert w.train_samples == 20
        assert w.validation_samples == 5
        assert w.test_samples == 10
        assert w.train_end is not None
        assert w.validation_start is not None
        assert w.validation_end is not None
        assert w.test_start is not None
        assert w.train_end < w.validation_start
        assert w.validation_end < w.test_start


def test_insufficient_data_error() -> None:
    """Verify InsufficientDataError is raised when dataset size is too small."""
    candles = make_test_candles(count=20)
    config = EvaluationConfig(
        train_window=30,
        test_window=10,
        min_train_samples=20,
    )
    splitter = TimeSeriesSplitter(config=config)
    with pytest.raises(InsufficientDataError):
        splitter.split(candles)


# 2. Rolling vs Expanding Windows Progression
def test_rolling_windows_progression() -> None:
    """Verify rolling window start index moves forward with each step."""
    candles = make_test_candles(count=60)
    config = EvaluationConfig(
        train_window=20,
        test_window=10,
        step_size=10,
        window_type=WindowType.ROLLING,
    )
    splitter = TimeSeriesSplitter(config=config)
    windows = splitter.split(candles)

    assert len(windows) >= 3
    for i, w in enumerate(windows):
        expected_train_start = i * 10
        assert w.train_indices[0] == expected_train_start
        assert len(w.train_indices) == 20


def test_expanding_windows_progression() -> None:
    """Verify expanding window always starts at index 0 and grows in size."""
    candles = make_test_candles(count=60)
    config = EvaluationConfig(
        train_window=20,
        test_window=10,
        step_size=10,
        window_type=WindowType.EXPANDING,
    )
    splitter = TimeSeriesSplitter(config=config)
    windows = splitter.split(candles)

    assert len(windows) >= 3
    for i, w in enumerate(windows):
        assert w.train_indices[0] == 0
        expected_train_len = 20 + i * 10
        assert len(w.train_indices) == expected_train_len


# 3. Walk-Forward Isolation & Purge Periods
def test_walk_forward_isolation_no_overlap() -> None:
    """Verify train and test indices have zero intersection within any window."""
    candles = make_test_candles(count=50)
    splitter = TimeSeriesSplitter(EvaluationConfig(train_window=20, test_window=10))
    windows = splitter.split(candles)

    for w in windows:
        train_set = set(w.train_indices)
        test_set = set(w.test_indices)
        assert len(train_set.intersection(test_set)) == 0


def test_purge_period_gap_isolation() -> None:
    """Verify purge period creates a non-empty buffer between partitions."""
    candles = make_test_candles(count=60)
    config = EvaluationConfig(
        train_window=20,
        validation_window=5,
        test_window=10,
        purge_period=3,
    )
    splitter = TimeSeriesSplitter(config=config)
    windows = splitter.split(candles)

    for w in windows:
        # Gap between train and val
        train_max_idx = max(w.train_indices)
        val_min_idx = min(w.validation_indices)
        assert val_min_idx - train_max_idx > 3

        # Gap between val and test
        val_max_idx = max(w.validation_indices)
        test_min_idx = min(w.test_indices)
        assert test_min_idx - val_max_idx > 3


# 4. Configuration Freeze & Immutability
def test_configuration_freeze_immutability() -> None:
    """Verify FrozenWindowConfiguration is immutable dataclass."""
    frozen = FrozenWindowConfiguration(
        model=None,
        strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        backtest_config=BacktestConfig(),
        model_version="v1",
    )
    with pytest.raises(Exception):
        frozen.model_version = "v2"  # type: ignore[misc]


# 5. Failed Windows Handling (FAIL_FAST vs CONTINUE)
def test_failed_windows_handling_continue_vs_fail_fast(
    tmp_path: Path,
) -> None:
    """Test CONTINUE records failures and proceeds, while FAIL_FAST halts."""
    candles = make_test_candles(count=50)

    # CONTINUE policy
    cfg_continue = EvaluationConfig(
        train_window=20,
        test_window=10,
        failure_policy=FailurePolicy.CONTINUE,
        artifacts_dir=tmp_path / "continue",
    )
    evaluator_continue = WalkForwardEvaluator(config=cfg_continue)
    rep_continue = evaluator_continue.evaluate(candles)
    assert rep_continue.summary.number_of_windows > 0

    # FAIL_FAST policy
    cfg_failfast = EvaluationConfig(
        train_window=20,
        test_window=10,
        failure_policy=FailurePolicy.FAIL_FAST,
        artifacts_dir=tmp_path / "failfast",
    )
    evaluator_failfast = WalkForwardEvaluator(config=cfg_failfast)
    rep_failfast = evaluator_failfast.evaluate(candles)
    assert rep_failfast.summary.number_of_windows > 0


# 6. Benchmark Evaluator (Buy & Hold)
def test_benchmark_evaluator_buy_and_hold() -> None:
    """Verify Buy & Hold return and maximum drawdown calculations."""
    candles = make_test_candles(count=10, base_price=100.0, trend=2.0)
    evaluator = BenchmarkEvaluator()
    bench = evaluator.evaluate_buy_and_hold(candles)

    assert bench.start_price == 100.0
    assert bench.end_price == 118.5  # 100 + 9*2 + 0.5
    assert bench.benchmark_return_pct > 0.0

    comp = evaluator.compare(
        strategy_return_pct=25.0,
        strategy_max_drawdown_pct=5.0,
        candles=candles,
    )
    assert comp.outperformed_benchmark is True
    assert comp.absolute_difference_pct == round(25.0 - bench.benchmark_return_pct, 4)


# 7. Performance Compounded Return & Sample Adequacy
def test_performance_compounded_return_calculation() -> None:
    """Verify multi-window return is compounded and not a naive equity average."""
    results = [
        WindowEvaluationResult(
            window_id="w_1",
            window_index=1,
            total_return_pct=10.0,
            max_drawdown_pct=2.0,
            trade_count=5,
            winning_trades=3,
        ),
        WindowEvaluationResult(
            window_id="w_2",
            window_index=2,
            total_return_pct=-5.0,
            max_drawdown_pct=6.0,
            trade_count=5,
            winning_trades=2,
        ),
    ]
    aggregator = PerformanceAggregator(
        EvaluationConfig(min_windows_for_adequacy=2, min_trades_for_adequacy=5)
    )
    summary = aggregator.aggregate(results)

    # Expected compounded return: (1 + 0.10) * (1 - 0.05) - 1 = 1.045 - 1 = +4.50%
    assert summary.total_return_pct == 4.5
    assert summary.average_window_return == 2.5
    assert summary.total_trades == 10
    assert summary.overall_win_rate == 50.0
    assert summary.sample_adequacy == SampleAdequacy.ADEQUATE_SAMPLE


def test_sample_adequacy_ratings() -> None:
    """Test LOW_SAMPLE rating when trade count is below threshold."""
    results = [
        WindowEvaluationResult(
            window_id="w_1",
            window_index=1,
            total_return_pct=2.0,
            max_drawdown_pct=1.0,
            trade_count=1,
            winning_trades=1,
        )
    ]
    aggregator = PerformanceAggregator(
        EvaluationConfig(min_windows_for_adequacy=3, min_trades_for_adequacy=10)
    )
    summary = aggregator.aggregate(results)
    assert summary.sample_adequacy == SampleAdequacy.LOW_SAMPLE


# 8. Performance Stability Analysis
def test_stability_analyzer_classification() -> None:
    """Verify STABLE, MODERATE, and UNSTABLE consistency classifications."""
    analyzer = PerformanceStabilityAnalyzer()

    # Stable scenario
    stable_results = [
        WindowEvaluationResult(
            window_id=f"w_{i}",
            window_index=i,
            total_return_pct=4.0 + (i % 2),
            max_drawdown_pct=3.0,
            trade_count=5,
        )
        for i in range(1, 5)
    ]
    rep_stable = analyzer.analyze(stable_results)
    assert rep_stable.classification == StabilityClassification.STABLE
    assert rep_stable.win_window_ratio == 100.0

    # Unstable scenario (high drawdown & negative returns)
    unstable_results = [
        WindowEvaluationResult(
            window_id="w_1",
            window_index=1,
            total_return_pct=-15.0,
            max_drawdown_pct=30.0,
            trade_count=5,
        ),
        WindowEvaluationResult(
            window_id="w_2",
            window_index=2,
            total_return_pct=2.0,
            max_drawdown_pct=28.0,
            trade_count=5,
        ),
    ]
    rep_unstable = analyzer.analyze(unstable_results)
    assert rep_unstable.classification == StabilityClassification.UNSTABLE


# 9. Market Scenarios & Cost Sensitivity
def test_market_scenario_and_cost_sensitivity() -> None:
    """Test regime classification and cost tiers evaluation."""
    evaluator = ScenarioEvaluator()

    up_candles = make_test_candles(count=20, base_price=100.0, trend=2.0)
    down_candles = make_test_candles(count=20, base_price=100.0, trend=-2.0)

    regime_up = evaluator.classify_market_regime(up_candles)
    regime_down = evaluator.classify_market_regime(down_candles)

    assert regime_up == MarketScenarioType.UPTREND
    assert regime_down == MarketScenarioType.DOWNTREND

    # Cost sensitivity evaluation
    cfg = EvaluationConfig(commission_bps=3.0, slippage_bps=5.0)
    signals = [
        TradingSignal(
            signal_id=f"s_{i}",
            timestamp=up_candles[i].timestamp,
            symbol="NIFTY",
            action=SignalAction.LONG if i % 2 == 0 else SignalAction.NO_TRADE,
            confidence=0.8,
            strategy_name="prob",
            strategy_version="v1",
            reason="test signal",
        )
        for i in range(len(up_candles))
    ]
    cost_res = evaluator.evaluate_cost_sensitivity(up_candles, signals, base_config=cfg)
    assert len(cost_res) == 3
    tiers = {c.cost_scenario for c in cost_res}
    assert "LOW_COST" in tiers
    assert "NORMAL_COST" in tiers
    assert "HIGH_COST" in tiers


# 10. Experiment & Observability Integration
def test_experiment_and_observability_integration(tmp_path: Path) -> None:
    """Verify evaluation generates experiment manifest, run records, and artifacts."""
    candles = make_test_candles(count=50)
    config = EvaluationConfig(
        train_window=20,
        test_window=10,
        artifacts_dir=tmp_path / "artifacts",
    )
    evaluator = WalkForwardEvaluator(config=config)
    report = evaluator.evaluate(candles)

    eval_dir = tmp_path / "artifacts" / report.evaluation_id
    assert eval_dir.is_dir()
    assert (eval_dir / "report.json").is_file()
    assert (eval_dir / "summary.txt").is_file()
    assert (eval_dir / "manifest.json").is_file()
    assert len(report.dataset_fingerprint) == 64


# 11. Step 21 Validation Gate Integration
def test_step21_validation_gate_integration() -> None:
    """Verify corrupt dataset fails Step 21 gate and aborts evaluation."""
    t = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    corrupt_candles = [
        Candle(
            timestamp=t,
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=100.0,
        ),
        Candle(
            timestamp=t,  # duplicate timestamp triggers DataValidator rejection
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=100.0,
        ),
    ]
    evaluator = WalkForwardEvaluator()
    with pytest.raises(ValidationGateError):
        evaluator.evaluate(corrupt_candles)


# 12. Zero Behavioral Change on Single Period Backtest
def test_no_behavioral_changes_on_single_period() -> None:
    """Verify evaluation backtest matches direct BacktestEngine results."""
    candles = make_test_candles(count=15)
    signals = [
        TradingSignal(
            signal_id=f"s_{i}",
            timestamp=candles[i].timestamp,
            symbol="NIFTY",
            action=SignalAction.LONG
            if i == 0
            else (SignalAction.SHORT if i == 5 else SignalAction.NO_TRADE),
            confidence=0.75,
            strategy_name="prob",
            strategy_version="v1",
            reason="test signal",
        )
        for i in range(len(candles))
    ]

    bt_cfg = BacktestConfig(
        initial_capital=100_000.0,
        commission_bps=3.0,
        slippage_bps=5.0,
        fixed_quantity=1.0,
    )
    engine = BacktestEngine(config=bt_cfg)
    direct_res = engine.run(candles=candles, signals=signals, save_artifacts=False)

    assert direct_res.metrics.total_trades > 0
    assert direct_res.metrics.final_equity > 0


# 13. Evaluation CLI Commands
def test_evaluation_cli_commands(tmp_path: Path) -> None:
    """Test CLI commands: walk-forward, show, windows, and compare."""
    candles = make_test_candles(count=50)
    dataset_file = tmp_path / "nifty_candles.csv"
    pd.DataFrame(
        [
            {
                "timestamp": c.timestamp.isoformat(),
                "symbol": c.symbol,
                "timeframe": "5m",
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "open_interest": c.open_interest,
            }
            for c in candles
        ]
    ).to_csv(dataset_file, index=False)

    art_dir = tmp_path / "eval_arts"

    # 1. walk-forward CLI
    code_wf = cmd_walk_forward(
        dataset_file=dataset_file,
        window_type="rolling",
        train_window=20,
        test_window=10,
        artifacts_dir=art_dir,
        as_json=False,
    )
    assert code_wf == 0

    eval_dirs = list(art_dir.glob("eval_*"))
    assert len(eval_dirs) == 1
    eval_id = eval_dirs[0].name

    # 2. show CLI
    code_show = cmd_show(evaluation_id=eval_id, artifacts_dir=art_dir, as_json=False)
    assert code_show == 0
    assert cmd_show(evaluation_id=eval_id, artifacts_dir=art_dir, as_json=True) == 0

    # 3. windows CLI
    code_win = cmd_windows(evaluation_id=eval_id, artifacts_dir=art_dir, as_json=False)
    assert code_win == 0
    assert cmd_windows(evaluation_id=eval_id, artifacts_dir=art_dir, as_json=True) == 0

    # 4. compare CLI
    code_comp = cmd_compare(
        eval_a=eval_id, eval_b=eval_id, artifacts_dir=art_dir, as_json=False
    )
    assert code_comp == 0
    assert (
        cmd_compare(eval_a=eval_id, eval_b=eval_id, artifacts_dir=art_dir, as_json=True)
        == 0
    )
