"""Unit tests for the Experiment, Dataset & Model Version Management subsystem."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.experiments.cli import (
    cmd_compare,
    cmd_list,
    cmd_manifest,
    cmd_show,
    cmd_verify,
)
from adaptive_trading.experiments.comparison import (
    ExperimentComparator,
)
from adaptive_trading.experiments.exceptions import (
    InvalidExperimentStateError,
)
from adaptive_trading.experiments.experiment import ExperimentRunner
from adaptive_trading.experiments.fingerprint import (
    compute_dict_fingerprint,
    compute_file_fingerprint,
)
from adaptive_trading.experiments.models import (
    ExperimentStatus,
    ExperimentType,
    ReproducibilityLevel,
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
from adaptive_trading.runtime.config import RuntimeConfig
from adaptive_trading.runtime.event_loop import (
    EventLoop,
    SimplePredictionService,
)
from adaptive_trading.runtime.events import MarketEvent

UTC_TZ = timezone.utc


def make_sample_candles(count: int = 10, base_price: float = 100.0) -> list[Candle]:
    """Helper to generate a deterministic list of test candles."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    return [
        Candle(
            timestamp=t0 + timedelta(minutes=5 * i),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=base_price + i,
            high=base_price + i + 2.0,
            low=base_price + i - 1.0,
            close=base_price + i + 1.0,
            volume=1000.0 + i * 50,
        )
        for i in range(count)
    ]


def test_fingerprint_determinism_and_dict_order() -> None:
    """Verify SHA-256 fingerprint stability and dictionary order invariance."""
    d1 = {"b": 2, "a": 1, "nested": {"y": "yes", "x": "no"}}
    d2 = {"a": 1, "b": 2, "nested": {"x": "no", "y": "yes"}}

    fp1 = compute_dict_fingerprint(d1)
    fp2 = compute_dict_fingerprint(d2)
    assert fp1 == fp2

    # Changing a value changes fingerprint
    d3 = {"a": 1, "b": 3, "nested": {"x": "no", "y": "yes"}}
    assert compute_dict_fingerprint(d3) != fp1

    # Sensitive credentials excluded/redacted
    d_secret = {"a": 1, "password": "supersecretpassword123"}
    fp_secret = compute_dict_fingerprint(d_secret)
    assert "supersecretpassword123" not in fp_secret


def test_dataset_fingerprint_and_versioning() -> None:
    """Verify DatasetVersion metadata generation and dataset fingerprinting."""
    candles1 = make_sample_candles(10, base_price=100.0)
    candles2 = make_sample_candles(10, base_price=100.0)
    candles_modified = make_sample_candles(10, base_price=105.0)

    ds1 = build_dataset_version(candles1, name="nifty_set1")
    ds2 = build_dataset_version(candles2, name="nifty_set2")
    ds_mod = build_dataset_version(candles_modified, name="nifty_mod")

    assert ds1.fingerprint == ds2.fingerprint
    assert ds1.fingerprint != ds_mod.fingerprint
    assert ds1.row_count == 10
    assert ds1.symbol == "NIFTY"
    assert ds1.source == "Angel One SmartAPI"
    assert ds1.exchange == "NSE"


def test_feature_versioning() -> None:
    """Verify FeatureVersion captures parameters and produces deterministic hash."""
    fv1 = build_feature_version(
        feature_set_name="tech_v1", parameters={"rsi_period": 14}
    )
    fv2 = build_feature_version(
        feature_set_name="tech_v1", parameters={"rsi_period": 14}
    )
    fv3 = build_feature_version(
        feature_set_name="tech_v1", parameters={"rsi_period": 21}
    )

    assert fv1.fingerprint == fv2.fingerprint
    assert fv1.fingerprint != fv3.fingerprint
    assert fv1.feature_version == "v1"


def test_model_versioning_and_artifact_fingerprint(tmp_path: Path) -> None:
    """Verify ModelVersion computes binary file SHA-256 fingerprint."""
    fake_artifact = tmp_path / "model.joblib"
    fake_artifact.write_bytes(b"FAKE_TRAINED_MODEL_BYTES_12345")

    mv = build_model_version(
        artifact_path=fake_artifact,
        model_name="logistic_regression",
        model_version="v1",
    )
    assert mv.artifact_fingerprint == compute_file_fingerprint(fake_artifact)
    assert mv.model_id == "logistic_regression_v1"
    assert len(mv.artifact_fingerprint) == 64


def test_strategy_risk_execution_runtime_versioning() -> None:
    """Verify versions for strategy, risk, execution (PAPER), and runtime."""
    sv = build_strategy_version(
        strategy_name="probability_strategy",
        parameters={"long_threshold": 0.60},
    )
    rv = build_risk_version()
    ev = build_execution_version()
    rtv = build_runtime_version()

    assert sv.strategy_name == "probability_strategy"
    assert ev.execution_mode == "PAPER"
    assert len(sv.fingerprint) == 64
    assert len(rv.fingerprint) == 64
    assert len(ev.fingerprint) == 64
    assert len(rtv.fingerprint) == 64


def test_environment_metadata_capture() -> None:
    """Verify capture_environment_metadata captures non-sensitive environment info."""
    env = capture_environment_metadata(random_seed=42)
    assert env.os_platform != ""
    assert env.python_version != ""
    assert env.application_version == "0.1.0"
    assert env.random_seed == 42


def test_experiment_manifest_construction_and_roundtrip() -> None:
    """Verify ExperimentManifest bundles contracts and serializes cleanly."""
    candles = make_sample_candles(15)
    ds = build_dataset_version(candles)
    fv = build_feature_version()
    mv = build_model_version()
    sv = build_strategy_version()
    rv = build_risk_version()
    ev = build_execution_version()
    rtv = build_runtime_version()

    manifest = build_experiment_manifest(
        experiment_id="exp_test_001",
        dataset=ds,
        features=fv,
        model=mv,
        strategy=sv,
        risk=rv,
        execution=ev,
        runtime=rtv,
    )
    assert manifest.experiment_id == "exp_test_001"
    assert len(manifest.manifest_fingerprint) == 64

    # JSON roundtrip
    dumped = manifest.model_dump(mode="json")
    json_str = json.dumps(dumped)
    loaded_dict = json.loads(json_str)
    reconstructed = manifest.__class__(**loaded_dict)

    assert reconstructed.experiment_id == manifest.experiment_id
    assert reconstructed.manifest_fingerprint == manifest.manifest_fingerprint
    assert reconstructed.dataset.fingerprint == manifest.dataset.fingerprint


def test_experiment_registry_lifecycle_and_persistence(tmp_path: Path) -> None:
    """Test full ExperimentRegistry lifecycle: CREATED -> RUNNING -> COMPLETED."""
    registry = ExperimentRegistry(base_dir=tmp_path)
    exp = registry.create_experiment(
        name="test_backtest",
        experiment_type=ExperimentType.BACKTEST,
        description="Backtest experiment",
    )
    assert exp.status == ExperimentStatus.CREATED
    assert (tmp_path / exp.experiment_id / "experiment.json").is_file()

    running = registry.start_experiment(exp.experiment_id)
    assert running.status == ExperimentStatus.RUNNING

    res = build_experiment_result(
        experiment_id=exp.experiment_id,
        result_data={
            "initial_equity": 100_000.0,
            "final_equity": 108_500.0,
            "net_pnl": 8_500.0,
            "trade_count": 4,
            "max_drawdown": 0.02,
        },
    )
    completed = registry.complete_experiment(exp.experiment_id, result=res)
    assert completed.status == ExperimentStatus.COMPLETED
    assert (tmp_path / exp.experiment_id / "result.json").is_file()

    retrieved = registry.get_experiment(exp.experiment_id)
    assert retrieved is not None
    assert retrieved.status == ExperimentStatus.COMPLETED

    listed = registry.list_experiments()
    assert len(listed) == 1
    assert listed[0].experiment_id == exp.experiment_id


def test_invalid_experiment_state_transitions(tmp_path: Path) -> None:
    """Verify invalid experiment transitions raise InvalidExperimentStateError."""
    registry = ExperimentRegistry(base_dir=tmp_path)
    exp = registry.create_experiment(
        name="invalid_exp", experiment_type=ExperimentType.ML_EXPERIMENT
    )

    with pytest.raises(InvalidExperimentStateError):
        # Cannot complete directly from CREATED
        registry.complete_experiment(exp.experiment_id)


def test_experiment_comparator_diff_detection() -> None:
    """Test ExperimentComparator detects config differences and performance deltas."""
    candles1 = make_sample_candles(10, base_price=100.0)
    candles2 = make_sample_candles(10, base_price=120.0)

    ds1 = build_dataset_version(candles1)
    ds2 = build_dataset_version(candles2)
    fv1 = build_feature_version(parameters={"rsi_period": 14})
    fv2 = build_feature_version(parameters={"rsi_period": 21})

    m1 = build_experiment_manifest(
        experiment_id="exp_A",
        dataset=ds1,
        features=fv1,
        model=build_model_version(model_version="v1"),
        strategy=build_strategy_version(),
        risk=build_risk_version(),
        execution=build_execution_version(),
        runtime=build_runtime_version(),
        result=build_experiment_result(
            "exp_A", {"final_equity": 105000.0, "net_pnl": 5000.0}
        ),
    )
    m2 = build_experiment_manifest(
        experiment_id="exp_B",
        dataset=ds2,
        features=fv2,
        model=build_model_version(model_version="v2"),
        strategy=build_strategy_version(),
        risk=build_risk_version(),
        execution=build_execution_version(),
        runtime=build_runtime_version(),
        result=build_experiment_result(
            "exp_B", {"final_equity": 110000.0, "net_pnl": 10000.0}
        ),
    )

    comp = ExperimentComparator.compare(m1, m2)
    assert not comp.is_identical_config
    assert not comp.is_identical_result
    assert "dataset" in comp.config_diffs
    assert "features" in comp.config_diffs
    assert "model" in comp.config_diffs
    assert comp.performance_diffs["delta_net_pnl"] == 5000.0
    assert comp.performance_diffs["delta_final_equity"] == 5000.0


def test_reproducibility_verification_levels() -> None:
    """Test EXACT, PARTIAL, and NOT_REPRODUCIBLE classifications."""
    candles = make_sample_candles(10)
    ds = build_dataset_version(candles)
    m_base = build_experiment_manifest(
        experiment_id="exp_orig",
        dataset=ds,
        features=build_feature_version(),
        model=build_model_version(),
        strategy=build_strategy_version(),
        risk=build_risk_version(),
        execution=build_execution_version(),
        runtime=build_runtime_version(),
    )

    # Identical copy -> EXACT
    level, reasons = ExperimentComparator.verify_reproducibility(m_base, m_base)
    assert level == ReproducibilityLevel.EXACT

    # Changed dataset -> NOT_REPRODUCIBLE
    ds_diff = build_dataset_version(make_sample_candles(10, base_price=200.0))
    m_diff_ds = m_base.model_copy(update={"dataset": ds_diff})
    level2, reasons2 = ExperimentComparator.verify_reproducibility(m_base, m_diff_ds)
    assert level2 == ReproducibilityLevel.NOT_REPRODUCIBLE

    # Changed only git commit -> PARTIAL
    env_diff = m_base.environment.model_copy(update={"git_commit": "1234567890abcdef"})
    m_diff_env = m_base.model_copy(update={"environment": env_diff})
    level3, reasons3 = ExperimentComparator.verify_reproducibility(m_base, m_diff_env)
    assert level3 == ReproducibilityLevel.PARTIAL


def test_version_compatibility_checker() -> None:
    """Verify compatibility flags mismatches when model expectations differ."""
    ds = build_dataset_version(make_sample_candles(5))
    fv = build_feature_version(feature_version="v1")
    mv = build_model_version(feature_version="v2")  # expects v2
    rtv = build_runtime_version(RuntimeConfig(warmup_period=20))

    manifest = build_experiment_manifest(
        experiment_id="exp_incompat",
        dataset=ds,
        features=fv,
        model=mv,
        strategy=build_strategy_version(),
        risk=build_risk_version(),
        execution=build_execution_version(),
        runtime=rtv,
    )
    issues = verify_compatibility(manifest)
    assert len(issues) >= 2  # Feature mismatch + insufficient rows


def test_experiment_runner_end_to_end_paper_replay(tmp_path: Path) -> None:
    """Test ExperimentRunner executes simulation replay, stores manifest and results."""
    candles = make_sample_candles(12)
    reg = ExperimentRegistry(base_dir=tmp_path)
    runner = ExperimentRunner(registry=reg)

    exp, manifest, res = runner.run_paper_replay_experiment(
        candles=candles,
        name="replay_test_exp",
        prediction_service=SimplePredictionService(fixed_probability_up=0.75),
        runtime_config=RuntimeConfig(warmup_period=5, initial_cash=100_000.0),
    )

    assert exp.status == ExperimentStatus.COMPLETED
    assert manifest.result is not None
    assert manifest.result.final_equity > 0.0
    assert (tmp_path / exp.experiment_id / "manifest.json").is_file()
    assert (tmp_path / exp.experiment_id / "result.json").is_file()


def test_no_behavioral_changes_with_experiment_tracking() -> None:
    """Verify that adding experiment tracking produces zero deviation in trading."""
    candles = make_sample_candles(12)

    # 1. Direct event loop run
    cfg = RuntimeConfig(warmup_period=5, initial_cash=100_000.0)
    loop = EventLoop(
        config=cfg,
        prediction_service=SimplePredictionService(fixed_probability_up=0.75),
    )
    for c in candles:
        loop.process_market_event(MarketEvent.from_candle(c))
    acct1 = loop.execution_service.broker.get_account()

    # 2. Managed ExperimentRunner run
    runner = ExperimentRunner()
    exp, manifest, res = runner.run_paper_replay_experiment(
        candles=candles,
        prediction_service=SimplePredictionService(fixed_probability_up=0.75),
        runtime_config=cfg,
    )

    assert acct1.equity == res.final_equity
    assert round(acct1.equity - 100_000.0, 4) == round(res.net_pnl, 4)
    assert loop.state.stats.orders_filled == res.trade_count


def test_experiments_cli_commands(tmp_path: Path) -> None:
    """Test all experiments CLI commands: list, show, manifest, compare, verify."""
    reg = ExperimentRegistry(base_dir=tmp_path)
    runner = ExperimentRunner(registry=reg)
    candles = make_sample_candles(10)

    exp1, m1, r1 = runner.run_paper_replay_experiment(
        candles=candles,
        name="cli_exp1",
        prediction_service=SimplePredictionService(fixed_probability_up=0.75),
        runtime_config=RuntimeConfig(warmup_period=5),
    )
    exp2, m2, r2 = runner.run_paper_replay_experiment(
        candles=candles,
        name="cli_exp2",
        prediction_service=SimplePredictionService(fixed_probability_up=0.25),
        runtime_config=RuntimeConfig(warmup_period=5),
    )

    assert cmd_list(base_dir=tmp_path, as_json=False) == 0
    assert cmd_list(base_dir=tmp_path, as_json=True) == 0

    assert (
        cmd_show(experiment_id=exp1.experiment_id, base_dir=tmp_path, as_json=False)
        == 0
    )
    assert (
        cmd_show(experiment_id=exp1.experiment_id, base_dir=tmp_path, as_json=True) == 0
    )

    assert (
        cmd_manifest(experiment_id=exp1.experiment_id, base_dir=tmp_path, as_json=False)
        == 0
    )

    assert (
        cmd_compare(
            exp_a_id=exp1.experiment_id,
            exp_b_id=exp2.experiment_id,
            base_dir=tmp_path,
            as_json=False,
        )
        == 0
    )
    assert (
        cmd_compare(
            exp_a_id=exp1.experiment_id,
            exp_b_id=exp2.experiment_id,
            base_dir=tmp_path,
            as_json=True,
        )
        == 0
    )

    assert (
        cmd_verify(
            experiment_id=exp1.experiment_id,
            target_exp_id=exp2.experiment_id,
            base_dir=tmp_path,
            as_json=False,
        )
        == 0
    )
    assert (
        cmd_verify(
            experiment_id=exp1.experiment_id,
            target_exp_id=exp2.experiment_id,
            base_dir=tmp_path,
            as_json=True,
        )
        == 0
    )
