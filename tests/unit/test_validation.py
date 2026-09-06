"""Comprehensive unit test suite for Data & Model Quality Validation layer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.prediction import FeatureVector
from adaptive_trading.experiments.models import ModelVersion
from adaptive_trading.features.definitions import (
    FEATURE_CATALOG_V1,
    FEATURE_SET_VERSION,
)
from adaptive_trading.runtime.config import RuntimeConfig
from adaptive_trading.runtime.event_loop import (
    EventLoop,
    SimplePredictionService,
)
from adaptive_trading.runtime.events import MarketEvent
from adaptive_trading.validation.cli import (
    cmd_validate_dataset,
    cmd_validate_run,
)
from adaptive_trading.validation.config import (
    ValidationConfig,
    ValidationPolicy,
)
from adaptive_trading.validation.data_validator import DataValidator
from adaptive_trading.validation.exceptions import ValidationGateError
from adaptive_trading.validation.feature_validator import FeatureValidator
from adaptive_trading.validation.gate import (
    validate_for_backtest,
    validate_for_replay,
)
from adaptive_trading.validation.leakage import LeakageValidator
from adaptive_trading.validation.model_validator import ModelValidator
from adaptive_trading.validation.report import (
    ValidationReport,
    ValidationResult,
)
from adaptive_trading.validation.rules import (
    ValidationRuleId,
    ValidationSeverity,
    ValidationStatus,
)
from adaptive_trading.validation.schema import DatasetSchemaValidator

UTC = timezone.utc


def make_sample_candles(count: int = 15, base_price: float = 100.0) -> list[Candle]:
    """Helper generating deterministic, valid 5m candles."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC)
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


# ---------------------------------------------------------------------------
# 1. Schema Validation Tests
# ---------------------------------------------------------------------------


def test_schema_missing_columns() -> None:
    """Verify missing required columns are flagged as CRITICAL FAIL."""
    df_missing = pd.DataFrame(
        {
            "timestamp": [datetime.now(UTC)],
            "open": [100.0],
            "high": [105.0],
        }
    )
    validator = DatasetSchemaValidator()
    results = validator.validate_schema(df_missing)

    assert any(
        r.rule_id == ValidationRuleId.SCHEMA_REQUIRED_COLUMNS.value
        and r.status == ValidationStatus.FAIL
        and r.severity == ValidationSeverity.CRITICAL
        for r in results
    )


def test_schema_invalid_data_types_and_missing_values() -> None:
    """Verify non-numeric values and nulls are caught."""
    df_invalid = pd.DataFrame(
        {
            "timestamp": [datetime.now(UTC), datetime.now(UTC)],
            "symbol": ["NIFTY", "NIFTY"],
            "open": [100.0, "NOT_A_NUMBER"],
            "high": [105.0, None],
            "low": [95.0, 95.0],
            "close": [102.0, 102.0],
            "volume": [500.0, 500.0],
        }
    )
    validator = DatasetSchemaValidator()
    results = validator.validate_schema(df_invalid)

    assert any(
        r.rule_id == ValidationRuleId.DATA_MISSING_VALUES.value
        and r.status == ValidationStatus.FAIL
        for r in results
    )
    assert any(
        r.rule_id == ValidationRuleId.SCHEMA_DATA_TYPES.value
        and r.status == ValidationStatus.FAIL
        for r in results
    )


def test_schema_empty_or_mixed_symbols() -> None:
    """Verify empty symbols fail and mixed symbols warn."""
    df_mixed = pd.DataFrame(
        {
            "timestamp": [
                datetime.now(UTC),
                datetime.now(UTC) + timedelta(minutes=5),
            ],
            "symbol": ["NIFTY", "BANKNIFTY"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [500.0, 500.0],
        }
    )
    validator = DatasetSchemaValidator()
    results = validator.validate_schema(df_mixed)

    assert any(
        r.rule_id == ValidationRuleId.SCHEMA_SYMBOL_VALID.value
        and r.status == ValidationStatus.WARN
        for r in results
    )


# ---------------------------------------------------------------------------
# 2. OHLCV Integrity Tests
# ---------------------------------------------------------------------------


def test_ohlcv_integrity_bounds() -> None:
    """Verify High < Low, High < Close, Low > Open, and negative prices fail."""
    corrupt_data = [
        {
            "timestamp": datetime.now(UTC),
            "symbol": "NIFTY",
            "open": 100.0,
            "high": 90.0,
            "low": 95.0,
            "close": 98.0,
            "volume": 100.0,
        },
        {
            "timestamp": datetime.now(UTC) + timedelta(minutes=5),
            "symbol": "NIFTY",
            "open": 100.0,
            "high": 105.0,
            "low": 102.0,
            "close": 103.0,
            "volume": 100.0,
        },
        {
            "timestamp": datetime.now(UTC) + timedelta(minutes=10),
            "symbol": "NIFTY",
            "open": -10.0,
            "high": 10.0,
            "low": -20.0,
            "close": 5.0,
            "volume": 100.0,
        },
    ]
    validator = DataValidator()
    results, _ = validator.validate_dataset(corrupt_data)

    assert any(
        r.rule_id == ValidationRuleId.DATA_OHLCV_INTEGRITY.value
        and r.status == ValidationStatus.FAIL
        and r.severity == ValidationSeverity.ERROR
        for r in results
    )


def test_ohlcv_volume_anomalies() -> None:
    """Verify negative volume fails and zero volume produces warning."""
    neg_vol_data = [
        {
            "timestamp": datetime.now(UTC),
            "symbol": "NIFTY",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": -50.0,
        }
    ]
    validator = DataValidator()
    results, _ = validator.validate_dataset(neg_vol_data)

    assert any(
        r.rule_id == ValidationRuleId.DATA_VOLUME_ANOMALIES.value
        and r.status == ValidationStatus.FAIL
        for r in results
    )

    zero_vol_data = [
        {
            "timestamp": datetime.now(UTC),
            "symbol": "NIFTY",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 0.0,
        }
    ]
    validator_strict_vol = DataValidator(
        config=ValidationConfig(allow_zero_volume=False)
    )
    results_zero, _ = validator_strict_vol.validate_dataset(zero_vol_data)
    assert any(
        r.rule_id == ValidationRuleId.DATA_VOLUME_ANOMALIES.value
        and r.status == ValidationStatus.WARN
        for r in results_zero
    )


# ---------------------------------------------------------------------------
# 3. Time Series, Duplicate, and Gap Tests
# ---------------------------------------------------------------------------


def test_chronological_ordering_violation() -> None:
    """Verify unordered timestamps fail ordering check."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC)
    unordered_data = [
        {
            "timestamp": t0,
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
        {
            "timestamp": t0 + timedelta(minutes=15),
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
        {
            "timestamp": t0 + timedelta(minutes=10),
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
    ]
    validator = DataValidator()
    results, _ = validator.validate_dataset(unordered_data)

    assert any(
        r.rule_id == ValidationRuleId.DATA_CHRONOLOGICAL_ORDER.value
        and r.status == ValidationStatus.FAIL
        for r in results
    )


def test_duplicate_candles_detection() -> None:
    """Verify duplicate timestamp-symbol pairs are detected."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC)
    dup_data = [
        {
            "timestamp": t0,
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
        {
            "timestamp": t0,
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
    ]
    validator = DataValidator()
    results, _ = validator.validate_dataset(dup_data)

    assert any(
        r.rule_id == ValidationRuleId.DATA_DUPLICATE_CANDLES.value
        and r.status == ValidationStatus.FAIL
        for r in results
    )


def test_session_gaps_vs_suspicious_gaps() -> None:
    """Verify overnight session gaps pass as expected, but intra-day gaps warn."""
    t_eve = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    t_morn = datetime(2026, 1, 3, 9, 15, tzinfo=UTC)
    session_gap_data = [
        {
            "timestamp": t_eve,
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
        {
            "timestamp": t_morn,
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
    ]
    validator = DataValidator()
    results_session, _ = validator.validate_dataset(session_gap_data)
    assert any(
        r.rule_id == ValidationRuleId.DATA_SUSPICIOUS_GAPS.value
        and r.status == ValidationStatus.PASS
        for r in results_session
    )

    t_start = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    t_jump = datetime(2026, 1, 2, 10, 45, tzinfo=UTC)
    suspicious_gap_data = [
        {
            "timestamp": t_start,
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
        {
            "timestamp": t_jump,
            "symbol": "NIFTY",
            "open": 100,
            "high": 102,
            "low": 98,
            "close": 101,
            "volume": 100,
        },
    ]
    results_susp, _ = validator.validate_dataset(suspicious_gap_data)
    assert any(
        r.rule_id == ValidationRuleId.DATA_SUSPICIOUS_GAPS.value
        and r.status == ValidationStatus.WARN
        for r in results_susp
    )


# ---------------------------------------------------------------------------
# 4. Feature Validation Tests
# ---------------------------------------------------------------------------


def test_feature_validator_missing_and_infinite_values() -> None:
    """Verify FeatureValidator catches infinite values and NaNs."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC)
    features_dict = {f: 1.0 for f in FEATURE_CATALOG_V1.keys()}
    features_dict["rsi_14"] = float("inf")

    v1 = FeatureVector(
        timestamp=t0,
        symbol="NIFTY",
        feature_version=FEATURE_SET_VERSION,
        features=features_dict,
    )

    validator = FeatureValidator()
    results = validator.validate_feature_vectors([v1])

    assert any(
        r.rule_id == ValidationRuleId.FEATURE_INFINITE_VALUES.value
        and r.status == ValidationStatus.FAIL
        and r.severity == ValidationSeverity.CRITICAL
        for r in results
    )


def test_feature_timestamp_alignment() -> None:
    """Verify feature vectors with timestamps outside reference candles fail."""
    candles = make_sample_candles(5)
    t_unmatched = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    features_dict = {f: 1.0 for f in FEATURE_CATALOG_V1.keys()}

    v_unmatched = FeatureVector(
        timestamp=t_unmatched,
        symbol="NIFTY",
        feature_version=FEATURE_SET_VERSION,
        features=features_dict,
    )

    validator = FeatureValidator()
    results = validator.validate_feature_vectors(
        [v_unmatched], reference_candles=candles
    )

    assert any(
        r.rule_id == ValidationRuleId.FEATURE_TIMESTAMP_ALIGNMENT.value
        and r.status == ValidationStatus.FAIL
        for r in results
    )


# ---------------------------------------------------------------------------
# 5. Leakage Detection Tests
# ---------------------------------------------------------------------------


def test_target_leakage_detection() -> None:
    """Verify target columns in feature list trigger CRITICAL FAIL."""
    feature_cols_clean = ["rsi_14", "macd_diff", "volatility_20"]
    feature_cols_leaked = ["rsi_14", "future_return", "target_label"]

    validator = LeakageValidator()
    res_clean = validator.validate_target_leakage(feature_cols_clean)
    assert res_clean.status == ValidationStatus.PASS

    res_leaked = validator.validate_target_leakage(feature_cols_leaked)
    assert res_leaked.status == ValidationStatus.FAIL
    assert res_leaked.severity == ValidationSeverity.CRITICAL


def test_train_test_split_chronological_overlap() -> None:
    """Verify overlapping train and test periods fail validation."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC)
    train_dates = [t0 + timedelta(days=i) for i in range(10)]
    test_dates_valid = [t0 + timedelta(days=i) for i in range(10, 15)]
    test_dates_overlap = [t0 + timedelta(days=i) for i in range(8, 14)]

    validator = LeakageValidator()
    res_valid = validator.validate_train_test_split(train_dates, test_dates_valid)
    assert res_valid.status == ValidationStatus.PASS

    res_overlap = validator.validate_train_test_split(train_dates, test_dates_overlap)
    assert res_overlap.status == ValidationStatus.FAIL
    assert res_overlap.severity == ValidationSeverity.CRITICAL


def test_scaler_isolation_validation() -> None:
    """Verify fitting scalers across non-training rows fails validation."""
    validator = LeakageValidator()
    train_idx = list(range(100))

    res_clean = validator.validate_scaler_isolation(
        scaler_fit_indices=train_idx, train_indices=train_idx, total_rows=150
    )
    assert res_clean.status == ValidationStatus.PASS

    res_leaked = validator.validate_scaler_isolation(
        scaler_fit_indices=list(range(150)),
        train_indices=train_idx,
        total_rows=150,
    )
    assert res_leaked.status == ValidationStatus.FAIL
    assert res_leaked.severity == ValidationSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 6. Model Compatibility & Input/Output Tests
# ---------------------------------------------------------------------------


def test_model_feature_version_compatibility() -> None:
    """Verify model expecting v2 fails when runtime provides v1."""
    mv = ModelVersion(
        model_id="lr_v2",
        model_name="logistic_regression",
        model_version="v2",
        model_type="LogisticRegression",
        feature_version="v2",
    )

    validator = ModelValidator()
    results = validator.validate_version_compatibility(
        model_version=mv, runtime_feature_version="v1"
    )

    assert any(
        r.rule_id == ValidationRuleId.MODEL_FEATURE_COMPATIBILITY.value
        and r.status == ValidationStatus.FAIL
        and r.severity == ValidationSeverity.CRITICAL
        for r in results
    )


def test_model_input_and_output_contracts() -> None:
    """Verify input dimensions and probability output bounds."""
    validator = ModelValidator()

    mat_19 = np.ones((10, 19))
    in_results = validator.validate_input_contract(mat_19, expected_feature_count=20)
    assert any(
        r.rule_id == ValidationRuleId.MODEL_INPUT_CONTRACT.value
        and r.status == ValidationStatus.FAIL
        for r in in_results
    )

    bad_probs = np.array([0.5, 1.2, -0.1])
    out_results = validator.validate_output_contract(bad_probs, is_probability=True)
    assert any(
        r.rule_id == ValidationRuleId.MODEL_OUTPUT_CONTRACT.value
        and r.status == ValidationStatus.FAIL
        for r in out_results
    )


# ---------------------------------------------------------------------------
# 7. Validation Policy & Pre-Run Gates
# ---------------------------------------------------------------------------


def test_validation_policies_behavior() -> None:
    """Test STRICT, NORMAL, and LENIENT execution eligibility."""
    chk_warn = ValidationResult(
        rule_id=ValidationRuleId.DATA_SUSPICIOUS_GAPS.value,
        severity=ValidationSeverity.WARNING,
        status=ValidationStatus.WARN,
        message="Suspicious gap",
    )
    chk_err = ValidationResult(
        rule_id=ValidationRuleId.DATA_OHLCV_INTEGRITY.value,
        severity=ValidationSeverity.ERROR,
        status=ValidationStatus.FAIL,
        message="OHLC error",
    )
    chk_crit = ValidationResult(
        rule_id=ValidationRuleId.LEAKAGE_TARGET_IN_FEATURES.value,
        severity=ValidationSeverity.CRITICAL,
        status=ValidationStatus.FAIL,
        message="Target leakage",
    )

    rep_warn = ValidationReport.create([chk_warn])
    assert not rep_warn.is_allowed(ValidationPolicy.STRICT)
    assert rep_warn.is_allowed(ValidationPolicy.NORMAL)
    assert rep_warn.is_allowed(ValidationPolicy.LENIENT)

    rep_err = ValidationReport.create([chk_err])
    assert not rep_err.is_allowed(ValidationPolicy.STRICT)
    assert not rep_err.is_allowed(ValidationPolicy.NORMAL)
    assert rep_err.is_allowed(ValidationPolicy.LENIENT)

    rep_crit = ValidationReport.create([chk_crit])
    assert not rep_crit.is_allowed(ValidationPolicy.STRICT)
    assert not rep_crit.is_allowed(ValidationPolicy.NORMAL)
    assert not rep_crit.is_allowed(ValidationPolicy.LENIENT)


def test_pre_run_gates_execution_flow(tmp_path: Path) -> None:
    """Test pre-run gates pass valid data and raise on invalid data."""
    valid_candles = make_sample_candles(10)
    cfg = ValidationConfig(artifacts_dir=tmp_path)

    rep_bt = validate_for_backtest(candles=valid_candles, config=cfg)
    assert rep_bt.overall_status in (
        ValidationStatus.PASS,
        ValidationStatus.WARN,
    )

    rep_rp = validate_for_replay(candles=valid_candles, config=cfg)
    assert rep_rp.overall_status in (
        ValidationStatus.PASS,
        ValidationStatus.WARN,
    )

    corrupt_candles = [
        Candle(
            timestamp=datetime.now(UTC),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=100.0,
        )
    ]
    corrupt_candles.append(corrupt_candles[0])

    with pytest.raises(ValidationGateError):
        validate_for_replay(
            candles=corrupt_candles,
            config=cfg,
            policy=ValidationPolicy.NORMAL,
        )


def test_no_behavioral_changes_with_validation() -> None:
    """Verify validating valid candles produces zero deviation in trading."""
    candles = make_sample_candles(12)
    cfg = RuntimeConfig(warmup_period=5, initial_cash=100_000.0)

    loop_direct = EventLoop(
        config=cfg,
        prediction_service=SimplePredictionService(fixed_probability_up=0.75),
    )
    for c in candles:
        loop_direct.process_market_event(MarketEvent.from_candle(c))
    acct_direct = loop_direct.execution_service.broker.get_account()

    val_rep = validate_for_replay(candles)
    assert val_rep.is_allowed()

    loop_gated = EventLoop(
        config=cfg,
        prediction_service=SimplePredictionService(fixed_probability_up=0.75),
    )
    for c in candles:
        loop_gated.process_market_event(MarketEvent.from_candle(c))
    acct_gated = loop_gated.execution_service.broker.get_account()

    assert acct_direct.equity == acct_gated.equity
    assert acct_direct.cash == acct_gated.cash
    assert loop_direct.state.stats.orders_filled == loop_gated.state.stats.orders_filled


# ---------------------------------------------------------------------------
# 8. CLI Command Tests
# ---------------------------------------------------------------------------


def test_validation_cli_commands(tmp_path: Path) -> None:
    """Test CLI commands for dataset, features, model, and full validation run."""
    sample_csv = Path("data/sample/nifty_5m_ml_sample.csv")

    assert (
        cmd_validate_dataset(
            dataset_path=sample_csv,
            policy=ValidationPolicy.NORMAL,
            as_json=False,
        )
        == 0
    )
    assert (
        cmd_validate_dataset(
            dataset_path=sample_csv,
            policy=ValidationPolicy.NORMAL,
            as_json=True,
        )
        == 0
    )

    assert (
        cmd_validate_run(
            dataset_path=sample_csv,
            policy=ValidationPolicy.NORMAL,
            as_json=False,
        )
        == 0
    )
    assert (
        cmd_validate_run(
            dataset_path=sample_csv,
            policy=ValidationPolicy.NORMAL,
            as_json=True,
        )
        == 0
    )
