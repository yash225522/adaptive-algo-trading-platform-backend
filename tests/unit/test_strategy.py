"""Unit tests for Strategy Engine, thresholds, signals, and CLI."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from adaptive_trading.domain.prediction import Prediction
from adaptive_trading.ml.validation.results import OOSPrediction
from adaptive_trading.strategy.cli import run_strategy_generation
from adaptive_trading.strategy.config import StrategyConfig
from adaptive_trading.strategy.engine import StrategyEngine
from adaptive_trading.strategy.exceptions import (
    DuplicateSignalError,
    StrategyConfigError,
)
from adaptive_trading.strategy.models import (
    SignalAction,
    StrategyPrediction,
    TradingSignal,
)
from adaptive_trading.strategy.rules import ProbabilityStrategy

UTC_TZ = timezone.utc


@pytest.fixture
def sample_strategy_predictions() -> list[StrategyPrediction]:
    """Create deterministic chronological prediction sequence."""
    base_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    probs = [0.72, 0.50, 0.27, 0.60, 0.40, 0.85, 0.15]
    preds: list[StrategyPrediction] = []

    for i, prob in enumerate(probs):
        preds.append(
            StrategyPrediction(
                timestamp=base_time + timedelta(minutes=5 * i),
                symbol="NIFTY",
                predicted_class="UP" if prob >= 0.5 else "DOWN",
                probability_up=prob,
                model_name="logistic_regression",
                model_version="v1",
            )
        )
    return preds


def test_strategy_config_valid() -> None:
    """Test valid StrategyConfig creation."""
    config = StrategyConfig(
        strategy_name="test_strat",
        strategy_version="v2",
        long_probability_threshold=0.65,
        short_probability_threshold=0.35,
    )
    assert config.strategy_name == "test_strat"
    assert config.long_probability_threshold == 0.65
    assert config.short_probability_threshold == 0.35


def test_strategy_config_invalid_thresholds() -> None:
    """Test invalid threshold bounds and relations are rejected."""
    # Reversed thresholds (short > long)
    with pytest.raises(StrategyConfigError, match="must be strictly less than"):
        StrategyConfig(
            long_probability_threshold=0.40,
            short_probability_threshold=0.60,
        )

    # Equal thresholds (short == long)
    with pytest.raises(StrategyConfigError, match="must be strictly less than"):
        StrategyConfig(
            long_probability_threshold=0.50,
            short_probability_threshold=0.50,
        )

    # Value out of [0.0, 1.0] bounds
    with pytest.raises(ValueError):
        StrategyConfig(long_probability_threshold=1.5)

    with pytest.raises(ValueError):
        StrategyConfig(short_probability_threshold=-0.1)


def test_probability_strategy_threshold_decisions() -> None:
    """Test standard decision logic for clear LONG, SHORT, and NO_TRADE cases."""
    config = StrategyConfig(
        long_probability_threshold=0.60,
        short_probability_threshold=0.40,
    )
    strategy = ProbabilityStrategy(config=config)

    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    # 1. Clear LONG (prob = 0.72)
    p_long = StrategyPrediction(
        timestamp=ts,
        symbol="NIFTY",
        probability_up=0.72,
    )
    sig_long = strategy.generate_signal(p_long)
    assert sig_long.action == SignalAction.LONG
    assert sig_long.confidence == pytest.approx(0.72)
    assert "LONG: probability_up" in sig_long.reason

    # 2. Clear SHORT (prob = 0.27)
    p_short = StrategyPrediction(
        timestamp=ts,
        symbol="NIFTY",
        probability_up=0.27,
    )
    sig_short = strategy.generate_signal(p_short)
    assert sig_short.action == SignalAction.SHORT
    assert sig_short.confidence == pytest.approx(0.73)  # 1.0 - 0.27
    assert "SHORT: probability_up" in sig_short.reason

    # 3. Neutral NO_TRADE (prob = 0.51)
    p_neutral = StrategyPrediction(
        timestamp=ts,
        symbol="NIFTY",
        probability_up=0.51,
    )
    sig_neutral = strategy.generate_signal(p_neutral)
    assert sig_neutral.action == SignalAction.NO_TRADE
    assert sig_neutral.confidence == 0.0
    assert "NO_TRADE: probability_up" in sig_neutral.reason


def test_probability_strategy_boundary_exact_values() -> None:
    """Test exact threshold boundary cases (inclusive boundaries)."""
    config = StrategyConfig(
        long_probability_threshold=0.60,
        short_probability_threshold=0.40,
    )
    strategy = ProbabilityStrategy(config=config)
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    # Exact upper boundary (0.60 -> LONG)
    sig_exact_long = strategy.generate_signal(
        StrategyPrediction(timestamp=ts, symbol="NIFTY", probability_up=0.60)
    )
    assert sig_exact_long.action == SignalAction.LONG
    assert sig_exact_long.confidence == pytest.approx(0.60)

    # Exact lower boundary (0.40 -> SHORT)
    sig_exact_short = strategy.generate_signal(
        StrategyPrediction(timestamp=ts, symbol="NIFTY", probability_up=0.40)
    )
    assert sig_exact_short.action == SignalAction.SHORT
    assert sig_exact_short.confidence == pytest.approx(0.60)  # 1.0 - 0.40


def test_strategy_prediction_constructors() -> None:
    """Test constructor adapters from Prediction and OOSPrediction."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    # From Domain Prediction
    domain_pred = Prediction(
        timestamp=ts,
        symbol="NIFTY",
        model_version="v1",
        probability_up=0.75,
        probability_down=0.25,
        expected_return=0.005,
    )
    sp1 = StrategyPrediction.from_prediction(
        domain_pred, model_name="logistic_regression"
    )
    assert sp1.timestamp == ts
    assert sp1.symbol == "NIFTY"
    assert sp1.probability_up == 0.75
    assert sp1.model_name == "logistic_regression"
    assert sp1.model_version == "v1"

    # From Walk-Forward OOSPrediction
    oos_pred = OOSPrediction(
        timestamp=ts,
        symbol="BANKNIFTY",
        actual_label=1,
        predicted_label=1,
        probability_up=0.68,
        fold_number=2,
    )
    sp2 = StrategyPrediction.from_oos_prediction(oos_pred)
    assert sp2.timestamp == ts
    assert sp2.symbol == "BANKNIFTY"
    assert sp2.probability_up == 0.68

    # Reject naive timestamp
    naive_ts = datetime(2026, 1, 2, 9, 15)
    with pytest.raises(ValueError, match="must be timezone-aware"):
        StrategyPrediction(
            timestamp=naive_ts,
            symbol="NIFTY",
            probability_up=0.50,
        )


def test_strategy_engine_batch_processing(
    sample_strategy_predictions: list[StrategyPrediction],
) -> None:
    """Test StrategyEngine batch processing and ordering."""
    engine = StrategyEngine()
    signals = engine.generate_signals(sample_strategy_predictions)

    assert len(signals) == len(sample_strategy_predictions)
    assert isinstance(signals[0], TradingSignal)

    # Verify chronological ordering
    for i in range(len(signals) - 1):
        assert signals[i].timestamp < signals[i + 1].timestamp

    # Verify distribution
    # probs = [0.72(LONG), 0.50(NO_TRADE), 0.27(SHORT), 0.60(LONG),
    #          0.40(SHORT), 0.85(LONG), 0.15(SHORT)]
    actions = [s.action for s in signals]
    assert actions == [
        SignalAction.LONG,
        SignalAction.NO_TRADE,
        SignalAction.SHORT,
        SignalAction.LONG,
        SignalAction.SHORT,
        SignalAction.LONG,
        SignalAction.SHORT,
    ]


def test_strategy_engine_duplicate_rejection() -> None:
    """Test duplicate (timestamp, symbol) predictions are rejected."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    duplicates = [
        StrategyPrediction(timestamp=ts, symbol="NIFTY", probability_up=0.70),
        StrategyPrediction(timestamp=ts, symbol="NIFTY", probability_up=0.80),
    ]
    engine = StrategyEngine()
    with pytest.raises(DuplicateSignalError, match="Duplicate prediction timestamp"):
        engine.generate_signals(duplicates)


def test_strategy_engine_signal_statistics(
    sample_strategy_predictions: list[StrategyPrediction],
) -> None:
    """Test signal statistics calculations."""
    engine = StrategyEngine()
    signals = engine.generate_signals(sample_strategy_predictions)
    stats = engine.calculate_statistics(signals)

    assert stats.total_predictions == 7
    assert stats.long_signals == 3
    assert stats.short_signals == 3
    assert stats.no_trade_signals == 1
    assert stats.long_percentage == pytest.approx(42.86, abs=0.01)
    assert stats.short_percentage == pytest.approx(42.86, abs=0.01)
    assert stats.no_trade_percentage == pytest.approx(14.29, abs=0.01)


def test_strategy_engine_empty_batch() -> None:
    """Test empty batch produces empty list and zero statistics."""
    engine = StrategyEngine()
    signals = engine.generate_signals([])
    assert signals == []
    stats = engine.calculate_statistics(signals)
    assert stats.total_predictions == 0
    assert stats.long_percentage == 0.0


def test_no_future_leakage_in_strategy(
    sample_strategy_predictions: list[StrategyPrediction],
) -> None:
    """Verify modifying future prediction items does NOT change previous signals."""
    engine = StrategyEngine()

    # Original signal for first item
    sig_0_orig = engine.generate_signal(sample_strategy_predictions[0])

    # Corrupt future items
    corrupted_predictions = [sample_strategy_predictions[0]] + [
        StrategyPrediction(
            timestamp=sample_strategy_predictions[i].timestamp,
            symbol="NIFTY",
            probability_up=0.99,
        )
        for i in range(1, len(sample_strategy_predictions))
    ]

    signals_mod = engine.generate_signals(corrupted_predictions)
    assert signals_mod[0].action == sig_0_orig.action
    assert signals_mod[0].confidence == sig_0_orig.confidence
    assert signals_mod[0].reason == sig_0_orig.reason


def test_strategy_cli_execution(tmp_path: Path) -> None:
    """Test CLI strategy signal generation and CSV artifact creation."""
    # Write dummy predictions CSV
    csv_file = tmp_path / "predictions.csv"
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02 09:15:00+00:00",
                "2026-01-02 09:20:00+00:00",
                "2026-01-02 09:25:00+00:00",
            ],
            "symbol": ["NIFTY", "NIFTY", "NIFTY"],
            "probability_up": [0.75, 0.50, 0.25],
            "predicted_label": [1, 1, 0],
            "model_name": [
                "logistic_regression",
                "logistic_regression",
                "logistic_regression",
            ],
            "model_version": ["v1", "v1", "v1"],
        }
    )
    df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "output_signals"
    exit_code = run_strategy_generation(
        predictions_file=csv_file,
        long_threshold=0.60,
        short_threshold=0.40,
        output_dir=out_dir,
        save_artifact=True,
    )

    assert exit_code == 0
    saved_csv = out_dir / "signals.csv"
    assert saved_csv.is_file()

    signals_df = pd.read_csv(saved_csv)
    assert len(signals_df) == 3
    assert list(signals_df["action"]) == ["LONG", "NO_TRADE", "SHORT"]
    assert "confidence" in signals_df.columns
    assert "reason" in signals_df.columns
