"""Unit tests for ML target generation, alignment, leakage prevention, and datasets."""

from datetime import datetime, timedelta, timezone

import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.targets.dataset import DatasetBuilder
from adaptive_trading.targets.definitions import (
    TARGET_VERSION,
    TargetConfig,
)
from adaptive_trading.targets.generator import TargetGenerator
from adaptive_trading.targets.models import DirectionLabel

UTC_TZ = timezone.utc


def make_test_candles(prices: list[float]) -> list[Candle]:
    """Helper to create sequential candles with specified close prices."""
    start_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    candles: list[Candle] = []

    for i, p in enumerate(prices):
        ts = start_time + timedelta(minutes=5 * i)
        candle = Candle(
            timestamp=ts,
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=p,
            high=p + 1.0,
            low=p - 1.0,
            close=p,
            volume=1000.0,
        )
        candles.append(candle)

    return candles


def test_future_return_calculation() -> None:
    """Test exact mathematical calculation of forward return."""
    # Prices: 100, 105, 110, 120 (4 candles, horizon=3)
    # Target at index 0: (120 - 100) / 100 = 0.20
    candles = make_test_candles([100.0, 105.0, 110.0, 120.0])
    generator = TargetGenerator(config=TargetConfig(horizon=3))
    targets = generator.generate_targets(candles)

    assert len(targets) == 1
    t0 = targets[0]
    assert t0.timestamp == candles[0].timestamp
    assert pytest.approx(t0.future_return, rel=1e-5) == 0.20
    assert t0.classification_label == DirectionLabel.UP
    assert t0.target_version == TARGET_VERSION


def test_binary_classification_thresholds() -> None:
    """Test binary classification logic with positive_threshold."""
    candles = make_test_candles([100.0, 100.0, 100.0, 102.0, 98.0, 100.0])
    # Horizon = 3:
    # index 0: close 100 -> 102 (+0.02) -> UP
    # index 1: close 100 -> 98 (-0.02) -> DOWN
    # index 2: close 100 -> 100 (0.00) -> DOWN (threshold=0.0)
    generator = TargetGenerator(config=TargetConfig(horizon=3, positive_threshold=0.0))
    targets = generator.generate_targets(candles)

    assert len(targets) == 3
    assert targets[0].classification_label == DirectionLabel.UP
    assert targets[1].classification_label == DirectionLabel.DOWN
    assert targets[2].classification_label == DirectionLabel.DOWN


def test_three_class_classification() -> None:
    """Test 3-class classification (UP/DOWN/NEUTRAL)."""
    candles = make_test_candles([100.0, 100.0, 100.0, 105.0, 95.0, 100.2])
    # Horizon = 3, pos=0.01 (+1%), neg=-0.01 (-1%):
    # index 0: 100 -> 105 (+5%) > +1% -> UP
    # index 1: 100 -> 95 (-5%) < -1% -> DOWN
    # index 2: 100 -> 100.2 (+0.2%) between -1% and +1% -> NEUTRAL
    config = TargetConfig(
        horizon=3,
        positive_threshold=0.01,
        negative_threshold=-0.01,
        use_neutral_class=True,
    )
    generator = TargetGenerator(config=config)
    targets = generator.generate_targets(candles)

    assert len(targets) == 3
    assert targets[0].classification_label == DirectionLabel.UP
    assert targets[1].classification_label == DirectionLabel.DOWN
    assert targets[2].classification_label == DirectionLabel.NEUTRAL


def test_horizon_configurability() -> None:
    """Test that changing horizon parameter adjusts the target window."""
    candles = make_test_candles([100.0, 102.0, 104.0, 106.0, 108.0])

    # Horizon = 1: 4 targets generated
    t_h1 = TargetGenerator(config=TargetConfig(horizon=1)).generate_targets(candles)
    assert len(t_h1) == 4
    assert pytest.approx(t_h1[0].future_return, rel=1e-5) == 0.02

    # Horizon = 4: 1 target generated
    t_h4 = TargetGenerator(config=TargetConfig(horizon=4)).generate_targets(candles)
    assert len(t_h4) == 1
    assert pytest.approx(t_h4[0].future_return, rel=1e-5) == 0.08


def test_final_rows_omitted() -> None:
    """Test that final h rows are never given fabricated targets."""
    candles = make_test_candles([100.0] * 10)
    horizon = 3
    targets = TargetGenerator(config=TargetConfig(horizon=horizon)).generate_targets(
        candles
    )

    # For 10 candles with horizon 3, exactly 7 targets must exist (0..6)
    assert len(targets) == 7
    # Prediction timestamps must match first 7 candles
    for i in range(7):
        assert targets[i].timestamp == candles[i].timestamp


def test_dataset_builder_and_dataframe() -> None:
    """Test building aligned training examples and pandas DataFrame."""
    prices = [100.0 + i for i in range(30)]
    candles = make_test_candles(prices)

    # Generate features (drop_warmup=True drops first 19 rows -> 11 feature vectors)
    feature_pipeline = FeaturePipeline()
    feature_vectors = feature_pipeline.generate_feature_vectors(
        candles, drop_warmup=True
    )

    # Generate targets (horizon=3 -> 27 targets)
    target_generator = TargetGenerator(config=TargetConfig(horizon=3))
    targets = target_generator.generate_targets(candles)

    # Build dataset (aligned on rows 19..26 = 8 aligned examples)
    builder = DatasetBuilder()
    examples = builder.build_training_examples(feature_vectors, targets)
    assert len(examples) == 8

    # Verify example properties
    ex0 = examples[0]
    assert ex0.timestamp == feature_vectors[0].timestamp
    assert "return_1" in ex0.features
    assert ex0.target.horizon == 3

    # Build DataFrame
    df = builder.build_training_dataframe(feature_vectors, targets)
    assert len(df) == 8
    assert "future_return" in df.columns
    assert "label" in df.columns
    assert "return_1" in df.columns
    assert "sma_20" in df.columns
    assert not df["future_return"].isna().any()
    assert not df["label"].isna().any()


def test_no_feature_leakage_with_targets() -> None:
    """Test that mutating future prices alters target at T, not features at T."""
    candles_orig = make_test_candles([100.0 + i for i in range(30)])

    pipeline = FeaturePipeline()
    vectors_orig = pipeline.generate_feature_vectors(candles_orig, drop_warmup=True)

    generator = TargetGenerator(config=TargetConfig(horizon=3))
    targets_orig = generator.generate_targets(candles_orig)

    # Clone candles and mutate future candles (index >= 25)
    candles_mutated = list(candles_orig)
    for i in range(25, 30):
        c = candles_orig[i]
        candles_mutated[i] = Candle(
            timestamp=c.timestamp,
            symbol=c.symbol,
            timeframe=c.timeframe,
            open=c.open * 5.0,
            high=c.high * 5.0,
            low=c.low * 5.0,
            close=c.close * 5.0,
            volume=c.volume,
        )

    vectors_mutated = pipeline.generate_feature_vectors(
        candles_mutated, drop_warmup=True
    )
    targets_mutated = generator.generate_targets(candles_mutated)

    # Check that feature vectors before index 25 are invariant
    for idx in range(len(vectors_orig)):
        if 19 + idx < 25:
            assert vectors_orig[idx].features == vectors_mutated[idx].features

    # Verify target at index 22 DID change to reflect future price
    assert targets_orig[22].future_return != targets_mutated[22].future_return, (
        "Target should reflect future price change!"
    )


def test_target_versioning_and_validation() -> None:
    """Test target versioning and timezone validation on Target model."""
    candles = make_test_candles([100.0, 101.0, 102.0, 103.0])
    generator = TargetGenerator(config=TargetConfig(target_version="v2"))
    targets = generator.generate_targets(candles)
    assert len(targets) == 1
    assert targets[0].target_version == "v2"

    # Test timezone awareness validator
    with pytest.raises(ValueError, match="timezone-aware"):
        from adaptive_trading.targets.models import Target

        Target(
            timestamp=datetime(2026, 1, 2, 9, 15),  # naive
            symbol="NIFTY",
            horizon=3,
            future_return=0.01,
            classification_label=DirectionLabel.UP,
            target_version="v1",
        )
