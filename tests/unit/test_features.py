"""Unit tests for feature definitions, calculation, validation, and pipeline."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.prediction import FeatureVector
from adaptive_trading.features.calculator import FeatureCalculator
from adaptive_trading.features.definitions import (
    FEATURE_CATALOG_V1,
    FEATURE_SET_VERSION,
)
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.features.validators import (
    FeatureValidationError,
    FeatureValidator,
)

UTC_TZ = timezone.utc


def make_synthetic_candles(
    count: int = 30,
    start_price: float = 100.0,
    price_step: float = 1.0,
    volume: float = 1000.0,
    with_oi: bool = True,
) -> list[Candle]:
    """Generate a deterministic sequence of valid 5m candles."""
    start_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    candles: list[Candle] = []

    for i in range(count):
        ts = start_time + timedelta(minutes=5 * i)
        close = start_price + i * price_step
        open_p = close - 0.5
        high = close + 1.0
        low = close - 1.0
        oi = 50000.0 + i * 100.0 if with_oi else None

        candle = Candle(
            timestamp=ts,
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=open_p,
            high=high,
            low=low,
            close=close,
            volume=volume + (i % 3) * 100.0,
            open_interest=oi,
        )
        candles.append(candle)

    return candles


def test_feature_catalog_metadata() -> None:
    """Test feature catalog integrity and version constant."""
    assert FEATURE_SET_VERSION == "v1"
    assert "return_1" in FEATURE_CATALOG_V1
    assert "sma_20" in FEATURE_CATALOG_V1
    assert "volatility_10" in FEATURE_CATALOG_V1
    assert "volume_ratio_10" in FEATURE_CATALOG_V1
    assert "vwap_distance" in FEATURE_CATALOG_V1


def test_return_calculations() -> None:
    """Test mathematical accuracy of 1, 3, and 6-period returns."""
    candles = make_synthetic_candles(count=10, start_price=100.0, price_step=10.0)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(calculator.candles_to_dataframe(candles))

    # Row 0: NaN returns
    assert math.isnan(df["return_1"].iloc[0])

    # Row 1: close moves from 100 -> 110: return_1 = 10 / 100 = 0.10
    assert pytest.approx(df["return_1"].iloc[1], rel=1e-5) == 0.10

    # Row 3: close moves from 100 -> 130: return_3 = (130 - 100) / 100 = 0.30
    assert pytest.approx(df["return_3"].iloc[3], rel=1e-5) == 0.30

    # Row 6: close moves from 100 -> 160: return_6 = (160 - 100) / 100 = 0.60
    assert pytest.approx(df["return_6"].iloc[6], rel=1e-5) == 0.60


def test_moving_averages_and_distances() -> None:
    """Test SMA and normalized price-to-SMA distance calculations."""
    # Prices: 100, 101, 102, 103, 104, 105, ...
    candles = make_synthetic_candles(count=25, start_price=100.0, price_step=1.0)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(calculator.candles_to_dataframe(candles))

    # Row 4 (5th candle, close=104): SMA5 of [100, 101, 102, 103, 104] = 102.0
    assert pytest.approx(df["sma_5"].iloc[4], rel=1e-5) == 102.0
    expected_dist_5 = (104.0 - 102.0) / 102.0
    assert pytest.approx(df["close_to_sma_5"].iloc[4], rel=1e-5) == expected_dist_5

    # Row 19 (20th candle, close=119): SMA20 of [100..119] = 109.5
    assert pytest.approx(df["sma_20"].iloc[19], rel=1e-5) == 109.5
    expected_dist_20 = (119.0 - 109.5) / 109.5
    assert pytest.approx(df["close_to_sma_20"].iloc[19], rel=1e-5) == expected_dist_20


def test_volatility_and_volume_ratio() -> None:
    """Test rolling volatility and relative volume ratio."""
    candles = make_synthetic_candles(count=25, volume=1000.0)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(calculator.candles_to_dataframe(candles))

    # volatility_10 at row 10 should be non-null and positive
    vol_10 = df["volatility_10"].iloc[10]
    assert not math.isnan(vol_10)
    assert vol_10 >= 0.0

    # volume_ratio_10 at row 9 should be near 1.0 for constant/periodic volume
    vr_10 = df["volume_ratio_10"].iloc[9]
    assert not math.isnan(vr_10)
    assert vr_10 > 0.0


def test_vwap_and_oi_features() -> None:
    """Test session VWAP distance and Open Interest change features."""
    candles = make_synthetic_candles(count=15, with_oi=True)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(calculator.candles_to_dataframe(candles))

    assert not math.isnan(df["vwap_distance"].iloc[0])
    assert not math.isnan(df["oi_change_1"].iloc[1])
    assert not math.isnan(df["oi_change_3"].iloc[3])


def test_no_look_ahead_bias() -> None:
    """Critical test: Modifying future candles MUST NOT alter past feature values."""
    candles_original = make_synthetic_candles(count=30, start_price=100.0)
    pipeline = FeaturePipeline()

    vectors_orig = pipeline.generate_feature_vectors(
        candles_original, drop_warmup=False, validate=False
    )

    # Clone candles and dramatically mutate future candles at index 20..29
    candles_mutated = list(candles_original)
    for i in range(20, 30):
        c = candles_original[i]
        candles_mutated[i] = Candle(
            timestamp=c.timestamp,
            symbol=c.symbol,
            timeframe=c.timeframe,
            open=c.open * 10.0,
            high=c.high * 10.0,
            low=c.low * 10.0,
            close=c.close * 10.0,
            volume=c.volume * 5.0,
            open_interest=c.open_interest,
        )

    vectors_mutated = pipeline.generate_feature_vectors(
        candles_mutated, drop_warmup=False, validate=False
    )

    # Check that all features at index 19 (and earlier) remain bit-for-bit identical!
    for idx in range(20):
        orig_features = vectors_orig[idx].features
        mutated_features = vectors_mutated[idx].features

        for fname in orig_features:
            v_orig = orig_features[fname]
            v_mutated = mutated_features[fname]
            if math.isnan(v_orig):
                assert math.isnan(v_mutated)
            else:
                assert v_orig == v_mutated, f"Feature '{fname}' leaked at index {idx}!"


def test_warmup_policy() -> None:
    """Test warm-up dropping vs retention."""
    candles = make_synthetic_candles(count=30)
    pipeline = FeaturePipeline()

    # Without warm-up filtering: returns 30 vectors, early vectors have missing features
    vectors_raw = pipeline.generate_feature_vectors(
        candles, drop_warmup=False, validate=False
    )
    assert len(vectors_raw) == 30
    assert "sma_20" not in vectors_raw[0].features

    # With warm-up filtering: drops first 19 rows, returns 11 complete vectors
    vectors_clean = pipeline.generate_feature_vectors(
        candles, drop_warmup=True, validate=True
    )
    assert len(vectors_clean) == 11
    for vec in vectors_clean:
        assert "sma_20" in vec.features
        assert "return_6" in vec.features
        assert not math.isnan(vec.features["sma_20"])


def test_feature_validation_errors() -> None:
    """Test validator detection of infinite values and version mismatch."""
    validator = FeatureValidator(expected_version="v1")

    # Mismatched version
    v_bad_ver = FeatureVector(
        timestamp=datetime.now(UTC_TZ),
        symbol="NIFTY",
        features={"return_1": 0.01},
        feature_version="v2",
    )
    errors = validator.validate_vector(v_bad_ver)
    assert any("version mismatch" in e for e in errors)

    # Infinite value
    v_inf = FeatureVector(
        timestamp=datetime.now(UTC_TZ),
        symbol="NIFTY",
        features={"return_1": float("inf")},
        feature_version="v1",
    )
    errors = validator.validate_vector(v_inf)
    assert any("infinite value" in e for e in errors)


def test_pipeline_raises_on_invalid_sequence() -> None:
    """Test that pipeline raises FeatureValidationError if output is invalid."""
    pipeline = FeaturePipeline()
    # Unordered duplicate timestamp input
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    c1 = Candle(
        timestamp=t0,
        symbol="NIFTY",
        timeframe=TimeFrame.FIVE_MINUTES,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=100.0,
    )
    c2 = Candle(
        timestamp=t0,  # duplicate timestamp
        symbol="NIFTY",
        timeframe=TimeFrame.FIVE_MINUTES,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=100.0,
    )

    with pytest.raises(FeatureValidationError):
        pipeline.generate_feature_vectors([c1, c2], drop_warmup=False, validate=True)
