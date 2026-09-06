"""Feature definitions, catalog, and versioning configuration."""

from dataclasses import dataclass

FEATURE_SET_VERSION = "v1"


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata specification for a single calculated feature."""

    name: str
    description: str
    lookback_periods: int
    required_columns: tuple[str, ...]
    formula: str
    is_optional: bool = False


# Centralized v1 Feature Catalog
FEATURE_CATALOG_V1: dict[str, FeatureDefinition] = {
    # Returns
    "return_1": FeatureDefinition(
        name="return_1",
        description="1-period fractional price return",
        lookback_periods=1,
        required_columns=("close",),
        formula="(close_t - close_{t-1}) / close_{t-1}",
    ),
    "return_3": FeatureDefinition(
        name="return_3",
        description="3-period fractional price return",
        lookback_periods=3,
        required_columns=("close",),
        formula="(close_t - close_{t-3}) / close_{t-3}",
    ),
    "return_6": FeatureDefinition(
        name="return_6",
        description="6-period fractional price return",
        lookback_periods=6,
        required_columns=("close",),
        formula="(close_t - close_{t-6}) / close_{t-6}",
    ),
    # Moving Averages
    "sma_5": FeatureDefinition(
        name="sma_5",
        description="5-period simple moving average of close price",
        lookback_periods=5,
        required_columns=("close",),
        formula="mean(close_{t-4..t})",
    ),
    "sma_10": FeatureDefinition(
        name="sma_10",
        description="10-period simple moving average of close price",
        lookback_periods=10,
        required_columns=("close",),
        formula="mean(close_{t-9..t})",
    ),
    "sma_20": FeatureDefinition(
        name="sma_20",
        description="20-period simple moving average of close price",
        lookback_periods=20,
        required_columns=("close",),
        formula="mean(close_{t-19..t})",
    ),
    "close_to_sma_5": FeatureDefinition(
        name="close_to_sma_5",
        description="Normalized distance of close to 5-period SMA",
        lookback_periods=5,
        required_columns=("close",),
        formula="(close_t - sma_5_t) / sma_5_t",
    ),
    "close_to_sma_20": FeatureDefinition(
        name="close_to_sma_20",
        description="Normalized distance of close to 20-period SMA",
        lookback_periods=20,
        required_columns=("close",),
        formula="(close_t - sma_20_t) / sma_20_t",
    ),
    # Volatility
    "volatility_10": FeatureDefinition(
        name="volatility_10",
        description="10-period rolling sample standard deviation of return_1",
        lookback_periods=11,  # 1 period for return_1 + 10 periods rolling std
        required_columns=("close",),
        formula="std(return_1_{t-9..t}, ddof=1)",
    ),
    # Volume
    "volume_ratio_10": FeatureDefinition(
        name="volume_ratio_10",
        description="Ratio of current volume to 10-period rolling mean volume",
        lookback_periods=10,
        required_columns=("volume",),
        formula="volume_t / mean(volume_{t-9..t})",
    ),
    # VWAP
    "vwap_distance": FeatureDefinition(
        name="vwap_distance",
        description="Normalized distance of close to intraday session VWAP",
        lookback_periods=1,
        required_columns=("high", "low", "close", "volume"),
        formula="(close_t - vwap_t) / vwap_t",
        is_optional=True,
    ),
    # Open Interest
    "oi_change_1": FeatureDefinition(
        name="oi_change_1",
        description="1-period fractional change in open interest",
        lookback_periods=1,
        required_columns=("open_interest",),
        formula="(oi_t - oi_{t-1}) / oi_{t-1}",
        is_optional=True,
    ),
    "oi_change_3": FeatureDefinition(
        name="oi_change_3",
        description="3-period fractional change in open interest",
        lookback_periods=3,
        required_columns=("open_interest",),
        formula="(oi_t - oi_{t-3}) / oi_{t-3}",
        is_optional=True,
    ),
}

MAX_LOOKBACK_PERIODS_V1: int = max(
    f.lookback_periods for f in FEATURE_CATALOG_V1.values()
)
