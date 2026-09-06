"""Deterministic and vectorized market data feature calculator."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.features.definitions import FEATURE_CATALOG_V1

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """Computes technical and statistical features deterministically."""

    def candles_to_dataframe(self, candles: Sequence[Candle]) -> pd.DataFrame:
        """Convert a sequence of Candle domain objects to a sorted DataFrame.

        Args:
            candles: Sequence of validated Candle instances.

        Returns:
            pd.DataFrame: Sorted DataFrame with DatetimeIndex.
        """
        if not candles:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "timeframe",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open_interest",
                ]
            )

        records = [
            {
                "timestamp": c.timestamp,
                "symbol": c.symbol,
                "timeframe": str(c.timeframe),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "open_interest": c.open_interest,
            }
            for c in candles
        ]
        df = pd.DataFrame.from_records(records)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all v1 features for the given market candles DataFrame.

        Args:
            df: DataFrame containing standard OHLCV columns.

        Returns:
            pd.DataFrame: DataFrame containing computed feature columns.
        """
        if df.empty:
            return df.copy()

        # Guarantee chronological order
        result_df = df.sort_values(by="timestamp").copy().reset_index(drop=True)

        close = result_df["close"].astype(float)
        volume = result_df["volume"].astype(float)

        # 1. Returns (simple fractional return)
        result_df["return_1"] = close.pct_change(1)
        result_df["return_3"] = close.pct_change(3)
        result_df["return_6"] = close.pct_change(6)

        # 2. Moving Averages
        result_df["sma_5"] = close.rolling(5).mean()
        result_df["sma_10"] = close.rolling(10).mean()
        result_df["sma_20"] = close.rolling(20).mean()

        # 3. Normalized Price-to-SMA Distances
        result_df["close_to_sma_5"] = (
            result_df["close"] - result_df["sma_5"]
        ) / result_df["sma_5"]
        result_df["close_to_sma_20"] = (
            result_df["close"] - result_df["sma_20"]
        ) / result_df["sma_20"]

        # 4. Volatility (10-period rolling sample std dev of 1-period returns)
        result_df["volatility_10"] = result_df["return_1"].rolling(10).std(ddof=1)

        # 5. Volume Ratio (current volume vs 10-period rolling average)
        vol_sma_10 = volume.rolling(10).mean()
        # Handle zero rolling volume gracefully
        result_df["volume_ratio_10"] = np.where(
            vol_sma_10 > 0, volume / vol_sma_10, np.nan
        )

        # 6. Intraday Session VWAP Distance
        # Typical Price = (High + Low + Close) / 3
        high = result_df["high"].astype(float)
        low = result_df["low"].astype(float)
        typical_price = (high + low + close) / 3.0

        # Session-aware group by date
        dates = pd.to_datetime(result_df["timestamp"]).dt.date
        vp = typical_price * volume

        cum_vp = vp.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()

        session_vwap = np.where(cum_vol > 0, cum_vp / cum_vol, np.nan)
        result_df["vwap_distance"] = np.where(
            session_vwap > 0, (close - session_vwap) / session_vwap, np.nan
        )

        # 7. Open Interest features (if column exists and has values)
        if (
            "open_interest" in result_df.columns
            and not result_df["open_interest"].isna().all()
        ):
            oi = result_df["open_interest"].astype(float)
            result_df["oi_change_1"] = oi.pct_change(1)
            result_df["oi_change_3"] = oi.pct_change(3)
        else:
            result_df["oi_change_1"] = np.nan
            result_df["oi_change_3"] = np.nan

        return result_df

    def get_feature_names(self, include_optional: bool = True) -> list[str]:
        """Return list of supported feature names."""
        if include_optional:
            return list(FEATURE_CATALOG_V1.keys())
        return [k for k, v in FEATURE_CATALOG_V1.items() if not v.is_optional]
