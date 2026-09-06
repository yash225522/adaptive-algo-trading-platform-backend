"""End-to-end feature engineering pipeline."""

import logging
from collections.abc import Sequence

import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.prediction import FeatureVector
from adaptive_trading.features.calculator import FeatureCalculator
from adaptive_trading.features.definitions import (
    FEATURE_CATALOG_V1,
    FEATURE_SET_VERSION,
)
from adaptive_trading.features.validators import (
    FeatureValidationError,
    FeatureValidator,
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates feature extraction from market candles into FeatureVectors."""

    def __init__(
        self,
        calculator: FeatureCalculator | None = None,
        validator: FeatureValidator | None = None,
        feature_version: str = FEATURE_SET_VERSION,
    ) -> None:
        self.calculator = calculator or FeatureCalculator()
        self.validator = validator or FeatureValidator(expected_version=feature_version)
        self.feature_version = feature_version

    def generate_feature_vectors(
        self,
        candles: Sequence[Candle],
        drop_warmup: bool = True,
        validate: bool = True,
    ) -> list[FeatureVector]:
        """Generate validated FeatureVector objects from Candle sequence.

        Args:
            candles: Sequence of validated Candle instances.
            drop_warmup: If True, filters out rows with NaN in required features.
            validate: If True, runs full sequence validation on output vectors.

        Returns:
            list[FeatureVector]: Chronologically ordered list of FeatureVectors.
        """
        if not candles:
            return []

        # 1. Convert to DataFrame and calculate features
        df = self.calculator.candles_to_dataframe(candles)
        features_df = self.calculator.calculate_features(df)

        # 2. Identify required vs optional feature columns
        required_features = [
            name
            for name, defn in FEATURE_CATALOG_V1.items()
            if not defn.is_optional and name in features_df.columns
        ]
        all_feature_cols = [
            name for name in FEATURE_CATALOG_V1 if name in features_df.columns
        ]

        # 3. Apply warm-up filtering
        if drop_warmup and required_features:
            initial_count = len(features_df)
            features_df = features_df.dropna(subset=required_features).reset_index(
                drop=True
            )
            dropped_count = initial_count - len(features_df)
            logger.debug(
                "Filtered %d warm-up period rows (remaining=%d)",
                dropped_count,
                len(features_df),
            )

        # 4. Construct FeatureVector objects
        vectors: list[FeatureVector] = []
        for _, row in features_df.iterrows():
            features_dict: dict[str, float] = {}
            for col in all_feature_cols:
                val = row.get(col)
                if val is not None and not pd.isna(val):
                    features_dict[col] = float(val)

            vector = FeatureVector(
                timestamp=row["timestamp"],
                symbol=str(row["symbol"]),
                features=features_dict,
                feature_version=self.feature_version,
            )
            vectors.append(vector)

        # 5. Validate output sequence
        if validate:
            errors = self.validator.validate_sequence(
                vectors, allow_nan=not drop_warmup
            )
            if errors:
                error_msg = "; ".join(errors)
                logger.error("Feature vector validation failed: %s", error_msg)
                raise FeatureValidationError(
                    f"Validation failed for generated feature vectors: {error_msg}"
                )

        logger.info(
            "Generated %d FeatureVectors for %s (version=%s, drop_warmup=%s)",
            len(vectors),
            candles[0].symbol if candles else "unknown",
            self.feature_version,
            drop_warmup,
        )
        return vectors
