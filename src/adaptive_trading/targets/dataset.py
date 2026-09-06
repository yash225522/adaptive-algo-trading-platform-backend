"""Dataset builder combining FeatureVectors and Targets into ML-ready datasets."""

import logging
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from adaptive_trading.domain.prediction import FeatureVector
from adaptive_trading.targets.models import Target, TrainingExample

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Combines FeatureVectors and Targets into aligned supervised ML datasets."""

    def build_training_examples(
        self,
        feature_vectors: Sequence[FeatureVector],
        targets: Sequence[Target],
    ) -> list[TrainingExample]:
        """Align FeatureVectors with their corresponding Targets by (timestamp, symbol).

        Args:
            feature_vectors: Sequence of calculated FeatureVectors.
            targets: Sequence of generated Targets.

        Returns:
            list[TrainingExample]: Aligned training examples.
        """
        if not feature_vectors or not targets:
            return []

        # Index targets by (timestamp, symbol)
        target_map: dict[tuple[datetime, str], Target] = {
            (t.timestamp, t.symbol): t for t in targets
        }

        examples: list[TrainingExample] = []
        for fv in feature_vectors:
            key = (fv.timestamp, fv.symbol)
            target = target_map.get(key)
            if target is not None:
                example = TrainingExample(
                    timestamp=fv.timestamp,
                    symbol=fv.symbol,
                    features=dict(fv.features),
                    feature_version=fv.feature_version,
                    target=target,
                )
                examples.append(example)

        # Sort chronologically
        examples.sort(key=lambda ex: ex.timestamp)
        logger.info("Built %d aligned training examples", len(examples))
        return examples

    def build_training_dataframe(
        self,
        feature_vectors: Sequence[FeatureVector],
        targets: Sequence[Target],
    ) -> pd.DataFrame:
        """Convert aligned FeatureVectors and Targets into a tabular training DataFrame.

        Args:
            feature_vectors: Sequence of calculated FeatureVectors.
            targets: Sequence of generated Targets.

        Returns:
            pd.DataFrame: Tabular dataset containing features, future_return, and label.
        """
        examples = self.build_training_examples(feature_vectors, targets)
        if not examples:
            return pd.DataFrame()

        records: list[dict[str, object]] = []
        for ex in examples:
            row: dict[str, object] = {
                "timestamp": ex.timestamp,
                "symbol": ex.symbol,
                **ex.features,
                "future_return": ex.target.future_return,
                "label": ex.target.classification_label.value,
                "target_version": ex.target.target_version,
                "feature_version": ex.feature_version,
            }
            records.append(row)

        df = pd.DataFrame.from_records(records)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
