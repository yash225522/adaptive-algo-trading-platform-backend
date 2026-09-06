"""Forward target and label generator for supervised learning."""

import logging
from collections.abc import Sequence

from adaptive_trading.domain.market import Candle
from adaptive_trading.targets.definitions import (
    TargetConfig,
)
from adaptive_trading.targets.models import DirectionLabel, Target

logger = logging.getLogger(__name__)


class TargetGenerator:
    """Generates future price return targets and direction labels."""

    def __init__(self, config: TargetConfig | None = None) -> None:
        self.config = config or TargetConfig()

    def generate_targets(
        self,
        candles: Sequence[Candle],
        config: TargetConfig | None = None,
    ) -> list[Target]:
        """Calculate forward targets from a sequence of market candles.

        Args:
            candles: Sequence of validated Candle instances.
            config: Optional TargetConfig override.

        Returns:
            list[Target]: Chronologically ordered list of Target objects.
        """
        cfg = config or self.config
        if not candles or len(candles) <= cfg.horizon:
            logger.debug(
                "Insufficient candles (%d) for horizon %d",
                len(candles) if candles else 0,
                cfg.horizon,
            )
            return []

        # Sort candles chronologically
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        n = len(sorted_candles)
        targets: list[Target] = []

        for i in range(n - cfg.horizon):
            current_candle = sorted_candles[i]
            future_candle = sorted_candles[i + cfg.horizon]

            if current_candle.close <= 0:
                logger.warning(
                    "Non-positive close price at %s; skipping target",
                    current_candle.timestamp,
                )
                continue

            future_return = (
                future_candle.close - current_candle.close
            ) / current_candle.close

            # Classification assignment
            if cfg.use_neutral_class and cfg.negative_threshold is not None:
                if future_return > cfg.positive_threshold:
                    label = DirectionLabel.UP
                elif future_return < cfg.negative_threshold:
                    label = DirectionLabel.DOWN
                else:
                    label = DirectionLabel.NEUTRAL
            else:
                if future_return > cfg.positive_threshold:
                    label = DirectionLabel.UP
                else:
                    label = DirectionLabel.DOWN

            target = Target(
                timestamp=current_candle.timestamp,
                symbol=current_candle.symbol,
                horizon=cfg.horizon,
                future_return=float(future_return),
                classification_label=label,
                target_version=cfg.target_version,
            )
            targets.append(target)

        logger.info(
            "Generated %d targets for %s (horizon=%d, version=%s)",
            len(targets),
            sorted_candles[0].symbol,
            cfg.horizon,
            cfg.target_version,
        )
        return targets
