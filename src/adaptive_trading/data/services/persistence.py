"""Market data database persistence service."""

import logging
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from adaptive_trading.database.models import MarketCandleModel
from adaptive_trading.domain.market import Candle

logger = logging.getLogger(__name__)


def _timestamp_matches(db_ts: datetime, candle_ts: datetime) -> bool:
    """Check if a database timestamp matches a candle timestamp."""
    if db_ts.tzinfo is not None and candle_ts.tzinfo is not None:
        return db_ts == candle_ts
    # Handle naive representation dialects (e.g. SQLite)
    candle_naive = candle_ts.replace(tzinfo=None)
    candle_astimezone = (
        candle_ts.astimezone().replace(tzinfo=None)
        if candle_ts.tzinfo is not None
        else candle_naive
    )
    return db_ts in (candle_naive, candle_astimezone)


class MarketDataPersistenceService:
    """Handles idempotent persistence of validated Candle domain models."""

    def persist_candles(
        self,
        candles: Sequence[Candle],
        session: Session,
    ) -> tuple[int, int]:
        """Persist a batch of validated Candle domain objects idempotently.

        Args:
            candles: Sequence of validated domain Candle instances.
            session: Active database Session.

        Returns:
            tuple[int, int]: (inserted_count, skipped_count)
        """
        if not candles:
            return 0, 0

        # 1. Deduplicate within the input batch first
        unique_batch: dict[tuple[datetime, str, str], Candle] = {}
        batch_duplicates = 0

        for candle in candles:
            key = (candle.timestamp, candle.symbol, str(candle.timeframe))
            if key in unique_batch:
                batch_duplicates += 1
            else:
                unique_batch[key] = candle

        symbols = {c.symbol for c in unique_batch.values()}
        timeframes = {str(c.timeframe) for c in unique_batch.values()}

        # 2. Query existing database records matching symbols and timeframes
        existing_rows = session.execute(
            select(
                MarketCandleModel.timestamp,
                MarketCandleModel.symbol,
                MarketCandleModel.timeframe,
            ).where(
                MarketCandleModel.symbol.in_(symbols),
                MarketCandleModel.timeframe.in_(timeframes),
            )
        ).all()

        # 3. Filter models to insert by comparing against existing DB rows
        models_to_insert: list[MarketCandleModel] = []
        skipped_count = batch_duplicates

        for candle in unique_batch.values():
            already_exists = False
            for db_ts, db_sym, db_tf in existing_rows:
                if (
                    db_sym == candle.symbol
                    and db_tf == str(candle.timeframe)
                    and _timestamp_matches(db_ts, candle.timestamp)
                ):
                    already_exists = True
                    break

            if already_exists:
                skipped_count += 1
            else:
                models_to_insert.append(
                    MarketCandleModel(
                        timestamp=candle.timestamp,
                        symbol=candle.symbol,
                        timeframe=str(candle.timeframe),
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        open_interest=candle.open_interest,
                    )
                )

        if models_to_insert:
            session.add_all(models_to_insert)
            session.flush()
            inserted_count = len(models_to_insert)
        else:
            inserted_count = 0

        logger.debug(
            "Persisted candles: inserted=%d, skipped=%d",
            inserted_count,
            skipped_count,
        )
        return inserted_count, skipped_count
