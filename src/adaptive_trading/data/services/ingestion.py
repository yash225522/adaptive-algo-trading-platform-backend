"""Market data ingestion orchestration service."""

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from adaptive_trading.data.models import IngestionStats, QualitySeverity
from adaptive_trading.data.normalizers.market_data import (
    MarketDataNormalizer,
)
from adaptive_trading.data.quality.checker import MarketDataQualityChecker
from adaptive_trading.data.readers.csv_reader import CSVMarketDataReader
from adaptive_trading.data.services.persistence import (
    MarketDataPersistenceService,
)
from adaptive_trading.database.session import get_db_session

logger = logging.getLogger(__name__)


class MarketDataIngestionService:
    """Orchestrates CSV reading, normalization, quality validation, and persistence."""

    def __init__(
        self,
        reader: CSVMarketDataReader | None = None,
        normalizer: MarketDataNormalizer | None = None,
        quality_checker: MarketDataQualityChecker | None = None,
        persistence_service: MarketDataPersistenceService | None = None,
    ) -> None:
        self.reader = reader or CSVMarketDataReader()
        self.normalizer = normalizer or MarketDataNormalizer()
        self.quality_checker = quality_checker or MarketDataQualityChecker()
        self.persistence_service = persistence_service or MarketDataPersistenceService()

    def ingest_csv_file(
        self,
        file_path: str | Path,
        session: Session | None = None,
    ) -> IngestionStats:
        """Ingest a market data CSV file end-to-end.

        Args:
            file_path: Path to the input CSV file.
            session: Optional existing DB Session. If None, uses managed session.

        Returns:
            IngestionStats: Statistics detailing processed, inserted, and skipped rows.
        """
        path = Path(file_path)
        stats = IngestionStats(source_file=str(path))
        logger.info("Starting market data ingestion from file: %s", path)

        # 1. Read raw CSV records
        try:
            raw_records = self.reader.read_records(path)
        except Exception as exc:
            logger.error("Failed reading CSV file '%s': %s", path, exc)
            raise

        stats.rows_read = len(raw_records)
        logger.info("Read %d rows from '%s'", stats.rows_read, path.name)

        if not raw_records:
            logger.warning("No data records found in file: %s", path)
            return stats

        # 2. Normalize records
        normalized_records, norm_issues = self.normalizer.normalize_batch(raw_records)

        # 3. Deep quality check & validation
        valid_candles, quality_report = self.quality_checker.check_batch(
            normalized_records
        )

        # Incorporate normalizer issues into quality report
        for issue in norm_issues:
            quality_report.add_issue(issue)

        stats.quality_report = quality_report
        stats.rows_valid = len(valid_candles)
        stats.rows_rejected = stats.rows_read - stats.rows_valid
        stats.warnings = quality_report.warning_count
        stats.errors = [
            i for i in quality_report.issues if i.severity == QualitySeverity.ERROR
        ]

        for issue in quality_report.issues:
            if issue.severity == QualitySeverity.ERROR:
                logger.warning(
                    "Data quality rejection on row %s [%s]: %s (raw_value: %s)",
                    issue.row_number,
                    issue.field or issue.issue_type,
                    issue.message,
                    issue.raw_value,
                )
            else:
                logger.info(
                    "Data quality warning [%s]: %s (symbol=%s, tf=%s)",
                    issue.issue_type,
                    issue.message,
                    issue.symbol,
                    issue.timeframe,
                )

        # 4. Persist valid candles
        if valid_candles:
            if session is not None:
                inserted, skipped = self.persistence_service.persist_candles(
                    valid_candles, session=session
                )
            else:
                with get_db_session() as managed_session:
                    inserted, skipped = self.persistence_service.persist_candles(
                        valid_candles, session=managed_session
                    )
            stats.rows_inserted = inserted
            stats.rows_skipped = skipped

        logger.info(
            "Ingestion completed for '%s': read=%d, valid=%d, "
            "inserted=%d, skipped=%d, rejected=%d, warnings=%d",
            path.name,
            stats.rows_read,
            stats.rows_valid,
            stats.rows_inserted,
            stats.rows_skipped,
            stats.rows_rejected,
            stats.warnings,
        )
        return stats
