"""Command-line interface for historical market data CSV ingestion."""

import argparse
import logging
import sys
from pathlib import Path

from adaptive_trading.data.models import QualitySeverity
from adaptive_trading.data.services.ingestion import (
    MarketDataIngestionService,
)


def main() -> None:
    """CLI entrypoint for CSV market data ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest historical market data from CSV into database."
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        type=str,
        help="Path to the market data CSV file",
    )
    parser.add_argument(
        "--timezone",
        "-t",
        type=str,
        default=None,
        help="Fallback timezone for naive timestamps (e.g. Asia/Kolkata)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging output",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: file not found '{file_path}'", file=sys.stderr)
        sys.exit(1)

    service = MarketDataIngestionService()
    try:
        stats = service.ingest_csv_file(file_path)
    except Exception as exc:
        print(f"Ingestion failed with error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Ingestion Summary")
    print("=" * 50)
    print(f"Source file       : {stats.source_file}")
    print(f"Rows read         : {stats.rows_read}")
    print(f"Rows valid        : {stats.rows_valid}")
    print(f"Rows inserted     : {stats.rows_inserted}")
    print(f"Rows skipped      : {stats.rows_skipped}")
    print(f"Rows rejected     : {stats.rows_rejected}")
    print(f"Warnings          : {stats.warnings}")

    if stats.quality_report:
        rep = stats.quality_report
        print("\nData Quality Metrics:")
        print(f"  - Missing candles  : {rep.missing_candle_count}")
        print(f"  - Duplicate candles: {rep.duplicate_count}")
        print(f"  - Warning findings : {rep.warning_count}")
        print(f"  - Error findings   : {rep.error_count}")

    if stats.errors:
        print(f"\nData Quality Errors ({len(stats.errors)}):")
        for err in stats.errors:
            row_info = f"Row {err.row_number} " if err.row_number is not None else ""
            field_info = f"[{err.field}] " if err.field else ""
            print(f"  - {row_info}{field_info}{err.message}")

    if stats.quality_report:
        warnings = [
            i
            for i in stats.quality_report.issues
            if i.severity == QualitySeverity.WARNING
        ]
        if warnings:
            print(f"\nData Quality Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"  - [{w.issue_type}] {w.message}")

    print("=" * 50)


if __name__ == "__main__":
    main()
