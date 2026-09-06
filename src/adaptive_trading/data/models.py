"""Data quality and ingestion tracking models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class QualitySeverity(StrEnum):
    """Severity of a data quality finding."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class QualityIssue:
    """Represents a data quality issue (error or warning)."""

    issue_type: str
    severity: QualitySeverity
    message: str
    timestamp: datetime | None = None
    symbol: str | None = None
    timeframe: str | None = None
    row_number: int | None = None
    field: str | None = None
    raw_value: Any = None

    @property
    def problem(self) -> str:
        """Alias for message for backward compatibility."""
        return self.message


@dataclass(frozen=True)
class DataQualityError(QualityIssue):
    """Backward-compatible error representation."""

    def __init__(
        self,
        row_number: int,
        field: str,
        problem: str,
        raw_value: Any = None,
        issue_type: str = "VALIDATION_ERROR",
        severity: QualitySeverity = QualitySeverity.ERROR,
    ) -> None:
        super().__init__(
            issue_type=issue_type,
            severity=severity,
            message=problem,
            row_number=row_number,
            field=field,
            raw_value=raw_value,
        )


@dataclass
class DataQualityReport:
    """Aggregated report of data quality checks across a market dataset."""

    rows_checked: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    duplicate_count: int = 0
    missing_candle_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    issues: list[QualityIssue] = field(default_factory=list)

    def add_issue(self, issue: QualityIssue) -> None:
        """Add an issue and increment corresponding metric counter."""
        self.issues.append(issue)
        if issue.severity == QualitySeverity.ERROR:
            self.error_count += 1
        elif issue.severity == QualitySeverity.WARNING:
            self.warning_count += 1


@dataclass
class IngestionStats:
    """Statistics collected during a market data ingestion run."""

    source_file: str
    rows_read: int = 0
    rows_valid: int = 0
    rows_inserted: int = 0
    rows_skipped: int = 0
    rows_rejected: int = 0
    warnings: int = 0
    quality_report: DataQualityReport | None = None
    errors: list[QualityIssue] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """True if at least one row was processed and no fatal errors occurred."""
        return self.rows_rejected == 0 and self.rows_valid > 0
