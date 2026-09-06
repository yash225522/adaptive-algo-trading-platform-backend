"""Market data normalization component."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.models import QualityIssue, QualitySeverity


class MarketDataNormalizer:
    """Performs deterministic normalization of market data rows."""

    def __init__(self, default_timezone: str | None = None) -> None:
        self.default_tz = ZoneInfo(default_timezone) if default_timezone else None

    def normalize_record(
        self, row_number: int, raw_record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[QualityIssue]]:
        """Normalize a single raw record dict.

        Returns:
            tuple[dict[str, Any] | None, list[QualityIssue]]:
                Tuple of normalized record dict and any detected quality issues.
        """
        issues: list[QualityIssue] = []
        normalized: dict[str, Any] = {}

        # 1. Normalize symbol
        raw_sym = raw_record.get("symbol")
        if raw_sym is None or str(raw_sym).strip() == "":
            issues.append(
                QualityIssue(
                    issue_type="MISSING_SYMBOL",
                    severity=QualitySeverity.ERROR,
                    message="Symbol is missing or empty",
                    row_number=row_number,
                    field="symbol",
                    raw_value=raw_sym,
                )
            )
        else:
            normalized["symbol"] = str(raw_sym).strip().upper()

        # 2. Normalize timeframe
        raw_tf = raw_record.get("timeframe")
        if raw_tf is None or str(raw_tf).strip() == "":
            issues.append(
                QualityIssue(
                    issue_type="MISSING_TIMEFRAME",
                    severity=QualitySeverity.ERROR,
                    message="Timeframe is missing or empty",
                    row_number=row_number,
                    field="timeframe",
                    raw_value=raw_tf,
                )
            )
        else:
            cleaned_tf = str(raw_tf).strip().lower()
            try:
                normalized["timeframe"] = TimeFrame(cleaned_tf)
            except ValueError:
                issues.append(
                    QualityIssue(
                        issue_type="INVALID_TIMEFRAME",
                        severity=QualitySeverity.ERROR,
                        message=f"Unsupported timeframe identifier: '{raw_tf}'",
                        row_number=row_number,
                        field="timeframe",
                        raw_value=raw_tf,
                    )
                )

        # 3. Normalize timestamp
        raw_ts = raw_record.get("timestamp")
        if not raw_ts or str(raw_ts).strip() == "":
            issues.append(
                QualityIssue(
                    issue_type="MISSING_TIMESTAMP",
                    severity=QualitySeverity.ERROR,
                    message="Timestamp is missing or empty",
                    row_number=row_number,
                    field="timestamp",
                    raw_value=raw_ts,
                )
            )
        else:
            parsed_ts: datetime | None = None
            try:
                if isinstance(raw_ts, datetime):
                    parsed_ts = raw_ts
                else:
                    parsed_ts = datetime.fromisoformat(str(raw_ts).strip())

                if (
                    parsed_ts.tzinfo is None
                    or parsed_ts.tzinfo.utcoffset(parsed_ts) is None
                ):
                    if self.default_tz:
                        parsed_ts = parsed_ts.replace(tzinfo=self.default_tz)
                    else:
                        issues.append(
                            QualityIssue(
                                issue_type="NAIVE_TIMESTAMP",
                                severity=QualitySeverity.ERROR,
                                message=(
                                    "Timestamp is naive; timezone-aware "
                                    "ISO-8601 timestamp required"
                                ),
                                row_number=row_number,
                                field="timestamp",
                                raw_value=raw_ts,
                            )
                        )
                        parsed_ts = None

                if parsed_ts is not None:
                    normalized["timestamp"] = parsed_ts
            except Exception as exc:
                issues.append(
                    QualityIssue(
                        issue_type="MALFORMED_TIMESTAMP",
                        severity=QualitySeverity.ERROR,
                        message=f"Cannot parse timestamp '{raw_ts}': {exc}",
                        row_number=row_number,
                        field="timestamp",
                        raw_value=raw_ts,
                    )
                )

        # 4. Normalize numeric prices and volume
        for field_name in ("open", "high", "low", "close", "volume"):
            raw_val = raw_record.get(field_name)
            if raw_val is None or str(raw_val).strip() == "":
                issues.append(
                    QualityIssue(
                        issue_type="MISSING_NUMERIC_FIELD",
                        severity=QualitySeverity.ERROR,
                        message=f"Required numeric field '{field_name}' is missing",
                        row_number=row_number,
                        field=field_name,
                        raw_value=raw_val,
                    )
                )
            else:
                try:
                    normalized[field_name] = float(str(raw_val).strip())
                except (ValueError, TypeError):
                    issues.append(
                        QualityIssue(
                            issue_type="INVALID_NUMERIC_FORMAT",
                            severity=QualitySeverity.ERROR,
                            message=(
                                f"Cannot convert '{raw_val}' to float for "
                                f"field '{field_name}'"
                            ),
                            row_number=row_number,
                            field=field_name,
                            raw_value=raw_val,
                        )
                    )

        # 5. Normalize optional open interest
        raw_oi = raw_record.get("open_interest")
        if raw_oi is not None and str(raw_oi).strip() != "":
            try:
                normalized["open_interest"] = float(str(raw_oi).strip())
            except (ValueError, TypeError):
                issues.append(
                    QualityIssue(
                        issue_type="INVALID_NUMERIC_FORMAT",
                        severity=QualitySeverity.ERROR,
                        message=f"Cannot convert '{raw_oi}' to float for open_interest",
                        row_number=row_number,
                        field="open_interest",
                        raw_value=raw_oi,
                    )
                )
        else:
            normalized["open_interest"] = None

        if issues:
            return None, issues

        return normalized, []

    def normalize_batch(
        self, raw_records: list[tuple[int, dict[str, Any]]]
    ) -> tuple[list[tuple[int, dict[str, Any]]], list[QualityIssue]]:
        """Normalize a batch of (row_number, raw_record) pairs."""
        normalized_records: list[tuple[int, dict[str, Any]]] = []
        all_issues: list[QualityIssue] = []

        for row_num, raw_rec in raw_records:
            norm_rec, issues = self.normalize_record(row_num, raw_rec)
            if issues:
                all_issues.extend(issues)
            elif norm_rec is not None:
                normalized_records.append((row_num, norm_rec))

        return normalized_records, all_issues
