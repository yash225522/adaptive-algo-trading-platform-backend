"""Market data validation and domain model construction."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.models import (
    DataQualityError,
    QualityIssue,
    QualitySeverity,
)
from adaptive_trading.domain.market import Candle


class MarketDataValidator:
    """Validates market data records, enforces invariants, and builds Candles."""

    def __init__(self, default_timezone: str | None = None) -> None:
        self.default_tz = ZoneInfo(default_timezone) if default_timezone else None

    def validate_record(
        self, row_number: int, record: dict[str, Any]
    ) -> tuple[Candle | None, list[QualityIssue]]:
        """Validate a single raw record and return (Candle, errors)."""
        errors: list[QualityIssue] = []

        # 1. Parse and validate timestamp
        raw_ts = record.get("timestamp")
        parsed_ts: datetime | None = None
        if not raw_ts:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="timestamp",
                    problem="Timestamp is missing or empty",
                    raw_value=raw_ts,
                )
            )
        else:
            try:
                if isinstance(raw_ts, datetime):
                    parsed_ts = raw_ts
                else:
                    parsed_ts = datetime.fromisoformat(str(raw_ts))

                if (
                    parsed_ts.tzinfo is None
                    or parsed_ts.tzinfo.utcoffset(parsed_ts) is None
                ):
                    if self.default_tz:
                        parsed_ts = parsed_ts.replace(tzinfo=self.default_tz)
                    else:
                        errors.append(
                            DataQualityError(
                                row_number=row_number,
                                field="timestamp",
                                problem=(
                                    "Timestamp is naive; timezone-aware "
                                    "ISO-8601 timestamp required"
                                ),
                                raw_value=raw_ts,
                            )
                        )
                        parsed_ts = None
            except Exception as exc:
                errors.append(
                    DataQualityError(
                        row_number=row_number,
                        field="timestamp",
                        problem=f"Invalid timestamp format: {exc}",
                        raw_value=raw_ts,
                    )
                )

        # 2. Validate symbol
        raw_symbol = record.get("symbol")
        symbol = str(raw_symbol).strip() if raw_symbol is not None else ""
        if not symbol:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="symbol",
                    problem="Symbol cannot be empty",
                    raw_value=raw_symbol,
                )
            )

        # 3. Validate timeframe
        raw_tf = record.get("timeframe")
        timeframe: TimeFrame | None = None
        if not raw_tf:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="timeframe",
                    problem="Timeframe cannot be empty",
                    raw_value=raw_tf,
                )
            )
        else:
            try:
                timeframe = TimeFrame(str(raw_tf).strip().lower())
            except ValueError:
                errors.append(
                    DataQualityError(
                        row_number=row_number,
                        field="timeframe",
                        problem=f"Unsupported timeframe: {raw_tf}",
                        raw_value=raw_tf,
                    )
                )

        # 4. Validate numerical OHLCV values
        def parse_float(field_name: str) -> float | None:
            raw_val = record.get(field_name)
            if raw_val is None or raw_val == "":
                errors.append(
                    DataQualityError(
                        row_number=row_number,
                        field=field_name,
                        problem=f"Missing required numeric field '{field_name}'",
                        raw_value=raw_val,
                    )
                )
                return None
            try:
                return float(raw_val)
            except (ValueError, TypeError):
                errors.append(
                    DataQualityError(
                        row_number=row_number,
                        field=field_name,
                        problem=f"Value cannot be parsed as float: '{raw_val}'",
                        raw_value=raw_val,
                    )
                )
                return None

        open_p = parse_float("open")
        high_p = parse_float("high")
        low_p = parse_float("low")
        close_p = parse_float("close")
        volume = parse_float("volume")

        # Open interest is optional
        raw_oi = record.get("open_interest")
        open_interest: float | None = None
        if raw_oi is not None and raw_oi != "":
            try:
                open_interest = float(raw_oi)
                if open_interest < 0:
                    errors.append(
                        DataQualityError(
                            row_number=row_number,
                            field="open_interest",
                            problem=(
                                f"Open interest cannot be negative "
                                f"(got {open_interest})"
                            ),
                            raw_value=raw_oi,
                        )
                    )
            except (ValueError, TypeError):
                errors.append(
                    DataQualityError(
                        row_number=row_number,
                        field="open_interest",
                        problem=f"Open interest cannot be parsed as float: '{raw_oi}'",
                        raw_value=raw_oi,
                    )
                )

        # Value checks
        if open_p is not None and open_p <= 0:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="open",
                    problem=f"Open price must be positive (got {open_p})",
                    raw_value=open_p,
                )
            )
        if high_p is not None and high_p <= 0:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="high",
                    problem=f"High price must be positive (got {high_p})",
                    raw_value=high_p,
                )
            )
        if low_p is not None and low_p <= 0:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="low",
                    problem=f"Low price must be positive (got {low_p})",
                    raw_value=low_p,
                )
            )
        if close_p is not None and close_p <= 0:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="close",
                    problem=f"Close price must be positive (got {close_p})",
                    raw_value=close_p,
                )
            )
        if volume is not None and volume < 0:
            errors.append(
                DataQualityError(
                    row_number=row_number,
                    field="volume",
                    problem=f"Volume cannot be negative (got {volume})",
                    raw_value=volume,
                )
            )

        # Relational OHLC checks
        if (
            open_p is not None
            and high_p is not None
            and low_p is not None
            and close_p is not None
        ):
            if high_p < open_p or high_p < close_p or high_p < low_p:
                errors.append(
                    DataQualityError(
                        row_number=row_number,
                        field="high",
                        problem=(
                            f"High ({high_p}) must be >= "
                            f"max(open={open_p}, close={close_p}, low={low_p})"
                        ),
                        raw_value=high_p,
                    )
                )
            if low_p > open_p or low_p > close_p or low_p > high_p:
                errors.append(
                    DataQualityError(
                        row_number=row_number,
                        field="low",
                        problem=(
                            f"Low ({low_p}) must be <= "
                            f"min(open={open_p}, close={close_p}, high={high_p})"
                        ),
                        raw_value=low_p,
                    )
                )

        has_missing = (
            parsed_ts is None
            or timeframe is None
            or open_p is None
            or high_p is None
            or low_p is None
            or close_p is None
            or volume is None
        )
        if errors or has_missing:
            return None, errors

        assert parsed_ts is not None
        assert timeframe is not None
        assert open_p is not None
        assert high_p is not None
        assert low_p is not None
        assert close_p is not None
        assert volume is not None

        try:
            candle = Candle(
                timestamp=parsed_ts,
                symbol=symbol,
                timeframe=timeframe,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume,
                open_interest=open_interest,
            )
            return candle, []
        except Exception as exc:
            errors.append(
                QualityIssue(
                    issue_type="CANDLE_CREATION_FAILED",
                    severity=QualitySeverity.ERROR,
                    message=f"Domain model construction failed: {exc}",
                    row_number=row_number,
                )
            )
            return None, errors

    def validate_batch(
        self, records: list[tuple[int, dict[str, Any]]]
    ) -> tuple[list[Candle], list[QualityIssue]]:
        """Validate a batch of records, returning sorted candles and errors."""
        valid_candles: list[Candle] = []
        all_errors: list[QualityIssue] = []

        for row_num, record in records:
            candle, errors = self.validate_record(row_num, record)
            if errors:
                all_errors.extend(errors)
            elif candle is not None:
                valid_candles.append(candle)

        valid_candles.sort(key=lambda c: c.timestamp)
        return valid_candles, all_errors
