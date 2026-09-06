"""Market data quality validation, missing candle detection, and heuristic checks."""

import logging
from datetime import datetime, time, timedelta
from typing import Any

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.models import (
    DataQualityReport,
    QualityIssue,
    QualitySeverity,
)
from adaptive_trading.domain.market import Candle

logger = logging.getLogger(__name__)

TIMEFRAME_DELTAS: dict[TimeFrame, timedelta] = {
    TimeFrame.ONE_MINUTE: timedelta(minutes=1),
    TimeFrame.THREE_MINUTES: timedelta(minutes=3),
    TimeFrame.FIVE_MINUTES: timedelta(minutes=5),
    TimeFrame.FIFTEEN_MINUTES: timedelta(minutes=15),
    TimeFrame.THIRTY_MINUTES: timedelta(minutes=30),
    TimeFrame.ONE_HOUR: timedelta(hours=1),
    TimeFrame.ONE_DAY: timedelta(days=1),
}


class MarketDataQualityChecker:
    """Performs deep quality checks, missing candle detection, and heuristics."""

    def __init__(
        self,
        price_jump_threshold_pct: float = 10.0,
        warn_on_zero_volume: bool = True,
        max_stale_bars: int = 3,
        session_start_time: time = time(9, 15),
        session_end_time: time = time(15, 30),
    ) -> None:
        self.price_jump_threshold_pct = price_jump_threshold_pct
        self.warn_on_zero_volume = warn_on_zero_volume
        self.max_stale_bars = max_stale_bars
        self.session_start_time = session_start_time
        self.session_end_time = session_end_time

    def check_record(
        self, row_number: int, record: dict[str, Any]
    ) -> tuple[Candle | None, list[QualityIssue]]:
        """Validate individual candle invariants and construct domain Candle."""
        issues: list[QualityIssue] = []

        ts: datetime | None = record.get("timestamp")
        symbol: str = record.get("symbol", "")
        timeframe: TimeFrame | None = record.get("timeframe")
        open_p: float | None = record.get("open")
        high_p: float | None = record.get("high")
        low_p: float | None = record.get("low")
        close_p: float | None = record.get("close")
        volume: float | None = record.get("volume")
        open_interest: float | None = record.get("open_interest")

        if ts is None:
            issues.append(
                QualityIssue(
                    issue_type="INVALID_TIMESTAMP",
                    severity=QualitySeverity.ERROR,
                    message="Missing or invalid timestamp",
                    row_number=row_number,
                    field="timestamp",
                )
            )

        if not symbol:
            issues.append(
                QualityIssue(
                    issue_type="INVALID_SYMBOL",
                    severity=QualitySeverity.ERROR,
                    message="Missing or empty symbol",
                    row_number=row_number,
                    field="symbol",
                )
            )

        if timeframe is None:
            issues.append(
                QualityIssue(
                    issue_type="INVALID_TIMEFRAME",
                    severity=QualitySeverity.ERROR,
                    message="Missing or invalid timeframe",
                    row_number=row_number,
                    field="timeframe",
                )
            )

        # 1. Price rules (must be > 0)
        for field_name, val in (
            ("open", open_p),
            ("high", high_p),
            ("low", low_p),
            ("close", close_p),
        ):
            if val is None or val <= 0:
                issues.append(
                    QualityIssue(
                        issue_type="NON_POSITIVE_PRICE",
                        severity=QualitySeverity.ERROR,
                        message=(
                            f"{field_name.capitalize()} price must be positive "
                            f"(got {val})"
                        ),
                        row_number=row_number,
                        field=field_name,
                        raw_value=val,
                    )
                )

        # 2. Volume rules (must be >= 0)
        if volume is None or volume < 0:
            issues.append(
                QualityIssue(
                    issue_type="NEGATIVE_VOLUME",
                    severity=QualitySeverity.ERROR,
                    message=f"Volume cannot be negative (got {volume})",
                    row_number=row_number,
                    field="volume",
                    raw_value=volume,
                )
            )

        # 3. Open Interest rules (must be >= 0 if present)
        if open_interest is not None and open_interest < 0:
            issues.append(
                QualityIssue(
                    issue_type="NEGATIVE_OPEN_INTEREST",
                    severity=QualitySeverity.ERROR,
                    message=(f"Open interest cannot be negative (got {open_interest})"),
                    row_number=row_number,
                    field="open_interest",
                    raw_value=open_interest,
                )
            )

        # 4. OHLC relationship checks
        if (
            open_p is not None
            and high_p is not None
            and low_p is not None
            and close_p is not None
            and open_p > 0
            and high_p > 0
            and low_p > 0
            and close_p > 0
        ):
            if high_p < open_p or high_p < close_p or high_p < low_p:
                issues.append(
                    QualityIssue(
                        issue_type="INVALID_OHLC_HIGH",
                        severity=QualitySeverity.ERROR,
                        message=(
                            f"High ({high_p}) must be >= "
                            f"max(open={open_p}, close={close_p}, low={low_p})"
                        ),
                        row_number=row_number,
                        field="high",
                        raw_value=high_p,
                    )
                )
            if low_p > open_p or low_p > close_p or low_p > high_p:
                issues.append(
                    QualityIssue(
                        issue_type="INVALID_OHLC_LOW",
                        severity=QualitySeverity.ERROR,
                        message=(
                            f"Low ({low_p}) must be <= "
                            f"min(open={open_p}, close={close_p}, high={high_p})"
                        ),
                        row_number=row_number,
                        field="low",
                        raw_value=low_p,
                    )
                )

        has_errors = any(i.severity == QualitySeverity.ERROR for i in issues)
        has_missing = (
            ts is None
            or timeframe is None
            or open_p is None
            or high_p is None
            or low_p is None
            or close_p is None
            or volume is None
        )
        if has_errors or has_missing:
            return None, issues

        assert ts is not None
        assert timeframe is not None
        assert open_p is not None
        assert high_p is not None
        assert low_p is not None
        assert close_p is not None
        assert volume is not None

        try:
            candle = Candle(
                timestamp=ts,
                symbol=symbol,
                timeframe=timeframe,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume,
                open_interest=open_interest,
            )
            return candle, issues
        except Exception as exc:
            issues.append(
                QualityIssue(
                    issue_type="CANDLE_CREATION_FAILED",
                    severity=QualitySeverity.ERROR,
                    message=f"Failed constructing Candle domain model: {exc}",
                    row_number=row_number,
                )
            )
            return None, issues

    def check_batch(
        self, records: list[tuple[int, dict[str, Any]]]
    ) -> tuple[list[Candle], DataQualityReport]:
        """Perform comprehensive quality check on batch."""
        report = DataQualityReport(rows_checked=len(records))
        valid_candidates: list[Candle] = []

        # Step 1: Check individual record invariants
        for row_num, rec in records:
            candle, row_issues = self.check_record(row_num, rec)
            for issue in row_issues:
                report.add_issue(issue)

            if candle is not None:
                valid_candidates.append(candle)
            else:
                report.rows_rejected += 1

        report.rows_valid = len(valid_candidates)

        if not valid_candidates:
            return [], report

        # Step 2: Sort chronologically
        sorted_candles = sorted(valid_candidates, key=lambda c: c.timestamp)

        # Step 3: Stream-level heuristics
        seen_keys: set[tuple[datetime, str, str]] = set()
        stale_streak = 0

        for idx, candle in enumerate(sorted_candles):
            key = (candle.timestamp, candle.symbol, str(candle.timeframe))
            if key in seen_keys:
                report.duplicate_count += 1
                report.add_issue(
                    QualityIssue(
                        issue_type="DUPLICATE_CANDLE",
                        severity=QualitySeverity.WARNING,
                        message=f"Duplicate candle found for key {key}",
                        timestamp=candle.timestamp,
                        symbol=candle.symbol,
                        timeframe=str(candle.timeframe),
                    )
                )
            else:
                seen_keys.add(key)

            # A. Check Zero volume
            if candle.volume == 0.0 and self.warn_on_zero_volume:
                report.add_issue(
                    QualityIssue(
                        issue_type="ZERO_VOLUME",
                        severity=QualitySeverity.WARNING,
                        message="Candle has zero traded volume",
                        timestamp=candle.timestamp,
                        symbol=candle.symbol,
                        timeframe=str(candle.timeframe),
                    )
                )

            # B. Check Stale flat OHLC
            if candle.open == candle.high == candle.low == candle.close:
                stale_streak += 1
                if stale_streak >= self.max_stale_bars:
                    report.add_issue(
                        QualityIssue(
                            issue_type="STALE_PRICES",
                            severity=QualitySeverity.WARNING,
                            message=(
                                f"{stale_streak} consecutive bars with "
                                f"identical flat OHLC ({candle.close})"
                            ),
                            timestamp=candle.timestamp,
                            symbol=candle.symbol,
                            timeframe=str(candle.timeframe),
                        )
                    )
            else:
                stale_streak = 0

            # C. Compare with previous candle (for non-duplicate sequence)
            if idx > 0:
                prev_candle = sorted_candles[idx - 1]

                if (
                    prev_candle.symbol == candle.symbol
                    and prev_candle.timeframe == candle.timeframe
                    and prev_candle.timestamp != candle.timestamp
                ):
                    # 1. Suspicious Price Jump
                    if prev_candle.close > 0:
                        pct_change = (
                            abs((candle.close - prev_candle.close) / prev_candle.close)
                            * 100.0
                        )
                        if pct_change >= self.price_jump_threshold_pct:
                            report.add_issue(
                                QualityIssue(
                                    issue_type="SUSPICIOUS_PRICE_JUMP",
                                    severity=QualitySeverity.WARNING,
                                    message=(
                                        f"Price changed by {pct_change:.2f}% "
                                        f"from {prev_candle.close} to {candle.close}"
                                    ),
                                    timestamp=candle.timestamp,
                                    symbol=candle.symbol,
                                    timeframe=str(candle.timeframe),
                                )
                            )

                    # 2. Missing Candle Detection
                    expected_step = TIMEFRAME_DELTAS.get(candle.timeframe)
                    if expected_step:
                        actual_gap = candle.timestamp - prev_candle.timestamp
                        prev_date = prev_candle.timestamp.date()
                        curr_date = candle.timestamp.date()

                        if prev_date == curr_date:
                            prev_t = prev_candle.timestamp.time()
                            curr_t = candle.timestamp.time()
                            is_in_session = (
                                self.session_start_time
                                <= prev_t
                                <= self.session_end_time
                                and self.session_start_time
                                <= curr_t
                                <= self.session_end_time
                            )
                            if is_in_session and actual_gap > expected_step:
                                missing_count = int(actual_gap / expected_step) - 1
                                if missing_count > 0:
                                    report.missing_candle_count += missing_count
                                    p_ts = prev_candle.timestamp.isoformat()
                                    c_ts = candle.timestamp.isoformat()
                                    msg = (
                                        f"Detected {missing_count} missing candle(s) "
                                        f"between {p_ts} and {c_ts}"
                                    )
                                    report.add_issue(
                                        QualityIssue(
                                            issue_type="MISSING_CANDLES",
                                            severity=QualitySeverity.WARNING,
                                            message=msg,
                                            timestamp=prev_candle.timestamp,
                                            symbol=candle.symbol,
                                            timeframe=str(candle.timeframe),
                                        )
                                    )

        return sorted_candles, report
