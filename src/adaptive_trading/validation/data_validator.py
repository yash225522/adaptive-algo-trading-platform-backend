"""Market data quality validator.

Checks OHLCV integrity, chronological order, duplicates, gaps, and anomalies.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.experiments.fingerprint import compute_dataset_fingerprint
from adaptive_trading.validation.config import ValidationConfig
from adaptive_trading.validation.report import ValidationResult
from adaptive_trading.validation.rules import (
    ValidationRuleId,
    ValidationSeverity,
    ValidationStatus,
)
from adaptive_trading.validation.schema import DatasetSchemaValidator


def _parse_timeframe_seconds(tf_str: str | TimeFrame | None) -> int:
    """Helper to convert timeframe representation to expected seconds."""
    if tf_str is None:
        return 300  # Default 5-minute
    if isinstance(tf_str, TimeFrame):
        tf_val = tf_str.value
    else:
        tf_val = str(tf_str).lower()

    if "1m" in tf_val or "1_min" in tf_val:
        return 60
    if "3m" in tf_val:
        return 180
    if "5m" in tf_val:
        return 300
    if "15m" in tf_val:
        return 900
    if "30m" in tf_val:
        return 1800
    if "1h" in tf_val or "60m" in tf_val:
        return 3600
    if "1d" in tf_val or "day" in tf_val:
        return 86400
    return 300


class DataValidator:
    """Comprehensive historical market data quality and integrity validator."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()
        self.schema_validator = DatasetSchemaValidator(config=self.config)

    def validate_dataset(
        self,
        candles_or_df: Sequence[Candle] | pd.DataFrame | Sequence[dict[str, Any]],
        expected_symbol: str | None = None,
        expected_timeframe: str | TimeFrame | None = None,
    ) -> tuple[list[ValidationResult], str | None]:
        """Run all data-quality validations across historical market candles."""
        results: list[ValidationResult] = []

        # 1. Schema Validation
        schema_results = self.schema_validator.validate_schema(candles_or_df)
        results.extend(schema_results)

        # If schema had critical errors, abort deep checks
        if any(
            r.severity == ValidationSeverity.CRITICAL
            and r.status == ValidationStatus.FAIL
            for r in schema_results
        ):
            return results, None

        # Standardize raw rows for deep checking
        raw_rows: list[dict[str, Any]] = []
        candles: list[Candle] = []

        if isinstance(candles_or_df, pd.DataFrame):
            for _, df_row in candles_or_df.iterrows():
                ts = df_row["timestamp"]
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts).to_pydatetime()
                raw_rows.append(
                    {
                        "timestamp": ts,
                        "symbol": str(
                            df_row.get("symbol", expected_symbol or "UNKNOWN")
                        ),
                        "open": float(df_row["open"]),
                        "high": float(df_row["high"]),
                        "low": float(df_row["low"]),
                        "close": float(df_row["close"]),
                        "volume": float(df_row.get("volume", 0.0)),
                    }
                )
        elif isinstance(candles_or_df, Sequence):
            for elem in candles_or_df:
                if isinstance(elem, Candle):
                    candles.append(elem)
                    raw_rows.append(
                        {
                            "timestamp": elem.timestamp,
                            "symbol": elem.symbol,
                            "open": elem.open,
                            "high": elem.high,
                            "low": elem.low,
                            "close": elem.close,
                            "volume": elem.volume,
                        }
                    )
                elif isinstance(elem, dict):
                    ts = elem["timestamp"]
                    if isinstance(ts, str):
                        ts = pd.to_datetime(ts).to_pydatetime()
                    raw_rows.append(
                        {
                            "timestamp": ts,
                            "symbol": str(
                                elem.get("symbol", expected_symbol or "UNKNOWN")
                            ),
                            "open": float(elem["open"]),
                            "high": float(elem["high"]),
                            "low": float(elem["low"]),
                            "close": float(elem["close"]),
                            "volume": float(elem.get("volume", 0.0)),
                        }
                    )

        if not raw_rows:
            return results, None

        # 2. Timestamp timezone-awareness & validity check
        tz_errors = 0
        for r_item in raw_rows:
            ts = r_item["timestamp"]
            if not isinstance(ts, datetime):
                tz_errors += 1
            elif self.config.strict_timezone and (
                ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None
            ):
                tz_errors += 1

        if tz_errors > 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_TIMESTAMP_VALID.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=f"Found {tz_errors} invalid or naive timestamps",
                    details={"invalid_timestamp_count": tz_errors},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_TIMESTAMP_VALID.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="All timestamps are valid and timezone-aware",
                    details={"count": len(raw_rows)},
                )
            )

        # 3. OHLCV Integrity Validation
        ohlcv_violations: list[str] = []
        for idx, r in enumerate(raw_rows):
            open_p, high_p, low_p, close_p, vol_val = (
                r["open"],
                r["high"],
                r["low"],
                r["close"],
                r["volume"],
            )
            ts_str = str(r["timestamp"])

            if high_p < max(open_p, close_p):
                ohlcv_violations.append(
                    f"Row {idx} ({ts_str}): High ({high_p}) < "
                    f"max(Open={open_p}, Close={close_p})"
                )
            if low_p > min(open_p, close_p):
                ohlcv_violations.append(
                    f"Row {idx} ({ts_str}): Low ({low_p}) > "
                    f"min(Open={open_p}, Close={close_p})"
                )
            if high_p < low_p:
                ohlcv_violations.append(
                    f"Row {idx} ({ts_str}): High ({high_p}) < Low ({low_p})"
                )
            if open_p <= 0 or high_p <= 0 or low_p <= 0 or close_p <= 0:
                ohlcv_violations.append(
                    f"Row {idx} ({ts_str}): Non-positive price "
                    f"(O={open_p}, H={high_p}, L={low_p}, C={close_p})"
                )
            if vol_val < 0:
                ohlcv_violations.append(
                    f"Row {idx} ({ts_str}): Negative volume ({vol_val})"
                )

        if ohlcv_violations:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_OHLCV_INTEGRITY.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Detected {len(ohlcv_violations)} OHLCV integrity violations"
                    ),
                    details={
                        "violations": ohlcv_violations[:20],
                        "total_violations": len(ohlcv_violations),
                    },
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_OHLCV_INTEGRITY.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="All OHLC relationships and price bounds are valid",
                    details={"row_count": len(raw_rows)},
                )
            )

        # 4. Chronological Ordering Validation
        is_strictly_ordered = True
        unordered_examples: list[str] = []
        for i in range(1, len(raw_rows)):
            t_prev = raw_rows[i - 1]["timestamp"]
            t_curr = raw_rows[i]["timestamp"]
            if t_curr <= t_prev:
                is_strictly_ordered = False
                if len(unordered_examples) < 10:
                    unordered_examples.append(
                        f"Row {i - 1} ({t_prev}) -> Row {i} ({t_curr})"
                    )

        if not is_strictly_ordered:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_CHRONOLOGICAL_ORDER.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message="Candles are not strictly chronologically ordered",
                    details={"unordered_transitions": unordered_examples},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_CHRONOLOGICAL_ORDER.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="Candles are strictly chronologically ordered",
                    details={"row_count": len(raw_rows)},
                )
            )

        # 5. Duplicate Candles Check
        seen_keys: set[tuple[str, Any]] = set()
        duplicate_count = 0
        duplicate_examples: list[str] = []
        for idx, r in enumerate(raw_rows):
            k = (r["symbol"], r["timestamp"])
            if k in seen_keys:
                duplicate_count += 1
                if len(duplicate_examples) < 10:
                    duplicate_examples.append(
                        f"Row {idx}: Symbol={k[0]}, Timestamp={k[1]}"
                    )
            else:
                seen_keys.add(k)

        if duplicate_count > 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_DUPLICATE_CANDLES.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=f"Found {duplicate_count} duplicate candle records",
                    details={
                        "duplicate_count": duplicate_count,
                        "examples": duplicate_examples,
                    },
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_DUPLICATE_CANDLES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="Zero duplicate candle records found",
                    details={"unique_candles": len(seen_keys)},
                )
            )

        # 6. Timeframe Consistency & Gap Detection
        expected_sec = _parse_timeframe_seconds(
            expected_timeframe or self.config.expected_timeframe
        )
        timeframe_mismatches = 0
        suspicious_gaps: list[dict[str, Any]] = []
        expected_gaps: list[dict[str, Any]] = []

        for i in range(1, len(raw_rows)):
            t_prev = raw_rows[i - 1]["timestamp"]
            t_curr = raw_rows[i]["timestamp"]
            delta = (t_curr - t_prev).total_seconds()

            if delta <= 0:
                continue

            if abs(delta - expected_sec) > (expected_sec * 0.1):
                timeframe_mismatches += 1

                # Classify gap: Session/Overnight/Weekend vs Suspicious
                if delta >= 3600 * 12:
                    expected_gaps.append(
                        {
                            "from": str(t_prev),
                            "to": str(t_curr),
                            "duration_hours": round(delta / 3600.0, 2),
                            "type": "OVERNIGHT_OR_WEEKEND",
                        }
                    )
                else:
                    suspicious_gaps.append(
                        {
                            "from": str(t_prev),
                            "to": str(t_curr),
                            "gap_seconds": delta,
                            "expected_seconds": expected_sec,
                            "missing_bars_approx": int(delta / expected_sec) - 1,
                        }
                    )

        if timeframe_mismatches > 0 and len(suspicious_gaps) > 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_SUSPICIOUS_GAPS.value,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARN,
                    message=(
                        f"Detected {len(suspicious_gaps)} suspicious intra-day gaps"
                    ),
                    details={
                        "suspicious_gap_count": len(suspicious_gaps),
                        "expected_gap_count": len(expected_gaps),
                        "examples": suspicious_gaps[:10],
                    },
                )
            )
        elif len(expected_gaps) > 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_SUSPICIOUS_GAPS.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=(
                        f"All {len(expected_gaps)} gaps correspond to "
                        "standard overnight/weekend session breaks"
                    ),
                    details={"session_gaps": len(expected_gaps)},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_SUSPICIOUS_GAPS.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="Continuous candle sequence with no detected gaps",
                    details={},
                )
            )

        # 7. Price Anomalies Check (Extreme percentage jumps)
        max_jump = 0.0
        price_anomalies: list[dict[str, Any]] = []
        for i in range(1, len(raw_rows)):
            c_prev = raw_rows[i - 1]["close"]
            c_curr = raw_rows[i]["close"]
            if c_prev > 0:
                ret = abs((c_curr - c_prev) / c_prev)
                if ret > max_jump:
                    max_jump = ret
                if ret > self.config.max_reasonable_return:
                    price_anomalies.append(
                        {
                            "timestamp": str(raw_rows[i]["timestamp"]),
                            "prev_close": c_prev,
                            "close": c_curr,
                            "return_pct": round(ret * 100.0, 2),
                        }
                    )

        if price_anomalies:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_PRICE_ANOMALIES.value,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARN,
                    message=(
                        f"Detected {len(price_anomalies)} single-candle price jumps "
                        f"exceeding {self.config.max_reasonable_return:.0%}"
                    ),
                    details={
                        "anomalies": price_anomalies[:10],
                        "max_jump_pct": round(max_jump * 100.0, 2),
                    },
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_PRICE_ANOMALIES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=f"All price jumps in normal bounds (max: {max_jump:.2%})",
                    details={"max_jump_pct": round(max_jump * 100.0, 2)},
                )
            )

        # 8. Volume Anomalies Check
        zero_vols = sum(1 for r in raw_rows if r["volume"] == 0)
        neg_vols = sum(1 for r in raw_rows if r["volume"] < 0)

        if neg_vols > 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_VOLUME_ANOMALIES.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=f"Detected {neg_vols} candles with negative volume",
                    details={"negative_volume_count": neg_vols},
                )
            )
        elif zero_vols > 0:
            severity = (
                ValidationSeverity.WARNING
                if not self.config.allow_zero_volume
                else ValidationSeverity.INFO
            )
            status = (
                ValidationStatus.WARN
                if not self.config.allow_zero_volume
                else ValidationStatus.PASS
            )
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_VOLUME_ANOMALIES.value,
                    severity=severity,
                    status=status,
                    message=f"Dataset contains {zero_vols} zero-volume candles",
                    details={
                        "zero_volume_count": zero_vols,
                        "total_rows": len(raw_rows),
                    },
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_VOLUME_ANOMALIES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="All candles have positive volume",
                    details={"row_count": len(raw_rows)},
                )
            )

        # 9. Deterministic Dataset Fingerprint
        if not candles:
            for r in raw_rows:
                ts = r["timestamp"]
                if isinstance(ts, datetime) and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                tf = TimeFrame.FIVE_MINUTES
                if expected_timeframe:
                    try:
                        tf = TimeFrame(str(expected_timeframe).lower())
                    except Exception:
                        pass
                try:
                    c = Candle(
                        timestamp=ts,
                        symbol=r["symbol"],
                        timeframe=tf,
                        open=r["open"],
                        high=r["high"],
                        low=r["low"],
                        close=r["close"],
                        volume=r["volume"],
                    )
                    candles.append(c)
                except Exception:
                    pass

        dataset_fp = compute_dataset_fingerprint(candles) if candles else None

        return results, dataset_fp
