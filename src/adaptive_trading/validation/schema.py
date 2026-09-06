"""Dataset schema, column presence, data types, and nullity validation."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.validation.config import ValidationConfig
from adaptive_trading.validation.report import ValidationResult
from adaptive_trading.validation.rules import (
    ValidationRuleId,
    ValidationSeverity,
    ValidationStatus,
)

REQUIRED_MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DatasetSchemaValidator:
    """Validates structural column schema, data types, and nulls in market data."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate_schema(
        self, data: pd.DataFrame | Sequence[Candle] | Sequence[dict[str, Any]]
    ) -> list[ValidationResult]:
        """Validate column names, data types, null values, and symbols in dataset."""
        results: list[ValidationResult] = []

        if isinstance(data, pd.DataFrame):
            df = data
        elif data and isinstance(data[0], Candle):
            df = pd.DataFrame([c.model_dump() for c in data if isinstance(c, Candle)])
        elif data and isinstance(data[0], dict):
            df = pd.DataFrame(list(data))
        else:
            if not data:
                return [
                    ValidationResult(
                        rule_id=ValidationRuleId.SCHEMA_REQUIRED_COLUMNS.value,
                        severity=ValidationSeverity.CRITICAL,
                        status=ValidationStatus.FAIL,
                        message="Dataset is empty; cannot validate schema",
                        details={"row_count": 0},
                    )
                ]
            df = pd.DataFrame(data)

        # 1. Required Columns Check
        cols_lower = {str(col).lower(): col for col in df.columns}
        missing_cols = [req for req in REQUIRED_MARKET_COLUMNS if req not in cols_lower]

        if missing_cols:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.SCHEMA_REQUIRED_COLUMNS.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=f"Missing required columns: {missing_cols}",
                    details={
                        "missing_columns": missing_cols,
                        "present_columns": list(df.columns),
                    },
                )
            )
            return results

        results.append(
            ValidationResult(
                rule_id=ValidationRuleId.SCHEMA_REQUIRED_COLUMNS.value,
                severity=ValidationSeverity.INFO,
                status=ValidationStatus.PASS,
                message="All required market columns are present",
                details={"columns": list(df.columns)},
            )
        )

        # 2. Missing Values Check (Null/NaN)
        missing_counts: dict[str, int] = {}
        for col in REQUIRED_MARKET_COLUMNS:
            actual_col = cols_lower[col]
            null_count = int(df[actual_col].isna().sum())
            if null_count > 0:
                missing_counts[col] = null_count

        if missing_counts:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_MISSING_VALUES.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=f"Missing/null values detected: {missing_counts}",
                    details={"missing_counts": missing_counts},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.DATA_MISSING_VALUES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="No missing or null values in market columns",
                    details={"row_count": len(df)},
                )
            )

        # 3. Data Types Check (Numeric OHLCV, valid Datetime)
        type_errors: list[str] = []
        for col in ["open", "high", "low", "close", "volume"]:
            actual_col = cols_lower[col]
            series = pd.to_numeric(df[actual_col], errors="coerce")
            nan_after_num = int(series.isna().sum()) - missing_counts.get(col, 0)
            if nan_after_num > 0:
                type_errors.append(
                    f"Column '{col}' has {nan_after_num} non-numeric values"
                )

        if type_errors:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.SCHEMA_DATA_TYPES.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message="Non-numeric values found in OHLCV columns",
                    details={"errors": type_errors},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.SCHEMA_DATA_TYPES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="All OHLCV price and volume columns are numeric",
                    details={},
                )
            )

        # 4. Symbol Validation
        sym_col = cols_lower.get("symbol")
        if sym_col is not None:
            unique_syms = [
                str(s) for s in df[sym_col].dropna().unique() if str(s).strip()
            ]
            if not unique_syms or int(df[sym_col].isna().sum()) > 0:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.SCHEMA_SYMBOL_VALID.value,
                        severity=ValidationSeverity.ERROR,
                        status=ValidationStatus.FAIL,
                        message="Dataset contains empty or null ticker symbols",
                        details={"unique_symbols": unique_syms},
                    )
                )
            elif len(unique_syms) > 1:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.SCHEMA_SYMBOL_VALID.value,
                        severity=ValidationSeverity.WARNING,
                        status=ValidationStatus.WARN,
                        message=f"Dataset contains mixed symbols: {unique_syms}",
                        details={"unique_symbols": unique_syms},
                    )
                )
            elif (
                self.config.expected_symbol
                and unique_syms[0] != self.config.expected_symbol
            ):
                exp_s = self.config.expected_symbol
                act_s = unique_syms[0]
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.SCHEMA_SYMBOL_VALID.value,
                        severity=ValidationSeverity.ERROR,
                        status=ValidationStatus.FAIL,
                        message=f"Symbol mismatch: expected '{exp_s}', got '{act_s}'",
                        details={
                            "expected": exp_s,
                            "actual": act_s,
                        },
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.SCHEMA_SYMBOL_VALID.value,
                        severity=ValidationSeverity.INFO,
                        status=ValidationStatus.PASS,
                        message=f"Valid symbol '{unique_syms[0]}'",
                        details={"symbol": unique_syms[0]},
                    )
                )

        return results
