"""Structured validation results, aggregate reports, and artifact persistence."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.validation.config import ValidationPolicy
from adaptive_trading.validation.rules import (
    ValidationSeverity,
    ValidationStatus,
)


class ValidationResult(BaseModel):
    """Outcome and diagnostics from a single validation rule check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(description="Identifier of the executed rule")
    severity: ValidationSeverity = Field(
        description="Severity classification of the rule"
    )
    status: ValidationStatus = Field(description="Pass/Warn/Fail outcome")
    message: str = Field(description="Human-readable explanation of result")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured diagnostic metadata and offending records",
    )


class ValidationReport(BaseModel):
    """Aggregate validation report combining multiple check outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validation_id: str = Field(
        default_factory=lambda: f"val_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this validation execution",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when validation was executed",
    )
    dataset_identifier: str = Field(
        default="unknown_dataset",
        description="Name or path of dataset or model being evaluated",
    )
    policy: ValidationPolicy = Field(
        default=ValidationPolicy.NORMAL,
        description="Gating policy applied to evaluate overall eligibility",
    )
    overall_status: ValidationStatus = Field(
        description="Overall aggregate validation status (PASS, WARN, FAIL)"
    )
    checks: list[ValidationResult] = Field(
        default_factory=list,
        description="List of individual rule validation outcomes",
    )
    summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of checks by status and severity",
    )
    dataset_fingerprint: str | None = Field(
        default=None,
        description="Deterministic SHA-256 fingerprint of the validated dataset",
    )

    @classmethod
    def create(
        cls,
        checks: list[ValidationResult],
        dataset_identifier: str = "dataset",
        policy: ValidationPolicy = ValidationPolicy.NORMAL,
        dataset_fingerprint: str | None = None,
        validation_id: str | None = None,
    ) -> "ValidationReport":
        """Compute aggregate status, counts, and instantiate ValidationReport."""
        passed = sum(1 for c in checks if c.status == ValidationStatus.PASS)
        warns = sum(1 for c in checks if c.status == ValidationStatus.WARN)
        fails = sum(1 for c in checks if c.status == ValidationStatus.FAIL)
        criticals = sum(
            1
            for c in checks
            if c.status == ValidationStatus.FAIL
            and c.severity == ValidationSeverity.CRITICAL
        )

        if fails > 0:
            overall = ValidationStatus.FAIL
        elif warns > 0:
            overall = ValidationStatus.WARN
        else:
            overall = ValidationStatus.PASS

        counts = {
            "total": len(checks),
            "passed": passed,
            "warnings": warns,
            "failed": fails,
            "critical": criticals,
        }

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        val_id = validation_id or f"val_{ts_str}_{uuid.uuid4().hex[:6]}"

        return cls(
            validation_id=val_id,
            timestamp=datetime.now(timezone.utc),
            dataset_identifier=dataset_identifier,
            policy=policy,
            overall_status=overall,
            checks=checks,
            summary=counts,
            dataset_fingerprint=dataset_fingerprint,
        )

    def is_allowed(self, policy: ValidationPolicy | None = None) -> bool:
        """Determine if execution is permitted under the active policy."""
        active_policy = policy or self.policy

        if active_policy == ValidationPolicy.STRICT:
            return self.overall_status == ValidationStatus.PASS

        if active_policy == ValidationPolicy.NORMAL:
            has_blocking = any(
                c.status == ValidationStatus.FAIL
                and c.severity
                in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                for c in self.checks
            )
            return not has_blocking

        if active_policy == ValidationPolicy.LENIENT:
            has_critical = any(
                c.status == ValidationStatus.FAIL
                and c.severity == ValidationSeverity.CRITICAL
                for c in self.checks
            )
            return not has_critical

        return True

    def to_text_summary(self) -> str:
        """Render a formatted human-readable summary table."""
        lines = []
        lines.append("=" * 65)
        lines.append("           DATA & MODEL QUALITY VALIDATION REPORT            ")
        lines.append("=" * 65)
        lines.append(f"Validation ID    : {self.validation_id}")
        lines.append(f"Target Identifier: {self.dataset_identifier}")
        lines.append(f"Policy           : {self.policy.value}")
        lines.append(f"Overall Status   : {self.overall_status.value}")
        if self.dataset_fingerprint:
            lines.append(f"Fingerprint      : {self.dataset_fingerprint[:16]}...")
        lines.append("-" * 65)
        lines.append(f"{'RULE ID':<30} {'STATUS':<8} {'SEVERITY':<10} {'MESSAGE'}")
        lines.append("-" * 65)
        for c in self.checks:
            lines.append(
                f"{c.rule_id:<30} {c.status.value:<8} "
                f"{c.severity.value:<10} {c.message}"
            )
        lines.append("-" * 65)
        s = self.summary
        lines.append(
            f"Summary: Total={s.get('total', 0)} | Passed={s.get('passed', 0)} | "
            f"Warnings={s.get('warnings', 0)} | Failed={s.get('failed', 0)} "
            f"(Critical={s.get('critical', 0)})"
        )
        lines.append(f"Execution Allowed ({self.policy.value}): {self.is_allowed()}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def save(self, base_dir: Path | str = "artifacts/validation") -> Path:
        """Persist validation report and summary text to artifact directory."""
        dir_path = Path(base_dir) / self.validation_id
        dir_path.mkdir(parents=True, exist_ok=True)

        report_file = dir_path / "report.json"
        summary_file = dir_path / "summary.txt"

        report_file.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        summary_file.write_text(self.to_text_summary(), encoding="utf-8")

        return dir_path
