"""Configuration contracts and defaults for observability and monitoring."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObservabilityConfig(BaseModel):
    """Configuration parameters governing logging, metrics, health, and tracking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    log_level: str = Field(
        default="INFO",
        description="Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    structured_logging: bool = Field(
        default=True,
        description="If True, formats logs as machine-readable JSON",
    )
    metrics_enabled: bool = Field(
        default=True,
        description="If True, records runtime counters, gauges, and latency histograms",
    )
    health_checks_enabled: bool = Field(
        default=True,
        description="If True, enables automated subsystem health checks",
    )
    run_tracking_enabled: bool = Field(
        default=True,
        description="If True, records and persists run lifecycles and metadata",
    )
    redact_sensitive_data: bool = Field(
        default=True,
        description="If True, masks API keys, JWT tokens, passwords, and secrets",
    )
    artifacts_dir: str = Field(
        default="artifacts/runs",
        description="Base filesystem directory for run artifacts and summaries",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure standard log level names."""
        v_upper = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v_upper not in allowed:
            raise ValueError(
                f"Invalid log level '{v}'. Allowed: {', '.join(sorted(allowed))}"
            )
        return v_upper
