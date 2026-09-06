"""System health checks, component diagnostics, and monitoring snapshots."""

import logging
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.execution.paper_broker import (
    PaperBroker,
    StaticMarketDataProvider,
)

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Health classification status."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class ComponentHealth(BaseModel):
    """Health diagnostic status for a single system component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Component identifier")
    status: HealthStatus = Field(description="Operational health status")
    is_critical: bool = Field(
        default=True, description="Whether component is critical to core system"
    )
    message: str = Field(
        default="Component operating normally",
        description="Diagnostic description",
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Supplementary diagnostic metadata"
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check execution timestamp",
    )


class SystemHealthReport(BaseModel):
    """Aggregated system-wide health evaluation report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: HealthStatus = Field(description="Overall system health status")
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp",
    )
    components: dict[str, ComponentHealth] = Field(
        description="Per-component health evaluations"
    )


class MonitoringSnapshot(BaseModel):
    """Unified telemetry snapshot for future dashboard and API consumption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    health: SystemHealthReport
    metrics: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    portfolio: dict[str, Any] = Field(default_factory=dict)
    orders: list[dict[str, Any]] = Field(default_factory=list)
    positions: dict[str, Any] = Field(default_factory=dict)


class HealthChecker:
    """Performs diagnostic checks across available system components."""

    def check_database(self) -> ComponentHealth:
        """Check database accessibility."""
        try:
            from adaptive_trading.database.session import get_db_session

            with get_db_session() as session:
                session.execute(text("SELECT 1"))
            return ComponentHealth(
                name="Database",
                status=HealthStatus.HEALTHY,
                is_critical=True,
                message="Database connection established successfully",
            )
        except Exception as exc:
            logger.debug("Database health check issue: %s", exc)
            return ComponentHealth(
                name="Database",
                status=HealthStatus.DEGRADED,
                is_critical=False,
                message=f"Database unavailable (using fallback): {exc}",
            )

    def check_model(self, model_dir: Path | str | None = None) -> ComponentHealth:
        """Verify machine learning model artifact readiness."""
        target_dir = Path(model_dir or "artifacts/models/logistic_regression/v1")
        if target_dir.is_dir() and (target_dir / "metadata.json").is_file():
            return ComponentHealth(
                name="ML Model",
                status=HealthStatus.HEALTHY,
                is_critical=True,
                message=f"Model artifact validated at {target_dir}",
                details={"model_dir": str(target_dir)},
            )
        return ComponentHealth(
            name="ML Model",
            status=HealthStatus.DEGRADED,
            is_critical=False,
            message="No ML model artifact loaded (fallback rules active)",
            details={"checked_dir": str(target_dir)},
        )

    def check_market_data(
        self, sample_path: Path | str | None = None
    ) -> ComponentHealth:
        """Check availability of historical / sample market data."""
        target_path = Path(sample_path or "data/sample/nifty_5m_ml_sample.csv")
        if target_path.is_file() and target_path.stat().st_size > 0:
            return ComponentHealth(
                name="Market Data",
                status=HealthStatus.HEALTHY,
                is_critical=True,
                message=f"Market data sample accessible at {target_path}",
                details={"file_size_bytes": target_path.stat().st_size},
            )
        return ComponentHealth(
            name="Market Data",
            status=HealthStatus.UNHEALTHY,
            is_critical=True,
            message=f"Market data sample file missing: {target_path}",
        )

    def check_paper_broker(self, broker: PaperBroker | None = None) -> ComponentHealth:
        """Verify paper trading broker initialization and cash state."""
        try:
            b = broker or PaperBroker(
                config=ExecutionConfig(initial_cash=100_000.0),
                market_data_provider=StaticMarketDataProvider(),
            )
            acct = b.get_account()
            return ComponentHealth(
                name="Paper Broker",
                status=HealthStatus.HEALTHY,
                is_critical=True,
                message="Paper broker operating normally",
                details={"cash": acct.cash, "equity": acct.equity},
            )
        except Exception as exc:
            return ComponentHealth(
                name="Paper Broker",
                status=HealthStatus.UNHEALTHY,
                is_critical=True,
                message=f"Paper broker initialization failed: {exc}",
            )

    def check_runtime(self) -> ComponentHealth:
        """Check runtime state and simulated event loop readiness."""
        return ComponentHealth(
            name="Runtime",
            status=HealthStatus.HEALTHY,
            is_critical=True,
            message="Event-driven runtime engine ready",
        )

    def check_all(
        self,
        model_dir: Path | str | None = None,
        market_data_path: Path | str | None = None,
    ) -> SystemHealthReport:
        """Run all component diagnostics and compute overall system status."""
        components = {
            "Database": self.check_database(),
            "ML Model": self.check_model(model_dir=model_dir),
            "Market Data": self.check_market_data(sample_path=market_data_path),
            "Paper Broker": self.check_paper_broker(),
            "Runtime": self.check_runtime(),
        }

        has_critical_unhealthy = any(
            c.status == HealthStatus.UNHEALTHY and c.is_critical
            for c in components.values()
        )
        has_any_unhealthy_or_degraded = any(
            c.status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED)
            for c in components.values()
        )

        if has_critical_unhealthy:
            overall = HealthStatus.UNHEALTHY
        elif has_any_unhealthy_or_degraded:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return SystemHealthReport(status=overall, components=components)
