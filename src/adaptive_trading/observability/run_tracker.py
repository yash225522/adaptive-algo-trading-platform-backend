"""Execution run tracking, lifecycle state machine, and persistence."""

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.observability.exceptions import (
    InvalidRunTransitionError,
    RunNotFoundError,
)

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    """Lifecycle states of a tracked execution run."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunType(StrEnum):
    """Classification of execution run types."""

    BACKTEST = "BACKTEST"
    PAPER_REPLAY = "PAPER_REPLAY"
    PAPER_TRADING = "PAPER_TRADING"
    ML_EXPERIMENT = "ML_EXPERIMENT"


class RunSummary(BaseModel):
    """Aggregated operational and financial statistics for a completed run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    events_processed: int = Field(default=0, ge=0)
    predictions_generated: int = Field(default=0, ge=0)
    signals_generated: int = Field(default=0, ge=0)
    risk_rejections: int = Field(default=0, ge=0)
    orders_submitted: int = Field(default=0, ge=0)
    orders_filled: int = Field(default=0, ge=0)
    orders_rejected: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    initial_equity: float = Field(default=0.0)
    final_equity: float = Field(default=0.0)
    net_pnl: float = Field(default=0.0)
    trade_count: int = Field(default=0, ge=0)
    max_drawdown: float = Field(default=0.0)


class RunRecord(BaseModel):
    """Auditable domain record tracking an execution run and its metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique execution run identifier",
    )
    run_type: RunType = Field(description="Run classification type")
    status: RunStatus = Field(
        default=RunStatus.CREATED, description="Active lifecycle state"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when run record was initiated",
    )
    ended_at: datetime | None = Field(
        default=None, description="Timestamp when run concluded"
    )
    duration_seconds: float | None = Field(
        default=None, description="Total execution duration in seconds"
    )
    model_name: str | None = Field(default=None, description="Name of ML model used")
    model_version: str | None = Field(
        default=None, description="Version of ML model used"
    )
    strategy_name: str | None = Field(
        default=None, description="Name of strategy evaluated"
    )
    strategy_version: str | None = Field(
        default=None, description="Version of strategy evaluated"
    )
    data_start: datetime | None = Field(
        default=None, description="Start timestamp of market data"
    )
    data_end: datetime | None = Field(
        default=None, description="End timestamp of market data"
    )
    summary: RunSummary | None = Field(
        default=None, description="Aggregated run summary"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Supplementary metadata"
    )
    error_summary: str | None = Field(
        default=None, description="Description of failure reason"
    )

    @field_validator("started_at", "ended_at", "data_start", "data_end")
    @classmethod
    def validate_timezone_aware(cls, v: datetime | None) -> datetime | None:
        """Ensure timestamps are timezone-aware if present."""
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("RunRecord timestamps must be timezone-aware")
        return v


# Valid state transitions
VALID_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.CANCELLED},
    RunStatus.RUNNING: {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}


class RunTracker:
    """Manages run lifecycle transitions, audit logging, and persistence."""

    def __init__(self, base_dir: Path | str = "artifacts/runs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_runs: dict[str, RunRecord] = {}

    def _persist_run(self, run: RunRecord) -> None:
        """Save run record to JSON on disk."""
        self._in_memory_runs[run.run_id] = run
        run_dir = self.base_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_file = run_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(run.model_dump(mode="json"), f, indent=2)

    def create_run(
        self,
        run_type: RunType,
        run_id: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        strategy_name: str | None = None,
        strategy_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Initiate a new execution run record in CREATED status."""
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        active_id = run_id or f"run_{now_str}_{uuid.uuid4().hex[:6]}"
        record = RunRecord(
            run_id=active_id,
            run_type=run_type,
            status=RunStatus.CREATED,
            started_at=datetime.now(timezone.utc),
            model_name=model_name,
            model_version=model_version,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            metadata=metadata or {},
        )
        self._persist_run(record)
        return record

    def start_run(self, run_id: str) -> RunRecord:
        """Transition run from CREATED to RUNNING."""
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"Run '{run_id}' not found")

        if RunStatus.RUNNING not in VALID_TRANSITIONS[run.status]:
            raise InvalidRunTransitionError(
                f"Cannot transition run '{run_id}' "
                f"from {run.status} to {RunStatus.RUNNING}"
            )

        updated = run.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "started_at": datetime.now(timezone.utc),
            }
        )
        self._persist_run(updated)
        return updated

    def complete_run(
        self,
        run_id: str,
        summary: RunSummary | None = None,
        data_start: datetime | None = None,
        data_end: datetime | None = None,
    ) -> RunRecord:
        """Transition run from RUNNING to COMPLETED."""
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"Run '{run_id}' not found")

        if RunStatus.COMPLETED not in VALID_TRANSITIONS[run.status]:
            raise InvalidRunTransitionError(
                f"Cannot transition run '{run_id}' "
                f"from {run.status} to {RunStatus.COMPLETED}"
            )

        ended = datetime.now(timezone.utc)
        duration = (ended - run.started_at).total_seconds()

        updated = run.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "ended_at": ended,
                "duration_seconds": duration,
                "summary": summary,
                "data_start": data_start,
                "data_end": data_end,
            }
        )
        self._persist_run(updated)
        return updated

    def fail_run(
        self,
        run_id: str,
        error: Exception | str,
        summary: RunSummary | None = None,
    ) -> RunRecord:
        """Transition run from RUNNING to FAILED."""
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"Run '{run_id}' not found")

        if RunStatus.FAILED not in VALID_TRANSITIONS[run.status]:
            raise InvalidRunTransitionError(
                f"Cannot transition run '{run_id}' "
                f"from {run.status} to {RunStatus.FAILED}"
            )

        ended = datetime.now(timezone.utc)
        duration = (ended - run.started_at).total_seconds()
        err_msg = str(error)

        updated = run.model_copy(
            update={
                "status": RunStatus.FAILED,
                "ended_at": ended,
                "duration_seconds": duration,
                "error_summary": err_msg,
                "summary": summary,
            }
        )
        self._persist_run(updated)
        return updated

    def cancel_run(
        self,
        run_id: str,
        reason: str = "",
        summary: RunSummary | None = None,
    ) -> RunRecord:
        """Transition run to CANCELLED."""
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"Run '{run_id}' not found")

        if RunStatus.CANCELLED not in VALID_TRANSITIONS[run.status]:
            raise InvalidRunTransitionError(
                f"Cannot transition run '{run_id}' "
                f"from {run.status} to {RunStatus.CANCELLED}"
            )

        ended = datetime.now(timezone.utc)
        duration = (ended - run.started_at).total_seconds()

        updated = run.model_copy(
            update={
                "status": RunStatus.CANCELLED,
                "ended_at": ended,
                "duration_seconds": duration,
                "error_summary": reason or "Run cancelled by user/system",
                "summary": summary,
            }
        )
        self._persist_run(updated)
        return updated

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by ID from memory or disk."""
        if run_id in self._in_memory_runs:
            return self._in_memory_runs[run_id]

        meta_file = self.base_dir / run_id / "metadata.json"
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                record = RunRecord(**data)
                self._in_memory_runs[run_id] = record
                return record
            except Exception as exc:
                logger.warning("Failed to load run metadata for %s: %s", run_id, exc)
                return None
        return None

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        """List recently recorded runs sorted chronologically descending."""
        runs: list[RunRecord] = []
        if self.base_dir.is_dir():
            for child in sorted(
                self.base_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                if child.is_dir():
                    rec = self.get_run(child.name)
                    if rec is not None:
                        runs.append(rec)
                        if len(runs) >= limit:
                            break
        return runs
