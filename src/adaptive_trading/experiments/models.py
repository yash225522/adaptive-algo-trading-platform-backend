"""Domain contracts for experiment manifests, versioning, and reproducibility."""

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExperimentType(StrEnum):
    """Classification of experiment types."""

    BACKTEST = "BACKTEST"
    PAPER_REPLAY = "PAPER_REPLAY"
    PAPER_TRADING = "PAPER_TRADING"
    ML_EXPERIMENT = "ML_EXPERIMENT"


class ExperimentStatus(StrEnum):
    """Lifecycle status of an experiment."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ReproducibilityLevel(StrEnum):
    """Classification level of run reproducibility."""

    EXACT = "EXACT"
    PARTIAL = "PARTIAL"
    NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"


class DatasetVersion(BaseModel):
    """Contract capturing the historical dataset metadata and fingerprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(description="Unique dataset version identifier")
    name: str = Field(description="Human-readable dataset name")
    source: str = Field(
        default="Angel One SmartAPI", description="Origin data provider"
    )
    symbol: str = Field(description="Target market symbol")
    exchange: str = Field(default="NSE", description="Exchange code")
    timeframe: str = Field(default="5m", description="Candle interval")
    start_timestamp: datetime = Field(description="Start time of data span")
    end_timestamp: datetime = Field(description="End time of data span")
    row_count: int = Field(ge=0, description="Total number of candle rows")
    schema_version: str = Field(default="v1", description="Dataset schema version")
    fingerprint: str = Field(description="Deterministic SHA-256 data fingerprint")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("start_timestamp", "end_timestamp", "created_at")
    @classmethod
    def validate_tz(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("DatasetVersion timestamps must be timezone-aware")
        return v


class FeatureVersion(BaseModel):
    """Contract capturing the feature pipeline metadata and fingerprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_set_name: str = Field(
        default="technical_features_v1", description="Feature catalog name"
    )
    feature_version: str = Field(default="v1", description="Feature version")
    feature_schema_version: str = Field(
        default="v1", description="Feature schema layout version"
    )
    feature_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Feature configuration parameters"
    )
    fingerprint: str = Field(description="Deterministic SHA-256 fingerprint")


class ModelVersion(BaseModel):
    """Contract capturing ML model architecture and artifact fingerprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Unique model identifier")
    model_name: str = Field(description="Model architecture name")
    model_version: str = Field(default="v1", description="Model release version")
    model_type: str = Field(description="Classifier / Regressor type")
    training_dataset_fingerprint: str = Field(
        default="", description="Fingerprint of dataset model was trained on"
    )
    feature_version: str = Field(
        default="v1", description="Feature set version expected by model"
    )
    model_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Model hyper-parameters"
    )
    artifact_path: str = Field(default="", description="Path to binary model file")
    artifact_fingerprint: str = Field(
        default="", description="SHA-256 hash of model artifact binary"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at")
    @classmethod
    def validate_tz(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("ModelVersion timestamp must be timezone-aware")
        return v


class StrategyVersion(BaseModel):
    """Contract capturing trading strategy rules and parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_name: str = Field(description="Strategy name")
    strategy_version: str = Field(default="v1", description="Strategy version")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters"
    )
    fingerprint: str = Field(description="Deterministic SHA-256 fingerprint")


class RiskConfigVersion(BaseModel):
    """Contract capturing risk management parameters and limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_version: str = Field(default="v1", description="Risk schema version")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Risk parameters and thresholds"
    )
    fingerprint: str = Field(description="Deterministic SHA-256 fingerprint")


class ExecutionConfigVersion(BaseModel):
    """Contract capturing execution engine parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_mode: str = Field(default="PAPER", description="Execution mode (PAPER)")
    commission_bps: float = Field(default=3.0, ge=0.0)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    default_exchange: str = Field(default="NSE")
    parameters: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str = Field(description="Deterministic SHA-256 fingerprint")


class RuntimeConfigVersion(BaseModel):
    """Contract capturing event loop and replay runner parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_mode: str = Field(default="REPLAY")
    warmup_period: int = Field(default=20, ge=0)
    deduplicate_events: bool = Field(default=True)
    checkpoint_enabled: bool = Field(default=True)
    fail_fast: bool = Field(default=True)
    parameters: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str = Field(description="Deterministic SHA-256 fingerprint")


class EnvironmentMetadata(BaseModel):
    """Non-sensitive runtime platform environment metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    os_platform: str = Field(description="Operating system name")
    python_version: str = Field(description="Python interpreter version")
    application_version: str = Field(
        default="0.1.0", description="Trading platform version"
    )
    git_commit: str | None = Field(default=None, description="Active git commit hash")
    git_branch: str | None = Field(default=None, description="Active git branch name")
    git_dirty: bool | None = Field(
        default=None, description="Whether git tree had uncommitted changes"
    )
    random_seed: int | None = Field(
        default=None, description="Deterministic random seed"
    )


class ExperimentResult(BaseModel):
    """Aggregated financial and statistical performance metrics of an experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Associated experiment ID")
    initial_equity: float = Field(default=100_000.0)
    final_equity: float = Field(default=100_000.0)
    net_pnl: float = Field(default=0.0)
    trade_count: int = Field(default=0, ge=0)
    win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_drawdown: float = Field(default=0.0, ge=0.0)
    sharpe_ratio: float | None = Field(default=None)
    result_fingerprint: str = Field(
        default="", description="Deterministic fingerprint of output results"
    )


class ExperimentManifest(BaseModel):
    """Complete, self-contained reproducibility manifest for an experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique experiment identifier")
    manifest_version: str = Field(default="v1", description="Manifest schema version")
    dataset: DatasetVersion
    features: FeatureVersion
    model: ModelVersion
    strategy: StrategyVersion
    risk: RiskConfigVersion
    execution: ExecutionConfigVersion
    runtime: RuntimeConfigVersion
    environment: EnvironmentMetadata
    result: ExperimentResult | None = Field(default=None)
    manifest_fingerprint: str = Field(
        default="", description="Deterministic SHA-256 manifest hash"
    )


class Experiment(BaseModel):
    """Top-level tracked experiment entity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(
        default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}",
        description="Unique experiment ID",
    )
    name: str = Field(description="Experiment name")
    description: str = Field(default="", description="Experiment rationale/description")
    experiment_type: ExperimentType = Field(
        description="Type classification of experiment"
    )
    status: ExperimentStatus = Field(
        default=ExperimentStatus.CREATED, description="Lifecycle status"
    )
    manifest: ExperimentManifest | None = Field(
        default=None, description="Reproducibility manifest"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    @field_validator("created_at", "started_at", "completed_at")
    @classmethod
    def validate_tz(cls, v: datetime | None) -> datetime | None:
        """Ensure timestamps are timezone-aware if present."""
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("Experiment timestamps must be timezone-aware")
        return v
