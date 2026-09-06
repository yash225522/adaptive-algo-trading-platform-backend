"""Configuration contracts for experiment management and reproducibility."""

from pydantic import BaseModel, ConfigDict, Field


class ExperimentConfig(BaseModel):
    """Configuration governing experiment manifests and reproducibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifacts_dir: str = Field(
        default="artifacts/experiments",
        description="Base directory where experiment manifests are stored",
    )
    strict_reproducibility: bool = Field(
        default=False,
        description="If True, reject runs without exact reproduction guarantee",
    )
    record_git_metadata: bool = Field(
        default=True,
        description="If True, capture git commit, branch, and status in manifests",
    )
    record_environment_metadata: bool = Field(
        default=True,
        description="If True, capture OS and Python runtime metadata",
    )
