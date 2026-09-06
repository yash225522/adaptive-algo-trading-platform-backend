"""Persistence registry for experiments, manifests, and comparative queries."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from adaptive_trading.experiments.exceptions import (
    ExperimentNotFoundError,
    InvalidExperimentStateError,
)
from adaptive_trading.experiments.models import (
    Experiment,
    ExperimentManifest,
    ExperimentResult,
    ExperimentStatus,
    ExperimentType,
)

logger = logging.getLogger(__name__)

# Valid experiment lifecycle transitions
VALID_EXPERIMENT_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.CREATED: {
        ExperimentStatus.RUNNING,
        ExperimentStatus.CANCELLED,
    },
    ExperimentStatus.RUNNING: {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
    },
    ExperimentStatus.COMPLETED: set(),
    ExperimentStatus.FAILED: set(),
    ExperimentStatus.CANCELLED: set(),
}


class ExperimentRegistry:
    """Manages experiment records, persistence, and manifest lookups."""

    def __init__(self, base_dir: Path | str = "artifacts/experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_experiments: dict[str, Experiment] = {}
        self._in_memory_manifests: dict[str, ExperimentManifest] = {}

    def _get_exp_dir(self, experiment_id: str) -> Path:
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _persist_experiment(self, exp: Experiment) -> None:
        self._in_memory_experiments[exp.experiment_id] = exp
        exp_dir = self._get_exp_dir(exp.experiment_id)
        exp_file = exp_dir / "experiment.json"
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(exp.model_dump(mode="json"), f, indent=2)

    def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        experiment_id: str | None = None,
        description: str = "",
        manifest: ExperimentManifest | None = None,
    ) -> Experiment:
        """Create a new experiment in CREATED state."""
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        active_id = experiment_id or f"exp_{now_str}_{uuid.uuid4().hex[:6]}"
        exp = Experiment(
            experiment_id=active_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            status=ExperimentStatus.CREATED,
            manifest=manifest,
            created_at=datetime.now(timezone.utc),
        )
        self._persist_experiment(exp)
        if manifest is not None:
            self.save_manifest(manifest)
        return exp

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Transition experiment to RUNNING state."""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found")

        if ExperimentStatus.RUNNING not in VALID_EXPERIMENT_TRANSITIONS[exp.status]:
            raise InvalidExperimentStateError(
                f"Cannot transition experiment '{experiment_id}' "
                f"from {exp.status} to {ExperimentStatus.RUNNING}"
            )

        updated = exp.model_copy(
            update={
                "status": ExperimentStatus.RUNNING,
                "started_at": datetime.now(timezone.utc),
            }
        )
        self._persist_experiment(updated)
        return updated

    def complete_experiment(
        self,
        experiment_id: str,
        result: ExperimentResult | None = None,
        manifest: ExperimentManifest | None = None,
    ) -> Experiment:
        """Transition experiment to COMPLETED state and record result."""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found")

        if ExperimentStatus.COMPLETED not in VALID_EXPERIMENT_TRANSITIONS[exp.status]:
            raise InvalidExperimentStateError(
                f"Cannot transition experiment '{experiment_id}' "
                f"from {exp.status} to {ExperimentStatus.COMPLETED}"
            )

        active_manifest = manifest or exp.manifest
        if active_manifest and result:
            active_manifest = active_manifest.model_copy(update={"result": result})
            self.save_manifest(active_manifest)

        updated = exp.model_copy(
            update={
                "status": ExperimentStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
                "manifest": active_manifest,
            }
        )
        self._persist_experiment(updated)
        if result is not None:
            self.save_result(result)
        return updated

    def fail_experiment(
        self,
        experiment_id: str,
        error: Exception | str,
    ) -> Experiment:
        """Transition experiment to FAILED state."""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found")

        if ExperimentStatus.FAILED not in VALID_EXPERIMENT_TRANSITIONS[exp.status]:
            raise InvalidExperimentStateError(
                f"Cannot transition experiment '{experiment_id}' "
                f"from {exp.status} to {ExperimentStatus.FAILED}"
            )

        updated = exp.model_copy(
            update={
                "status": ExperimentStatus.FAILED,
                "completed_at": datetime.now(timezone.utc),
                "description": f"{exp.description} [FAILED: {error}]".strip(),
            }
        )
        self._persist_experiment(updated)
        return updated

    def save_manifest(self, manifest: ExperimentManifest) -> None:
        """Persist experiment manifest JSON."""
        self._in_memory_manifests[manifest.experiment_id] = manifest
        exp_dir = self._get_exp_dir(manifest.experiment_id)
        man_file = exp_dir / "manifest.json"
        with open(man_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2)

    def save_result(self, result: ExperimentResult) -> None:
        """Persist experiment result JSON."""
        exp_dir = self._get_exp_dir(result.experiment_id)
        res_file = exp_dir / "result.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Retrieve experiment record by ID."""
        if experiment_id in self._in_memory_experiments:
            return self._in_memory_experiments[experiment_id]

        exp_file = self.base_dir / experiment_id / "experiment.json"
        if exp_file.is_file():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                exp = Experiment(**data)
                self._in_memory_experiments[experiment_id] = exp
                return exp
            except Exception as exc:
                logger.warning("Failed to parse experiment file %s: %s", exp_file, exc)
                return None
        return None

    def get_manifest(self, experiment_id: str) -> ExperimentManifest | None:
        """Retrieve manifest by experiment ID."""
        if experiment_id in self._in_memory_manifests:
            return self._in_memory_manifests[experiment_id]

        man_file = self.base_dir / experiment_id / "manifest.json"
        if man_file.is_file():
            try:
                with open(man_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                manifest = ExperimentManifest(**data)
                self._in_memory_manifests[experiment_id] = manifest
                return manifest
            except Exception as exc:
                logger.warning("Failed to parse manifest file %s: %s", man_file, exc)
                return None
        return None

    def list_experiments(self, limit: int = 20) -> list[Experiment]:
        """List tracked experiments sorted chronologically descending."""
        experiments: list[Experiment] = []
        if self.base_dir.is_dir():
            for child in sorted(
                self.base_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                if child.is_dir():
                    exp = self.get_experiment(child.name)
                    if exp is not None:
                        experiments.append(exp)
                        if len(experiments) >= limit:
                            break
        return experiments
