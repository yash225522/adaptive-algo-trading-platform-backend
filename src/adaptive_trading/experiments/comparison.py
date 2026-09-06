"""Experiment comparison, configuration diffing, and reproducibility verification."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.experiments.models import (
    ExperimentManifest,
    ReproducibilityLevel,
)


class ExperimentComparison(BaseModel):
    """Comparison results between two experiment manifests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exp_a_id: str = Field(description="First experiment ID")
    exp_b_id: str = Field(description="Second experiment ID")
    config_diffs: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Differences in configurations"
    )
    performance_diffs: dict[str, Any] = Field(
        default_factory=dict, description="Differences in performance"
    )
    is_identical_config: bool = Field(
        default=False, description="Whether all configuration hashes match"
    )
    is_identical_result: bool = Field(
        default=False, description="Whether output result hashes match"
    )
    summary: str = Field(default="", description="Human-readable comparison summary")


class ExperimentComparator:
    """Compares experiment manifests and verifies reproducibility."""

    @staticmethod
    def compare(
        manifest_a: ExperimentManifest,
        manifest_b: ExperimentManifest,
    ) -> ExperimentComparison:
        """Compare two manifests for configuration and performance differences."""
        config_diffs: dict[str, dict[str, Any]] = {}

        # Dataset diff
        if manifest_a.dataset.fingerprint != manifest_b.dataset.fingerprint:
            config_diffs["dataset"] = {
                "a": {
                    "symbol": manifest_a.dataset.symbol,
                    "rows": manifest_a.dataset.row_count,
                    "fp": manifest_a.dataset.fingerprint[:8],
                },
                "b": {
                    "symbol": manifest_b.dataset.symbol,
                    "rows": manifest_b.dataset.row_count,
                    "fp": manifest_b.dataset.fingerprint[:8],
                },
            }

        # Features diff
        if manifest_a.features.fingerprint != manifest_b.features.fingerprint:
            config_diffs["features"] = {
                "a": {
                    "version": manifest_a.features.feature_version,
                    "fp": manifest_a.features.fingerprint[:8],
                },
                "b": {
                    "version": manifest_b.features.feature_version,
                    "fp": manifest_b.features.fingerprint[:8],
                },
            }

        # Model diff
        if (
            manifest_a.model.artifact_fingerprint
            != manifest_b.model.artifact_fingerprint
            or manifest_a.model.model_id != manifest_b.model.model_id
        ):
            config_diffs["model"] = {
                "a": {
                    "model_id": manifest_a.model.model_id,
                    "artifact_fp": (
                        manifest_a.model.artifact_fingerprint[:8]
                        if manifest_a.model.artifact_fingerprint
                        else "N/A"
                    ),
                },
                "b": {
                    "model_id": manifest_b.model.model_id,
                    "artifact_fp": (
                        manifest_b.model.artifact_fingerprint[:8]
                        if manifest_b.model.artifact_fingerprint
                        else "N/A"
                    ),
                },
            }

        # Strategy diff
        if manifest_a.strategy.fingerprint != manifest_b.strategy.fingerprint:
            config_diffs["strategy"] = {
                "a": {
                    "name": manifest_a.strategy.strategy_name,
                    "fp": manifest_a.strategy.fingerprint[:8],
                },
                "b": {
                    "name": manifest_b.strategy.strategy_name,
                    "fp": manifest_b.strategy.fingerprint[:8],
                },
            }

        # Risk diff
        if manifest_a.risk.fingerprint != manifest_b.risk.fingerprint:
            config_diffs["risk"] = {
                "a": {"fp": manifest_a.risk.fingerprint[:8]},
                "b": {"fp": manifest_b.risk.fingerprint[:8]},
            }

        # Execution diff
        if manifest_a.execution.fingerprint != manifest_b.execution.fingerprint:
            config_diffs["execution"] = {
                "a": {"fp": manifest_a.execution.fingerprint[:8]},
                "b": {"fp": manifest_b.execution.fingerprint[:8]},
            }

        # Runtime diff
        if manifest_a.runtime.fingerprint != manifest_b.runtime.fingerprint:
            config_diffs["runtime"] = {
                "a": {"fp": manifest_a.runtime.fingerprint[:8]},
                "b": {"fp": manifest_b.runtime.fingerprint[:8]},
            }

        # Performance diffs
        perf_diffs: dict[str, Any] = {}
        res_a = manifest_a.result
        res_b = manifest_b.result

        if res_a and res_b:
            perf_diffs["delta_net_pnl"] = res_b.net_pnl - res_a.net_pnl
            perf_diffs["delta_final_equity"] = res_b.final_equity - res_a.final_equity
            perf_diffs["delta_trade_count"] = res_b.trade_count - res_a.trade_count
            perf_diffs["delta_max_drawdown"] = res_b.max_drawdown - res_a.max_drawdown
            if res_a.win_rate is not None and res_b.win_rate is not None:
                perf_diffs["delta_win_rate"] = res_b.win_rate - res_a.win_rate
            if res_a.sharpe_ratio is not None and res_b.sharpe_ratio is not None:
                perf_diffs["delta_sharpe_ratio"] = (
                    res_b.sharpe_ratio - res_a.sharpe_ratio
                )

        is_ident_cfg = len(config_diffs) == 0
        is_ident_res = bool(
            res_a and res_b and res_a.result_fingerprint == res_b.result_fingerprint
        )

        diff_count = len(config_diffs)
        pnl_diff = perf_diffs.get("delta_net_pnl", 0.0)
        summary = (
            f"Compared {manifest_a.experiment_id} vs {manifest_b.experiment_id}: "
            f"{diff_count} config diffs; Net P&L Delta: {pnl_diff:+,.2f}"
        )

        return ExperimentComparison(
            exp_a_id=manifest_a.experiment_id,
            exp_b_id=manifest_b.experiment_id,
            config_diffs=config_diffs,
            performance_diffs=perf_diffs,
            is_identical_config=is_ident_cfg,
            is_identical_result=is_ident_res,
            summary=summary,
        )

    @staticmethod
    def verify_reproducibility(
        manifest: ExperimentManifest,
        target_manifest: ExperimentManifest | None = None,
    ) -> tuple[ReproducibilityLevel, list[str]]:
        """Verify whether an experiment can be reproduced exactly or partially."""
        reasons: list[str] = []

        if target_manifest is None:
            # Self-consistency check
            if not manifest.dataset.fingerprint:
                reasons.append("Dataset fingerprint is missing")
            if not manifest.strategy.fingerprint:
                reasons.append("Strategy fingerprint is missing")
            if not manifest.risk.fingerprint:
                reasons.append("Risk fingerprint is missing")
            if not manifest.execution.fingerprint:
                reasons.append("Execution fingerprint is missing")

            if reasons:
                return ReproducibilityLevel.NOT_REPRODUCIBLE, reasons
            return ReproducibilityLevel.EXACT, ["Manifest self-consistent"]

        # Cross-manifest comparison check
        core_mismatches = []
        if manifest.dataset.fingerprint != target_manifest.dataset.fingerprint:
            core_mismatches.append("Dataset fingerprint mismatch")
        if manifest.features.fingerprint != target_manifest.features.fingerprint:
            core_mismatches.append("Feature fingerprint mismatch")
        if (
            manifest.model.artifact_fingerprint
            != target_manifest.model.artifact_fingerprint
            or manifest.model.model_id != target_manifest.model.model_id
        ):
            core_mismatches.append("Model version or artifact hash mismatch")
        if manifest.strategy.fingerprint != target_manifest.strategy.fingerprint:
            core_mismatches.append("Strategy configuration mismatch")
        if manifest.risk.fingerprint != target_manifest.risk.fingerprint:
            core_mismatches.append("Risk configuration mismatch")
        if manifest.execution.fingerprint != target_manifest.execution.fingerprint:
            core_mismatches.append("Execution configuration mismatch")
        if manifest.runtime.fingerprint != target_manifest.runtime.fingerprint:
            core_mismatches.append("Runtime configuration mismatch")

        if core_mismatches:
            return ReproducibilityLevel.NOT_REPRODUCIBLE, core_mismatches

        # Check environment divergence
        env_diffs = []
        env_a = manifest.environment
        env_b = target_manifest.environment

        if env_a.git_commit != env_b.git_commit:
            c_a = env_a.git_commit[:7] if env_a.git_commit else "None"
            c_b = env_b.git_commit[:7] if env_b.git_commit else "None"
            env_diffs.append(f"Git commit differs: {c_a} vs {c_b}")

        if env_a.python_version != env_b.python_version:
            env_diffs.append(
                f"Python differs: {env_a.python_version} vs {env_b.python_version}"
            )

        if env_a.os_platform != env_b.os_platform:
            env_diffs.append(f"OS differs: {env_a.os_platform} vs {env_b.os_platform}")

        if env_diffs:
            return (
                ReproducibilityLevel.PARTIAL,
                ["Core match; environment differs: " + "; ".join(env_diffs)],
            )

        return (
            ReproducibilityLevel.EXACT,
            ["All dataset, model, config, and environment hashes match"],
        )
