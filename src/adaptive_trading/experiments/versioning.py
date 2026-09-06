"""Factory builders for version contracts, manifests, and environment capture."""

import platform
import subprocess
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.experiments.fingerprint import (
    compute_dataset_fingerprint,
    compute_dict_fingerprint,
    compute_file_fingerprint,
)
from adaptive_trading.experiments.models import (
    DatasetVersion,
    EnvironmentMetadata,
    ExecutionConfigVersion,
    ExperimentManifest,
    ExperimentResult,
    FeatureVersion,
    ModelVersion,
    RiskConfigVersion,
    RuntimeConfigVersion,
    StrategyVersion,
)
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.runtime.config import RuntimeConfig
from adaptive_trading.strategy.config import StrategyConfig


def build_dataset_version(
    candles: Sequence[Candle],
    name: str = "dataset_v1",
    source: str = "Angel One SmartAPI",
    dataset_id: str | None = None,
) -> DatasetVersion:
    """Build a DatasetVersion contract from a sequence of market candles."""
    if not candles:
        now = datetime.now(timezone.utc)
        return DatasetVersion(
            dataset_id=dataset_id or f"ds_{uuid.uuid4().hex[:6]}",
            name=name,
            source=source,
            symbol="UNKNOWN",
            exchange="NSE",
            timeframe="5m",
            start_timestamp=now,
            end_timestamp=now,
            row_count=0,
            fingerprint=compute_dataset_fingerprint([]),
        )

    symbol = candles[0].symbol
    timeframe = (
        candles[0].timeframe.value
        if isinstance(candles[0].timeframe, TimeFrame)
        else str(candles[0].timeframe)
    )
    start_ts = min(c.timestamp for c in candles)
    end_ts = max(c.timestamp for c in candles)
    fp = compute_dataset_fingerprint(candles)

    return DatasetVersion(
        dataset_id=dataset_id or f"ds_{fp[:8]}",
        name=name,
        source=source,
        symbol=symbol,
        exchange="NSE",
        timeframe=timeframe,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        row_count=len(candles),
        fingerprint=fp,
    )


def build_feature_version(
    feature_set_name: str = "technical_features_v1",
    feature_version: str = "v1",
    parameters: dict[str, Any] | None = None,
) -> FeatureVersion:
    """Build FeatureVersion contract capturing feature configuration."""
    params = parameters or {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14,
        "volatility_lookback": 20,
    }
    fp = compute_dict_fingerprint(
        {"set_name": feature_set_name, "version": feature_version, **params}
    )
    return FeatureVersion(
        feature_set_name=feature_set_name,
        feature_version=feature_version,
        feature_schema_version="v1",
        feature_parameters=params,
        fingerprint=fp,
    )


def build_model_version(
    artifact_path: Path | str | None = None,
    model_name: str = "logistic_regression",
    model_version: str = "v1",
    model_type: str = "LogisticRegression",
    training_dataset_fingerprint: str = "",
    feature_version: str = "v1",
    parameters: dict[str, Any] | None = None,
) -> ModelVersion:
    """Build ModelVersion contract capturing model metadata and artifact hash."""
    params = parameters or {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    art_path_str = str(artifact_path) if artifact_path else ""
    art_fp = ""

    if artifact_path:
        p = Path(artifact_path)
        if p.is_file():
            art_fp = compute_file_fingerprint(p)
        elif (p / "model.joblib").is_file():
            art_fp = compute_file_fingerprint(p / "model.joblib")
        elif (p / "metadata.json").is_file():
            art_fp = compute_file_fingerprint(p / "metadata.json")

    return ModelVersion(
        model_id=f"{model_name}_{model_version}",
        model_name=model_name,
        model_version=model_version,
        model_type=model_type,
        training_dataset_fingerprint=training_dataset_fingerprint,
        feature_version=feature_version,
        model_parameters=params,
        artifact_path=art_path_str,
        artifact_fingerprint=art_fp,
    )


def build_strategy_version(
    config: StrategyConfig | None = None,
    strategy_name: str = "probability_strategy",
    strategy_version: str = "v1",
    parameters: dict[str, Any] | None = None,
) -> StrategyVersion:
    """Build StrategyVersion contract capturing strategy rules."""
    params = parameters or (
        config.model_dump(mode="json")
        if config
        else {
            "long_threshold": 0.60,
            "short_threshold": 0.40,
            "min_confidence": 0.60,
            "allow_short": True,
        }
    )
    fp = compute_dict_fingerprint(
        {"name": strategy_name, "version": strategy_version, **params}
    )
    return StrategyVersion(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        parameters=params,
        fingerprint=fp,
    )


def build_risk_version(
    config: RiskConfig | None = None,
    risk_version: str = "v1",
) -> RiskConfigVersion:
    """Build RiskConfigVersion contract capturing risk limits."""
    cfg = config or RiskConfig()
    params = cfg.model_dump(mode="json")
    fp = compute_dict_fingerprint({"version": risk_version, **params})
    return RiskConfigVersion(
        risk_version=risk_version,
        parameters=params,
        fingerprint=fp,
    )


def build_execution_version(
    config: ExecutionConfig | None = None,
) -> ExecutionConfigVersion:
    """Build ExecutionConfigVersion contract."""
    cfg = config or ExecutionConfig()
    params = cfg.model_dump(mode="json")
    fp = compute_dict_fingerprint(params)
    return ExecutionConfigVersion(
        execution_mode="PAPER",
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        default_exchange=cfg.default_exchange,
        parameters=params,
        fingerprint=fp,
    )


def build_runtime_version(
    config: RuntimeConfig | None = None,
) -> RuntimeConfigVersion:
    """Build RuntimeConfigVersion contract."""
    cfg = config or RuntimeConfig()
    params = cfg.model_dump(mode="json")
    fp = compute_dict_fingerprint(params)
    return RuntimeConfigVersion(
        runtime_mode=cfg.mode.value if hasattr(cfg.mode, "value") else str(cfg.mode),
        warmup_period=cfg.warmup_period,
        deduplicate_events=cfg.deduplicate_events,
        checkpoint_enabled=cfg.checkpoint_enabled,
        fail_fast=cfg.fail_fast,
        parameters=params,
        fingerprint=fp,
    )


def capture_environment_metadata(
    random_seed: int | None = 42,
) -> EnvironmentMetadata:
    """Capture non-sensitive environment and git metadata."""
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None

    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if commit_res.returncode == 0:
            git_commit = commit_res.stdout.strip()

        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if branch_res.returncode == 0:
            git_branch = branch_res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if status_res.returncode == 0:
            git_dirty = bool(status_res.stdout.strip())
    except Exception:
        pass

    return EnvironmentMetadata(
        os_platform=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        application_version="0.1.0",
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        random_seed=random_seed,
    )


def build_experiment_result(
    experiment_id: str,
    result_data: Any,
) -> ExperimentResult:
    """Build ExperimentResult contract from RunSummary or dict."""
    if hasattr(result_data, "model_dump"):
        data = result_data.model_dump(mode="json")
    elif isinstance(result_data, dict):
        data = result_data
    else:
        data = {
            "final_equity": getattr(result_data, "final_equity", 100000.0),
            "net_pnl": getattr(result_data, "net_pnl", 0.0),
            "trade_count": getattr(result_data, "trade_count", 0),
            "max_drawdown": getattr(result_data, "max_drawdown", 0.0),
            "win_rate": getattr(result_data, "win_rate", None),
            "sharpe_ratio": getattr(result_data, "sharpe_ratio", None),
        }

    init_eq = float(data.get("initial_equity", 100_000.0))
    fin_eq = float(data.get("final_equity", init_eq))
    net_pnl = float(data.get("net_pnl", fin_eq - init_eq))
    tc = int(data.get("trade_count", data.get("orders_filled", 0)))
    wr = float(data["win_rate"]) if data.get("win_rate") is not None else None
    dd = float(data.get("max_drawdown", 0.0))
    sharpe = (
        float(data["sharpe_ratio"]) if data.get("sharpe_ratio") is not None else None
    )

    res_dict = {
        "experiment_id": experiment_id,
        "initial_equity": init_eq,
        "final_equity": fin_eq,
        "net_pnl": net_pnl,
        "trade_count": tc,
        "win_rate": wr,
        "max_drawdown": dd,
        "sharpe_ratio": sharpe,
    }
    fp = compute_dict_fingerprint(res_dict)

    return ExperimentResult(
        experiment_id=experiment_id,
        initial_equity=init_eq,
        final_equity=fin_eq,
        net_pnl=net_pnl,
        trade_count=tc,
        win_rate=wr,
        max_drawdown=dd,
        sharpe_ratio=sharpe,
        result_fingerprint=fp,
    )


def build_experiment_manifest(
    experiment_id: str,
    dataset: DatasetVersion,
    features: FeatureVersion,
    model: ModelVersion,
    strategy: StrategyVersion,
    risk: RiskConfigVersion,
    execution: ExecutionConfigVersion,
    runtime: RuntimeConfigVersion,
    environment: EnvironmentMetadata | None = None,
    result: ExperimentResult | None = None,
) -> ExperimentManifest:
    """Bundle all component versions into a manifest and calculate hash."""
    env = environment or capture_environment_metadata()
    manifest_dict = {
        "experiment_id": experiment_id,
        "dataset_fp": dataset.fingerprint,
        "features_fp": features.fingerprint,
        "model_fp": model.artifact_fingerprint or model.model_id,
        "strategy_fp": strategy.fingerprint,
        "risk_fp": risk.fingerprint,
        "execution_fp": execution.fingerprint,
        "runtime_fp": runtime.fingerprint,
    }
    fp = compute_dict_fingerprint(manifest_dict)

    return ExperimentManifest(
        experiment_id=experiment_id,
        manifest_version="v1",
        dataset=dataset,
        features=features,
        model=model,
        strategy=strategy,
        risk=risk,
        execution=execution,
        runtime=runtime,
        environment=env,
        result=result,
        manifest_fingerprint=fp,
    )


def verify_compatibility(manifest: ExperimentManifest) -> list[str]:
    """Verify cross-version compatibility for models, features, and datasets."""
    issues: list[str] = []

    # Check model feature version expectation vs feature version
    if manifest.model.feature_version != manifest.features.feature_version:
        issues.append(
            f"Feature version mismatch: Model '{manifest.model.model_id}' "
            f"expects feature version '{manifest.model.feature_version}', "
            f"but experiment specifies '{manifest.features.feature_version}'"
        )

    # Check dataset row count sufficiency
    if manifest.dataset.row_count < manifest.runtime.warmup_period:
        issues.append(
            f"Dataset insufficient: Dataset has {manifest.dataset.row_count} rows, "
            f"which is less than runtime warmup period {manifest.runtime.warmup_period}"
        )

    return issues
