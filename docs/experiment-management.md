# Experiment, Dataset & Model Version Management

The Experiment & Version Management layer (`src/adaptive_trading/experiments/`) provides comprehensive tracking, cryptographic fingerprinting, manifest creation, reproducibility verification, and comparative analytics across backtests, paper-trading runs, and ML experiments.

---

## 1. Core Architecture

An **Experiment** encapsulates every artifact, parameter, and environmental attribute that produced a specific trading outcome:

```text
                    Experiment
                         │
        ┌────────────────┼─────────────────┐
        ↓                ↓                 ↓
     Dataset           Model            Strategy
     Version           Version           Version
        │                │                 │
        └────────────────┼─────────────────┘
                         ↓
                  Risk Configuration
                         ↓
               Execution Configuration
                         ↓
                Runtime Configuration
                         ↓
                    Experiment
                      Result
```

---

## 2. Version Contracts & Fingerprinting

### Cryptographic Fingerprinting
All components generate deterministic **SHA-256** digests:
- **Datasets**: Derived from ordered canonical representations of `timestamp|symbol|open|high|low|close|volume`.
- **Configurations**: Derived from canonical JSON representations (alphabetically sorted keys, consistent separators, volatile fields excluded).
- **Model Artifacts**: Derived from binary file content hashes (`model.joblib`).

```python
from adaptive_trading.experiments.fingerprint import (
    compute_dataset_fingerprint,
    compute_dict_fingerprint,
    compute_file_fingerprint,
)
```

### Version Models

| Contract | Description | Key Attributes |
| :--- | :--- | :--- |
| `DatasetVersion` | Historical candle dataset metadata | `dataset_id`, `symbol`, `exchange`, `timeframe`, `row_count`, `fingerprint` |
| `FeatureVersion` | Technical indicator pipeline parameters | `feature_set_name`, `feature_version`, `parameters`, `fingerprint` |
| `ModelVersion` | Machine learning model architecture | `model_id`, `model_type`, `artifact_fingerprint`, `training_dataset_fingerprint` |
| `StrategyVersion` | Strategy thresholds and rules | `strategy_name`, `strategy_version`, `parameters`, `fingerprint` |
| `RiskConfigVersion` | Portfolio protection and sizing limits | `risk_version`, `parameters`, `fingerprint` |
| `ExecutionConfigVersion`| Broker execution simulation parameters | `execution_mode` (PAPER), `commission_bps`, `slippage_bps`, `fingerprint` |
| `RuntimeConfigVersion` | Event loop replay and warmup settings | `runtime_mode`, `warmup_period`, `deduplicate_events`, `fingerprint` |
| `EnvironmentMetadata` | Non-sensitive platform environment | `os_platform`, `python_version`, `git_commit`, `git_branch`, `git_dirty` |
| `ExperimentResult` | Financial and statistical performance | `initial_equity`, `final_equity`, `net_pnl`, `trade_count`, `win_rate`, `sharpe` |
| `ExperimentManifest` | Unified self-contained reproducibility contract | All sub-contracts + `manifest_fingerprint` |

---

## 3. Reproducibility Levels

When verifying an experiment against another execution or target manifest, the system categorizes reproducibility into three distinct levels:

1. **`EXACT`**:
   - All critical fingerprints (dataset, model artifact, features, strategy, risk, execution, runtime) and environment metadata match identically.
2. **`PARTIAL`**:
   - Core dataset, model, strategy, and risk configurations match, but non-critical environment attributes (e.g. git commit hash or Python interpreter patch version) diverge.
3. **`NOT_REPRODUCIBLE`**:
   - Core component fingerprints (dataset, model artifact, strategy rules, or risk limits) do not match or are missing.

---

## 4. Experiment Registry & Persistence

Experiments and manifests are stored locally under `artifacts/experiments/<experiment_id>/` and synchronized to the database:
- `experiment.json`: Metadata, lifecycle status, timestamps.
- `manifest.json`: Full reproducibility manifest.
- `result.json`: Financial performance summary and `result_fingerprint`.

---

## 5. CLI Reference

### List Experiments
```bash
python -m adaptive_trading experiments list [--limit 20] [--json]
```

### Show Experiment Details
```bash
python -m adaptive_trading experiments show <experiment_id> [--json]
```

### Export Manifest
```bash
python -m adaptive_trading experiments manifest <experiment_id> [--json]
```

### Compare Experiments
```bash
python -m adaptive_trading experiments compare <experiment_a> <experiment_b> [--json]
```

### Verify Reproducibility
```bash
python -m adaptive_trading experiments verify <experiment_id> [--target <target_id>] [--json]
```

### Run Replay Experiment
```bash
python -m adaptive_trading experiments create --candles data/sample/nifty_5m_ml_sample.csv --name my_replay_exp [--json]
```

---

## 6. Security and Redaction

To prevent credential leakage:
- Authentication tokens, API keys, passwords, TOTP secrets, and database connection URIs are scrubbed prior to computing configuration fingerprints or writing manifests to disk.
- Environment variable capture is strictly restricted to non-sensitive runtime platform properties (`os_platform`, `python_version`, `git_commit`).

