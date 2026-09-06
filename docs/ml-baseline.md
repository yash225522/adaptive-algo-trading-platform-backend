# Machine Learning Baseline Model Guide

This document describes the first supervised machine learning baseline model, chronological train/test methodology, training-only scaling, evaluation metrics, majority-class comparison, and model artifact management in `adaptive-algo-trading-platform`.

> [!IMPORTANT]
> **Trading Disclaimer**: Classification performance does not imply trading profitability. The baseline model evaluates purely whether financial indicators carry predictive directional signal on unseen chronological partitions. Execution costs, slippage, risk management, and portfolio returns are evaluated in subsequent backtesting layers.

---

## 1. Why Logistic Regression is the First Baseline

Logistic Regression serves as the fundamental benchmark in quantitative trading architectures because:
- **Interpretability**: Linear combination of features allows direct inspection of learned coefficients and odds ratios.
- **Fast & Deterministic Convergence**: Solves convex optimization problems reliably with $L_2$ regularization.
- **Calibrated Probabilities**: Provides bounded $[0, 1]$ confidence estimates via the sigmoid function.
- **Sanity Anchor**: Any future complex model (e.g. Tree ensembles, Neural Networks) must demonstrably beat this baseline to justify operational complexity.

---

## 2. Input Features & Target Definition

- **Feature Set (`FEATURE_SET_VERSION = "v1"`)**:
  13 normalized technical/statistical features calculated at prediction time $T$:
  `return_1`, `return_3`, `return_6`, `sma_5`, `sma_10`, `sma_20`, `close_to_sma_5`, `close_to_sma_20`, `volatility_10`, `volume_ratio_10`, `vwap_distance`, `oi_change_1`, `oi_change_3`.
- **Target Label (`TARGET_VERSION = "v1"`)**:
  Binary directional movement over horizon $h = 3$ (15 minutes forward on 5m bars):
  - `UP` (1): Forward return $> 0.0$
  - `DOWN` (0): Forward return $\le 0.0$

---

## 3. Chronological Train/Test Split

Because financial asset prices exhibit autocorrelation and non-stationarity, **random k-fold shuffling introduces catastrophic look-ahead bias**.

```text
Chronological Time Axis:
├── Training Partition (Early 80%) ──────────►├── Test Partition (Held-Out Final 20%) ──►
│   - Used exclusively for fitting scaler     │   - Strictly unseen out-of-sample data
│   - Used for training Logistic Regression   │   - Evaluates generalization performance
```

- **Split Ratio**: Default `train_ratio = 0.80`
- **Boundary Verification**: Invariant check asserts $t_{\text{train\_end}} < t_{\text{test\_start}}$ with 0 overlap.

---

## 4. Preprocessing & Scaling Isolation

Feature scaling ($z = \frac{x - \mu}{\sigma}$) is required for regularized linear models:
1. **Fit**: `StandardScaler` calculates mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ **solely on the training set**.
2. **Transform**: The fitted parameters are applied to $X_{\text{train}}$ and $X_{\text{test}}$ without re-estimating parameters on the test set.
3. **Target Protection**: Target metadata columns (`future_return`, `label`, `target_version`, `timestamp`, `symbol`) are strictly filtered out prior to preprocessing.

---

## 5. Evaluation Metrics & Majority Baseline

Performance on the unseen chronological test partition is quantified across 6 dimensions:

| Metric | Formula / Meaning | Edge Case Handling |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Baseline comparison against naive majority class |
| **Precision** | $\frac{TP}{TP + FP}$ | `zero_division=0.0` prevents division-by-zero crashes |
| **Recall** | $\frac{TP}{TP + FN}$ | `zero_division=0.0` prevents division-by-zero crashes |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under Receiver Operating Characteristic curve | Returns `None` gracefully if test partition contains a single class |
| **Confusion Matrix** | $\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$ | Explicit $2 \times 2$ matrix for error analysis |

### Majority-Class Baseline Comparison
Alongside model metrics, the system calculates the performance of a naive heuristic that blindly predicts the training set's majority class. This directly answers:
*“Does the machine learning model outperform a naive coin toss or constant trend bias?”*

---

## 6. Model Artifact Structure

Model binaries and metadata are persisted locally:

```text
artifacts/
└── models/
    └── logistic_regression/
        └── v1/
            ├── model.joblib      # Serialized model, preprocessor, and feature names bundle
            └── metadata.json     # Structured JSON capturing split windows, sample counts, and metrics
```

### Metadata JSON Schema
```json
{
  "model_name": "logistic_regression",
  "artifact_version": "v1",
  "feature_version": "v1",
  "target_version": "v1",
  "feature_names": ["return_1", "return_3", "..."],
  "training_timestamp": "2026-08-30T14:20:00Z",
  "split": {
    "train_ratio": 0.8,
    "train_samples": 80,
    "test_samples": 20,
    "train_start": "2026-01-02T09:15:00+05:30",
    "train_end": "2026-01-02T13:45:00+05:30",
    "test_start": "2026-01-02T13:50:00+05:30",
    "test_end": "2026-01-02T15:25:00+05:30"
  },
  "metrics": {
    "accuracy": 0.6500,
    "precision": 0.6250,
    "recall": 0.7142,
    "f1": 0.6667,
    "roc_auc": 0.6850,
    "confusion_matrix": [[8, 4], [3, 5]],
    "baseline_metrics": {
      "majority_class": 1.0,
      "accuracy": 0.5500,
      "f1": 0.7096
    }
  }
}
```

---

## 7. Running Baseline Model Training via CLI

```bash
# Train baseline model on sample market candles
python -m adaptive_trading.ml.cli --file data/sample/nifty_5m_sample.csv --train-ratio 0.80 --horizon 3
```

