# Walk-Forward Validation & ML Evaluation Infrastructure

This document describes the design, splitting strategy, leakage prevention, independent preprocessing, majority baseline comparison, out-of-sample prediction generation, and CLI commands for the **Walk-Forward Validation Engine** in `adaptive-algo-trading-platform`.

---

## 1. Why Random Cross-Validation Fails on Financial Data

In standard machine learning, datasets often consist of independent and identically distributed (I.I.D.) samples where random K-Fold cross-validation or randomized train/test splits are valid.

In financial market time series, observations have strong temporal dependencies, autocorrelation, and non-stationary regimes. Using random splits causes severe **Lookahead Bias** and **Information Leakage**:
- Future information leaks into training folds.
- Technical indicators (e.g. moving averages, return lookbacks) and future target horizons overlap across train and test sets.
- Results appear artificially optimistic but fail in live market execution.

Walk-Forward Validation preserves the arrow of time: **the model is only ever trained on the past and evaluated on the future**.

---

## 2. Walk-Forward Architecture

```text
Historical Timeline
─────────────────────────────────────────────────────────────────────────>

Fold 1:
TRAIN      |========================|
PURGE GAP                           |---|
VALIDATE                                |===========|

Fold 2 (Expanding):
TRAIN      |====================================|
PURGE GAP                                       |---|
VALIDATE                                            |===========|

Fold 3 (Expanding):
TRAIN      |================================================|
PURGE GAP                                                   |---|
VALIDATE                                                        |===========|
```

---

## 3. Window Strategies

The framework supports two window progression strategies:
1. **Expanding Window (Default)**:
   - Starts with `initial_train_size` (e.g. 50% of the dataset).
   - In each subsequent fold, the training window expands to incorporate all historical observations up to that fold's validation start.
   - Preserves long-term historical regimes and maximizes training sample volume.
2. **Rolling Window (`window_type="ROLLING"`)**:
   - Maintains a fixed-size training window that rolls forward with each step.
   - Discards distant historical data to adapt rapidly to regime changes.

---

## 4. Temporal Purge Gap & Horizon Awareness

Step 8 target generation uses a forward prediction horizon (e.g. $h=3$ bars into the future).
- If bar $t$ has a target calculated from $t+1 \dots t+h$, training on bar $t$ involves future price data up to $t+h$.
- To prevent this future information from bleeding into a validation window starting at $t+1$, the `WalkForwardConfig` supports a configurable `gap` parameter (default 0 or $h$).
- The splitter strictly guarantees:
  $$\max(\text{train\_timestamp}) < \min(\text{validation\_timestamp})$$

---

## 5. Leak-Free Preprocessing Rules

A common mistake in ML workflows is fitting scalers (e.g. `StandardScaler`) on the entire dataset prior to splitting.

Our pipeline strictly isolates preprocessing per fold:
```text
Fold k:
  1. train_df, val_df = split.extract(dataset)
  2. preprocessor = MLPreprocessor()
  3. X_train_scaled = preprocessor.fit_transform(X_train_df)    <-- FITTED ONLY ON TRAIN
  4. model.fit(X_train_scaled, y_train)
  5. X_val_scaled = preprocessor.transform(X_val_df)             <-- TRANSFORMED USING TRAIN FIT
  6. y_pred = model.predict(X_val_scaled)
```

---

## 6. Majority-Class Baseline Comparison

To determine whether the ML model learns genuine signal beyond class imbalance, each fold evaluates against a **Majority-Class Baseline**:
- The majority class is computed strictly from that fold's `y_train` distribution (not the global dataset or validation fold).
- Constant predictions of this majority class are evaluated on the validation fold to establish baseline accuracy and F1 score.

---

## 7. Out-of-Sample (OOS) Predictions

For every validation fold, individual predictions are recorded:
- `timestamp`: Bar timestamp
- `symbol`: Ticker symbol
- `actual_label`: Ground truth binary label (1 for UP, 0 for DOWN)
- `predicted_label`: Model prediction
- `probability_up`: Model predicted probability $P(\text{UP})$
- `fold_number`: Fold index

This creates a continuous out-of-sample prediction stream that can later be consumed by strategy and backtesting modules.

---

## 8. Experiment Artifact Persistence

Each validation run generates an isolated artifact directory under `artifacts/experiments/<experiment_id>/`:
```text
artifacts/experiments/wf_20260830_225000_a1b2c3/
├── metadata.json       # Run configuration, versions, timestamp
├── metrics.json        # Per-fold and aggregate summary metrics
└── predictions.csv     # Out-of-sample predictions table
```

---

## 9. CLI Usage

```bash
# Run walk-forward validation with default expanding window
python -m adaptive_trading.ml.cli walk-forward \
    --file data/sample/nifty_5m_ml_sample.csv \
    --initial-train-size 0.50 \
    --validation-size 0.10 \
    --horizon 3

# Run with rolling window and 3-bar purge gap
python -m adaptive_trading.ml.cli walk-forward \
    --file data/sample/nifty_5m_ml_sample.csv \
    --initial-train-size 0.50 \
    --validation-size 0.10 \
    --window-type ROLLING \
    --gap 3
```

### Sample CLI Output:
```text
============================================================
      ADAPTIVE TRADING PLATFORM - WALK-FORWARD VALIDATION   
============================================================
Dataset File     : data/sample/nifty_5m_ml_sample.csv
Prediction Horizon: 3 bars
Window Type      : EXPANDING
Initial Train    : 0.5
Validation Size  : 0.1
Gap (Purge)      : 0 bars
------------------------------------------------------------
Aligned Samples  : 167
------------------------------------------------------------
Experiment ID    : wf_20260830_230000_123456
Total Folds      : 5
Total Val Samples: 80
Mean Accuracy    : 0.5500 (±0.0450)
Mean Precision   : 0.5300 (±0.0400)
Mean Recall      : 0.5800 (±0.0500)
Mean F1 Score    : 0.5520 (±0.0420)
Mean ROC-AUC     : 0.5650
------------------------------------------------------------
Majority Baseline: 0.5125
OOS Predictions  : 80 rows
============================================================
```

---

## 10. Limitations & Non-Goals

- **No Trading Interpretation**: Higher accuracy does not guarantee strategy profitability. P&L, transaction costs, and slippage are evaluated strictly in backtesting.
- **No In-Sample Contamination**: Hyperparameter tuning across validation folds without nested cross-validation is avoided to prevent overfitting.

