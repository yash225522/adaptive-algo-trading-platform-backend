# Machine Learning Target & Label Generation Guide

This document defines the supervised learning target formulation, forward horizon calculation, classification thresholds, alignment, and data leakage prevention rules in `adaptive-algo-trading-platform`.

---

## 1. What the Prediction Target Means

In supervised machine learning for algorithmic trading:
- **Feature Vector ($X_t$)**: Represents information available at prediction time $t$ (e.g. past returns, SMAs, volatility).
- **Target ($Y_t$)**: Represents the future market outcome between time $t$ and $t + h$.
- **Training Example**: Combines $(X_t, Y_t)$ so that the model learns the relationship between past market patterns and future price movement.

---

## 2. Prediction Horizon & Versioning

$$\text{TARGET\_VERSION} = \text{"v1"}$$

- **Horizon ($h$)**: Configurable integer number of candle bars (default $h = 3$).
- On a 5-minute timeframe bar, $h = 3$ corresponds to a **15-minute forward prediction window**.

---

## 3. Mathematical Formulas

### Forward Return (Regression Target)

$$\text{future\_return}_t = \frac{C_{t+h} - C_t}{C_t}$$

Where:
- $C_t$: Close price of the candle at prediction time $t$.
- $C_{t+h}$: Close price of the candle $h$ bars into the future.

### Direction Classification Label

#### Binary Classification (Default)
$$\text{label}_t = \begin{cases} \text{UP} & \text{if } \text{future\_return}_t > \text{positive\_threshold} \\ \text{DOWN} & \text{if } \text{future\_return}_t \le \text{positive\_threshold} \end{cases}$$
*(Default `positive_threshold = 0.0`)*

#### Three-Class Classification (Optional)
$$\text{label}_t = \begin{cases} \text{UP} & \text{if } \text{future\_return}_t > \text{positive\_threshold} \\ \text{DOWN} & \text{if } \text{future\_return}_t < \text{negative\_threshold} \\ \text{NEUTRAL} & \text{otherwise} \end{cases}$$

---

## 4. Final-Row Handling

Because forward return requires observing price at $t + h$:
- The final $h$ observations in any historical dataset lack future market data.
- **Strict Policy**: The final $h$ rows receive no targets and are excluded from the training dataset.
- **Zero Fabrication**: Future prices are **never** fabricated, imputed, or wrapped around.

---

## 5. Feature & Target Timestamp Alignment

In the dataset builder:
- A `FeatureVector` calculated at timestamp $T$ is aligned with the `Target` that has prediction timestamp $T$.
- Even though the target is evaluated using the close price at $T + h$, the training row is anchored at timestamp $T$.

```text
Timestamp T: 09:30
  ├── FeatureVector at 09:30 (Calculated from candles up to 09:30)
  └── Target at 09:30        (Evaluates return between 09:30 close and 09:45 close)
```

---

## 6. Data Leakage Prevention

| Component | Allowed Information Window | Purpose |
| :--- | :--- | :--- |
| **Feature Vector** | Data $\le T$ strictly | Input features provided to the model at inference time |
| **Target Outcome** | Close price at $T + h$ | Ground-truth supervision label for model training |

Automated unit tests (`test_no_feature_leakage_with_targets`) verify that modifying future prices at $T + h$ affects **only** the target at $T$, while all feature values at time $T$ remain completely invariant.

---

## 7. Example Training Dataset Row

```json
{
  "timestamp": "2026-01-02T09:30:00+05:30",
  "symbol": "NIFTY",
  "return_1": 0.0012,
  "return_3": 0.0035,
  "return_6": 0.0051,
  "sma_5": 26010.2,
  "sma_20": 25985.4,
  "volatility_10": 0.0015,
  "volume_ratio_10": 1.12,
  "future_return": 0.0028,
  "label": "UP",
  "feature_version": "v1",
  "target_version": "v1"
}
```

