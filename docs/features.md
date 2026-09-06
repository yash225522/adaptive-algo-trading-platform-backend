# Feature Engineering Foundation Guide

This document describes the feature engineering pipeline, versioning, mathematical formulas, warm-up policies, and look-ahead bias prevention in `adaptive-algo-trading-platform`.

---

## 1. Role of Feature Engineering in the Platform

In algorithmic trading, **Feature Engineering** transforms raw financial time series (OHLCV candles) into stationary, normalized, and predictive numeric representations suitable for machine learning models.

The pipeline guarantees:
- **Reproducibility**: Features are versioned and computed with deterministic math.
- **Look-Ahead Safety**: Features at time $T$ rely strictly on observations at or before $T$.
- **Warm-Up Governance**: Insufficient lookback periods produce `NaN` and are filtered systematically.

---

## 2. Feature Set Versioning

Features are governed by a version identifier attached to every `FeatureVector`:

$$\text{FEATURE\_SET\_VERSION} = \text{"v1"}$$

This version stamp guarantees that models trained against `v1` features are never inadvertently evaluated against incompatible feature definitions (`v2`, `v3`).

---

## 3. Feature Catalog (`v1`)

| Feature Name | Category | Lookback | Formula | Description |
| :--- | :--- | :---: | :--- | :--- |
| `return_1` | Returns | 1 | $\frac{C_t - C_{t-1}}{C_{t-1}}$ | 1-bar simple price return |
| `return_3` | Returns | 3 | $\frac{C_t - C_{t-3}}{C_{t-3}}$ | 3-bar simple price return |
| `return_6` | Returns | 6 | $\frac{C_t - C_{t-6}}{C_{t-6}}$ | 6-bar simple price return |
| `sma_5` | Moving Average | 5 | $\frac{1}{5}\sum_{i=0}^4 C_{t-i}$ | 5-bar simple moving average of close |
| `sma_10` | Moving Average | 10 | $\frac{1}{10}\sum_{i=0}^9 C_{t-i}$ | 10-bar simple moving average of close |
| `sma_20` | Moving Average | 20 | $\frac{1}{20}\sum_{i=0}^{19} C_{t-i}$ | 20-bar simple moving average of close |
| `close_to_sma_5` | Distance Ratio | 5 | $\frac{C_t - \text{SMA5}_t}{\text{SMA5}_t}$ | Percentage distance of close to 5-bar SMA |
| `close_to_sma_20` | Distance Ratio | 20 | $\frac{C_t - \text{SMA20}_t}{\text{SMA20}_t}$ | Percentage distance of close to 20-bar SMA |
| `volatility_10` | Volatility | 11 | $\text{std}(R_{1, t-9..t}, \text{ddof}=1)$ | 10-bar sample standard deviation of 1-bar returns |
| `volume_ratio_10` | Volume | 10 | $\frac{V_t}{\text{SMA}(V, 10)_t}$ | Relative volume vs 10-bar volume mean |
| `vwap_distance` | Price / Volume | 1 | $\frac{C_t - \text{VWAP}_t}{\text{VWAP}_t}$ | Normalized distance to session VWAP (where $TP = \frac{H+L+C}{3}$) |
| `oi_change_1` | Open Interest | 1 | $\frac{OI_t - OI_{t-1}}{OI_{t-1}}$ | 1-bar fractional change in open interest |
| `oi_change_3` | Open Interest | 3 | $\frac{OI_t - OI_{t-3}}{OI_{t-3}}$ | 3-bar fractional change in open interest |

---

## 4. Warm-Up Period Policy

- **Maximum Lookback**: For `v1`, the longest rolling window is 20 bars (`sma_20`).
- **Initial Bars**: Because the first 19 observations lack sufficient historical window, rolling calculations produce `NaN`.
- **No Artificial Zero-Filling**: The system **never** fills initial rolling values with `0.0` or arbitrary interpolations.
- **Warm-Up Filtering**: When generating ML-ready datasets with `drop_warmup=True` (default in `FeaturePipeline`), the initial warm-up rows containing NaNs in required features are dropped cleanly.

---

## 5. Prevention of Look-Ahead Bias

Look-ahead bias occurs when future data $T+k$ leaks into feature calculations for time $T$. In our pipeline:
1. **Strict Shift Indexing**: All lagged returns use positive integer shifts (`close.pct_change(k)` or `.shift(k)`).
2. **Right-Closed Rolling Windows**: All rolling metrics compute over $[t-W+1, t]$ with closed right boundaries.
3. **No Centered Windows**: Centered or forward-looking rolling windows (`center=True`) are prohibited.
4. **Verification**: Automated test suites explicitly verify that perturbing or replacing future candles leaves past feature vectors bit-for-bit identical.

---

## 6. How to Generate Features from Market Candles

```python
from adaptive_trading.features import FeaturePipeline

# 1. Instantiate feature pipeline
pipeline = FeaturePipeline()

# 2. Extract validated FeatureVectors from domain Candle sequence
feature_vectors = pipeline.generate_feature_vectors(
    candles=validated_candles,
    drop_warmup=True,  # Discards warm-up rows containing NaNs
    validate=True,  # Validates absence of Inf/NaN and strictly chronological order
)

# 3. Inspect generated vector
sample_vector = feature_vectors[0]
print(sample_vector.timestamp)  # 2026-01-02 10:50:00+05:30
print(sample_vector.symbol)  # NIFTY
print(sample_vector.feature_version)  # v1
print(sample_vector.features)  # {'return_1': 0.0008, 'sma_20': 26045.2, ...}
```

