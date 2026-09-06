# Strategy Engine & Signal Generation

This document details the design, configuration, probability thresholds, signal data contracts, confidence calculations, batch processing, and CLI for the **Strategy Engine** in `adaptive-algo-trading-platform`.

---

## 1. Purpose of the Strategy Engine

The Strategy Engine serves as the pure decision-making bridge between **Machine Learning inference** and **downstream risk/execution management**.

Its sole responsibility is to answer:
> *"Given the model prediction and probability of market direction, should the system recommend a `LONG`, `SHORT`, or `NO_TRADE` action?"*

```text
Historical / Streaming Data
            ↓
    Feature Pipeline
            ↓
        ML Model
            ↓
  StrategyPrediction
            ↓
      Strategy Engine
            ↓
       TradingSignal
            ↓
   [Future Risk Engine]
            ↓
 [Future Portfolio / Orders]
            ↓
    [Future Broker Execution]
```

---

## 2. Decision Rules & Probability Thresholds

A common trap in quantitative trading is naively trading whenever directional probability $P(\text{UP}) > 0.50$. In reality, model outputs around 50% indicate high uncertainty and market noise.

The Strategy Engine uses dual configurable probability thresholds:
- **Upper Threshold (`long_probability_threshold`, default: 0.60)**:
  $$P(\text{UP}) \ge \text{long\_threshold} \implies \mathbf{LONG}$$
- **Lower Threshold (`short_probability_threshold`, default: 0.40)**:
  $$P(\text{UP}) \le \text{short\_threshold} \implies \mathbf{SHORT}$$
- **Neutral Zone**:
  $$\text{short\_threshold} < P(\text{UP}) < \text{long\_threshold} \implies \mathbf{NO\_TRADE}$$

### Boundary Invariant
Configurations must strictly satisfy:
$$0.0 \le \text{short\_probability\_threshold} < \text{long\_probability\_threshold} \le 1.0$$

Any configuration where thresholds are reversed (e.g. short=0.60, long=0.40) or equal (short=0.50, long=0.50) is rejected immediately at configuration time.

---

## 3. Confidence Calculation

Confidence reflects the strength of directional conviction:
- **LONG Signal**:
  $$\text{Confidence} = P(\text{UP})$$
  *(e.g., $P(\text{UP}) = 0.72 \implies \text{Confidence} = 0.72$)*
- **SHORT Signal**:
  $$\text{Confidence} = 1.0 - P(\text{UP})$$
  *(e.g., $P(\text{UP}) = 0.27 \implies \text{Confidence} = 0.73$)*
- **NO_TRADE Signal**:
  $$\text{Confidence} = 0.0$$
  *(Uncommitted / neutral observation inside threshold boundary)*

> [!NOTE]
> High model confidence is a statistical probability metric from the machine learning model, not a guarantee of trade profitability.

---

## 4. Signal Data Contract (`TradingSignal`)

Every signal is an immutable, structured record:
- `signal_id`: Unique UUID identifier.
- `timestamp`: Timezone-aware timestamp.
- `symbol`: Market asset symbol (e.g. `NIFTY`).
- `action`: `SignalAction.LONG`, `SignalAction.SHORT`, or `SignalAction.NO_TRADE`.
- `confidence`: Calibrated conviction score in $[0.0, 1.0]$.
- `strategy_name`: Identifier of the strategy (e.g. `probability_threshold`).
- `strategy_version`: Identifier of the strategy rule version (e.g. `v1`).
- `reason`: Human-readable audit explanation.
- `model_name`: Name of the model generating the upstream prediction.
- `model_version`: Version of the model generating the upstream prediction.

---

## 5. Strategy Abstraction & Extensibility

```python
class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def generate_signal(self, prediction: StrategyPrediction) -> TradingSignal: ...
```

The concrete `ProbabilityStrategy` implements `BaseStrategy`. Future strategy models (e.g. volatility-regime strategies, multi-timeframe strategies) can be plugged into `StrategyEngine` without altering downstream risk or backtesting components.

---

## 6. Batch Processing & Determinism

`StrategyEngine.generate_signals(predictions)`:
1. Validates prediction input contracts.
2. Orders predictions chronologically by timestamp.
3. Detects and rejects unexpected duplicate timestamps for identical `(timestamp, symbol)` pairs.
4. Generates signals sequentially and returns an ordered list.
5. Computes diagnostic summary statistics via `StrategyEngine.calculate_statistics(signals)`.

---

## 7. Artifact Persistence

When executed via CLI or python API, generated signals are saved as CSV files under:
```text
artifacts/strategy/<experiment_id>/signals.csv
```
Fields saved: `signal_id`, `timestamp`, `symbol`, `action`, `confidence`, `strategy_name`, `strategy_version`, `reason`, `model_name`, `model_version`.

---

## 8. CLI Usage

```bash
# Generate signals from out-of-sample predictions CSV
python -m adaptive_trading.strategy.cli generate \
    --file artifacts/experiments/wf_20260830_173247_dfd7fb/predictions.csv \
    --long-threshold 0.60 \
    --short-threshold 0.40 \
    --strategy-name probability_threshold \
    --strategy-version v1
```

### Sample Output:
```text
============================================================
      ADAPTIVE TRADING PLATFORM - STRATEGY ENGINE           
============================================================
Strategy Name    : probability_threshold
Strategy Version : v1
Long Threshold   : 0.6000
Short Threshold  : 0.4000
------------------------------------------------------------
Total Predictions: 10,000
LONG Signals     : 1,800 (18.0%)
SHORT Signals    : 1,650 (16.5%)
NO_TRADE Signals : 6,550 (65.5%)
------------------------------------------------------------
Artifact Saved   : artifacts/strategy/strat_20260830_231500_123abc/signals.csv
============================================================
```

---

## 9. Architectural Boundaries & Non-Goals

1. **No Order Placement**: The Strategy Engine never sends orders to Angel One or brokers.
2. **No Position Sizing**: It does not determine number of contracts or shares.
3. **No Stop-Loss / Take-Profit**: Stop-loss and profit target policies belong to the Risk Management Engine.
4. **No Backtesting / P&L**: It computes signal distributions, not financial returns, Sharpe ratios, or drawdowns.

