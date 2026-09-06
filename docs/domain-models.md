# Domain Models & Data Contracts

This document defines the core domain models and data contracts for the `adaptive-algo-trading-platform`.

---

## 1. What is a Domain Model?

In this platform, a **Domain Model** is a pure, strongly-typed, and validated Python representation of a core business concept (e.g., market candles, ML predictions, trading signals, orders, fills, and portfolio positions).

Domain models serve as **contracts** between distinct decoupled components in the platform. They validate invariants, ensure type safety, and provide standard JSON serialization across asynchronous pipelines.

---

## 2. Separation from Database and Persistence Models

Domain models are strictly decoupled from database models (such as ORM entities or SQL tables) for several key reasons:

1. **Independent Evolution**: The business concepts and inter-component contracts should not be bound to database schema design, column naming conventions, or ORM-specific decorators.
2. **Performance & Purity**: Domain models have no database drivers, network connections, or lazy-loading overhead, making them lightweight and ideal for in-memory computations and unit testing.
3. **Multi-Source Flexibility**: Domain models can be constructed from external broker APIs, CSV backtest datasets, WebSocket streams, or database queries without altering their definitions.

---

## 3. Models Overview and Producer/Consumer Matrix

| Domain Model | Purpose | Producer Component | Consumer Component |
| :--- | :--- | :--- | :--- |
| **`Candle`** | Represents one validated OHLCV interval bar with positive prices, non-negative volume, and valid OHLC bounds. | Market Data Ingestion / Historic Loader | Feature Engineering, Strategy Engine, Backtester |
| **`FeatureVector`** | Maps computed numerical indicators/features at a specific timestamp for a symbol. | Feature Engineering Pipeline | ML Inference Engine, Model Training Pipeline |
| **`Prediction`** | Encapsulates model probability distributions (summing to ~1.0) and expected returns. | ML Inference Engine | Strategy Engine, Risk Evaluator |
| **`Signal`** | Captures strategy-generated intent (`BUY`, `SELL`, `FLAT`) with optional target, stop loss, and take profit prices. | Strategy Engine | Risk Engine |
| **`RiskDecision`** | Encapsulates risk assessment approval (`approved: bool`) and policy versioning. | Risk Engine | Execution Engine, Order Router |
| **`Order`** | Represents a validated order request (`MARKET` or `LIMIT` with price) submitted for execution. | Execution Engine | Broker Adapter / Simulator |
| **`Fill`** | Execution confirmation record detailing executed quantity, price, fees, and slippage. | Broker Adapter / Execution Simulator | Portfolio Manager, Position Tracker, Backtester |
| **`Position`** | Tracks open/closed inventory, average cost, unrealized PnL, and realized PnL. | Portfolio Manager | Risk Engine, Dashboard / Reporting |
| **`Trade`** | Auditable record of a completed round-trip trade, including timestamps, prices, fees, slippage, and net PnL. | Portfolio Manager / Trade Reconciler | Performance Analytics, Model Evaluation |

---

## 4. Key Invariants & Validation Rules

- **Timezone Awareness**: All timestamps across all models must be timezone-aware (rejecting naive datetimes).
- **Market Price Bounds**:
  - `High >= max(Open, Close, Low)`
  - `Low <= min(Open, Close, High)`
  - Prices (`Open`, `High`, `Low`, `Close`) must be strictly positive (`> 0`).
  - `Volume` and `Open Interest` must be non-negative (`>= 0`).
- **Probability Distributions**: `probability_up` and `probability_down` must each be in `[0.0, 1.0]` and their sum must equal `1.0 ± 0.001`.
- **Trading Quantities**: `BUY` and `SELL` signals and orders require strictly positive quantities (`> 0`).
- **Order Types**: `LIMIT` orders must provide a limit price.
- **Trade Timestamps**: Completed trades require `exit_timestamp >= entry_timestamp`.

