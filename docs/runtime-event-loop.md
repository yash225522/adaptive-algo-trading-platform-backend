# Paper Trading Event Loop & End-to-End Simulation

This document describes the orchestration architecture, event-time semantics, warm-up policies, risk/execution integration, audit logging, and CLI tools for the **`adaptive-algo-trading-platform`** runtime.

---

## 1. Runtime Architecture

The runtime acts strictly as a deterministic orchestrator, coordinating existing components without embedding domain math or trading formulas:

```text
             Market Data
                  ↓
            Market Event
                  ↓
            Feature State
                  ↓
             Prediction
                  ↓
              Strategy
                  ↓
                 Risk
                  ↓
          Execution Service
                  ↓
             Paper Broker
                  ↓
                Fill
                  ↓
              Portfolio
                  ↓
            Event / Audit Log
```

---

## 2. Event Model & Types

The lifecycle is tracked via discrete, serializable `EventRecord` items:
- `MARKET_DATA`: Ingestion of historical or simulated candle bar.
- `PREDICTION`: Output of ML prediction service (`probability_up`).
- `SIGNAL`: Directional decision (`LONG`, `SHORT`, `NO_TRADE`) emitted by strategy rules.
- `RISK_DECISION`: Approval/rejection record from `RiskEngine`.
- `ORDER`: Execution request dispatched to `ExecutionService`.
- `FILL`: Executed trade fill confirmed by `PaperBroker`.
- `PORTFOLIO_UPDATE`: Mark-to-market revaluation snapshot.
- `ERROR`: System exception record.

---

## 3. Event-Time Clock Semantics

- Uses `TradingClock` to maintain simulated trading time driven strictly by `MarketEvent.timestamp`.
- Real computer wall-clock time (`datetime.now()`) is never used inside market decision branches.
- Chronological ordering is strictly enforced; time cannot move backwards.

---

## 4. Feature Warm-Up & Lookahead Prevention

- Moving average, return, and volatility indicators require historical observations before calculating feature values.
- During the initial warm-up period (e.g. 20 bars), incoming candles update feature state, but no predictions or trading orders are emitted.
- Rolling windows only contain past observations $[T-k, T]$, guaranteeing zero future leakage.

---

## 5. Risk & Execution Integration

- Every trading signal must pass through `RiskEngine.evaluate()` before dispatching.
- If rejected by risk rules (e.g. daily loss limit, max drawdown, exposure limits), `EventLoop` records the rejection and halts trade submission for that bar without interrupting the loop.
- Only approved quantities are submitted via `ExecutionService` to `PaperBroker`.

---

## 6. Audit Trail & Traceability

Every fill can be traced backward using `correlation_id`:
$$\text{Fill} \longrightarrow \text{Order} \longrightarrow \text{RiskDecision} \longrightarrow \text{TradingSignal} \longrightarrow \text{Prediction} \longrightarrow \text{MarketEvent}$$

---

## 7. Artifact Outputs

Runs save complete auditable artifacts under `artifacts/runtime/<run_id>/`:
- `metadata.json`: Configuration, data span, start/end timestamps.
- `events.jsonl`: Complete chronological stream of all lifecycle event records.
- `runtime_stats.json`: Operational counters (events processed, predictions, signals, orders, fills, rejections, errors).
- `checkpoint.json`: Periodic / final state snapshot for recovery.
- `final_state.json`: Final cash, equity, realized/unrealized P&L, and open positions.

---

## 8. CLI Usage

```bash
# 1. Run historical replay simulation
python -m adaptive_trading.runtime.cli replay \
    --candles data/sample/nifty_5m_ml_sample.csv \
    --initial-cash 100000 \
    --warmup 20

# 2. Inspect runtime status and results from run directory
python -m adaptive_trading.runtime.cli status \
    --run-dir artifacts/runtime/run_20260831_000000_123456
```

---

## 9. Safety Restrictions

> [!CAUTION]
> **Live Trading Disabled**:
> Setting `RuntimeMode.LIVE` immediately raises `LiveTradingNotEnabledError`. The runtime only operates in `REPLAY` and `PAPER` modes.

