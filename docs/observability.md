# Observability, Monitoring & Run Tracking

This document outlines the architecture, structured logging, metrics subsystem, health diagnostics, run tracking, event audit tracing, and security safeguards in the **`adaptive-algo-trading-platform`**.

---

## 1. Observability Architecture

The observability layer observes the system without altering trading logic or execution state:

```text
                    ┌──────────────┐
                    │   Runtime    │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
       Logging          Metrics          Events
          │                │                │
          └────────────────┼────────────────┘
                           ↓
                     Run Tracker
                           ↓
                    Monitoring Layer
                           ↓
              CLI / API / Future Dashboard
```

---

## 2. Structured Logging & Sensitive Credential Redaction

- **Format**: Produces single-line JSON log events capturing `timestamp`, `level`, `logger`, `message`, `component`, `run_id`, `event_id`, `correlation_id`, `symbol`, and exception traces.
- **Security & Redaction**:
  - Automatically scrubs sensitive fields such as `password`, `secret`, `jwt`, `token`, `api_key`, `auth_token`, `client_code`, `totp_secret`, and `pin`.
  - Applies regex filters to scrub bearer tokens and JWT payloads from log messages and exception dumps.

---

## 3. Metrics System

Provides lightweight in-memory metrics without requiring external infrastructure:
- **Counters**: `market_events_processed`, `predictions_generated`, `signals_generated`, `risk_rejections`, `orders_submitted`, `orders_filled`, `orders_rejected`, `runtime_errors`.
- **Gauges**: `current_equity`, `current_cash`, `open_positions`, `current_exposure`, `daily_pnl`, `drawdown`.
- **Histograms & Latency**: `feature_generation_latency_ms`, `prediction_latency_ms`, `strategy_latency_ms`, `risk_latency_ms`, `execution_latency_ms`.

---

## 4. Run Tracking & Lifecycles

Tracks simulation and live sessions across distinct states:

```text
CREATED  ───►  RUNNING  ───►  COMPLETED
                  │
                  ├───►  FAILED
                  │
                  └───►  CANCELLED
```

Stores structured run records containing execution duration, model version, strategy version, financial summary (net P&L, final equity, max drawdown), and operational statistics.

---

## 5. Health Checks & Diagnostics

The `HealthChecker` validates platform readiness across subsystems:
- **Database**: Verifies connectivity (falls back to file storage if unavailable).
- **ML Model**: Checks availability of trained model binaries and metadata.
- **Market Data**: Validates access to market candles.
- **Paper Broker**: Verifies broker state and capital initialization.
- **Runtime**: Checks event loop readiness.

Status levels: `HEALTHY`, `DEGRADED`, `UNHEALTHY`.

---

## 6. Audit Tracing & Causality Reversal

Reconstructs the full causal chain for any trade using `correlation_id` or `order_id`:
$$\text{Fill} \longrightarrow \text{Order} \longrightarrow \text{Risk Decision} \longrightarrow \text{Trading Signal} \longrightarrow \text{Prediction} \longrightarrow \text{Market Event}$$

---

## 7. CLI Reference

```bash
# 1. Evaluate system health
python -m adaptive_trading monitoring health
python -m adaptive_trading monitoring health --json

# 2. Inspect runtime metrics
python -m adaptive_trading monitoring metrics
python -m adaptive_trading monitoring metrics --json

# 3. List recent execution runs
python -m adaptive_trading monitoring runs
python -m adaptive_trading monitoring runs --json

# 4. View detailed run summary
python -m adaptive_trading monitoring run <run_id>
python -m adaptive_trading monitoring run <run_id> --json

# 5. Query lifecycle event records for a run
python -m adaptive_trading monitoring events <run_id> --type ORDER
```

---

## 8. Behavioral Transparency

Observability components are strictly read-only and telemetry-focused:
- Adding logging or metrics does not modify predictions, strategy signals, risk decisions, order quantities, or portfolio equity calculations.
- Verified through automated regression tests comparing runs with and without telemetry.

