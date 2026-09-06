# Market Data Quality & Normalization Architecture

This document describes the data quality, normalization, validation, and anomaly detection layer in `adaptive-algo-trading-platform`.

---

## 1. Why Data Quality is Critical in Algorithmic Trading

In algorithmic trading and quantitative machine learning:
- **Garbage in, Garbage out**: Corrupted prices, negative volumes, or inverted OHLC candles corrupt technical features, mislead ML inference, and distort backtest results.
- **Look-Ahead & Timing Bugs**: Unsorted or misaligned timestamps distort chronological sequence modeling and lookback calculations.
- **False Signals & Liquidity Illusion**: Stale data or zero volume periods can generate deceptive strategy triggers.

The quality layer guarantees that all market data stored in PostgreSQL is structurally verified, normalized, chronologically consistent, and thoroughly audited.

---

## 2. Normalization vs. Validation

| Stage | Responsibility | Examples |
| :--- | :--- | :--- |
| **Normalization** (`MarketDataNormalizer`) | Transforms raw, heterogeneous inputs into standardized, deterministic Python types without changing market meaning. | • Upper-casing ticker (`" nifty "` $\rightarrow$ `"NIFTY"`)<br>• Standardizing timeframe (`"5M"` $\rightarrow$ `TimeFrame.FIVE_MINUTES`)<br>• Parsing ISO-8601 strings into timezone-aware `datetime`<br>• Stripping whitespace and casting numeric strings to `float` |
| **Quality Validation** (`MarketDataQualityChecker`) | Evaluates business invariants, financial consistency, and stream-level heuristics. | • Verifying $High \ge \max(Open, Close, Low)$<br>• Verifying $Low \le \min(Open, Close, High)$<br>• Detecting missing intraday candles<br>• Flagging suspicious price jumps and zero-volume bars |

---

## 3. Fatal Errors vs. Non-Fatal Warnings

The platform strictly differentiates between **fatal errors** (which reject rows) and **non-fatal warnings** (which audit anomalies without dropping valid data):

### Fatal Errors (`QualitySeverity.ERROR`)
Records with fatal errors cannot be accepted into the database:
- Missing required fields (`timestamp`, `symbol`, `timeframe`, `open`, `high`, `low`, `close`, `volume`).
- Non-positive prices ($Open, High, Low, Close \le 0$).
- Physically impossible OHLC relationships ($High < Open$, $High < Close$, $High < Low$, $Low > Open$, $Low > Close$).
- Negative volume ($Volume < 0$) or negative open interest ($OI < 0$).
- Naive or unparseable timestamps.

### Non-Fatal Warnings (`QualitySeverity.WARNING`)
Records that are structurally valid but exhibit suspicious market behavior are saved to the database while producing audit warnings in `DataQualityReport`:
- **Missing Candles**: Intraday gap exceeding the expected timeframe step during market session hours.
- **Suspicious Price Jump**: Single-candle close-to-close percentage movement exceeding threshold (default: $\ge 10\%$).
- **Zero Volume**: Valid candle with zero traded volume ($Volume = 0$).
- **Stale OHLC Prices**: Consecutive identical flat bars ($Open = High = Low = Close$) spanning $\ge 3$ periods.
- **In-Batch Duplicate**: Duplicate identical logical candles encountered in the same CSV stream.

---

## 4. Missing-Candle Detection Logic

- The system maps timeframes to exact time delta steps:
  - `1m`: 1 min, `3m`: 3 min, `5m`: 5 min, `15m`: 15 min, `30m`: 30 min, `1h`: 1 hour, `1d`: 1 day.
- **Intraday Gaps**: If consecutive candles on the same trading date within market hours (`09:15` to `15:30`) have $\Delta t > \text{step}$, the missing candle count is computed:
  $$\text{Missing Count} = \left\lfloor \frac{\Delta t}{\text{step}} \right\rfloor - 1$$
- **Session Transitions**: Overnight transitions (e.g. `15:30` on Day 1 to `09:15` on Day 2) and weekend gaps are recognized as normal market session closures and **not** flagged as missing candles.
- **No Data Fabrication**: Missing candles are strictly reported for auditability; **no fake or interpolated candles are synthesized**.

---

## 5. Quality Report Structure

The `DataQualityReport` aggregates complete diagnostic information:

```text
DataQualityReport(
    rows_checked=9,
    rows_valid=8,
    rows_rejected=1,
    duplicate_count=1,
    missing_candle_count=1,
    warning_count=2,
    error_count=2,
    issues=[
        QualityIssue(issue_type='INVALID_OHLC_LOW', severity=ERROR, message='...'),
        QualityIssue(issue_type='MISSING_CANDLES', severity=WARNING, message='Detected 1 missing candle(s)...')
    ]
)
```

---

## 6. Pipeline Integration

```
Raw CSV File
     │
     ▼
CSVMarketDataReader (Reads raw rows with line numbers)
     │
     ▼
MarketDataNormalizer (Normalizes types, timezones, & strings)
     │
     ▼
MarketDataQualityChecker (Applies invariant checks & stream heuristics)
     │
     ├── DataQualityReport (Errors & Warnings recorded)
     │
     ▼
Valid MarketCandle Objects (Time-sorted immutable contracts)
     │
     ▼
MarketDataPersistenceService (Idempotent batch insert into PostgreSQL)
```

