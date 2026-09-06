# Historical Market Data Ingestion Guide

This document details the historical market data ingestion pipeline for `adaptive-algo-trading-platform`.

---

## 1. Pipeline Architecture

The ingestion pipeline strictly decouples file ingestion, validation, domain modeling, and database persistence:

```
┌─────────────────────────┐
│     CSV Market File     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   CSVMarketDataReader   │  Reads raw CSV & applies configurable column mapping
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   MarketDataValidator   │  Validates OHLCV, timezones, & builds Candle domain models
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   MarketCandle (Domain) │  Time-sorted, validated immutable domain contracts
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│PersistenceService (SQL) │  Idempotent batch insertion to PostgreSQL market_candles
└─────────────────────────┘
```

---

## 2. Supported CSV Format & Expected Columns

The standard input format is a UTF-8 encoded CSV file with the following column structure:

```csv
timestamp,symbol,timeframe,open,high,low,close,volume,open_interest
2026-01-02T09:15:00+05:30,NIFTY,5m,26000.0,26050.0,25980.0,26030.0,120000,1500000
```

| Field Name | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `timestamp` | ISO-8601 String | Yes | Timezone-aware timestamp (e.g. `2026-01-02T09:15:00+05:30` or `2026-01-02T03:45:00Z`) |
| `symbol` | String | Yes | Market ticker identifier (e.g. `NIFTY`, `BANKNIFTY`) |
| `timeframe` | String | Yes | Controlled timeframe (`1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `1d`) |
| `open` | Float | Yes | Bar opening price (`> 0`) |
| `high` | Float | Yes | Bar highest price (`> 0`) |
| `low` | Float | Yes | Bar lowest price (`> 0`) |
| `close` | Float | Yes | Bar closing price (`> 0`) |
| `volume` | Float | Yes | Total traded volume (`>= 0`) |
| `open_interest`| Float | No | Total open interest (`>= 0` if provided) |

---

## 3. Timestamp & Chronology Requirements

- **Timezone Awareness**: Timestamps must contain an explicit timezone offset (`+05:30`, `Z`, `+00:00`). Naive datetimes are strictly rejected unless an explicit fallback timezone is supplied to the ingestion service.
- **Chronological Sorting**: Validated candles are sorted chronologically by timestamp before persistence, ensuring ordered downstream feature computation.

---

## 4. Validation Rules

A row is rejected and added to `IngestionStats.errors` with detailed row-level context if:
1. Missing any required column or null value in required fields.
2. Prices (`open`, `high`, `low`, `close`) are `<= 0`.
3. High price is lower than Open, Close, or Low (`high < max(open, close, low)`).
4. Low price is higher than Open, Close, or High (`low > min(open, close, high)`).
5. Volume or Open Interest is negative (`< 0`).
6. Timestamp is unparseable or naive.

---

## 5. Duplicate Handling & Idempotency

Logical market candle identity is defined by:
$$\text{Candle Identity} = (\text{timestamp}, \text{symbol}, \text{timeframe})$$

The persistence service guarantees idempotency:
- Duplicate rows within the same CSV batch are automatically deduplicated.
- Existing database records matching the `(timestamp, symbol, timeframe)` key are skipped without raising errors or altering existing data.
- Re-running ingestion on an already-ingested CSV file results in `rows_inserted = 0` and `rows_skipped = N`.

---

## 6. Running Ingestion from the Command Line

### Usage

```bash
# Using module execution
python -m adaptive_trading.data.cli --file data/sample/nifty_5m_sample.csv

# Using script runner
python scripts/ingest_csv.py --file data/sample/nifty_5m_sample.csv --verbose
```

### Example CLI Output

```text
==================================================
Ingestion Summary
==================================================
Source file    : data/sample/nifty_5m_sample.csv
Rows read      : 9
Rows valid     : 8
Rows inserted  : 7
Rows skipped   : 1
Rows rejected  : 1
Data Quality Errors (1):
  - Row 10 [close]: Close price must be positive (got -26090.0) (got: -26090.0)
==================================================
```

