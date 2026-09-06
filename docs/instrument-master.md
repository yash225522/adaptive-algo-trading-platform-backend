# Angel One Instrument Master & Symbol Token Manager

This document describes the design, database storage, synchronization, resolution logic, historical API integration, and CLI commands for the **Instrument Master / Symbol Token Manager** in `adaptive-algo-trading-platform`.

---

## 1. Why the Instrument Master Exists

Angel One SmartAPI identifies tradable instruments via external numerical identifiers known as `symboltoken` (e.g. `"99926000"` for NIFTY 50 index, `"3045"` for SBIN equity, `"35938"` for NIFTY monthly futures).

Application layers, ML pipelines, and trading strategies operate on human-readable tickers (such as `NIFTY`, `RELIANCE`, `BANKNIFTY`) and structured derivative criteria (`expiry`, `strike`, `option_type`). The **Instrument Master** bridges this gap by maintaining a local, queryable catalog of all Angel One market instruments.

```text
User / Strategy Query
        ↓
InstrumentQuery (symbol="NIFTY", exchange="NSE", type=INDEX)
        ↓
InstrumentResolver
        ↓
Instrument Domain Model (symbol_token="99926000")
        ↓
AngelOneHistoricalService / SmartAPI
```

---

## 2. Meaning of `symboltoken`

- `symboltoken` is an external, broker-assigned identifier.
- Tokens can change over time (e.g., when derivative contracts expire or new instruments are listed).
- The application never hard-codes `symboltoken` values in code, configuration, or strategy logic; instead, it dynamically resolves tokens from the local PostgreSQL instrument master.

---

## 3. Instrument Domain Model

The `Instrument` domain model represents standardized market securities across four asset classes:
- **Equity**: Cash equities traded on NSE/BSE (e.g. `SBIN-EQ`).
- **Index**: Broad market indices (e.g. `Nifty 50`, `Nifty Bank`, `Sensex`).
- **Futures**: Index and stock futures contracts with expiration dates (e.g. `NIFTY26JANFUT`).
- **Options**: Call (`CE`) and Put (`PE`) option contracts with strike prices and expiration dates.

### Field Specification:
| Field | Type | Description |
| :--- | :--- | :--- |
| `symbol_token` | `str` | Angel One instrument identifier token |
| `exchange` | `str` | Exchange segment (`NSE`, `BSE`, `NFO`, `MCX`) |
| `trading_symbol` | `str` | Exchange trading symbol (`Nifty 50`, `SBIN-EQ`) |
| `symbol` | `str` | Normalized underlying symbol (`NIFTY`, `SBIN`) |
| `name` | `str` | Full descriptive security name |
| `instrument_type` | `InstrumentType` | `EQUITY`, `INDEX`, `FUTURES`, `OPTIONS`, `OTHER` |
| `expiry` | `date \| None` | Contract expiration date (for derivatives) |
| `strike` | `float \| None` | Option strike price |
| `option_type` | `OptionType \| None` | Option style (`CE` or `PE`) |
| `lot_size` | `int` | Minimum order quantity |
| `tick_size` | `float` | Minimum price fluctuation increment |

---

## 4. Database Schema & Storage

Instruments are stored in PostgreSQL in the `instruments` table created via Alembic migration `0002_instruments_table.py`:

```sql
CREATE TABLE instruments (
    id SERIAL PRIMARY KEY,
    symbol_token VARCHAR(32) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    trading_symbol VARCHAR(64) NOT NULL,
    symbol VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL DEFAULT '',
    instrument_type VARCHAR(16) NOT NULL,
    expiry DATE,
    strike FLOAT,
    option_type VARCHAR(8),
    lot_size INTEGER NOT NULL DEFAULT 1,
    tick_size FLOAT NOT NULL DEFAULT 0.05,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_instruments_exchange_token UNIQUE (exchange, symbol_token)
);

CREATE INDEX ix_instruments_exch_sym ON instruments (exchange, symbol);
CREATE INDEX ix_instruments_exch_tsym ON instruments (exchange, trading_symbol);
CREATE INDEX ix_instruments_lookup ON instruments (exchange, symbol, instrument_type);
```

---

## 5. Instrument Master Source & Refresh Process

The master catalog is published daily by Angel One at:
`https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json`

The `AngelOneInstrumentImporter` executes idempotent synchronization:
1. **Fetch**: Downloads JSON from the official endpoint or reads a local JSON cache file.
2. **Parse & Normalize**: Maps Angel One fields (`token` $\rightarrow$ `symbol_token`, `symbol` $\rightarrow$ `trading_symbol`, `name` $\rightarrow$ `symbol`, instrument types, dates).
3. **Validate**: Enforces presence of `token`, `exchange`, and `trading_symbol`.
4. **Idempotent Upsert**: Performs bulk upsert against PostgreSQL based on `(exchange, symbol_token)`. Existing instruments are updated; unchanged instruments are skipped; duplicates are prevented.

---

## 6. Instrument Resolver Behavior

The `InstrumentResolver` handles symbol resolution:
- **Index Disambiguation**: Queries for `"NIFTY"` automatically resolve to the `"Nifty 50"` cash index on `NSE` rather than picking an arbitrary derivative.
- **Cash Equity Matching**: Queries for `"SBIN"` resolve directly to `"SBIN-EQ"`.
- **Missing Instruments**: Raises `InstrumentNotFoundError` if no matching instrument is found.
- **Ambiguous Queries**: If a query matches multiple contracts (e.g. requesting `NIFTY` on `NFO` without specifying `expiry` or `strike`), the resolver refuses to pick arbitrarily and raises `InstrumentResolutionError` listing the missing filter criteria.

---

## 7. Historical API Integration

`AngelOneHistoricalService` supports automatic token resolution:
```python
# User specifies human-readable symbol without manual token lookup
request = HistoricalDataRequest(
    exchange="NSE",
    symbol="NIFTY",
    interval=AngelOneInterval.FIVE_MINUTE,
    from_datetime=datetime(2026, 1, 2, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata")),
    to_datetime=datetime(2026, 1, 2, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
)

service = AngelOneHistoricalService()
result = service.get_candles(request)
# result.symbol_token is automatically resolved to '99926000'
```

---

## 8. CLI Usage

### Refresh Master
```bash
# Download and synchronize instrument master from Angel One
python -m adaptive_trading.instruments.cli refresh

# Or import from a local JSON file
python -m adaptive_trading.instruments.cli refresh --source sample_scrip_master.json
```

### Resolve Symbol to Token
```bash
# Resolve NIFTY index
python -m adaptive_trading.instruments.cli resolve --symbol NIFTY --exchange NSE

# Resolve derivative contract
python -m adaptive_trading.instruments.cli resolve \
    --symbol NIFTY \
    --exchange NFO \
    --type OPTIONS \
    --expiry 2026-01-29 \
    --strike 26000 \
    --option-type CE
```

---

## 9. Security & Error Handling

- **No Secrets Stored**: The `instruments` table contains public market security metadata only; no API keys, credentials, or session tokens are stored.
- **Deterministic & Hermetic Testing**: Tests use isolated SQLite databases with deterministic fixtures containing sample index, equity, futures, and options records with dummy tokens (e.g. `TEST_TOKEN_001`).

