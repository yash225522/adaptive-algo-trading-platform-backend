# Historical Backtesting Engine

This document details the architecture, execution conventions, position state transitions, cash accounting, slippage and commission models, drawdown calculations, performance metrics, and CLI for the **Backtesting Engine** in `adaptive-algo-trading-platform`.

---

## 1. Purpose of the Backtesting Engine

The Backtesting Engine evaluates pre-computed `TradingSignal` sequences against historical `MarketCandle` data to simulate strategy performance without lookahead bias, broker dependencies, or live risk exposure.

```text
Market Candles + Out-of-Sample Signals
                  ↓
           Backtest Engine
                  ↓
            Simulated Fill
                  ↓
           Portfolio State
                  ↓
                Trade
                  ↓
             Equity Curve
                  ↓
         Performance Metrics
```

---

## 2. Execution Timing & Look-Ahead Protection

To eliminate lookahead bias:
- A `TradingSignal` generated at candle $T$ (based on that candle's close) is executed at **candle $T+1$'s open price**.
- Executing at the same candle's close would be invalid because the signal was calculated from that exact close.
- If no subsequent candle $T+1$ exists in the dataset, no simulated order is executed.

---

## 3. Position State Machine & Reversal Logic

The portfolio tracks a single position per symbol:
- $\mathbf{FLAT \rightarrow LONG}$: Opens long position with `fixed_quantity` at $T+1$ open (adjusted for slippage and commission).
- $\mathbf{FLAT \rightarrow SHORT}$: Opens short position with `fixed_quantity` at $T+1$ open.
- $\mathbf{LONG \rightarrow SHORT}$ *(Reversal)*:
  1. Closes the existing LONG position at $T+1$ open (SELL), records completed `BacktestTrade`.
  2. Opens new SHORT position at $T+1$ open (SELL).
- $\mathbf{SHORT \rightarrow LONG}$ *(Reversal)*:
  1. Closes the existing SHORT position at $T+1$ open (BUY to cover), records completed `BacktestTrade`.
  2. Opens new LONG position at $T+1$ open (BUY).
- $\mathbf{NO\_TRADE}$ or duplicate directional signals maintain existing position without generating redundant fills.

---

## 4. Slippage & Commission Models

### Slippage Model
Applied to the base execution price (candle open):
- **BUY (or Cover)**:
  $$\text{Executed Price} = \text{Open Price} \times \left(1 + \frac{\text{slippage\_bps}}{10\,000}\right)$$
- **SELL (or Short)**:
  $$\text{Executed Price} = \text{Open Price} \times \left(1 - \frac{\text{slippage\_bps}}{10\,000}\right)$$

### Commission Model
Proportional fee per transaction:
$$\text{Commission} = \text{Executed Price} \times \text{Quantity} \times \left(\frac{\text{commission\_bps}}{10\,000}\right)$$

---

## 5. P&L & Cash Accounting

### Completed Trade P&L
- **LONG Trade**:
  $$\text{Gross P&L} = (\text{Exit Price} - \text{Entry Price}) \times \text{Quantity}$$
  $$\text{Net P&L} = \text{Gross P&L} - \text{Entry Commission} - \text{Exit Commission}$$
- **SHORT Trade**:
  $$\text{Gross P&L} = (\text{Entry Price} - \text{Exit Price}) \times \text{Quantity}$$
  $$\text{Net P&L} = \text{Gross P&L} - \text{Entry Commission} - \text{Exit Commission}$$

### Portfolio Equity & Mark-to-Market
At each candle close:
- For LONG position: $\text{Unrealized P&L} = (\text{Close Price} - \text{Entry Price}) \times \text{Quantity}$
- For SHORT position: $\text{Unrealized P&L} = (\text{Entry Price} - \text{Close Price}) \times \text{Quantity}$
- Total Equity:
  $$\text{Equity} = \text{Initial Capital} + \text{Realized P&L} + \text{Unrealized P&L}$$

---

## 6. Drawdown Calculation

Calculated continuously along the equity curve:
$$\text{Peak}_t = \max(\text{Initial Capital}, \max_{0 \le i \le t} \text{Equity}_i)$$
$$\text{Drawdown}_t = \text{Peak}_t - \text{Equity}_t$$
$$\text{Max Drawdown (\%)} = \max_t \left( \frac{\text{Drawdown}_t}{\text{Peak}_t} \times 100 \right)$$

---

## 7. Performance Metrics Summary

- `initial_capital`, `final_equity`, `net_pnl`, `total_return_pct`
- `total_trades`, `winning_trades`, `losing_trades`, `scratch_trades`, `win_rate`
- `gross_profit`, `gross_loss`, `profit_factor`
- `average_trade_pnl`, `average_winner`, `average_loser`, `largest_winner`, `largest_loser`
- `max_drawdown_abs`, `max_drawdown_pct`

---

## 8. Artifacts Format

Persisted under `artifacts/backtests/<backtest_id>/`:
- `metadata.json`: Run timestamps, configuration, strategy/model identifiers.
- `metrics.json`: Summary performance and drawdown metrics.
- `trades.csv`: List of completed round-trip trades.
- `equity_curve.csv`: Chronological equity curve series.

---

## 9. CLI Usage

```bash
python -m adaptive_trading.backtesting.cli run \
    --candles data/sample/nifty_5m_ml_sample.csv \
    --signals artifacts/strategy/strat_20260830_174319_50996b/signals.csv \
    --initial-capital 100000 \
    --commission-bps 3.0 \
    --slippage-bps 5.0
```

### Sample Output:
```text
============================================================
      ADAPTIVE TRADING PLATFORM - BACKTEST RESULTS          
============================================================
Strategy         : probability_threshold (v1)
Model            : logistic_regression (v1)
Period           : 2026-01-02 09:15 to 2026-01-02 13:20
------------------------------------------------------------
Initial Capital  : 100,000.00
Final Equity     : 100,124.50
Net P&L          : +124.50
Total Return     : +0.12%
Max Drawdown     : 45.20 (0.05%)
------------------------------------------------------------
Total Trades     : 4
Winning / Losing : 3 / 1
Win Rate         : 75.0%
Profit Factor    : 2.85
Avg Trade P&L    : +31.13
Largest Win/Loss : +85.00 / -30.50
------------------------------------------------------------
Artifacts Saved  : artifacts/backtests/bt_20260830_232000_123abc
============================================================
```

---

## 10. Financial Disclaimer

> [!WARNING]
> **Backtest results are historical simulations and do not guarantee future live trading performance.**
> The simulation assumes fills at the open price with linear slippage and commission models. It does not account for intraday liquidity exhaustion, order queue priority, partial fills, margin calls, or broker downtime.

