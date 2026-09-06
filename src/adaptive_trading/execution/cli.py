"""Command line interface for paper trading execution and account status."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.execution.models import (
    OrderRequest,
    OrderSide,
    OrderType,
    ProductType,
)
from adaptive_trading.execution.paper_broker import (
    PaperBroker,
    StaticMarketDataProvider,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Shared state file for local paper trading CLI state persistence
PAPER_STATE_FILE = Path("artifacts/paper/account_state.json")


def _load_paper_broker(config: ExecutionConfig | None = None) -> PaperBroker:
    """Load or initialize persistent paper broker state."""
    cfg = config or ExecutionConfig()
    provider = StaticMarketDataProvider()
    broker = PaperBroker(config=cfg, market_data_provider=provider)

    if PAPER_STATE_FILE.exists():
        try:
            with open(PAPER_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                broker.cash = float(data.get("cash", cfg.initial_cash))
                broker.realized_pnl = float(data.get("realized_pnl", 0.0))
        except Exception as exc:
            logger.warning("Could not read paper state file: %s", exc)

    return broker


def _save_paper_broker(broker: PaperBroker) -> None:
    """Save paper broker state to disk."""
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    account = broker.get_account()
    data = {
        "cash": account.cash,
        "equity": account.equity,
        "realized_pnl": account.realized_pnl,
        "unrealized_pnl": account.unrealized_pnl,
        "positions": {k: v.model_dump() for k, v in account.positions.items()},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def cmd_status(config: ExecutionConfig) -> int:
    """Display execution environment status."""
    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - EXECUTION STATUS          ")
    print("=" * 60)
    print(f"Execution Mode   : {config.mode.value}")
    print(f"Default Exchange : {config.default_exchange}")
    comm_pct = config.commission_bps * 0.01
    slip_pct = config.slippage_bps * 0.01
    print(f"Commission (bps) : {config.commission_bps} ({comm_pct:.2f}%)")
    print(f"Slippage (bps)   : {config.slippage_bps} ({slip_pct:.2f}%)")
    print("Live Trading     : DISABLED (Safety Enforced)")
    print("=" * 60)
    return 0


def cmd_account(config: ExecutionConfig) -> int:
    """Display paper account balances and positions."""
    broker = _load_paper_broker(config)
    account = broker.get_account()

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - PAPER ACCOUNT             ")
    print("=" * 60)
    print(f"Cash Balance     : {account.cash:,.2f}")
    print(f"Total Equity     : {account.equity:,.2f}")
    print(f"Buying Power     : {account.buying_power:,.2f}")
    print(f"Realized P&L     : {account.realized_pnl:+,.2f}")
    print(f"Unrealized P&L   : {account.unrealized_pnl:+,.2f}")
    print("-" * 60)
    print(f"Open Positions   : {len(account.positions)}")
    for sym, pos in account.positions.items():
        if pos.quantity > 0:
            print(
                f"  • {sym} | {pos.side.value} | Qty: {pos.quantity:.1f} | "
                f"Avg: {pos.average_entry_price:,.1f} | "
                f"PnL: {pos.unrealized_pnl:+,.1f}"
            )
    print("=" * 60)
    return 0


def cmd_submit(
    symbol: str,
    side: str,
    quantity: float,
    price: float | None = None,
    exchange: str = "NSE",
    order_type: str = "MARKET",
    config: ExecutionConfig | None = None,
) -> int:
    """Submit a paper order and print the execution fill result."""
    cfg = config or ExecutionConfig()
    broker = _load_paper_broker(cfg)

    # Set mock reference price if provided
    if price is not None and price > 0:
        if isinstance(broker.market_data_provider, StaticMarketDataProvider):
            broker.market_data_provider.set_price(symbol, price)

    order_side = OrderSide(side.upper())
    ord_type = OrderType(order_type.upper())

    request = OrderRequest(
        symbol=symbol,
        exchange=exchange,
        side=order_side,
        quantity=quantity,
        order_type=ord_type,
        price=price,
        product_type=ProductType.INTRADAY,
    )

    order = broker.submit_order(request)
    _save_paper_broker(broker)

    fill_price_str = (
        f"{order.average_fill_price:,.2f}"
        if order.average_fill_price is not None
        else "N/A"
    )

    print("=" * 60)
    print("                  PAPER ORDER CONFIRMATION                  ")
    print("=" * 60)
    print(f"Symbol             : {order.symbol}")
    print(f"Exchange           : {order.exchange}")
    print(f"Side               : {order.side.value}")
    print(f"Quantity           : {order.quantity}")
    print(f"Type               : {order.order_type.value}")
    print("-" * 60)
    print(f"Status             : {order.status.value}")
    print(f"Average Fill Price : {fill_price_str}")
    if order.rejection_reason:
        print(f"Rejection Reason   : {order.rejection_reason}")
    print(f"Order ID           : {order.order_id}")
    print(f"Client Order ID    : {order.client_order_id}")
    print("=" * 60)
    return 0 if order.status.value == "FILLED" else 1


def main() -> None:
    """CLI entrypoint for paper trading execution commands."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Platform Execution & Paper Trading CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 1. status
    subparsers.add_parser("status", help="Show execution environment status")

    # 2. account
    subparsers.add_parser("account", help="Display paper trading account details")

    # 3. submit
    submit_parser = subparsers.add_parser("submit", help="Submit a manual paper order")
    submit_parser.add_argument(
        "--symbol", "-s", type=str, required=True, help="Market ticker symbol"
    )
    submit_parser.add_argument(
        "--side",
        type=str,
        required=True,
        choices=["BUY", "SELL", "buy", "sell"],
        help="Order side",
    )
    submit_parser.add_argument(
        "--quantity", "-q", type=float, required=True, help="Order quantity"
    )
    submit_parser.add_argument(
        "--price", "-p", type=float, default=None, help="Reference market price"
    )
    submit_parser.add_argument(
        "--exchange", "-e", type=str, default="NSE", help="Target exchange"
    )
    submit_parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="MARKET",
        choices=["MARKET", "LIMIT", "market", "limit"],
        help="Order type",
    )

    args = parser.parse_args()
    config = ExecutionConfig()

    if args.command == "status":
        code = cmd_status(config)
    elif args.command == "account":
        code = cmd_account(config)
    elif args.command == "submit":
        code = cmd_submit(
            symbol=args.symbol,
            side=args.side,
            quantity=args.quantity,
            price=args.price,
            exchange=args.exchange,
            order_type=args.type,
            config=config,
        )
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
