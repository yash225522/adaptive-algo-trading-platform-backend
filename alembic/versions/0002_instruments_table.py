"""Add instruments table for Angel One master.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-30

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol_token", sa.String(length=32), nullable=False),
        sa.Column("exchange", sa.String(length=16), nullable=False),
        sa.Column("trading_symbol", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("instrument_type", sa.String(length=16), nullable=False),
        sa.Column("expiry", sa.Date(), nullable=True),
        sa.Column("strike", sa.Float(), nullable=True),
        sa.Column("option_type", sa.String(length=8), nullable=True),
        sa.Column("lot_size", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("tick_size", sa.Float(), nullable=False, server_default="0.05"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "exchange",
            "symbol_token",
            name="uq_instruments_exchange_token",
        ),
    )
    op.create_index(
        "ix_instruments_exch_sym",
        "instruments",
        ["exchange", "symbol"],
    )
    op.create_index(
        "ix_instruments_exch_tsym",
        "instruments",
        ["exchange", "trading_symbol"],
    )
    op.create_index(
        "ix_instruments_lookup",
        "instruments",
        ["exchange", "symbol", "instrument_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_instruments_lookup", table_name="instruments")
    op.drop_index("ix_instruments_exch_tsym", table_name="instruments")
    op.drop_index("ix_instruments_exch_sym", table_name="instruments")
    op.drop_table("instruments")
