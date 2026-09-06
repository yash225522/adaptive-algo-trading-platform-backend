"""Add observability columns to system runs table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-30

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(
            sa.Column("run_id", sa.String(length=128), nullable=True)
        )
        batch_op.create_unique_constraint("uq_runs_run_id", ["run_id"])
        batch_op.add_column(
            sa.Column("duration_seconds", sa.Float(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("run_metadata", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("error_summary", sa.Text(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("final_equity", sa.Float(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("net_pnl", sa.Float(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_column("net_pnl")
        batch_op.drop_column("final_equity")
        batch_op.drop_column("error_summary")
        batch_op.drop_column("run_metadata")
        batch_op.drop_column("duration_seconds")
        batch_op.drop_constraint("uq_runs_run_id", type_="unique")
        batch_op.drop_column("run_id")

