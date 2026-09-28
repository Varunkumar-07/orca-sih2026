"""create history table

Additive: adds the new `history` table only, touches nothing else in the
schema (there is nothing else in the schema yet). Fully reversible via
`alembic downgrade -1`, which drops exactly what upgrade() created.

Revision ID: a1761619abbf
Revises:
Create Date: 2026-09-13 21:44:24.438380

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1761619abbf'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("page_source", sa.String(length=32), nullable=False),
        sa.Column("query_summary", sa.Text(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint(
            "page_source IN ('chat','zones','weather','route','alerts','analytics','download')",
            name="ck_history_page_source",
        ),
    )
    op.create_index("ix_history_page_source_timestamp", "history", ["page_source", "timestamp"])
    op.create_index("ix_history_session_id", "history", ["session_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_history_session_id", table_name="history")
    op.drop_index("ix_history_page_source_timestamp", table_name="history")
    op.drop_table("history")
