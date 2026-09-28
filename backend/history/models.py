"""ORM model for the `history` table — one row per lookup/query across the
8 frontend pages. See backend/history/middleware.py for how rows get here."""
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON

from backend.history.db import Base

PAGE_SOURCES = ("chat", "zones", "weather", "route", "alerts", "analytics", "download")

# JSONB on Postgres, plain JSON elsewhere (e.g. SQLite in a future local-dev
# fallback) — same column, no dialect-specific code at the call sites.
_JSONVariant = JSON().with_variant(JSONB(), "postgresql")


class HistoryRecord(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    page_source = Column(String(32), nullable=False)
    query_summary = Column(Text, nullable=False)
    session_id = Column(String(64), nullable=False)
    payload = Column(_JSONVariant, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "page_source IN ('chat','zones','weather','route','alerts','analytics','download')",
            name="ck_history_page_source",
        ),
        Index("ix_history_page_source_timestamp", "page_source", "timestamp"),
        Index("ix_history_session_id", "session_id"),
    )
