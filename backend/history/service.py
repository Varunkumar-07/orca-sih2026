"""Read/write helpers for the `history` table.

log_history() is the one reusable logging hook — called only from
HistoryLoggingMiddleware, never duplicated per-page. list_history() /
get_history_record() back the GET /history and GET /history/{id} endpoints.

Every function is a no-op (empty result / silently skipped write) when the
history DB isn't configured, so none of the 8 pages — or the existing test
suite — depend on a database being present.
"""
import json
import logging
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from backend.history.db import get_session_factory, is_enabled
from backend.history.models import HistoryRecord

logger = logging.getLogger("orca.history")

# start_date/end_date are calendar days as the (India-based) user means
# them, but HistoryRecord.timestamp is stored/compared in UTC — comparing
# a naive date directly against it (as this module used to) is the same
# class of IST day-boundary bug already fixed elsewhere in this codebase
# (see backend/services/weather_service.py's own _IST) for up to 5.5
# hours around midnight IST.
_IST = timezone(timedelta(hours=5, minutes=30))
_MAX_LIST_LIMIT = 200


def _sanitize_non_finite(value: Any) -> Any:
    """NaN/Infinity/-Infinity floats round-trip fine through Python's own
    json module (it accepts them by default) but aren't valid JSON and are
    rejected by Postgres's strict JSONB — replace them with null so one
    non-finite value doesn't fail the whole insert (caught only by
    log_history's blanket except, silently losing the entire history row)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_finite(v) for v in value]
    return value


def _json_safe(value: Any) -> Any:
    """Round-trip through json.dumps(default=str) so datetimes, etc. inside
    arbitrary request/response payloads never blow up the JSONB write."""
    return _sanitize_non_finite(json.loads(json.dumps(value, default=str)))


async def log_history(
    *,
    page_source: str,
    query_summary: str,
    session_id: str,
    request_payload: Any,
    response_payload: Any,
) -> None:
    if not is_enabled():
        return
    session_factory = get_session_factory()
    try:
        async with session_factory() as db:
            record = HistoryRecord(
                page_source=page_source,
                query_summary=query_summary[:1000],
                session_id=session_id[:64],
                payload=_json_safe({"request": request_payload, "response": response_payload}),
            )
            db.add(record)
            await db.commit()
    except Exception as exc:
        logger.warning("Failed to log history record for page_source=%s: %s", page_source, exc)


async def list_history(
    *,
    page_source: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    # Defense in depth — the router layer also validates these, but this is
    # a public function other callers could reach directly.
    limit = max(1, min(limit, _MAX_LIST_LIMIT))
    offset = max(0, offset)

    if not is_enabled():
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    session_factory = get_session_factory()
    async with session_factory() as db:
        filters = []
        if page_source:
            filters.append(HistoryRecord.page_source == page_source)
        if start_date:
            filters.append(HistoryRecord.timestamp >= datetime.combine(start_date, time.min, tzinfo=_IST))
        if end_date:
            # end_date is a calendar day with no time component — compare
            # against the start of the *next* IST day so today's own rows
            # aren't excluded.
            filters.append(
                HistoryRecord.timestamp < datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=_IST)
            )

        count_stmt = select(func.count()).select_from(HistoryRecord)
        list_stmt = select(HistoryRecord)
        for condition in filters:
            count_stmt = count_stmt.where(condition)
            list_stmt = list_stmt.where(condition)
        list_stmt = list_stmt.order_by(HistoryRecord.timestamp.desc()).limit(limit).offset(offset)

        total = (await db.execute(count_stmt)).scalar_one()
        rows = (await db.execute(list_stmt)).scalars().all()
        return {"items": list(rows), "total": total, "limit": limit, "offset": offset}


async def get_history_record(record_id: int) -> HistoryRecord | None:
    if not is_enabled():
        return None
    session_factory = get_session_factory()
    async with session_factory() as db:
        return await db.get(HistoryRecord, record_id)
