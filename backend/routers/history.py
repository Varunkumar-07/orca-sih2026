"""History endpoints — Phase 8 persistence layer, backing the History page.

Split out of main.py (P4 codebase-audit cleanup) purely for file size; no
behavior change. See backend/history/ for the middleware that populates
this data and backend/history/service.py for the read logic itself.
"""
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.history.schemas import HistoryListResponse, HistoryRecordDetail
from backend.history.service import IST_OFFSET_MINUTES, get_history_record, list_history

router = APIRouter()


@router.get("/history", response_model=HistoryListResponse)
async def history_list(
    page_source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    tz_offset_minutes: int = Query(IST_OFFSET_MINUTES, ge=-12 * 60, le=14 * 60),
) -> dict:
    """Paginated history list for the History page, optionally filtered by
    page_source (chat/zones/weather/route/alerts/analytics/download) and/or
    a date range. The range is calendar days in the caller's timezone,
    `tz_offset_minutes` ahead of UTC (the History page sends the browser's
    own; IST when omitted). See backend/history/service.py."""
    try:
        parsed_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC).date() if start_date else None
        parsed_end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC).date() if end_date else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"start_date/end_date must be YYYY-MM-DD: {exc}") from exc
    return await list_history(
        page_source=page_source,
        start_date=parsed_start,
        end_date=parsed_end,
        limit=limit,
        offset=offset,
        utc_offset_minutes=tz_offset_minutes,
    )


@router.get("/history/{record_id}", response_model=HistoryRecordDetail)
async def history_detail(record_id: int) -> Any:
    """Single history record detail — the raw request+response payload
    captured by HistoryLoggingMiddleware for one past lookup/query."""
    record = await get_history_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="history record not found")
    return record
