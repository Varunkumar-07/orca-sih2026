"""One reusable ASGI middleware that logs every lookup/query across the 8
frontend pages into the `history` table — no per-endpoint duplication in
main.py, and zero changes to the 9 core agents.

Only requests whose path is in PAGE_SOURCE_BY_PATH are logged; everything
else (health check, fixtures, the History API itself) passes through
untouched, and no history endpoint about the History page's own reads
gets logged back into history.
"""
import asyncio
import json
import logging
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.history.service import log_history

logger = logging.getLogger("orca.history")

# Holds strong references to the fire-and-forget log_history() tasks below
# — asyncio only holds a weak reference to a task, so with nothing else
# referencing it, it can be garbage-collected mid-write (same asyncio
# gotcha documented in backend/main.py's own _background_tasks).
_background_tasks: set[asyncio.Task] = set()


def _log_in_background(**kwargs: Any) -> None:
    """Fire-and-forget log_history() — audit finding: awaiting it inline
    added a DB round-trip to every page request's user-visible latency.
    log_history already catches and logs its own failures internally, so
    nothing here needs to observe the task's result."""
    task = asyncio.create_task(log_history(**kwargs))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


PAGE_SOURCE_BY_PATH = {
    "/query": "chat",
    "/query/full": "chat",
    "/query/demo": "chat",
    "/zones": "zones",
    "/weather": "weather",
    "/weather/forecast": "weather",
    "/route": "route",
    "/alerts": "alerts",
    "/analytics/historical": "analytics",
    "/export": "download",
}


def _build_summary(request: Request, request_json: Any) -> str:
    parts = [request.method, request.url.path]
    if request.query_params:
        parts.append(str(dict(request.query_params)))
    elif isinstance(request_json, (dict, list)):
        parts.append(str(request_json)[:150])
    return " ".join(parts)[:500]


class HistoryLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        page_source = PAGE_SOURCE_BY_PATH.get(request.url.path)
        if page_source is None:
            return await call_next(request)

        request_json: Any = None
        try:
            body = await request.body()
            if body:
                request_json = json.loads(body)
        except Exception:
            request_json = None
        if request_json is None and request.query_params:
            request_json = dict(request.query_params)

        response = await call_next(request)

        if request.url.path == "/export":
            # /export can return multi-MB pdf/docx/csv files — buffering
            # the full body into memory just to log a byte count (the only
            # thing logged for non-JSON responses anyway, see below) isn't
            # worth the memory/latency cost. This is a plain Response built
            # from in-memory content (not streamed), so Content-Length is
            # already set — read the size from there instead, and pass the
            # response through untouched.
            try:
                size_bytes = int(response.headers.get("content-length", "0"))
            except ValueError:
                size_bytes = 0
            session_id = (
                request.headers.get("x-session-id")
                or (request_json.get("session_id") if isinstance(request_json, dict) else None)
                or str(uuid.uuid4())
            )
            _log_in_background(
                page_source=page_source,
                query_summary=_build_summary(request, request_json),
                session_id=str(session_id),
                request_payload=request_json,
                response_payload={"content_type": response.headers.get("content-type", ""), "size_bytes": size_bytes},
            )
            return response

        try:
            body_chunks = [chunk async for chunk in response.body_iterator]
            response_body = b"".join(body_chunks)
        except Exception as exc:
            # A failure mid-stream (e.g. client disconnect) shouldn't turn
            # an otherwise-successful downstream response into an unhandled
            # 500 just because our own logging step choked on it.
            logger.warning("Failed to buffer response body for history logging (page_source=%s): %s", page_source, exc)
            return response

        headers = dict(response.headers)
        headers.pop("content-length", None)
        rebuilt = Response(
            content=response_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                response_payload: Any = json.loads(response_body)
            except Exception:
                response_payload = {"content_type": content_type, "size_bytes": len(response_body)}
        else:
            response_payload = {"content_type": content_type, "size_bytes": len(response_body)}

        session_id = (
            response.headers.get("x-session-id")
            or request.headers.get("x-session-id")
            or (request_json.get("session_id") if isinstance(request_json, dict) else None)
            or str(uuid.uuid4())
        )

        _log_in_background(
            page_source=page_source,
            query_summary=_build_summary(request, request_json),
            session_id=str(session_id),
            request_payload=request_json,
            response_payload=response_payload,
        )

        return rebuilt
