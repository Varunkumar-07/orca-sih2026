"""
ORCA backend — FastAPI app composition root.

Endpoint implementations live in backend/routers/ (split out from this file
during a P4 codebase-audit cleanup — see each router module's own
docstring for what it covers and why the split happened; no behavior
change from the split itself):
  backend/routers/chat.py     -> POST /query, /query/full, /query/demo,
                                  GET /fixtures
  backend/routers/pages.py    -> GET /zones, /weather, /weather/forecast,
                                  POST /route, GET /alerts,
                                  /analytics/historical, /export
  backend/routers/history.py  -> GET /history, /history/{id}

This file owns only: app/middleware setup, the startup lifespan (PFZ cache
pre-warm), and the two endpoints too trivial to need their own router.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Anchored to this file's own location, not the process cwd: load_dotenv()'s
# default search only walks upward from cwd, so it silently finds nothing
# when the server is run the documented way (from the project root, one
# level above backend/ — see this repo's own README "Common mistake" note)
# even though backend/.env sits right here. That failure is silent — every
# credential just reads as unset — and was caught only because PFZ zones
# failed instantly with no Copernicus auth attempt at all.
load_dotenv(Path(__file__).resolve().parent / ".env")

from backend.auth import ApiKeyAuthMiddleware
from backend.auth import is_enabled as api_key_auth_enabled
from backend.history.db import dispose_engine
from backend.history.middleware import HistoryLoggingMiddleware
from backend.rate_limit import RateLimitMiddleware
from backend.rate_limit import is_enabled as rate_limiting_enabled
from backend.routers import chat, history, pages
from backend.services.pfz_service import get_cached_zones

# Audit finding: no cap on request body size anywhere in the app: combined
# with every route being unauthenticated (see backend/auth.py) and
# unbounded, this made large-body DoS trivial. Checked via Content-Length
# rather than reading the body, so oversized requests are rejected before
# any downstream middleware (notably HistoryLoggingMiddleware) reads them.
_MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(2 * 1024 * 1024)))


class _MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > _MAX_REQUEST_BODY_BYTES
            except ValueError:
                too_large = False
            if too_large:
                return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

# Holds the one strong reference to the fire-and-forget prewarm task below —
# asyncio only holds a weak reference to a task, so with nothing else
# referencing it, it can be garbage-collected mid-execution (a documented
# asyncio gotcha), silently vanishing with just a "Task was destroyed but
# it is pending" warning and none of _prewarm's own error handling ever
# running. Discarded via the task's own done-callback once it finishes.
_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Phase 4 (Chat-PFZ Live-Data Integration): pre-warm the shared live
    PFZ zone cache (pfz_service.get_cached_zones) in the background at
    startup, so the first real user — whether they hit chat or GET
    /zones — doesn't eat the ~55s cold-cache recompute across all 11
    anchors (see get_cached_zones's own docstring).

    asyncio.create_task fires this without awaiting it: the lifespan
    startup phase (and therefore Uvicorn binding/serving) proceeds
    immediately, the actual fetch continues on the event loop in the
    background. A failure here (no Copernicus credentials, network
    issue, etc.) is logged only — every endpoint's own per-request live
    lookup (chat's _live_pfz_candidate_zones, Zones Explorer's per-anchor
    fetch) still runs regardless of whether pre-warm succeeded; it just
    won't have a warm cache to read.
    """

    async def _prewarm() -> None:
        try:
            await get_cached_zones()
            logging.getLogger("orca.pfz").info("PFZ zone cache pre-warmed at startup")
        except Exception as exc:
            logging.getLogger("orca.pfz").warning(
                "PFZ zone cache pre-warm failed at startup (per-request fallback still applies): %s", exc
            )

    task = asyncio.create_task(_prewarm())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    yield
    # Audit finding: the history DB engine was created at import time with
    # no corresponding cleanup — dispose its connection pool on shutdown
    # rather than leaking open connections when the process exits.
    await dispose_engine()


app = FastAPI(title="ORCA Backend", version="0.3.0", lifespan=_lifespan)

# Added first (before rate limiting/CORS) so it ends up the innermost of
# this app's middlewares — a rejected request still counts against
# RateLimitMiddleware's per-IP quota (see backend/auth.py's own docstring)
# and still carries CORS headers / gets logged like any other. Opt-in (not
# opt-out) — see backend/auth.py's own docstring for why, and for the
# ORCA_API_KEY env var.
if api_key_auth_enabled():
    app.add_middleware(ApiKeyAuthMiddleware)

# Added next so it ends up the innermost of the remaining three
# middlewares below — CORSMiddleware still wraps it, so a 429 it returns
# still carries CORS headers the browser needs to actually see the
# response, and it never intercepts CORS's own OPTIONS-preflight handling.
# Opt-out (not opt-in) — see backend/rate_limit.py's own docstring for the
# RATE_LIMIT_ENABLED / RATE_LIMIT_PER_MINUTE env vars.
if rate_limiting_enabled():
    app.add_middleware(RateLimitMiddleware)

# CORS for frontend (Vite default 5173). Explicit origins required here —
# allow_origins=["*"] combined with allow_credentials=True is invalid per
# the CORS spec and browsers reject it for any credentialed request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Phase 8: logs every lookup/query on the 8 page-facing endpoints below into
# the `history` table (see backend/history/). Added after CORSMiddleware so
# it wraps outermost of these four — it sees the final response, CORS
# headers included — and is a no-op entirely when DATABASE_URL isn't
# configured.
app.add_middleware(HistoryLoggingMiddleware)
# Added last so it's the outermost middleware overall — oversized requests
# are rejected before anything else (notably HistoryLoggingMiddleware's own
# request.body() read) touches the body.
app.add_middleware(_MaxBodySizeMiddleware)

app.include_router(chat.router)
app.include_router(pages.router)
app.include_router(history.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
