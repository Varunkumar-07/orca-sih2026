"""Shared API-key authentication — audit finding "no auth on any route"
(backend/main.py) and the resulting IDOR on GET /history, GET /history/{id}
(backend/routers/history.py): with no auth, a sequential integer id lets
any client enumerate and read every session's raw query/location history.

Opt-in, not opt-out, matching this repo's own convention for every other
credential (GROQ_API_KEY, COPERNICUSMARINE_*, DATABASE_URL): unset ->
degrade to the existing zero-config demo behavior, never unset -> an
insecure default. Setting ORCA_API_KEY is what turns this on; leaving it
unset (the default) keeps every existing test and the current frontend
working exactly as before. The frontend sends the matching header once
VITE_ORCA_API_KEY is set at build time (frontend/src/lib/apiFetch.ts) —
the two values must be kept in sync (render.yaml).

Every request must then carry a matching `X-API-Key` header, except
GET /health (uptime probes have no way to know a shared secret).
Constant-time comparison (hmac.compare_digest) avoids leaking the key
byte-by-byte through response-time differences.
"""
import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_EXEMPT_PATHS = {"/health"}
_PLACEHOLDER_VALUES = {"", "your-api-key-here", "your_api_key_here"}


def _configured_key() -> str:
    return os.getenv("ORCA_API_KEY", "").strip()


def is_enabled() -> bool:
    return _configured_key().lower() not in _PLACEHOLDER_VALUES


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Rejects any request lacking a matching X-API-Key header with 401.
    Registered only when is_enabled() is true (see main.py) — placed
    innermost of this app's middlewares so a rejected request still counts
    against RateLimitMiddleware's per-IP quota (deterring brute-force key
    guessing) and still carries CORS headers / gets logged like any other."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        expected = _configured_key()
        provided = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(provided, expected):
            return JSONResponse(
                {"detail": "Missing or invalid API key — set the X-API-Key header."},
                status_code=401,
            )
        return await call_next(request)
