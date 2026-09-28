"""Per-client request-rate limiting middleware — audit backlog item #10
("no request-level rate limiting on any endpoint").

In-memory sliding window, keyed by client IP (request.client.host). No
external store (Redis, etc.) — this is a single-process/single-instance
demo deployment, so process memory is sufficient; a horizontally-scaled
deployment would need a shared store instead, out of scope here.

Same env-var-driven, degrade-by-default convention as backend/history/db.py:
RATE_LIMIT_ENABLED controls whether the middleware does anything at all
(backend/tests/conftest.py sets it to "false" so the existing test suite,
which fires hundreds of requests through one shared TestClient/app instance
in a single process, isn't itself rate-limited), and RATE_LIMIT_PER_MINUTE
overrides the default capacity.
"""
import os
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_DEFAULT_LIMIT_PER_MINUTE = 120
_WINDOW_SECONDS = 60.0

# Never rate limit the health check — uptime probes/monitoring must not be
# able to trip into 429s just from polling frequently.
_EXEMPT_PATHS = {"/health"}


def is_enabled() -> bool:
    return os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() not in ("false", "0", "no")


def _configured_limit() -> int:
    return int(os.getenv("RATE_LIMIT_PER_MINUTE", str(_DEFAULT_LIMIT_PER_MINUTE)))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-capacity sliding window per client IP: each client may make up
    to `limit` requests in any trailing `window_seconds` interval. Requests
    over the limit get a 429 with a Retry-After header instead of reaching
    any route handler.

    Registered as the innermost of this app's three middlewares (see
    main.py) — CORSMiddleware still wraps it, so a 429 response still
    carries CORS headers the browser needs to actually see it, and
    HistoryLoggingMiddleware still wraps it too, so a rejected request is
    still logged like any other.
    """

    def __init__(self, app, limit: int | None = None, window_seconds: float = _WINDOW_SECONDS):
        super().__init__(app)
        self._limit = limit if limit is not None else _configured_limit()
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_sweep = time.monotonic()

    @staticmethod
    def _client_key(request: Request) -> str:
        # request.client is None for some proxied/test-client setups,
        # which previously collapsed every such client into one shared
        # "unknown" bucket (either trivially exhausted by one client or
        # wrongly throttling all of them together). Fall back to the
        # first X-Forwarded-For entry when present — note this is only
        # trustworthy when the app sits behind a proxy that sets/overwrites
        # this header itself; a directly-exposed deployment must not rely
        # on it, since a direct client can forge it.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _sweep_idle_buckets(self, now: float) -> None:
        """Opportunistic full-dict cleanup, run at most once per window.
        Per-request trimming above only touches the current request's own
        bucket, so without this, self._hits grows by one entry per distinct
        client key ever seen and never shrinks — unbounded memory growth
        over the life of the process, easily driven by IP/header rotation."""
        if now - self._last_sweep < self._window_seconds:
            return
        self._last_sweep = now
        stale_keys = [key for key, bucket in self._hits.items() if not bucket or now - bucket[-1] > self._window_seconds]
        for key in stale_keys:
            del self._hits[key]

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        now = time.monotonic()
        self._sweep_idle_buckets(now)

        client_key = self._client_key(request)
        bucket = self._hits[client_key]

        while bucket and now - bucket[0] > self._window_seconds:
            bucket.popleft()

        if len(bucket) >= self._limit:
            retry_after = max(1, int(self._window_seconds - (now - bucket[0])))
            return JSONResponse(
                {"detail": "Too many requests — please slow down and try again shortly."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)
