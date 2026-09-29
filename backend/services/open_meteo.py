"""
Shared Open-Meteo access for every live-weather path (Weather page, chat's
weather/marine agents, Alerts, Analytics).

Why this exists: in production (Render free tier, whose outbound IPs are
shared with other tenants) Open-Meteo's keyless tier answered with 429s —
both "Daily API request limit exceeded" and "Too many concurrent requests"
(GET /alerts alone fanned out ~30 forecast calls at once). Every caller
used to open its own client and fire immediately, so the same point was
re-fetched by chat, the Weather page and Alerts within seconds of each
other, and a burst could exceed the concurrency limit on its own.

What this module adds, in one place:
- A short in-process TTL cache per exact request (current conditions only
  change every ~15 minutes upstream), with concurrent identical requests
  coalesced onto one upstream call.
- A process-wide cap on concurrent Open-Meteo requests (_MAX_CONCURRENT).
- Bounded retry with exponential backoff for genuinely transient failures:
  timeouts, connection errors, 5xx, and the "Too many concurrent requests"
  429.
- A per-host circuit breaker for quota 429s (daily/hourly/minutely). Those
  are never retried — a retry is a guaranteed second 429 that just burns
  more quota — and further calls to that host fail fast until the block
  expires, instead of every request re-hitting a host that already said no.
- Failures are raised as OpenMeteoError carrying a short, user-safe
  message; the raw upstream detail (URL, body) goes to the log only.

Optional OPEN_METEO_API_KEY switches every call to Open-Meteo's commercial
"customer-" hosts with the key attached — the only way to get a quota that
isn't shared with every other tenant behind the same outbound IP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import weakref
from urllib.parse import urlsplit

import httpx

from backend.error_utils import describe_exception

logger = logging.getLogger("orca.open_meteo")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# The free-tier forecast host has a hard daily request cap. The Historical
# Forecast API serves the same near-real-time model output (confirmed:
# matching current temperature/wind and today's daily aggregates) from a
# separate quota bucket and accepts identical parameters — a safe drop-in
# fallback for current conditions once the primary host's quota is gone.
FORECAST_FALLBACK_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_HTTP_TIMEOUT = 8.0
_MAX_ATTEMPTS = 2
_BACKOFF_BASE_SECONDS = 0.5
# Open-Meteo documents no exact concurrency limit for the free tier; 4 is
# enough to keep GET /alerts' 30-zone sweep to a few seconds while staying
# far below the burst that triggered "Too many concurrent requests".
_MAX_CONCURRENT = 4

# Current conditions: Open-Meteo's own "current" values update every 15
# minutes, so caching for 10 never serves anything staler than upstream
# itself would by more than one update.
CURRENT_TTL_SECONDS = 10 * 60
# Coordinates for current-condition lookups are rounded to 0.01° (~1 km) —
# finer than any Open-Meteo model grid — so GPS jitter between chat turns
# and the Weather page's zone points share cache entries.
_COORD_DECIMALS = 2

# Quota blocks. Open-Meteo doesn't document exactly when a daily window
# resets, so a daily block is re-probed hourly (at most ~24 wasted calls a
# day per host) rather than assumed to last until some fixed clock time.
_BLOCK_SECONDS_DAILY = 60 * 60
_BLOCK_SECONDS_HOURLY = 15 * 60
_BLOCK_SECONDS_MINUTELY = 60
_BLOCK_SECONDS_UNKNOWN_429 = 60


class OpenMeteoError(Exception):
    """An Open-Meteo request that could not be served. `user_message` is
    safe to show end users; str(exc) is the log-level detail."""

    def __init__(self, detail: str, *, kind: str, user_message: str):
        super().__init__(detail)
        self.kind = kind  # "quota" | "unavailable"
        self.user_message = user_message


_QUOTA_USER_MESSAGE = "the weather service's request limit has been reached — please try again later"
_UNAVAILABLE_USER_MESSAGE = "the weather service could not be reached — please try again shortly"

_cache: dict[tuple, tuple[float, dict]] = {}
_blocked_until: dict[str, float] = {}
# Per event loop: asyncio primitives are bound to the loop they're first
# used on (tests run many short-lived loops via asyncio.run).
_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()
_inflight: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple, asyncio.Task]] = weakref.WeakKeyDictionary()


def reset_state() -> None:
    """Clear the cache and every quota block (tests; never needed in prod)."""
    _cache.clear()
    _blocked_until.clear()


def _api_key() -> str | None:
    key = (os.getenv("OPEN_METEO_API_KEY") or "").strip()
    return key or None


def _resolve(url: str, params: dict) -> tuple[str, dict]:
    key = _api_key()
    if key is None:
        return url, params
    parts = urlsplit(url)
    return f"{parts.scheme}://customer-{parts.netloc}{parts.path}", {**params, "apikey": key}


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = _semaphores[loop] = asyncio.Semaphore(_MAX_CONCURRENT)
    return sem


def _inflight_for_loop() -> dict[tuple, asyncio.Task]:
    loop = asyncio.get_running_loop()
    tasks = _inflight.get(loop)
    if tasks is None:
        tasks = _inflight[loop] = {}
    return tasks


def _quota_block_seconds(body: str) -> float | None:
    """Seconds to stop calling a host after this 429, or None when the 429
    is a transient "too many concurrent requests" worth a backoff retry."""
    lowered = body.lower()
    if "concurrent" in lowered:
        return None
    if "daily" in lowered:
        return _BLOCK_SECONDS_DAILY
    if "hourly" in lowered:
        return _BLOCK_SECONDS_HOURLY
    if "minutely" in lowered:
        return _BLOCK_SECONDS_MINUTELY
    return _BLOCK_SECONDS_UNKNOWN_429


async def _backoff(attempt: int) -> None:
    if _BACKOFF_BASE_SECONDS > 0:
        delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        await asyncio.sleep(delay + random.uniform(0, delay / 2))


async def _fetch(url: str, params: dict, label: str) -> dict:
    host = urlsplit(url).netloc
    request_url, request_params = _resolve(url, params)
    last_detail = "no attempt made"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        blocked_until = _blocked_until.get(host, 0.0)
        if time.monotonic() < blocked_until:
            raise OpenMeteoError(
                f"{label}: {host} quota block active for another {blocked_until - time.monotonic():.0f}s",
                kind="quota",
                user_message=_QUOTA_USER_MESSAGE,
            )
        try:
            async with _semaphore(), httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(request_url, params=request_params)
        except httpx.TransportError as exc:  # timeouts, connection/protocol errors
            last_detail = describe_exception(exc)
        else:
            if resp.status_code == 429:
                body = resp.text[:300]
                block_seconds = _quota_block_seconds(body)
                if block_seconds is not None:
                    _blocked_until[host] = time.monotonic() + block_seconds
                    logger.warning("%s quota exceeded (HTTP 429), pausing %s for %ds: %s", label, host, block_seconds, body)
                    raise OpenMeteoError(
                        f"{label}: HTTP 429 quota exceeded: {body}", kind="quota", user_message=_QUOTA_USER_MESSAGE
                    )
                last_detail = f"HTTP 429 (transient): {body}"
            elif resp.status_code >= 500:
                last_detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
            else:
                try:
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.HTTPStatusError, ValueError) as exc:
                    # 4xx other than 429 (bad parameters etc.) or an
                    # unparseable body — not transient, never retried.
                    raise OpenMeteoError(
                        f"{label}: {describe_exception(exc)}", kind="unavailable", user_message=_UNAVAILABLE_USER_MESSAGE
                    ) from exc
        if attempt < _MAX_ATTEMPTS:
            await _backoff(attempt)
    logger.warning("%s failed after %d attempt(s): %s", label, _MAX_ATTEMPTS, last_detail)
    raise OpenMeteoError(f"{label}: {last_detail}", kind="unavailable", user_message=_UNAVAILABLE_USER_MESSAGE)


async def get_json(url: str, params: dict, *, label: str, ttl: float) -> dict:
    """GET an Open-Meteo endpoint through the shared cache/limits (see the
    module docstring). Raises OpenMeteoError on failure; never caches one."""
    key = (url, tuple(sorted((k, str(v)) for k, v in params.items())))
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]

    inflight = _inflight_for_loop()
    task = inflight.get(key)
    if task is None:

        async def _run() -> dict:
            try:
                data = await _fetch(url, params, label)
                _cache[key] = (time.monotonic(), data)
                return data
            finally:
                inflight.pop(key, None)

        task = inflight[key] = asyncio.ensure_future(_run())
    return await asyncio.shield(task)


def _round(value: float) -> float:
    return round(value, _COORD_DECIMALS)


async def fetch_current_forecast(lat: float, lon: float) -> dict:
    """Current conditions plus today's daily totals in one request —
    `daily.wind_speed_10m_max` is today's forecast peak, a genuinely
    different number from `current.wind_speed_10m`'s right-now reading.
    One superset parameter set for every caller (Weather page, chat,
    Alerts) so they share cache entries. Falls back to the Historical
    Forecast host (separate quota) when the primary host fails."""
    params = {
        "latitude": _round(lat),
        "longitude": _round(lon),
        "current": "temperature_2m,wind_speed_10m,weather_code",
        "daily": "precipitation_sum,wind_speed_10m_max",
        "forecast_days": 1,
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    try:
        return await get_json(FORECAST_URL, params, label="Open-Meteo forecast", ttl=CURRENT_TTL_SECONDS)
    except OpenMeteoError as primary_exc:
        logger.info("Falling back to historical-forecast-api for current weather (%s)", primary_exc)
        return await get_json(
            FORECAST_FALLBACK_URL, params, label="Open-Meteo historical-forecast fallback", ttl=CURRENT_TTL_SECONDS
        )


async def fetch_current_marine(lat: float, lon: float) -> dict:
    """Current wave height + sea surface temperature in one request (the
    superset every marine caller needs). Returns the `current` block."""
    data = await get_json(
        MARINE_URL,
        {
            "latitude": _round(lat),
            "longitude": _round(lon),
            "current": "wave_height,sea_surface_temperature",
        },
        label="Open-Meteo marine",
        ttl=CURRENT_TTL_SECONDS,
    )
    return data.get("current") or {}
