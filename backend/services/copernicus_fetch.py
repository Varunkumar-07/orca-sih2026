"""
Shared runner for the small, request-path Copernicus Marine reads — the
Weather page / chat chlorophyll point reading and the Analytics page's
chlorophyll / salinity / dissolved-oxygen series. (The PFZ zone refresh
has its own background path in pfz_service.py and doesn't use this.)

Why this exists: every one of those reads is a copernicusmarine
open_dataset() call, which costs ~1s of pure CPU plus ~8s of serial
network round-trips (auth check, catalogue, Zarr metadata) even on a fast
machine. On Render's free tier (0.1 CPU) that measured ~30s for a single
read, so the old wait_for budgets (15s for chlorophyll, 20s for
analytics) timed out every time in production: the blank "after 2
attempt(s): " log lines were asyncio.TimeoutError, whose str() is empty.
Worse, a timed-out read's worker thread keeps running (threads can't be
cancelled), so the immediate retry ran a second copy alongside the first
on the same starved CPU — making the retry slower still — and both
results were thrown away.

What this runner does instead:
- One background job per distinct read (identical concurrent requests
  share it). A caller waits up to its own timeout; if that passes, the
  job keeps going and its result is cached, so the next request (a page
  refresh, the next chat turn) gets it instantly instead of starting over.
- Successful results are cached for a per-call TTL (these are daily
  products; re-opening the dataset for every request is pure waste).
- Retries only after a real error (e.g. a transient auth hiccup), never
  after a timeout.
- A hard cap per attempt so a genuinely hung upstream call eventually
  frees its job slot and a later request can try again.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from backend.error_utils import describe_exception

logger = logging.getLogger("orca.copernicus")

_MAX_ATTEMPTS = 2
_HARD_TIMEOUT_SECONDS = 180.0

_cache: dict[tuple, tuple[float, Any]] = {}
_jobs: dict[tuple, asyncio.Task] = {}


def reset_state() -> None:
    """Drop cached results (tests; never needed in prod)."""
    _cache.clear()
    _jobs.clear()


async def _run_job(key: tuple, fn: Callable[..., Any], args: tuple, label: str) -> Any:
    last_exc: BaseException | None = None
    attempt = 0
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            value = await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=_HARD_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            last_exc = exc
            break  # a hung upstream isn't un-hung by piling a second call on top
        except Exception as exc:  # noqa: BLE001 - any read failure is retried, then re-raised
            last_exc = exc
        else:
            _cache[key] = (time.monotonic(), value)
            return value
    logger.warning(
        "%s failed after %d attempt(s): %s",
        label, attempt, describe_exception(last_exc, timeout=_HARD_TIMEOUT_SECONDS),
    )
    raise last_exc  # type: ignore[misc]


def _forget_job(key: tuple, task: asyncio.Task) -> None:
    if _jobs.get(key) is task:
        del _jobs[key]
    # Retrieve the outcome so a job whose waiter already gave up doesn't
    # log "Task exception was never retrieved" (its failure is logged above).
    if not task.cancelled():
        task.exception()


async def fetch(
    key: tuple, fn: Callable[..., Any], args: tuple, *, wait_timeout: float, ttl: float, label: str
) -> Any:
    """Run blocking `fn(*args)` (a Copernicus read) via the shared job for
    `key`, waiting at most `wait_timeout` seconds. Returns its value, or
    raises: TimeoutError if the job is still running (it continues, and
    its result is cached for the next caller), otherwise the job's own
    exception after its retries."""
    hit = _cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < ttl:
        return hit[1]

    loop = asyncio.get_running_loop()
    task = _jobs.get(key)
    if task is None or task.done() or task.get_loop() is not loop:
        task = loop.create_task(_run_job(key, fn, args, label))
        _jobs[key] = task
        task.add_done_callback(lambda t: _forget_job(key, t))

    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=wait_timeout)
    except TimeoutError:
        logger.warning(
            "%s: %s — still running in the background; its result will be cached for the next request",
            label, describe_exception(TimeoutError(), timeout=wait_timeout),
        )
        raise
