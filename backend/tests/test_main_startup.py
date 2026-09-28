"""Integration coverage for backend/main.py's Phase 4 startup pre-warm
(Chat-PFZ Live-Data Integration): the shared live PFZ zone cache
(pfz_service.get_cached_zones) is kicked off in the background via
asyncio.create_task in main._lifespan, NOT awaited before the app starts
serving requests.

This was previously verified only by an ad-hoc manual script in-session
(TestClient + wall-clock timing), never by an actual pytest test — this
fills that gap so the "pre-warm doesn't block startup" claim is checked
on every test run, not just once by hand.

Drives the real FastAPI app's lifespan via Starlette's TestClient used as
a context manager (`with TestClient(app) as client:`), which is what
actually triggers ASGI lifespan startup/shutdown events — a bare
`TestClient(app)` (as test_main_geolocation.py uses, for endpoints that
don't care about lifespan) does not reliably do this.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
import threading
import time

from fastapi.testclient import TestClient

import backend.main as main_module

app = main_module.app

# Generous upper bound for the artificial pre-warm delay used below — real
# startup should return orders of magnitude faster than this; used only to
# prove the app doesn't wait for it, not as a tuned/tight timing assertion.
_SLOW_PREWARM_SECONDS = 0.3


def test_lifespan_startup_does_not_block_on_prewarm(monkeypatch):
    """The literal Phase 4 requirement: asyncio.create_task fires the
    pre-warm without awaiting it, so entering the app's lifespan (what
    Uvicorn does before it starts accepting connections) must return
    near-instantly even when the underlying fetch is slow — never wait
    for it to finish."""
    prewarm_started = threading.Event()
    prewarm_finished = threading.Event()

    async def slow_get_cached_zones():
        prewarm_started.set()
        await asyncio.sleep(_SLOW_PREWARM_SECONDS)
        prewarm_finished.set()
        return {"zones": [], "generated_at": "irrelevant"}

    monkeypatch.setattr(main_module, "get_cached_zones", slow_get_cached_zones)

    t0 = time.monotonic()
    with TestClient(app) as client:
        elapsed = time.monotonic() - t0
        assert elapsed < _SLOW_PREWARM_SECONDS, (
            f"lifespan startup took {elapsed:.3f}s — should return immediately, "
            "not wait for the background pre-warm fetch"
        )
        # Confirmed not just "fast" but genuinely still in flight — the
        # server is already serving while pre-warm continues.
        assert not prewarm_finished.is_set()

        resp = client.get("/health")
        assert resp.status_code == 200

        # Let the background task actually run to completion on its own
        # event loop (TestClient runs the app in a background thread).
        deadline = time.monotonic() + 5.0
        while not prewarm_finished.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)

    assert prewarm_started.is_set(), "pre-warm task never started in the background"
    assert prewarm_finished.is_set(), "pre-warm task never completed"


def test_prewarm_failure_is_logged_only_and_does_not_crash_the_app(monkeypatch, caplog):
    """Task requirement: pre-warm failing (no credentials, network issue,
    etc.) must degrade silently — logged, not raised — and never take the
    app down or block any endpoint's own per-request fallback."""

    async def broken_get_cached_zones():
        raise RuntimeError("simulated Copernicus/cache outage")

    monkeypatch.setattr(main_module, "get_cached_zones", broken_get_cached_zones)

    with caplog.at_level("WARNING", logger="orca.pfz"), TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        # Give the background task a moment to run (and fail) on its own
        # event loop before the context exits.
        time.sleep(0.2)

    assert any("pre-warm failed" in record.message for record in caplog.records)


def test_prewarm_actually_calls_get_cached_zones(monkeypatch):
    """Sanity: the background task really is get_cached_zones() (or an
    equivalent full-catalog fetch) — not a no-op that merely looks
    non-blocking."""
    called = threading.Event()

    async def spy_get_cached_zones():
        called.set()
        return {"zones": [], "generated_at": "irrelevant"}

    monkeypatch.setattr(main_module, "get_cached_zones", spy_get_cached_zones)

    with TestClient(app) as client:
        client.get("/health")
        deadline = time.monotonic() + 5.0
        while not called.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)

    assert called.is_set()
