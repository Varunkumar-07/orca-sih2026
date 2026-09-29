"""Pytest coverage for backend/services/copernicus_fetch.py — the shared
runner for request-path Copernicus reads (Weather/chat chlorophyll,
Analytics series), plus the chlorophyll caller's logging.

Mocking strategy: plain blocking functions stand in for the Copernicus
read (the runner only ever sees a callable). State is reset by conftest's
autouse fixture.

Covers:
  - a successful result is cached for its TTL (one read, two callers)
  - a read that outlives the caller's timeout keeps running and its
    result is served to the next caller (no second read)
  - a timeout is never retried; a real error is retried once
  - failure/timeout logs name the exception type (previously a bare,
    empty asyncio.TimeoutError message)

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from backend.agents.reasoning import marine_data_agent as mda
from backend.schemas.contracts import GeoPoint
from backend.services import copernicus_fetch


def _counting(fn):
    calls = {"n": 0}

    def wrapper(*args):
        calls["n"] += 1
        return fn(*args)

    return wrapper, calls


def test_success_is_cached_for_ttl():
    fn, calls = _counting(lambda x: x * 2)

    async def run():
        first = await copernicus_fetch.fetch(("k",), fn, (21,), wait_timeout=1, ttl=60, label="t")
        second = await copernicus_fetch.fetch(("k",), fn, (21,), wait_timeout=1, ttl=60, label="t")
        return first, second

    assert asyncio.run(run()) == (42, 42)
    assert calls["n"] == 1


def test_timed_out_read_keeps_running_and_serves_the_next_caller(caplog):
    release = threading.Event()

    def slow(value):
        release.wait(5)
        return value

    fn, calls = _counting(slow)

    async def run():
        with pytest.raises(TimeoutError):
            await copernicus_fetch.fetch(("slow",), fn, (7,), wait_timeout=0.05, ttl=60, label="Slow read")
        release.set()
        # Same event loop: the background job finishes and caches its result.
        return await copernicus_fetch.fetch(("slow",), fn, (7,), wait_timeout=2, ttl=60, label="Slow read")

    with caplog.at_level(logging.WARNING, logger="orca.copernicus"):
        assert asyncio.run(run()) == 7
    assert calls["n"] == 1  # the timed-out read was reused, not re-run or retried
    assert any("TimeoutError (no result within 0.05s)" in r.getMessage() for r in caplog.records)


def test_error_is_retried_once_then_logged_with_its_type(caplog):
    def broken(_):
        raise ConnectionError("auth server hiccup")

    fn, calls = _counting(broken)

    async def run():
        await copernicus_fetch.fetch(("err",), fn, (1,), wait_timeout=2, ttl=60, label="Broken read")

    with caplog.at_level(logging.WARNING, logger="orca.copernicus"), pytest.raises(ConnectionError):
        asyncio.run(run())
    assert calls["n"] == copernicus_fetch._MAX_ATTEMPTS
    assert any(
        "Broken read failed after 2 attempt(s): ConnectionError: auth server hiccup" in r.getMessage()
        for r in caplog.records
    )


def test_failure_is_not_cached():
    outcomes = iter([RuntimeError("x"), RuntimeError("x"), 5])

    def flaky(_):
        item = next(outcomes)
        if isinstance(item, Exception):
            raise item
        return item

    async def run():
        with pytest.raises(RuntimeError):
            await copernicus_fetch.fetch(("f",), flaky, (0,), wait_timeout=2, ttl=60, label="t")
        return await copernicus_fetch.fetch(("f",), flaky, (0,), wait_timeout=2, ttl=60, label="t")

    assert asyncio.run(run()) == 5


def test_chlorophyll_timeout_is_logged_with_exception_type(monkeypatch, caplog):
    """Regression: production logged "Chlorophyll fetch failed ... after 2
    attempt(s): " with an empty message — an asyncio.TimeoutError."""
    monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "test-user")
    monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "test-pass")
    monkeypatch.setattr(mda, "_CHL_TIMEOUT", 0.05)
    monkeypatch.setattr(mda, "_fetch_chlorophyll_sync", lambda *_a: time.sleep(0.3) or 0.2)

    with caplog.at_level(logging.WARNING, logger="orca.copernicus"):
        assert asyncio.run(mda._fetch_live_chlorophyll(GeoPoint(lat=13.08, lon=80.35))) is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("Chlorophyll fetch for (13.08,80.35)" in m and "TimeoutError" in m for m in messages), messages
