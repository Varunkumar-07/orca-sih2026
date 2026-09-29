"""Pytest coverage for backend/services/open_meteo.py — the shared
Open-Meteo client every live-weather path goes through.

Mocking strategy: httpx.AsyncClient.get patched at the class level (same
as test_weather_service.py), counting upstream calls. State (cache, quota
blocks) and retry backoff are reset/zeroed by conftest's autouse fixture.

Covers:
  - identical requests within the TTL hit upstream once (cache)
  - concurrent identical requests are coalesced onto one upstream call
  - concurrency is capped at _MAX_CONCURRENT
  - a daily-quota 429 is not retried and blocks further calls to that
    host (fail fast, zero upstream calls) until the block expires
  - a "Too many concurrent requests" 429 is transient: retried
  - failures are never cached; the error carries a user-safe message
    without the upstream URL
  - OPEN_METEO_API_KEY switches to the customer- host with the key

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from backend.services import open_meteo

_PARAMS = {"latitude": 13.08, "longitude": 80.27, "current": "wave_height"}


class _FakeResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json = json_data if json_data is not None else {"current": {"wave_height": 1.0}}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _install(monkeypatch, responses, *, delay: float = 0.0):
    """`responses`: list consumed in order (last one repeats) of either a
    _FakeResponse or an exception to raise."""
    calls: list[str] = []
    active = {"now": 0, "max": 0}

    async def fake_get(self, url, params=None, headers=None):
        calls.append(url)
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        try:
            if delay:
                await asyncio.sleep(delay)
            item = responses[min(len(calls) - 1, len(responses) - 1)]
            if isinstance(item, Exception):
                raise item
            return item
        finally:
            active["now"] -= 1

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    return calls, active


def _get(url=open_meteo.MARINE_URL, params=None):
    return open_meteo.get_json(url, params or _PARAMS, label="test", ttl=600)


def test_identical_requests_within_ttl_hit_upstream_once(monkeypatch):
    calls, _ = _install(monkeypatch, [_FakeResponse()])

    async def run():
        first = await _get()
        second = await _get()
        return first, second

    first, second = asyncio.run(run())
    assert first == second == {"current": {"wave_height": 1.0}}
    assert len(calls) == 1


def test_concurrent_identical_requests_are_coalesced(monkeypatch):
    calls, _ = _install(monkeypatch, [_FakeResponse()], delay=0.05)

    async def run():
        return await asyncio.gather(*(_get() for _ in range(5)))

    results = asyncio.run(run())
    assert all(r == results[0] for r in results)
    assert len(calls) == 1


def test_concurrency_is_capped(monkeypatch):
    calls, active = _install(monkeypatch, [_FakeResponse()], delay=0.02)

    async def run():
        await asyncio.gather(*(_get(params={**_PARAMS, "latitude": i}) for i in range(30)))

    asyncio.run(run())
    assert len(calls) == 30
    assert active["max"] <= open_meteo._MAX_CONCURRENT


def test_daily_quota_429_is_not_retried_and_blocks_the_host(monkeypatch):
    calls, _ = _install(
        monkeypatch,
        [_FakeResponse(status_code=429, text='{"error":true,"reason":"Daily API request limit exceeded. Please try again tomorrow."}')],
    )

    with pytest.raises(open_meteo.OpenMeteoError) as first:
        asyncio.run(_get())
    assert first.value.kind == "quota"
    assert len(calls) == 1  # not retried

    # A different request to the same host now fails fast, with no upstream call.
    with pytest.raises(open_meteo.OpenMeteoError) as second:
        asyncio.run(_get(params={**_PARAMS, "latitude": 1.0}))
    assert second.value.kind == "quota"
    assert len(calls) == 1

    # Other hosts are unaffected by this host's block.
    _install(monkeypatch, [_FakeResponse({"current": {"temperature_2m": 30}})])
    assert asyncio.run(_get(url=open_meteo.FORECAST_FALLBACK_URL)) == {"current": {"temperature_2m": 30}}


def test_quota_block_expires(monkeypatch):
    calls, _ = _install(
        monkeypatch,
        [_FakeResponse(status_code=429, text="Minutely API request limit exceeded."), _FakeResponse()],
    )
    with pytest.raises(open_meteo.OpenMeteoError):
        asyncio.run(_get())

    host = "marine-api.open-meteo.com"
    open_meteo._blocked_until[host] = 0.0  # simulate the block running out
    assert asyncio.run(_get()) == {"current": {"wave_height": 1.0}}
    assert len(calls) == 2


def test_too_many_concurrent_requests_429_is_retried(monkeypatch):
    calls, _ = _install(
        monkeypatch,
        [_FakeResponse(status_code=429, text='{"reason":"Too many concurrent requests"}'), _FakeResponse()],
    )
    assert asyncio.run(_get()) == {"current": {"wave_height": 1.0}}
    assert len(calls) == 2
    assert "marine-api.open-meteo.com" not in open_meteo._blocked_until


def test_transport_error_is_retried_then_raises_clean_error_and_is_not_cached(monkeypatch):
    calls, _ = _install(monkeypatch, [httpx.ConnectError("boom")])

    with pytest.raises(open_meteo.OpenMeteoError) as excinfo:
        asyncio.run(_get())
    assert len(calls) == open_meteo._MAX_ATTEMPTS
    assert excinfo.value.kind == "unavailable"
    assert "http" not in excinfo.value.user_message
    assert "ConnectError" in str(excinfo.value)  # the log-level detail keeps the type

    # Not cached: once upstream recovers, the next call goes through.
    _install(monkeypatch, [_FakeResponse()])
    assert asyncio.run(_get()) == {"current": {"wave_height": 1.0}}


def test_api_key_switches_to_customer_host(monkeypatch):
    seen = {}

    async def fake_get(self, url, params=None, headers=None):
        seen["url"], seen["params"] = url, params
        return _FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setenv("OPEN_METEO_API_KEY", "test-key")

    asyncio.run(_get())
    assert seen["url"] == "https://customer-marine-api.open-meteo.com/v1/marine"
    assert seen["params"]["apikey"] == "test-key"


def test_fetch_current_forecast_falls_back_to_historical_host(monkeypatch):
    urls: list[str] = []

    async def fake_get(self, url, params=None, headers=None):
        urls.append(url)
        if url == open_meteo.FORECAST_URL:
            return _FakeResponse(status_code=429, text="Daily API request limit exceeded.")
        return _FakeResponse({"current": {"wind_speed_10m": 5.0}})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = asyncio.run(open_meteo.fetch_current_forecast(13.0827, 80.2707))
    assert result == {"current": {"wind_speed_10m": 5.0}}
    assert urls == [open_meteo.FORECAST_URL, open_meteo.FORECAST_FALLBACK_URL]
