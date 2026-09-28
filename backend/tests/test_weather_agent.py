"""Pytest coverage for backend/agents/reasoning/weather_agent.py's live
Open-Meteo integration (added this session — previously untested).

Mocking strategy: httpx.AsyncClient.get is patched at the class level with a
fake that dispatches on URL (forecast vs marine endpoint), the same "patch
the network boundary, exercise the real orchestration logic" approach
test_language_agent.py already uses for _post_json — no real HTTP traffic,
no live credentials needed. Tests are plain sync functions that drive the
async code via asyncio.run() rather than pulling in pytest-asyncio as a new
dependency just for this.

Covers:
  - _fetch_live_weather: normal case, wave-height-missing (coastal grid gap
    -> partial, not a failure), wind-missing (raises -> caller's error path),
    cyclone_alert threshold (>=62 km/h), lightning_alert (thunderstorm codes)
  - run_weather_agent: status="partial" when wave is unavailable, status=
    "error" when query_location is None, and the intent-classification
    decoupling fix — a broken/failing _classify_intent must never discard
    otherwise-good live weather data (this is the exact bug fixed this
    session: Groq occasionally returns invalid JSON for this call, which
    used to wipe out perfectly good weather data along with it)

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from backend.agents.reasoning import weather_agent as wa
from backend.schemas.contracts import GeoPoint

_LOCATION = GeoPoint(lat=13.08, lon=80.27)


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _install_fake_get(monkeypatch, *, wind, weather_code, wave):
    """Dispatches on which endpoint is being hit, matching _fetch_live_weather's
    two concurrent calls (forecast API for wind/weather_code, marine API for
    wave_height)."""

    async def fake_get(self, url, params=None, headers=None):
        if url == wa._FORECAST_URL:
            return _FakeResponse(
                {"current": {"wind_speed_10m": wind, "weather_code": weather_code}}
            )
        if url == wa._MARINE_URL:
            return _FakeResponse({"current": {"wave_height": wave}})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


# ---------------------------------------------------------------------------
# _fetch_live_weather
# ---------------------------------------------------------------------------


def test_fetch_live_weather_normal_case(monkeypatch):
    _install_fake_get(monkeypatch, wind=18.5, weather_code=1, wave=1.2)

    snapshot = asyncio.run(wa._fetch_live_weather(_LOCATION))

    assert snapshot["wind_kmh"] == 18.5
    assert snapshot["wave_height_m"] == 1.2
    assert snapshot["cyclone_alert"] is False
    assert snapshot["lightning_alert"] is False
    # No free/keyless tide source exists — must be honestly None, never faked.
    assert snapshot["tide_info"] is None


def test_fetch_live_weather_wave_missing_is_not_a_failure(monkeypatch):
    """A coastal marine-grid gap (confirmed live for Goa) — wind/lightning
    are still real, only wave_height_m degrades to None."""
    _install_fake_get(monkeypatch, wind=9.6, weather_code=0, wave=None)

    snapshot = asyncio.run(wa._fetch_live_weather(_LOCATION))

    assert snapshot["wind_kmh"] == 9.6
    assert snapshot["wave_height_m"] is None


def test_fetch_live_weather_wind_missing_raises(monkeypatch):
    """Unlike a missing wave reading, a missing wind reading means the
    forecast API itself failed for this location — that's a real failure,
    not a coverage gap, so this must raise (caller turns it into
    status="error")."""
    _install_fake_get(monkeypatch, wind=None, weather_code=0, wave=1.0)

    with pytest.raises(ValueError):
        asyncio.run(wa._fetch_live_weather(_LOCATION))


def test_cyclone_alert_threshold(monkeypatch):
    _install_fake_get(monkeypatch, wind=62.0, weather_code=0, wave=1.0)
    snapshot = asyncio.run(wa._fetch_live_weather(_LOCATION))
    assert snapshot["cyclone_alert"] is True

    _install_fake_get(monkeypatch, wind=61.9, weather_code=0, wave=1.0)
    snapshot = asyncio.run(wa._fetch_live_weather(_LOCATION))
    assert snapshot["cyclone_alert"] is False


def test_lightning_alert_from_thunderstorm_weather_code(monkeypatch):
    for code in (95, 96, 99):
        _install_fake_get(monkeypatch, wind=10.0, weather_code=code, wave=1.0)
        snapshot = asyncio.run(wa._fetch_live_weather(_LOCATION))
        assert snapshot["lightning_alert"] is True, f"code {code} should mean lightning_alert"

    _install_fake_get(monkeypatch, wind=10.0, weather_code=3, wave=1.0)
    snapshot = asyncio.run(wa._fetch_live_weather(_LOCATION))
    assert snapshot["lightning_alert"] is False


# ---------------------------------------------------------------------------
# run_weather_agent
# ---------------------------------------------------------------------------


def test_run_weather_agent_no_location_is_error():
    trace = []
    result = asyncio.run(wa.run_weather_agent("is it safe?", None, trace))
    assert result.status == "error"
    assert trace[-1].agent_name == "weather_agent"


def test_run_weather_agent_partial_status_when_wave_unavailable(monkeypatch):
    _install_fake_get(monkeypatch, wind=9.6, weather_code=0, wave=None)
    monkeypatch.setattr(wa, "_classify_intent", _async_return("general"))

    trace = []
    result = asyncio.run(wa.run_weather_agent("is it safe near Goa?", _LOCATION, trace))

    assert result.status == "partial"
    assert result.wind_kmh == 9.6
    assert result.wave_height_m is None


def test_intent_classification_failure_does_not_discard_real_weather_data(monkeypatch):
    """The exact bug fixed this session: Groq occasionally returns invalid
    JSON for the (purely cosmetic) intent-classification call. That must
    degrade to intent="general" in the trace, never take the whole weather
    result down with it — cyclone_alert no longer depends on intent at all
    (it's derived from real wind now), so there's no reason for a
    classification hiccup to matter here."""
    _install_fake_get(monkeypatch, wind=18.5, weather_code=1, wave=1.2)

    async def broken_classify(_query_text):
        raise RuntimeError("simulated Groq JSON validation failure")

    monkeypatch.setattr(wa, "_classify_intent", broken_classify)

    trace = []
    result = asyncio.run(wa.run_weather_agent("is it safe to fish?", _LOCATION, trace))

    assert result.status == "ok"
    assert result.wind_kmh == 18.5
    assert result.wave_height_m == 1.2
    assert "intent=general" in trace[-1].output_summary
