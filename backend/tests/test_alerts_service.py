"""Pytest coverage for backend/services/alerts_service.py (previously
untested — added as part of the codebase audit's P1 coverage gaps).

Mocking strategy: _fetch_forecast (imported into this module's own
namespace from weather_service.py) is patched directly — same "patch our
own function, not the network" principle as everywhere else in this suite.
Tests are plain sync functions driving async code via asyncio.run().

Covers:
  - _check_zone: no coordinates -> skipped (no alert, no fetch); forecast
    fetch failing -> skipped, not an error; wind at/above the cyclone
    threshold -> cyclone alert; a thunderstorm weather code -> lightning
    alert; both conditions on the same zone -> two distinct alerts
  - get_active_alerts: combines alerts across multiple zones; the
    short-TTL in-memory cache actually serves cached results within the
    window and recomputes once it's stale

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

import pytest

from backend.services import alerts_service as svc


@pytest.fixture(autouse=True)
def _reset_cache():
    svc._cache = None
    svc._cache_at = 0.0
    yield
    svc._cache = None
    svc._cache_at = 0.0


def _pfz_zone(zone_id="KOCHI-PFZ-001", lat=10.0, lon=76.3, near="Kochi") -> dict:
    return {"id": zone_id, "name": "PFZ-001", "type": "pfz", "near": near, "coordinates": {"lat": lat, "lon": lon}}


def _forecast(wind_kmh=10.0, weather_code=0) -> dict:
    return {"current": {"wind_speed_10m": wind_kmh, "weather_code": weather_code}}


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


# ---------------------------------------------------------------------------
# _check_zone
# ---------------------------------------------------------------------------


def test_zone_without_coordinates_is_skipped_no_fetch(monkeypatch):
    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not fetch a forecast for a zone with no coordinates")

    monkeypatch.setattr(svc, "_fetch_forecast", fail_if_called)

    result = asyncio.run(svc._check_zone({"id": "restricted-x", "name": "X", "type": "restricted"}))
    assert result == []


def test_forecast_fetch_failure_is_skipped_not_an_error(monkeypatch):
    monkeypatch.setattr(svc, "_fetch_forecast", _async_return(None))

    result = asyncio.run(svc._check_zone(_pfz_zone()))
    assert result == []


def test_cyclone_alert_at_threshold(monkeypatch):
    monkeypatch.setattr(svc, "_fetch_forecast", _async_return(_forecast(wind_kmh=svc._CYCLONE_WIND_THRESHOLD_KMH)))

    result = asyncio.run(svc._check_zone(_pfz_zone()))
    assert len(result) == 1
    assert result[0]["alert_type"] == "cyclone"
    assert result[0]["zone_id"] == "KOCHI-PFZ-001"
    assert result[0]["near"] == "Kochi"


def test_no_cyclone_alert_just_below_threshold(monkeypatch):
    monkeypatch.setattr(svc, "_fetch_forecast", _async_return(_forecast(wind_kmh=svc._CYCLONE_WIND_THRESHOLD_KMH - 0.1)))

    result = asyncio.run(svc._check_zone(_pfz_zone()))
    assert result == []


def test_lightning_alert_from_thunderstorm_code(monkeypatch):
    for code in svc._THUNDERSTORM_CODES:
        monkeypatch.setattr(svc, "_fetch_forecast", _async_return(_forecast(wind_kmh=5.0, weather_code=code)))
        result = asyncio.run(svc._check_zone(_pfz_zone()))
        assert len(result) == 1
        assert result[0]["alert_type"] == "lightning"


def test_no_lightning_alert_for_non_thunderstorm_code(monkeypatch):
    monkeypatch.setattr(svc, "_fetch_forecast", _async_return(_forecast(wind_kmh=5.0, weather_code=1)))
    result = asyncio.run(svc._check_zone(_pfz_zone()))
    assert result == []


def test_both_cyclone_and_lightning_produce_two_alerts(monkeypatch):
    monkeypatch.setattr(
        svc, "_fetch_forecast", _async_return(_forecast(wind_kmh=svc._CYCLONE_WIND_THRESHOLD_KMH + 5, weather_code=99))
    )
    result = asyncio.run(svc._check_zone(_pfz_zone()))
    assert {a["alert_type"] for a in result} == {"cyclone", "lightning"}


# ---------------------------------------------------------------------------
# get_active_alerts
# ---------------------------------------------------------------------------


def test_combines_alerts_across_multiple_zones(monkeypatch):
    calm = _pfz_zone(zone_id="CALM-001", lat=1.0, lon=1.0)
    stormy = _pfz_zone(zone_id="STORMY-001", lat=2.0, lon=2.0)

    async def fake_fetch(lat, lon):
        if lat == 2.0:
            return _forecast(wind_kmh=svc._CYCLONE_WIND_THRESHOLD_KMH + 1)
        return _forecast(wind_kmh=5.0)

    monkeypatch.setattr(svc, "_fetch_forecast", fake_fetch)

    result = asyncio.run(svc.get_active_alerts([calm, stormy]))

    assert result["checked_zones"] == 2
    assert len(result["alerts"]) == 1
    assert result["alerts"][0]["zone_id"] == "STORMY-001"
    assert "generated_at" in result


def test_cache_hit_within_ttl_does_not_refetch(monkeypatch):
    call_count = {"n": 0}

    async def fake_fetch(lat, lon):
        call_count["n"] += 1
        return _forecast(wind_kmh=5.0)

    monkeypatch.setattr(svc, "_fetch_forecast", fake_fetch)

    zones = [_pfz_zone()]
    first = asyncio.run(svc.get_active_alerts(zones))
    second = asyncio.run(svc.get_active_alerts(zones))

    assert call_count["n"] == 1  # second call served from cache
    assert first is second


def test_cache_miss_after_ttl_expires_recomputes(monkeypatch):
    call_count = {"n": 0}

    async def fake_fetch(lat, lon):
        call_count["n"] += 1
        return _forecast(wind_kmh=5.0)

    monkeypatch.setattr(svc, "_fetch_forecast", fake_fetch)

    zones = [_pfz_zone()]
    asyncio.run(svc.get_active_alerts(zones))
    svc._cache_at -= svc._CACHE_TTL_SECONDS + 1  # force staleness without a real sleep

    asyncio.run(svc.get_active_alerts(zones))

    assert call_count["n"] == 2
