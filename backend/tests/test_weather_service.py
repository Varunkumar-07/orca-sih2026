"""Pytest coverage for backend/services/weather_service.py (previously
untested — added as part of the codebase audit's P1 coverage gaps).

Mocking strategy: httpx.AsyncClient.get is patched at the class level,
dispatching on URL (forecast vs marine endpoint) — same approach
test_weather_agent.py already uses for the chat-flow equivalent of this
same live-fetch logic. _fetch_live_chlorophyll (imported from
marine_data_agent.py) is patched directly rather than faking Copernicus
credentials. _compute_astronomical (imported from analytics_service.py) is
left real — it's pure, deterministic, no-network computation, already
covered on its own terms in test_analytics_service.py.

Covers:
  - get_current_weather: normal case (status="ok", every field present);
    forecast endpoint failing degrades only the forecast-derived fields,
    not marine/chlorophyll (status="partial"); marine endpoint failing
    likewise; chlorophyll rounding to 3 decimals; cyclone_alert/
    lightning_alert derived from the same IMD threshold/WMO codes
    weather_agent.py uses; status="error" when nothing came back at all

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from backend.services import weather_service as svc

_LAT, _LON = 13.08, 80.27


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json = json_data
        self.status_code = status_code
        self.text = "" if status_code == 200 else str(json_data)

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _install_fake_get(
    monkeypatch,
    *,
    forecast_json=None,
    marine_json=None,
    forecast_fails=False,
    marine_fails=False,
    forecast_status=200,
    fallback_json=None,
    fallback_fails=False,
):
    call_counts = {"forecast": 0, "marine": 0, "fallback": 0}

    async def fake_get(self, url, params=None, headers=None):
        if url == svc._FORECAST_URL:
            call_counts["forecast"] += 1
            if forecast_status == 429:
                return _FakeResponse({"error": True, "reason": "Daily API request limit exceeded."}, status_code=429)
            if forecast_fails:
                raise httpx.ConnectError("simulated network failure")
            return _FakeResponse(
                forecast_json
                or {
                    "current": {"temperature_2m": 29.5, "wind_speed_10m": 15.0, "weather_code": 1},
                    "daily": {"precipitation_sum": [2.5], "wind_speed_10m_max": [22.0]},
                }
            )
        if url == svc._FORECAST_FALLBACK_URL:
            call_counts["fallback"] += 1
            if fallback_fails:
                raise httpx.ConnectError("simulated network failure")
            return _FakeResponse(
                fallback_json
                or {
                    "current": {"temperature_2m": 27.5, "wind_speed_10m": 5.5, "weather_code": 0},
                    "daily": {"precipitation_sum": [2.0], "wind_speed_10m_max": [10.3]},
                }
            )
        if url == svc._MARINE_URL:
            call_counts["marine"] += 1
            if marine_fails:
                raise httpx.ConnectError("simulated network failure")
            return _FakeResponse(marine_json or {"current": {"wave_height": 1.1, "sea_surface_temperature": 28.7}})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    return call_counts


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


def test_normal_case_all_fields_present(monkeypatch):
    _install_fake_get(monkeypatch)
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(0.6354018449783325))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))

    assert result["status"] == "ok"
    assert result["air_temperature_c"] == 29.5
    assert result["wind_kmh"] == 15.0
    assert result["wave_height_m"] == 1.1
    assert result["sst_celsius"] == 28.7
    assert result["chlorophyll_mg_m3"] == 0.635  # rounded to 3 decimals
    assert result["precipitation_mm"] == 2.5
    assert result["wind_max_kmh"] == 22.0
    assert result["lat"] == _LAT and result["lon"] == _LON
    assert result["sunrise_hour_ist"] is not None
    assert result["sunset_hour_ist"] is not None


def test_forecast_failure_falls_back_and_recovers(monkeypatch):
    """A genuine transient failure (connection error) is retried once
    against the primary host; if that's still down, the Historical
    Forecast API fallback (separate quota, same shape) is tried next and
    recovers the fields instead of leaving them "unavailable"."""
    call_counts = _install_fake_get(monkeypatch, forecast_fails=True)
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(0.5))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))

    assert result["status"] == "ok"
    assert result["air_temperature_c"] == 27.5  # from the fallback host
    assert result["wind_kmh"] == 5.5
    assert result["wave_height_m"] == 1.1
    assert result["sst_celsius"] == 28.7
    assert call_counts["forecast"] == 2  # primary retried once before falling back
    assert call_counts["fallback"] == 1


def test_forecast_429_skips_retry_and_falls_back_immediately(monkeypatch):
    """Regression test for a real production incident: Open-Meteo's free
    tier returns 429 for both short-lived rate-limiting and a hard daily
    quota ("Daily API request limit exceeded" — the actual body observed).
    Retrying the latter immediately only burns latency for a guaranteed
    second 429, so a 429 must skip the retry and go straight to the
    fallback host instead."""
    call_counts = _install_fake_get(monkeypatch, forecast_status=429)
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(0.5))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))

    assert result["status"] == "ok"
    assert result["air_temperature_c"] == 27.5
    assert result["wind_kmh"] == 5.5
    assert call_counts["forecast"] == 1  # not retried
    assert call_counts["fallback"] == 1


def test_forecast_and_fallback_both_down_degrades_to_partial(monkeypatch):
    """If the fallback host is also unavailable, fields degrade to None
    exactly as before this fallback existed — marine fields are untouched
    either way."""
    call_counts = _install_fake_get(monkeypatch, forecast_status=429, fallback_fails=True)
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(0.5))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))

    assert result["status"] == "partial"
    assert result["air_temperature_c"] is None
    assert result["wind_kmh"] is None
    assert result["wave_height_m"] == 1.1
    assert result["sst_celsius"] == 28.7
    assert call_counts["fallback"] == 2  # the fallback call itself is retried once too


def test_marine_failure_degrades_to_partial_without_losing_forecast_fields(monkeypatch):
    _install_fake_get(monkeypatch, marine_fails=True)
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(0.5))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))

    assert result["status"] == "partial"
    assert result["wave_height_m"] is None
    assert result["sst_celsius"] is None
    assert result["wind_kmh"] == 15.0
    assert result["air_temperature_c"] == 29.5


def test_everything_missing_is_status_error(monkeypatch):
    _install_fake_get(monkeypatch, forecast_fails=True, marine_fails=True, fallback_fails=True)
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(None))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))

    assert result["status"] == "error"
    assert all(
        result[k] is None
        for k in ("air_temperature_c", "wind_kmh", "wave_height_m", "sst_celsius", "chlorophyll_mg_m3")
    )


@pytest.mark.parametrize("wind,expected", [(svc._CYCLONE_WIND_THRESHOLD_KMH, True), (svc._CYCLONE_WIND_THRESHOLD_KMH - 0.1, False)])
def test_cyclone_alert_threshold(monkeypatch, wind, expected):
    _install_fake_get(monkeypatch, forecast_json={"current": {"temperature_2m": 29.0, "wind_speed_10m": wind, "weather_code": 0}, "daily": {}})
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(None))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))
    assert result["cyclone_alert"] is expected


@pytest.mark.parametrize("code,expected", [(95, True), (96, True), (99, True), (1, False)])
def test_lightning_alert_from_thunderstorm_code(monkeypatch, code, expected):
    _install_fake_get(monkeypatch, forecast_json={"current": {"temperature_2m": 29.0, "wind_speed_10m": 10.0, "weather_code": code}, "daily": {}})
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(None))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))
    assert result["lightning_alert"] is expected


def test_chlorophyll_none_is_not_rounded_and_does_not_crash(monkeypatch):
    _install_fake_get(monkeypatch)
    monkeypatch.setattr(svc, "_fetch_live_chlorophyll", _async_return(None))

    result = asyncio.run(svc.get_current_weather(_LAT, _LON))
    assert result["chlorophyll_mg_m3"] is None
    assert result["status"] == "partial"  # 4 of 5 core fields present
