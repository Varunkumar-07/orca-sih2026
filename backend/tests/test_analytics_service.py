"""Pytest coverage for backend/services/analytics_service.py (previously
untested — added as part of the codebase audit's P1 coverage gaps).

Mocking strategy: the three network-fetch functions (_fetch_marine_daily,
_fetch_marine_current_hourly, _fetch_archive_daily) are patched directly
(same "patch our own function, not the network" principle as everywhere
else in this suite); _fetch_copernicus_series_sync (the blocking Copernicus
call run via asyncio.to_thread) is patched directly too, same approach as
test_pfz_service.py's Copernicus tests. The pure computation functions
(_status_for, _aggregate_current_daily, sunrise/sunset, moon phase) need no
mocking at all — no I/O, deterministic.

Covers:
  - _status_for: empty/all-missing -> "error", some-missing -> "partial",
    all-present -> "ok"
  - _aggregate_current_daily: picks each day's peak-velocity reading (and
    its paired direction), not an average
  - _sunrise_sunset_utc_hours / get_historical_analytics's sunrise/sunset
    output: a real solar-position sanity check against Chennai's known
    (roughly) sunrise/sunset time, not just "returns a number"
  - _moon_phase: known new-moon reference date -> fraction ~0; wraps
    correctly across a synodic month
  - get_historical_analytics: only fetches the sources a variable
    selection actually needs (e.g. requesting only moon_phase never calls
    the network); invalid variables are dropped, empty selection falls
    back to every VALID_VARIABLES; identical calls are served from the
    in-process cache (no re-fetch); Copernicus variables degrade to an
    empty series (not a crash) when credentials are unset
  - build_export_notes: known-variable caveats included; partial/error
    status produces its own dynamic note; a fully-clean series produces
    the single "full coverage" fallback note, not silence

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
from datetime import date

import pytest

from backend.services import analytics_service as svc

_CHENNAI_LAT, _CHENNAI_LON = 13.0827, 80.2707


@pytest.fixture(autouse=True)
def _reset_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def _async_return(value):
    async def _inner(*_a, **_k):
        return value

    return _inner


# ---------------------------------------------------------------------------
# _status_for
# ---------------------------------------------------------------------------


def test_status_for_empty_list_is_error():
    assert svc._status_for([]) == "error"


def test_status_for_all_none_is_error():
    assert svc._status_for([None, None]) == "error"


def test_status_for_some_none_is_partial():
    assert svc._status_for([1.0, None, 2.0]) == "partial"


def test_status_for_all_present_is_ok():
    assert svc._status_for([1.0, 2.0, 3.0]) == "ok"


# ---------------------------------------------------------------------------
# _aggregate_current_daily
# ---------------------------------------------------------------------------


def test_aggregate_current_daily_picks_peak_velocity_per_day():
    hourly = {
        "time": ["2026-01-01T00:00", "2026-01-01T06:00", "2026-01-01T12:00", "2026-01-02T00:00"],
        "ocean_current_velocity": [1.0, 3.5, 2.0, 0.8],
        "ocean_current_direction": [10.0, 90.0, 180.0, 270.0],
    }
    days, velocities, directions = svc._aggregate_current_daily(hourly)

    assert days == ["2026-01-01", "2026-01-02"]
    assert velocities == [3.5, 0.8]
    assert directions == [90.0, 270.0]  # direction paired with the peak-velocity hour, not averaged


def test_aggregate_current_daily_skips_none_velocity_readings():
    hourly = {
        "time": ["2026-01-01T00:00", "2026-01-01T06:00"],
        "ocean_current_velocity": [None, 1.2],
        "ocean_current_direction": [0.0, 45.0],
    }
    days, velocities, directions = svc._aggregate_current_daily(hourly)
    assert days == ["2026-01-01"]
    assert velocities == [1.2]
    assert directions == [45.0]


# ---------------------------------------------------------------------------
# Astronomical — sunrise/sunset (real solar-position formula, no mocking)
# ---------------------------------------------------------------------------


def test_chennai_sunrise_sunset_are_plausible_ist_hours():
    """Sanity check against Chennai's real, roughly-known sunrise/sunset —
    not an exact-second match (that would be fragile), just confirms the
    formula lands in the right ballpark (sunrise ~5:30-6:30 IST, sunset
    ~17:30-18:30 IST) rather than returning nonsense."""
    sr_utc, ss_utc = svc._sunrise_sunset_utc_hours(_CHENNAI_LAT, _CHENNAI_LON, date(2026, 3, 20))
    assert sr_utc is not None and ss_utc is not None

    sunrise_ist = (sr_utc + svc._IST_OFFSET_HOURS) % 24
    sunset_ist = (ss_utc + svc._IST_OFFSET_HOURS) % 24
    assert 5.0 <= sunrise_ist <= 7.0
    assert 17.0 <= sunset_ist <= 19.0
    assert sunset_ist > sunrise_ist


def test_compute_astronomical_only_computes_what_was_asked():
    result = svc._compute_astronomical(_CHENNAI_LAT, _CHENNAI_LON, "2026-01-01", "2026-01-02", {"moon_phase"})
    assert result["sunrise"] == []
    assert result["sunset"] == []
    assert len(result["moon_fraction"]) == 2
    assert len(result["dates"]) == 2


# ---------------------------------------------------------------------------
# _moon_phase
# ---------------------------------------------------------------------------


def test_moon_phase_at_known_new_moon_reference_is_near_zero():
    fraction, name = svc._moon_phase(date(2000, 1, 6))  # the algorithm's own reference new moon
    assert fraction < 0.03
    assert name == "New Moon"


def test_moon_phase_wraps_across_a_synodic_month():
    from datetime import timedelta

    synodic_month_days = 29.53058867
    frac_start, _ = svc._moon_phase(date(2000, 1, 6))
    frac_one_month_later, _ = svc._moon_phase(date(2000, 1, 6) + timedelta(days=round(synodic_month_days)))
    # One full synodic month later should be back near the same fraction (new moon again).
    assert abs(frac_one_month_later - frac_start) < 0.05


# ---------------------------------------------------------------------------
# get_historical_analytics — source selection, caching, degradation
# ---------------------------------------------------------------------------


def test_moon_phase_only_request_never_touches_the_network(monkeypatch):
    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not fetch any network source for a moon_phase-only request")

    monkeypatch.setattr(svc, "_fetch_marine_daily", fail_if_called)
    monkeypatch.setattr(svc, "_fetch_marine_current_hourly", fail_if_called)
    monkeypatch.setattr(svc, "_fetch_archive_daily", fail_if_called)
    monkeypatch.delenv("COPERNICUSMARINE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_PASSWORD", raising=False)

    result = asyncio.run(svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-02", ["moon_phase"]))

    assert "moon_phase" in result["series"]
    assert "names" in result["series"]["moon_phase"]


def test_sst_request_only_calls_marine_daily(monkeypatch):
    async def fail_if_called(*_a, **_k):
        raise AssertionError("should not be called for an sst-only request")

    async def fake_marine_daily(lat, lon, start_date, end_date):
        return {"time": ["2026-01-01"], "sea_surface_temperature_max": [28.5]}

    monkeypatch.setattr(svc, "_fetch_marine_daily", fake_marine_daily)
    monkeypatch.setattr(svc, "_fetch_marine_current_hourly", fail_if_called)
    monkeypatch.setattr(svc, "_fetch_archive_daily", fail_if_called)

    result = asyncio.run(svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-01", ["sst_celsius"]))

    assert result["series"]["sst_celsius"]["values"] == [28.5]
    assert result["series"]["sst_celsius"]["status"] == "ok"
    assert result["series"]["sst_celsius"]["unit"] == "°C"


def test_invalid_variables_are_dropped():
    result = asyncio.run(
        svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-01", ["not_a_real_variable", "moon_phase"])
    )
    assert set(result["series"].keys()) == {"moon_phase"}


def test_empty_variable_list_falls_back_to_every_valid_variable(monkeypatch):
    async def empty(*_a, **_k):
        return {}

    monkeypatch.setattr(svc, "_fetch_marine_daily", empty)
    monkeypatch.setattr(svc, "_fetch_marine_current_hourly", empty)
    monkeypatch.setattr(svc, "_fetch_archive_daily", empty)
    monkeypatch.delenv("COPERNICUSMARINE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_PASSWORD", raising=False)

    result = asyncio.run(svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-01", []))

    assert set(result["series"].keys()) == svc.VALID_VARIABLES


def test_identical_calls_are_served_from_cache(monkeypatch):
    call_count = {"n": 0}

    async def fake_marine_daily(lat, lon, start_date, end_date):
        call_count["n"] += 1
        return {"time": ["2026-01-01"], "sea_surface_temperature_max": [28.0]}

    monkeypatch.setattr(svc, "_fetch_marine_daily", fake_marine_daily)

    args = (5.0, 6.0, "2026-01-01", "2026-01-01", ["sst_celsius"])
    first = asyncio.run(svc.get_historical_analytics(*args))
    second = asyncio.run(svc.get_historical_analytics(*args))

    assert call_count["n"] == 1
    assert first is second


def test_copernicus_variable_without_credentials_degrades_to_empty_series(monkeypatch):
    monkeypatch.delenv("COPERNICUSMARINE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_PASSWORD", raising=False)

    result = asyncio.run(svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-01", ["chlorophyll_mg_m3"]))

    assert result["series"]["chlorophyll_mg_m3"]["values"] == []
    assert result["series"]["chlorophyll_mg_m3"]["status"] == "error"


def test_copernicus_variable_success_path(monkeypatch):
    monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "u")
    monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "p")

    def fake_sync(dataset_id, variable, lat, lon, start_date, end_date, username, password):
        return ["2026-01-01", "2026-01-02"], [0.4, None]

    monkeypatch.setattr(svc, "_fetch_copernicus_series_sync", fake_sync)

    result = asyncio.run(svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-02", ["chlorophyll_mg_m3"]))

    series = result["series"]["chlorophyll_mg_m3"]
    assert series["values"] == [0.4, None]
    assert series["status"] == "partial"


def test_copernicus_fetch_exception_degrades_to_empty_series(monkeypatch):
    monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "u")
    monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "p")

    def broken(*_a, **_k):
        raise RuntimeError("simulated Copernicus outage")

    monkeypatch.setattr(svc, "_fetch_copernicus_series_sync", broken)

    result = asyncio.run(svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-01", ["salinity_psu"]))

    assert result["series"]["salinity_psu"]["values"] == []
    assert result["series"]["salinity_psu"]["status"] == "error"


def test_current_variables_derive_from_hourly_aggregation(monkeypatch):
    async def fake_current_hourly(lat, lon, start_date, end_date):
        return {
            "time": ["2026-01-01T00:00", "2026-01-01T12:00"],
            "ocean_current_velocity": [0.5, 1.8],
            "ocean_current_direction": [10.0, 200.0],
        }

    monkeypatch.setattr(svc, "_fetch_marine_current_hourly", fake_current_hourly)

    result = asyncio.run(
        svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-01", ["current_velocity_kmh", "current_direction_deg"])
    )

    assert result["series"]["current_velocity_kmh"]["values"] == [1.8]
    assert result["series"]["current_direction_deg"]["values"] == [200.0]


# ---------------------------------------------------------------------------
# build_export_notes
# ---------------------------------------------------------------------------


def test_build_export_notes_includes_known_variable_caveats():
    series = {"chlorophyll_mg_m3": {"status": "ok", "dates": ["2026-01-01"], "values": [0.5], "unit": "mg/m³"}}
    notes = svc.build_export_notes(series)
    assert any("Coverage-limited" in n for n in notes)


def test_build_export_notes_flags_partial_and_error_status_dynamically():
    series = {
        "wind_kmh": {"status": "partial", "dates": [], "values": [], "unit": "km/h"},
        "wave_height_m": {"status": "error", "dates": [], "values": [], "unit": "m"},
    }
    notes = svc.build_export_notes(series)
    assert any("some dates in this range had no data" in n for n in notes)
    assert any("no data was available from the source" in n for n in notes)


def test_build_export_notes_fallback_when_fully_clean():
    series = {"wind_kmh": {"status": "ok", "dates": ["2026-01-01"], "values": [10.0], "unit": "km/h"}}
    notes = svc.build_export_notes(series)
    assert notes == ["All variables are sourced directly from their upstream APIs with full coverage for this range."]
