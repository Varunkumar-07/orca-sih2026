"""Pytest coverage for backend/routers/pages.py's _build_export_csv
(previously untested at all — added per finding #8 of the shape-mismatch
audit; see the audit report for why this specific gap mattered).

Mocking strategy — deliberately NOT a hand-written fixture: the whole
point of this file is that _build_export_csv's input (the `data` dict) is
never independently invented here. Every test drives the REAL
analytics_service.get_historical_analytics() (network fetches mocked at
the same boundary test_analytics_service.py already uses) and feeds its
REAL, actual return value straight into _build_export_csv. This is the
exact fix for the trap the original zone_id bug fell into: a hand-typed
mock that encodes the same wrong shape assumption as the bug itself would
never have caught it. Deriving the input from the real producer's output
means a future shape change in get_historical_analytics's series dict
breaks this test for the right reason, automatically.

Covers:
  - a normal variable's series (dates/values/unit all present, status=ok)
    renders correctly: header row includes the unit, each date row has
    the right value, standing csv/note-building behavior
  - the moon_phase-only case, which is a REAL producer shape variant
    ordinary variables don't have (an extra "names" key alongside
    dates/values) — confirms _build_export_csv doesn't choke on it and
    faithfully transcribes the (numeric) values/dates it actually reads,
    not an assumption about what "should" be there
  - a variable that comes back status="partial" (some dates missing data)
    — confirms missing dates render as a blank CSV cell (not "None", not
    a KeyError) and the partial-status note actually appears

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
import csv

import pytest

from backend.routers.pages import _build_export_csv
from backend.services import analytics_service as analytics_svc


@pytest.fixture(autouse=True)
def _reset_cache():
    analytics_svc._cache.clear()
    yield
    analytics_svc._cache.clear()


def _rows(csv_text: str) -> list[list[str]]:
    """Parsed data rows only — strips the leading '# ...' comment lines
    _build_export_csv writes before the real CSV header."""
    lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    return list(csv.reader(lines))


def test_normal_variable_series_renders_correct_header_and_rows(monkeypatch):
    async def fake_archive(lat, lon, start_date, end_date):
        return {"time": ["2026-01-01", "2026-01-02"], "wind_speed_10m_max": [10.0, 12.0]}

    monkeypatch.setattr(analytics_svc, "_fetch_archive_daily", fake_archive)

    # Real producer output — ground truth, not a hand-typed shape.
    data = asyncio.run(analytics_svc.get_historical_analytics(1.0, 2.0, "2026-01-01", "2026-01-02", ["wind_kmh"]))
    assert data["series"]["wind_kmh"]["status"] == "ok"  # sanity: this really is the "normal" case

    csv_text = _build_export_csv("TEST-ZONE", data)

    assert "# zone: TEST-ZONE" in csv_text
    assert "# location: 1.0, 2.0" in csv_text
    assert "# range: 2026-01-01 to 2026-01-02" in csv_text

    rows = _rows(csv_text)
    assert rows[0] == ["date", "wind_kmh (km/h)"]  # unit comes from the real series dict, not hardcoded
    assert rows[1] == ["2026-01-01", "10.0"]
    assert rows[2] == ["2026-01-02", "12.0"]


def test_moon_phase_only_names_key_does_not_break_csv_and_uses_real_values(monkeypatch):
    """moon_phase's series dict carries an extra "names" key
    (display-name strings) alongside dates/values that no other variable
    has — a real shape variant, not a hypothetical. _build_export_csv has
    no special-casing for it (unlike export_render.py's PDF/DOCX table,
    which does substitute the display name) — this test documents and
    locks in that actual, current behavior: the CSV cell holds the raw
    numeric fraction _build_export_csv actually read out of the real
    dict, not a hand-guessed value."""
    data = asyncio.run(analytics_svc.get_historical_analytics(5.0, 6.0, "2026-01-01", "2026-01-02", ["moon_phase"]))
    series = data["series"]["moon_phase"]
    assert "names" in series  # confirms this really is the shape variant under test
    assert series["dates"] == ["2026-01-01", "2026-01-02"]

    csv_text = _build_export_csv("TEST-ZONE", data)

    rows = _rows(csv_text)
    assert rows[0] == ["date", "moon_phase ()"]  # moon_phase's real unit is "" (see _UNITS)
    # Values transcribed straight from the real series dict — not from
    # its "names" list, and not from any value this test invented itself.
    assert rows[1] == ["2026-01-01", str(series["values"][0])]
    assert rows[2] == ["2026-01-02", str(series["values"][1])]

    # The real VARIABLE_CAVEATS note for moon_phase must still show up.
    assert "# note: Moon Phase: Computed from the synodic lunar month" in csv_text


def test_partial_status_variable_renders_blank_cell_and_partial_note(monkeypatch):
    async def fake_archive_partial(lat, lon, start_date, end_date):
        return {
            "time": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "wind_speed_10m_max": [10.0, None, 12.0],
        }

    monkeypatch.setattr(analytics_svc, "_fetch_archive_daily", fake_archive_partial)

    data = asyncio.run(analytics_svc.get_historical_analytics(3.0, 4.0, "2026-01-01", "2026-01-03", ["wind_kmh"]))
    assert data["series"]["wind_kmh"]["status"] == "partial"  # sanity: this really is the partial case
    assert data["series"]["wind_kmh"]["values"][1] is None

    csv_text = _build_export_csv("TEST-ZONE", data)

    rows = _rows(csv_text)
    assert rows[0] == ["date", "wind_kmh (km/h)"]
    assert rows[1] == ["2026-01-01", "10.0"]
    assert rows[2] == ["2026-01-02", ""]  # missing data -> blank cell, not "None" or a crash
    assert rows[3] == ["2026-01-03", "12.0"]

    assert "# note: Wind Speed: some dates in this range had no data available" in csv_text


def test_multi_variable_series_aligns_independent_date_ranges(monkeypatch):
    """Two variables with genuinely different available date ranges (a
    realistic case, not contrived) — confirms _build_export_csv's
    date-union logic reads each variable's own real "dates" list rather
    than assuming they're identical across variables."""
    async def fake_marine(lat, lon, start_date, end_date):
        return {"time": ["2026-01-01", "2026-01-02", "2026-01-03"], "sea_surface_temperature_max": [28.0, 28.5, 29.0]}

    async def fake_archive(lat, lon, start_date, end_date):
        # Archive coverage starts a day later than marine, a real-world gap.
        return {"time": ["2026-01-02", "2026-01-03"], "wind_speed_10m_max": [11.0, 13.0]}

    monkeypatch.setattr(analytics_svc, "_fetch_marine_daily", fake_marine)
    monkeypatch.setattr(analytics_svc, "_fetch_archive_daily", fake_archive)

    data = asyncio.run(
        analytics_svc.get_historical_analytics(7.0, 8.0, "2026-01-01", "2026-01-03", ["sst_celsius", "wind_kmh"])
    )
    csv_text = _build_export_csv("TEST-ZONE", data)

    rows = _rows(csv_text)
    assert rows[0] == ["date", "sst_celsius (°C)", "wind_kmh (km/h)"]
    assert rows[1] == ["2026-01-01", "28.0", ""]  # wind_kmh has no reading for this date at all
    assert rows[2] == ["2026-01-02", "28.5", "11.0"]
    assert rows[3] == ["2026-01-03", "29.0", "13.0"]
