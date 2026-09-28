"""Pytest coverage for run_planning_agent's location-resolution priority
added this session: place name in query text > live GPS (user_location,
the frontend's captured browser geolocation) > session's last-known
location > none.

Context: the chat UI was already sending a real captured GPS coordinate
as `user_location` on every /query/full call, but run_planning_agent had
no parameter to receive it, so a query with no place name (e.g. "where is
the nearest PFZ today?") always fell straight to the "location_error"
degradation instead of using the device's actual position. This file
covers the fix: the new `user_location` parameter, its priority against
text/session, that it persists into session memory for later turns (same
mechanism as any other resolved location — no new code needed there, see
update_session's bundle.query_location read), and that the trace's
`location_source` label (text/gps/session/None) matches what actually
happened.

Mocking strategy: matches test_planning_agent_location.py's approach —
patch `call_groq_json` (so _classify_query never hits real Groq) and the
three downstream agent calls planning_agent imported into its own module
namespace (`run_marine_data_agent`, `run_weather_agent`,
`run_risk_assessment_agent`), so each test exercises the real location-
resolution logic in run_planning_agent without any network access.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

from backend.agents.reasoning import planning_agent as pa
from backend.schemas.contracts import (
    GeoPoint,
    MarineDataResult,
    RiskAssessment,
    WeatherDataResult,
)

_GPS_COORD = GeoPoint(lat=13.1689, lon=77.5334)
_SESSION_COORD = GeoPoint(lat=9.9312, lon=76.2673)  # Kochi, distinct from GPS coord


def _fake_marine_ok(query_text, query_location, trace):
    return MarineDataResult(
        status="ok",
        pfz_zones=[],
        sst_celsius=27.5,
        chlorophyll_mg_m3=0.5,
        source_timestamp="2026-01-01T00:00:00Z",
    )


def _fake_weather_ok(query_text, query_location, trace):
    return WeatherDataResult(
        status="ok",
        wind_kmh=10.0,
        wave_height_m=1.0,
        cyclone_alert=False,
        lightning_alert=False,
        tide_info=None,
        source_timestamp="2026-01-01T00:00:00Z",
    )


def _fake_risk_ok(bundle, trace):
    return RiskAssessment(status="ok", safe_to_go=True, confidence=0.9, explanation="calm")


def _install_fakes(monkeypatch, classification: dict):
    async def fake_call_groq_json(system_prompt, user_content, max_tokens, model=None):
        return classification

    async def fake_marine(query_text, query_location, trace):
        return _fake_marine_ok(query_text, query_location, trace)

    async def fake_weather(query_text, query_location, trace):
        return _fake_weather_ok(query_text, query_location, trace)

    async def fake_risk(bundle, trace):
        return _fake_risk_ok(bundle, trace)

    monkeypatch.setattr(pa, "call_groq_json", fake_call_groq_json)
    monkeypatch.setattr(pa, "run_marine_data_agent", fake_marine)
    monkeypatch.setattr(pa, "run_weather_agent", fake_weather)
    monkeypatch.setattr(pa, "run_risk_assessment_agent", fake_risk)


_NO_PLACE_NAMED = {"intent": "marine_query", "place_name": None}


def test_gps_used_when_no_place_named_and_no_session(monkeypatch):
    """No place in the query, no session, GPS provided -> GPS wins."""
    _install_fakes(monkeypatch, _NO_PLACE_NAMED)

    bundle, _session_id, greeting, api_failure, _turn = asyncio.run(
        pa.run_planning_agent(
            "where is the nearest fishing zone today?",
            session_id=None,
            user_location=_GPS_COORD,
        )
    )

    assert greeting is None
    assert not api_failure
    assert bundle.query_location == _GPS_COORD
    planning_step = next(s for s in bundle.trace if s.agent_name == "planning_agent")
    assert "location_source=gps" in planning_step.output_summary
    assert "session_context_used=False" in planning_step.output_summary


def test_text_place_name_wins_over_gps(monkeypatch):
    """A place explicitly named in the query must win over GPS, even when
    GPS is present — never silently substitute a different location for
    one the user actually typed."""
    _install_fakes(monkeypatch, {"intent": "marine_query", "place_name": "chennai"})

    bundle, _session_id, _greeting, _api_failure, _turn = asyncio.run(
        pa.run_planning_agent(
            "is it safe near chennai?",
            session_id=None,
            user_location=_GPS_COORD,
        )
    )

    assert bundle.query_location == pa._KNOWN_LOCATIONS["chennai"]
    assert bundle.query_location != _GPS_COORD
    planning_step = next(s for s in bundle.trace if s.agent_name == "planning_agent")
    assert "location_source=text" in planning_step.output_summary


def test_session_fallback_used_when_no_gps_sent(monkeypatch):
    """No place named, no GPS this turn, but a prior turn in the same
    session had a location -> falls back to session (existing behavior,
    must survive the GPS change untouched)."""
    _install_fakes(monkeypatch, {"intent": "marine_query", "place_name": "kochi"})
    # Turn 1: establish a session location via a named place, no GPS.
    bundle1, session_id, _greeting1, _api1, _turn1 = asyncio.run(
        pa.run_planning_agent("is it safe near kochi?", session_id=None, user_location=None)
    )
    assert bundle1.query_location == pa._KNOWN_LOCATIONS["kochi"]

    # Turn 2: genuine follow-up, no place named, no GPS this time.
    _install_fakes(monkeypatch, _NO_PLACE_NAMED)
    bundle2, _session_id2, _greeting2, _api2, _turn2 = asyncio.run(
        pa.run_planning_agent("what about tomorrow?", session_id=session_id, user_location=None)
    )

    assert bundle2.query_location == pa._KNOWN_LOCATIONS["kochi"]
    planning_step = next(s for s in bundle2.trace if s.agent_name == "planning_agent")
    assert "location_source=session" in planning_step.output_summary
    assert "session_context_used=True" in planning_step.output_summary


def test_gps_wins_over_session_when_both_available(monkeypatch):
    """Priority is text > GPS > session — a fresh GPS fix should be trusted
    over a possibly-stale session location (e.g. a boat that has moved)."""
    _install_fakes(monkeypatch, {"intent": "marine_query", "place_name": "kochi"})
    bundle1, session_id, _greeting1, _api1, _turn1 = asyncio.run(
        pa.run_planning_agent("is it safe near kochi?", session_id=None, user_location=None)
    )
    assert bundle1.query_location == pa._KNOWN_LOCATIONS["kochi"]

    _install_fakes(monkeypatch, _NO_PLACE_NAMED)
    bundle2, _session_id2, _greeting2, _api2, _turn2 = asyncio.run(
        pa.run_planning_agent(
            "where is the nearest PFZ now?", session_id=session_id, user_location=_GPS_COORD
        )
    )

    assert bundle2.query_location == _GPS_COORD
    assert bundle2.query_location != pa._KNOWN_LOCATIONS["kochi"]
    planning_step = next(s for s in bundle2.trace if s.agent_name == "planning_agent")
    assert "location_source=gps" in planning_step.output_summary
    assert "session_context_used=False" in planning_step.output_summary


def test_no_location_anywhere_degrades_cleanly(monkeypatch):
    """No place named, no GPS, no session -> query_location stays None and
    the existing clean "no location" degradation still fires (Phase 6a:
    confirms the GPS wiring didn't change this pre-existing path)."""

    async def fake_marine_error(query_text, query_location, trace):
        assert query_location is None
        return MarineDataResult(
            status="error",
            pfz_zones=[],
            sst_celsius=None,
            chlorophyll_mg_m3=None,
            source_timestamp="2026-01-01T00:00:00Z",
            error_message="query_location is required for a PFZ lookup",
        )

    async def fake_weather_error(query_text, query_location, trace):
        assert query_location is None
        return WeatherDataResult(
            status="error",
            wind_kmh=None,
            wave_height_m=None,
            cyclone_alert=False,
            lightning_alert=False,
            tide_info=None,
            source_timestamp="2026-01-01T00:00:00Z",
            error_message="query_location is required for a weather lookup",
        )

    _install_fakes(monkeypatch, _NO_PLACE_NAMED)
    monkeypatch.setattr(pa, "run_marine_data_agent", fake_marine_error)
    monkeypatch.setattr(pa, "run_weather_agent", fake_weather_error)

    bundle, _session_id, _greeting, _api_failure, _turn = asyncio.run(
        pa.run_planning_agent(
            "where is the nearest fishing zone today?",
            session_id=None,
            user_location=None,
        )
    )

    assert bundle.query_location is None
    assert bundle.marine.status == "error"
    assert bundle.weather.status == "error"
    planning_step = next(s for s in bundle.trace if s.agent_name == "planning_agent")
    assert "location_source=None" in planning_step.output_summary


def test_gps_derived_location_persists_into_session(monkeypatch):
    """A GPS-resolved location must be saved into session history too (same
    bundle.query_location -> SessionTurn.query_location path as any other
    resolved location), so a later follow-up in the same session can reuse
    it via session fallback."""
    _install_fakes(monkeypatch, _NO_PLACE_NAMED)
    _bundle1, session_id, _greeting1, _api1, turn1 = asyncio.run(
        pa.run_planning_agent(
            "where is the nearest fishing zone today?",
            session_id=None,
            user_location=_GPS_COORD,
        )
    )
    assert turn1.query_location == _GPS_COORD

    session = pa.get_session(session_id)
    assert session is not None
    assert session.last_known_location() == _GPS_COORD
