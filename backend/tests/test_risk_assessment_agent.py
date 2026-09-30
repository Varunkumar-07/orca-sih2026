"""Pytest coverage for backend/agents/reasoning/risk_assessment_agent.py
(previously untested — added as part of the codebase audit's P1 coverage
gaps).

Mocking strategy: call_groq_json is patched directly (same "patch our own
function, not the vendor SDK" principle as every other reasoning-agent test
file in this suite) rather than mocking the Groq client itself. Tests are
plain sync functions driving async code via asyncio.run() — no
pytest-asyncio dependency, matching the rest of the suite.

Covers:
  - run_risk_assessment_agent: missing/None marine or weather -> "partial"
    degradation without calling Groq at all; marine/weather status="error"
    -> same degradation; marine/weather status="partial" (still usable
    data) -> Groq IS called and the result itself degrades to "partial";
    normal ok path -> status="ok", safe_to_go/confidence/explanation
    passed through untouched; a genuine Groq failure (exception, or a
    non-bool safe_to_go) -> status="error", never raises across the
    agent boundary
  - the trace step is always appended, exactly once, regardless of which
    path was taken

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

from backend.agents.reasoning import risk_assessment_agent as raa
from backend.schemas.contracts import (
    EvidenceBundle,
    GeoPoint,
    MarineDataResult,
    WeatherDataResult,
)

_LOCATION = GeoPoint(lat=13.08, lon=80.27)


def _marine(status="ok", **overrides) -> MarineDataResult:
    defaults = {
        "status": status,
        "pfz_zones": [{"zone_id": "PFZ-001", "center": {"lat": 13.1, "lon": 80.3}}],
        "sst_celsius": 28.4,
        "chlorophyll_mg_m3": 0.5,
        "source_timestamp": "2026-09-14T00:00:00Z",
    }
    defaults.update(overrides)
    return MarineDataResult(**defaults)


def _weather(status="ok", **overrides) -> WeatherDataResult:
    defaults = {
        "status": status,
        "wind_kmh": 15.0,
        "wave_height_m": 0.8,
        "cyclone_alert": False,
        "lightning_alert": False,
        "tide_info": None,
        "source_timestamp": "2026-09-14T00:00:00Z",
    }
    defaults.update(overrides)
    return WeatherDataResult(**defaults)


def _bundle(marine, weather) -> EvidenceBundle:
    return EvidenceBundle(
        query_text="is it safe to go out tomorrow?",
        query_location=_LOCATION,
        marine=marine,
        weather=weather,
        risk=None,
    )


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


# ---------------------------------------------------------------------------
# Degradation before any Groq call — missing or hard-error input
# ---------------------------------------------------------------------------


def test_marine_none_degrades_to_partial_without_calling_groq(monkeypatch):
    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not call Groq when marine data is entirely missing")

    monkeypatch.setattr(raa, "call_groq_json", fail_if_called)

    trace = []
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(None, _weather()), trace))

    assert result.status == "partial"
    assert result.safe_to_go is None
    assert "marine data missing" in result.explanation
    assert trace[-1].agent_name == "risk_assessment_agent"


def test_weather_none_degrades_to_partial_without_calling_groq(monkeypatch):
    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not call Groq when weather data is entirely missing")

    monkeypatch.setattr(raa, "call_groq_json", fail_if_called)

    trace = []
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), None), trace))

    assert result.status == "partial"
    assert result.safe_to_go is None
    assert "weather data missing" in result.explanation


def test_marine_status_error_degrades_to_partial_without_calling_groq(monkeypatch):
    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not call Groq when marine.status is 'error'")

    monkeypatch.setattr(raa, "call_groq_json", fail_if_called)

    trace = []
    bundle = _bundle(_marine(status="error", sst_celsius=None, chlorophyll_mg_m3=None), _weather())
    result = asyncio.run(raa.run_risk_assessment_agent(bundle, trace))

    assert result.status == "partial"
    assert result.confidence == 0.4


def test_weather_status_error_degrades_to_partial_without_calling_groq(monkeypatch):
    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not call Groq when weather.status is 'error'")

    monkeypatch.setattr(raa, "call_groq_json", fail_if_called)

    trace = []
    bundle = _bundle(_marine(), _weather(status="error", wind_kmh=None, wave_height_m=None))
    result = asyncio.run(raa.run_risk_assessment_agent(bundle, trace))

    assert result.status == "partial"


# ---------------------------------------------------------------------------
# marine/weather status="partial" — still usable, Groq IS called
# ---------------------------------------------------------------------------


def test_partial_marine_status_still_reasons_and_result_is_partial(monkeypatch):
    """A coastal grid gap (e.g. chlorophyll unavailable) is real, usable
    data — not a reason to skip Groq reasoning entirely, just a reason the
    final RiskAssessment.status itself reports 'partial'."""
    called = {"n": 0}

    async def fake_call(_system, _user, max_tokens=200):
        called["n"] += 1
        return {"safe_to_go": True, "confidence": 0.75, "explanation": "Calm seas."}

    monkeypatch.setattr(raa, "call_groq_json", fake_call)

    trace = []
    bundle = _bundle(_marine(status="partial", chlorophyll_mg_m3=None), _weather())
    result = asyncio.run(raa.run_risk_assessment_agent(bundle, trace))

    assert called["n"] == 1
    assert result.status == "partial"
    assert result.safe_to_go is True
    assert "(partial input)" in trace[-1].output_summary


# ---------------------------------------------------------------------------
# Normal ok path
# ---------------------------------------------------------------------------


def test_ok_path_passes_through_groq_output_untouched(monkeypatch):
    monkeypatch.setattr(
        raa,
        "call_groq_json",
        _async_return({"safe_to_go": False, "confidence": 0.92, "explanation": "Active cyclone alert."}),
    )

    trace = []
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), _weather(cyclone_alert=True)), trace))

    assert result.status == "ok"
    assert result.safe_to_go is False
    assert result.confidence == 0.92
    assert result.explanation == "Active cyclone alert."
    # The cyclone alert is also a hard limit (safety_limits.py): the rules
    # agree with the LLM's UNSAFE verdict, so its result stays untouched and
    # a safety_rules step records the agreement after the agent's own step.
    assert [t.agent_name for t in trace] == ["risk_assessment_agent", "safety_rules"]
    assert "safe_to_go=False, confidence=0.92" in trace[0].output_summary
    assert "agrees" in trace[1].output_summary
    assert result.verdict_source == "llm"


def test_reason_over_data_sends_both_marine_and_weather_fields(monkeypatch):
    """Sanity: the payload actually sent to Groq carries the real values
    from both results, not stale/default placeholders."""
    seen = {}

    async def fake_call(_system, user_content, max_tokens=200):
        import json

        seen["payload"] = json.loads(user_content)
        return {"safe_to_go": True, "confidence": 0.6, "explanation": "ok"}

    monkeypatch.setattr(raa, "call_groq_json", fake_call)

    marine = _marine(sst_celsius=29.5, chlorophyll_mg_m3=0.33)
    weather = _weather(wind_kmh=40.0, wave_height_m=2.1, lightning_alert=True)
    asyncio.run(raa.run_risk_assessment_agent(_bundle(marine, weather), []))

    assert seen["payload"]["marine"]["sst_celsius"] == 29.5
    assert seen["payload"]["marine"]["chlorophyll_mg_m3"] == 0.33
    assert seen["payload"]["weather"]["wind_kmh"] == 40.0
    assert seen["payload"]["weather"]["wave_height_m"] == 2.1
    assert seen["payload"]["weather"]["lightning_alert"] is True


# ---------------------------------------------------------------------------
# Genuine failures — never raise across the agent boundary
# ---------------------------------------------------------------------------


def test_groq_exception_degrades_to_error(monkeypatch):
    async def broken(*_a, **_k):
        raise RuntimeError("simulated Groq API failure")

    monkeypatch.setattr(raa, "call_groq_json", broken)

    trace = []
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), _weather()), trace))

    assert result.status == "error"
    assert result.safe_to_go is None
    assert result.confidence == 0.0
    assert result.error_message == "simulated Groq API failure"
    assert trace[-1].output_summary.startswith("error:")


def test_non_bool_safe_to_go_degrades_to_error(monkeypatch):
    """Groq occasionally returns malformed JSON (a string instead of a
    bool, etc.) — this must be caught, not propagated as a crash."""
    monkeypatch.setattr(
        raa,
        "call_groq_json",
        _async_return({"safe_to_go": "yes", "confidence": 0.5, "explanation": "malformed"}),
    )

    trace = []
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), _weather()), trace))

    assert result.status == "error"


def test_missing_expected_key_in_groq_response_degrades_to_error(monkeypatch):
    monkeypatch.setattr(raa, "call_groq_json", _async_return({"confidence": 0.5}))

    trace = []
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), _weather()), trace))

    assert result.status == "error"
