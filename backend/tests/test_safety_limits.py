"""Pytest coverage for backend/agents/deterministic/safety_limits.py — the
hard safety limits enforced in code on top of the Risk Assessment Agent's
LLM verdict — and for where they're wired in: the risk agent itself, the
deterministic pipeline (PROHIBITED), reporting's verdict line, and
/query/full's Groq-failure snapshot fallback.

The rule under test everywhere: code can move a verdict toward caution,
never toward safety.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.agents.deterministic import safety_limits as sl
from backend.agents.deterministic.reporting import run_reporting
from backend.agents.reasoning import risk_assessment_agent as raa
from backend.main import app
from backend.routers import chat
from backend.schemas.contracts import (
    EvidenceBundle,
    GeoPoint,
    GeospatialResult,
    MarineDataResult,
    RiskAssessment,
    WeatherDataResult,
)
from backend.schemas.test_fixtures import FIXTURE_1_HAPPY_PATH, FIXTURE_4_RESTRICTED_ZONE

_LOCATION = GeoPoint(lat=13.08, lon=80.27)


def _weather(**overrides) -> WeatherDataResult:
    defaults = {
        "status": "ok",
        "wind_kmh": 15.0,
        "wave_height_m": 0.8,
        "cyclone_alert": False,
        "lightning_alert": False,
        "tide_info": None,
        "source_timestamp": "2026-09-14T00:00:00Z",
    }
    defaults.update(overrides)
    return WeatherDataResult(**defaults)


def _marine(status="ok") -> MarineDataResult:
    return MarineDataResult(
        status=status,
        pfz_zones=[{"zone_id": "PFZ-001", "center": {"lat": 13.1, "lon": 80.3}}],
        sst_celsius=28.4,
        chlorophyll_mg_m3=0.5,
        source_timestamp="2026-09-14T00:00:00Z",
    )


def _risk(safe_to_go, confidence=0.9, status="ok", error_message=None) -> RiskAssessment:
    return RiskAssessment(
        status=status,
        safe_to_go=safe_to_go,
        confidence=confidence,
        explanation="LLM explanation.",
        error_message=error_message,
    )


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


# ---------------------------------------------------------------------------
# weather_limit_breaches / prohibited_area_breaches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        ({"cyclone_alert": True}, "cyclone"),
        ({"lightning_alert": True}, "lightning"),
        ({"wind_kmh": sl.WIND_LIMIT_KMH + 0.1}, "wind"),
        ({"wave_height_m": sl.WAVE_HEIGHT_LIMIT_M + 0.1}, "wave height"),
    ],
)
def test_each_weather_limit_alone_is_a_breach(overrides, expected_fragment):
    breaches = sl.weather_limit_breaches(_weather(**overrides))
    assert len(breaches) == 1
    assert expected_fragment in breaches[0]


def test_values_exactly_at_a_limit_are_not_a_breach():
    weather = _weather(wind_kmh=sl.WIND_LIMIT_KMH, wave_height_m=sl.WAVE_HEIGHT_LIMIT_M)
    assert sl.weather_limit_breaches(weather) == []


def test_missing_readings_never_count_as_a_breach():
    assert sl.weather_limit_breaches(_weather(wind_kmh=None, wave_height_m=None)) == []
    assert sl.weather_limit_breaches(None) == []


def test_several_breaches_are_all_listed():
    weather = _weather(cyclone_alert=True, wind_kmh=70.0, wave_height_m=4.5)
    assert len(sl.weather_limit_breaches(weather)) == 3


def test_prohibited_area_is_a_breach_only_when_inside():
    inside = GeospatialResult(
        nearest_zone_name=None, distance_km=None, inside_restricted_area=True, restricted_area_name="Gulf of Mannar"
    )
    near = GeospatialResult(
        nearest_zone_name=None,
        distance_km=None,
        inside_restricted_area=False,
        near_restricted_area_name="Gulf of Mannar",
        near_restricted_area_km=5.0,
    )
    assert "Gulf of Mannar" in sl.prohibited_area_breaches(inside)[0]
    assert sl.prohibited_area_breaches(near) == []
    assert sl.prohibited_area_breaches(None) == []


# ---------------------------------------------------------------------------
# enforce_hard_limits
# ---------------------------------------------------------------------------


def test_no_breach_leaves_the_llm_verdict_untouched_and_traces_nothing():
    risk = _risk(True, 0.88)
    trace = []
    assert sl.enforce_hard_limits(risk, [], trace) is risk
    assert trace == []


def test_breach_overrides_a_safe_llm_verdict_as_rule_based():
    trace = []
    result = sl.enforce_hard_limits(_risk(True, 0.88), ["active cyclone alert"], trace)
    assert result.safe_to_go is False
    assert result.verdict_source == "rules"
    assert result.confidence is None
    assert "active cyclone alert" in result.explanation
    assert len(trace) == 1
    assert trace[0].agent_name == "safety_rules"
    assert "safe_to_go=True, confidence=0.88" in trace[0].input_summary
    assert "OVERRIDE" in trace[0].output_summary


def test_breach_overrides_an_inconclusive_verdict():
    result = sl.enforce_hard_limits(_risk(None, 0.4, status="partial"), ["active lightning alert"], [])
    assert result.safe_to_go is False
    assert result.status == "partial"
    assert result.verdict_source == "rules"


def test_breach_with_an_already_unsafe_verdict_keeps_the_llm_result_and_traces_agreement():
    risk = _risk(False, 0.95)
    trace = []
    assert sl.enforce_hard_limits(risk, ["active cyclone alert"], trace) is risk
    assert len(trace) == 1
    assert "agrees" in trace[0].output_summary


def test_an_unsafe_llm_verdict_is_never_turned_safe():
    risk = _risk(False, 0.6)
    assert sl.enforce_hard_limits(risk, [], []).safe_to_go is False


def test_override_of_a_failed_assessment_becomes_partial_and_drops_the_raw_error():
    risk = _risk(None, 0.0, status="error", error_message="Error code: 429 - rate limited")
    trace = []
    result = sl.enforce_hard_limits(risk, ["active cyclone alert"], trace)
    assert result.status == "partial"
    assert result.safe_to_go is False
    assert result.error_message is None
    # The raw detail stays in the internal trace only.
    assert "Error code: 429" in trace[0].input_summary


def test_breach_with_no_risk_assessment_at_all_is_unsafe():
    result = sl.enforce_hard_limits(None, ["active cyclone alert"], [])
    assert result.safe_to_go is False
    assert result.verdict_source == "rules"


# ---------------------------------------------------------------------------
# Risk Assessment Agent wiring
# ---------------------------------------------------------------------------


def _bundle(marine, weather) -> EvidenceBundle:
    return EvidenceBundle(query_text="is it safe?", query_location=_LOCATION, marine=marine, weather=weather, risk=None)


def test_prompt_thresholds_come_from_the_constants():
    assert f"{sl.WAVE_HEIGHT_LIMIT_M:g} meters" in raa._SYSTEM_PROMPT
    assert f"{sl.WIND_LIMIT_KMH:g} km/h" in raa._SYSTEM_PROMPT


def test_risk_agent_overrides_llm_safe_verdict_when_wave_limit_breached(monkeypatch):
    monkeypatch.setattr(
        raa, "call_groq_json", _async_return({"safe_to_go": True, "confidence": 0.8, "explanation": "Looks fine."})
    )
    trace = []
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), _weather(wave_height_m=4.2)), trace))
    assert result.safe_to_go is False
    assert result.verdict_source == "rules"
    assert [t.agent_name for t in trace] == ["risk_assessment_agent", "safety_rules"]


def test_risk_agent_partial_data_branch_with_cyclone_is_unsafe(monkeypatch):
    """Marine data missing used to give 'inconclusive' even with a live
    cyclone alert; a breached limit now makes it UNSAFE."""

    async def fail_if_called(*_a, **_k):
        raise AssertionError("Groq must not be called when marine data is unusable")

    monkeypatch.setattr(raa, "call_groq_json", fail_if_called)
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(None, _weather(cyclone_alert=True)), []))
    assert result.safe_to_go is False
    assert result.status == "partial"
    assert result.verdict_source == "rules"


def test_risk_agent_groq_failure_with_breach_is_unsafe(monkeypatch):
    async def broken(*_a, **_k):
        raise RuntimeError("Error code: 429 - rate limited")

    monkeypatch.setattr(raa, "call_groq_json", broken)
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), _weather(lightning_alert=True)), []))
    assert result.safe_to_go is False
    assert result.error_message is None


def test_risk_agent_groq_failure_without_breach_is_unchanged(monkeypatch):
    async def broken(*_a, **_k):
        raise RuntimeError("Error code: 429 - rate limited")

    monkeypatch.setattr(raa, "call_groq_json", broken)
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), _weather()), []))
    assert result.status == "error"
    assert result.safe_to_go is None
    assert result.verdict_source == "llm"


# ---------------------------------------------------------------------------
# Deterministic pipeline (PROHIBITED) + reporting
# ---------------------------------------------------------------------------


def test_prohibited_overrides_a_safe_verdict_in_the_deterministic_pipeline():
    """Fixture 4 sits inside the Gulf of Mannar with the happy-path 'safe'
    risk verdict — the pipeline must end UNSAFE, rule-based, and traced."""
    bundle = FIXTURE_4_RESTRICTED_ZONE.model_copy(deep=True)
    assert bundle.risk.safe_to_go is True
    final = chat._run_deterministic_pipeline(bundle)

    assert bundle.risk.safe_to_go is False
    assert bundle.risk.verdict_source == "rules"
    steps = [t for t in final.reasoning_trace if t.agent_name == "safety_rules"]
    assert len(steps) == 1 and "OVERRIDE" in steps[0].output_summary
    assert "PROHIBITED" in final.answer_text
    assert "confidence 88%" not in final.answer_text


def test_outside_protected_areas_the_pipeline_adds_no_safety_step():
    final = chat._run_deterministic_pipeline(FIXTURE_1_HAPPY_PATH.model_copy(deep=True))
    assert not any(t.agent_name == "safety_rules" for t in final.reasoning_trace)
    assert "Safe to go (confidence 88%)" in final.answer_text


def test_reporting_shows_limit_breached_instead_of_a_percentage():
    bundle = FIXTURE_1_HAPPY_PATH.model_copy(deep=True)
    bundle.risk = RiskAssessment(
        status="ok", safe_to_go=False, confidence=None, explanation="Hard safety limit breached: x.", verdict_source="rules"
    )
    text = run_reporting(bundle).answer_text
    assert "⚠️ Not safe to go (limit breached) — Hard safety limit breached: x." in text
    assert "%" not in text.split("\n")[2]


def test_reporting_handles_a_missing_confidence_without_a_percentage():
    bundle = FIXTURE_1_HAPPY_PATH.model_copy(deep=True)
    bundle.risk = RiskAssessment(status="partial", safe_to_go=None, confidence=None, explanation="Incomplete.")
    assert "Safety assessment inconclusive — Incomplete." in run_reporting(bundle).answer_text


# ---------------------------------------------------------------------------
# /query/full — Groq-failure snapshot fallback
# ---------------------------------------------------------------------------


def _live_bundle(weather: WeatherDataResult) -> EvidenceBundle:
    trace = []
    risk = sl.enforce_hard_limits(
        _risk(None, 0.0, status="error", error_message="Error code: 429 - rate limited"),
        sl.weather_limit_breaches(weather),
        trace,
    )
    return EvidenceBundle(
        query_text="is it safe near chennai", query_location=_LOCATION, marine=_marine(), weather=weather, risk=risk, trace=trace
    )


class _Turn:
    detected_language = None


def _patch_planning(monkeypatch, bundle: EvidenceBundle, api_failure: bool):
    async def fake_planning(query, session_id=None, user_location=None):
        return bundle, "session-1", None, api_failure, _Turn()

    monkeypatch.setattr(chat, "run_planning_agent", fake_planning)
    monkeypatch.setattr(chat, "set_last_turn_language", lambda *_a: None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-a-real-key")


def test_groq_failure_with_a_live_hazard_never_serves_a_snapshot(monkeypatch):
    _patch_planning(monkeypatch, _live_bundle(_weather(cyclone_alert=True, wind_kmh=70.0)), api_failure=True)
    resp = TestClient(app).post("/query/full", json={"query": "is it safe near chennai"})
    assert resp.status_code == 200
    body = resp.json()
    agents = [t["agent_name"] for t in body["reasoning_trace"]]
    assert "demo_fallback" not in agents
    assert "safety_rules" in agents
    assert "Not safe to go (limit breached)" in body["answer_text"]
    assert "Error code" not in body["answer_text"]


def test_groq_failure_without_a_hazard_still_serves_the_snapshot(monkeypatch):
    _patch_planning(monkeypatch, _live_bundle(_weather()), api_failure=True)
    resp = TestClient(app).post("/query/full", json={"query": "is it safe near chennai"})
    assert resp.status_code == 200
    assert "demo_fallback" in [t["agent_name"] for t in resp.json()["reasoning_trace"]]


# ---------------------------------------------------------------------------
# "Safe" needs both a wave and a wind reading (cap_unsupported_safe)
# ---------------------------------------------------------------------------


def test_missing_readings_lists_each_missing_one():
    assert sl.missing_safety_readings(_weather()) == []
    assert sl.missing_safety_readings(_weather(wave_height_m=None)) == ["wave height reading unavailable"]
    assert sl.missing_safety_readings(_weather(wind_kmh=None)) == ["wind reading unavailable"]
    assert len(sl.missing_safety_readings(_weather(wave_height_m=None, wind_kmh=None))) == 2


def test_no_weather_or_failed_weather_counts_as_both_missing():
    assert len(sl.missing_safety_readings(None)) == 2
    assert len(sl.missing_safety_readings(_weather(status="error", wind_kmh=15.0, wave_height_m=0.8))) == 2


@pytest.mark.parametrize("overrides", [{"wave_height_m": None}, {"wind_kmh": None}])
def test_a_safe_verdict_missing_a_reading_is_capped_at_inconclusive(overrides):
    trace = []
    missing = sl.missing_safety_readings(_weather(**overrides))
    result = sl.cap_unsupported_safe(_risk(True, 0.9), missing, trace)
    assert result.safe_to_go is None
    assert result.verdict_source == "rules"
    assert result.confidence is None
    assert result.status == "partial"
    assert "Not enough data to call it safe" in result.explanation
    assert len(trace) == 1 and "CAPPED" in trace[0].output_summary
    assert "safe_to_go=True, confidence=0.9" in trace[0].input_summary


def test_both_missing_readings_are_listed_in_the_explanation():
    result = sl.cap_unsupported_safe(_risk(True), sl.missing_safety_readings(None), [])
    assert "wave height reading unavailable" in result.explanation
    assert "wind reading unavailable" in result.explanation


def test_an_unsafe_verdict_with_a_missing_reading_is_untouched():
    risk = _risk(False, 0.7)
    trace = []
    assert sl.cap_unsupported_safe(risk, ["wave height reading unavailable"], trace) is risk
    assert trace == []


def test_an_inconclusive_verdict_with_a_missing_reading_is_untouched():
    risk = _risk(None, 0.4, status="partial")
    assert sl.cap_unsupported_safe(risk, ["wind reading unavailable"], []) is risk


def test_complete_readings_leave_a_safe_verdict_unchanged():
    risk = _risk(True, 0.9)
    trace = []
    assert sl.cap_unsupported_safe(risk, [], trace) is risk
    assert trace == []


def test_failed_assessment_with_a_missing_reading_becomes_rule_based_and_drops_the_raw_error():
    risk = _risk(None, 0.0, status="error", error_message="Error code: 429 - rate limited")
    trace = []
    result = sl.cap_unsupported_safe(risk, ["wave height reading unavailable"], trace)
    assert result.safe_to_go is None
    assert result.status == "partial"
    assert result.verdict_source == "rules"
    assert result.error_message is None
    assert "Error code: 429" in trace[0].input_summary


def test_risk_agent_caps_llm_safe_verdict_when_wave_reading_is_missing(monkeypatch):
    monkeypatch.setattr(
        raa, "call_groq_json", _async_return({"safe_to_go": True, "confidence": 0.85, "explanation": "Calm."})
    )
    trace = []
    weather = _weather(status="partial", wave_height_m=None)
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), weather), trace))
    assert result.safe_to_go is None
    assert result.verdict_source == "rules"
    assert [t.agent_name for t in trace] == ["risk_assessment_agent", "safety_rules"]


def test_breach_wins_over_the_cap_when_a_reading_is_also_missing(monkeypatch):
    """A cyclone alert with no wave reading is UNSAFE, not inconclusive."""
    monkeypatch.setattr(
        raa, "call_groq_json", _async_return({"safe_to_go": True, "confidence": 0.8, "explanation": "Looks fine."})
    )
    weather = _weather(status="partial", wave_height_m=None, cyclone_alert=True)
    result = asyncio.run(raa.run_risk_assessment_agent(_bundle(_marine(), weather), []))
    assert result.safe_to_go is False
    assert result.verdict_source == "rules"


def test_reporting_labels_a_rule_based_inconclusive_verdict_as_missing_readings():
    bundle = FIXTURE_1_HAPPY_PATH.model_copy(deep=True)
    bundle.risk = RiskAssessment(
        status="partial",
        safe_to_go=None,
        confidence=None,
        explanation="Not enough data to call it safe: wave height reading unavailable.",
        verdict_source="rules",
    )
    text = run_reporting(bundle).answer_text
    assert "❓ Safety assessment inconclusive (missing readings) — Not enough data" in text
    assert "limit breached" not in text


def _groq_failed_bundle(weather: WeatherDataResult) -> EvidenceBundle:
    trace = []
    risk = _risk(None, 0.0, status="error", error_message="Error code: 429 - rate limited")
    risk = sl.enforce_hard_limits(risk, sl.weather_limit_breaches(weather), trace)
    risk = sl.cap_unsupported_safe(risk, sl.missing_safety_readings(weather), trace)
    return EvidenceBundle(
        query_text="is it safe near kolkata", query_location=_LOCATION, marine=_marine(), weather=weather, risk=risk, trace=trace
    )


def test_groq_failure_with_a_missing_reading_never_serves_a_snapshot(monkeypatch):
    _patch_planning(monkeypatch, _groq_failed_bundle(_weather(status="partial", wave_height_m=None)), api_failure=True)
    resp = TestClient(app).post("/query/full", json={"query": "is it safe near kolkata"})
    assert resp.status_code == 200
    body = resp.json()
    agents = [t["agent_name"] for t in body["reasoning_trace"]]
    assert "demo_fallback" not in agents
    assert "safety_rules" in agents
    assert "Safety assessment inconclusive (missing readings)" in body["answer_text"]
    assert "Error code" not in body["answer_text"]
