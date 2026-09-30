"""Hard safety limits — deterministic, no LLM call.

The one place ORCA's go/no-go thresholds live. The Risk Assessment Agent's
LLM gives the first verdict; these limits are then enforced in code on top
of it, and can only move that verdict toward caution:

- any limit breached -> the verdict is UNSAFE, whatever the LLM said
- no limit breached -> the LLM's verdict stands unchanged (the code never
  turns an unsafe or inconclusive verdict into a safe one)
- a missing reading never counts as a breach, but "safe" needs both a wave
  and a wind reading: with either one missing, a SAFE verdict is capped at
  inconclusive (cap_unsupported_safe). Breaches are checked first, so a
  cyclone alert with no wave reading is still UNSAFE.

A verdict the code overrode is marked verdict_source="rules" with
confidence=None: it's a rule outcome, not a probability, so reporting shows
"limit breached" (UNSAFE) or "missing readings" (inconclusive) instead of a
percentage. When the LLM already said unsafe and a limit is breached too,
its own result is kept as-is and the trace records that the rules agreed.

Every check here returns reasons; enforce_hard_limits and
cap_unsupported_safe append a "safety_rules" TraceStep whenever they act.
"""
from __future__ import annotations

from backend.schemas.contracts import (
    GeospatialResult,
    RiskAssessment,
    TraceStep,
    WeatherDataResult,
)
from backend.time_utils import now_iso

# Sea-state limits. These match the thresholds the Risk Assessment Agent's
# prompt used before they were enforced in code; the prompt is now built
# from these same constants (risk_assessment_agent.py), so the two can't
# drift apart. A reading strictly above the limit is a breach.
WAVE_HEIGHT_LIMIT_M = 3.0
WIND_LIMIT_KMH = 45.0

# How the weather alert flags themselves are raised (weather_agent.py,
# weather_service.py, alerts_service.py all read these). IMD classifies a
# "Cyclonic Storm" at sustained wind >= 62 km/h; no free feed publishes an
# explicit cyclone-alert flag, so live wind against this threshold stands in.
CYCLONE_WIND_THRESHOLD_KMH = 62.0
# WMO weather codes (used by Open-Meteo) for thunderstorm conditions.
THUNDERSTORM_WMO_CODES = frozenset({95, 96, 99})


def weather_limit_breaches(weather: WeatherDataResult | None) -> list[str]:
    """Plain-language reasons, one per breached weather limit (empty when
    none are, or when there's no weather data)."""
    if weather is None:
        return []
    reasons: list[str] = []
    if weather.cyclone_alert:
        reasons.append("active cyclone alert")
    if weather.lightning_alert:
        reasons.append("active lightning alert")
    if weather.wind_kmh is not None and weather.wind_kmh > WIND_LIMIT_KMH:
        reasons.append(f"wind {weather.wind_kmh:.1f} km/h is above the {WIND_LIMIT_KMH:g} km/h limit")
    if weather.wave_height_m is not None and weather.wave_height_m > WAVE_HEIGHT_LIMIT_M:
        reasons.append(f"wave height {weather.wave_height_m:.1f} m is above the {WAVE_HEIGHT_LIMIT_M:g} m limit")
    return reasons


def missing_safety_readings(weather: WeatherDataResult | None) -> list[str]:
    """What a SAFE verdict would be missing: wave height and/or wind. No
    weather data at all (or a failed lookup) is both."""
    if weather is None or weather.status == "error":
        return ["wave height reading unavailable", "wind reading unavailable"]
    missing: list[str] = []
    if weather.wave_height_m is None:
        missing.append("wave height reading unavailable")
    if weather.wind_kmh is None:
        missing.append("wind reading unavailable")
    return missing


def prohibited_area_breaches(geospatial: GeospatialResult | None) -> list[str]:
    """One reason when the query location is inside a protected area."""
    if geospatial is None or not geospatial.inside_restricted_area:
        return []
    name = geospatial.restricted_area_name or "a protected area"
    return [f"inside {name} (protected area — PROHIBITED)"]


def _describe(risk: RiskAssessment | None) -> str:
    if risk is None:
        return "no risk assessment"
    confidence = "n/a" if risk.confidence is None else f"{risk.confidence}"
    return f"safe_to_go={risk.safe_to_go}, confidence={confidence}, source={risk.verdict_source}"


def enforce_hard_limits(
    risk: RiskAssessment | None, breaches: list[str], trace: list[TraceStep]
) -> RiskAssessment | None:
    """Apply `breaches` (from the functions above) to `risk`.

    No breaches: `risk` is returned unchanged and nothing is traced.
    Breaches and the verdict is already UNSAFE: `risk` is returned
    unchanged; the trace records that the rules agreed. Otherwise (the LLM
    said safe, or was inconclusive, or failed): a rule-based UNSAFE verdict
    replaces it, and the trace records the original next to the override.

    An overridden verdict that came from a failed assessment (status
    "error") becomes "partial": a verdict now exists, but the automated
    reasoning behind the rest of it didn't run. Its raw error_message is
    kept out of the result (reporting.py shows error_message to the user)
    and recorded in the trace instead.
    """
    if not breaches:
        return risk

    reasons = "; ".join(breaches)
    if risk is not None and risk.safe_to_go is False:
        trace.append(
            TraceStep(
                agent_name="safety_rules",
                input_summary=f"verdict: {_describe(risk)}",
                output_summary=f"hard limits breached ({reasons}) — agrees with UNSAFE verdict, no override",
                timestamp=now_iso(),
            )
        )
        return risk

    status = "partial" if risk is None or risk.status == "error" else risk.status
    overridden = RiskAssessment(
        status=status,
        safe_to_go=False,
        confidence=None,
        explanation=f"Hard safety limit breached: {reasons}.",
        verdict_source="rules",
    )
    input_summary = f"verdict: {_describe(risk)}"
    if risk is not None and risk.error_message:
        input_summary += f", error={risk.error_message}"
    trace.append(
        TraceStep(
            agent_name="safety_rules",
            input_summary=input_summary[:500],
            output_summary=f"OVERRIDE -> safe_to_go=False (rule-based): {reasons}"[:500],
            timestamp=now_iso(),
        )
    )
    return overridden


def cap_unsupported_safe(
    risk: RiskAssessment | None, missing: list[str], trace: list[TraceStep]
) -> RiskAssessment | None:
    """With `missing` readings (from missing_safety_readings), a verdict can
    be at most inconclusive.

    Acts only on a SAFE verdict, turning it into a rule-based inconclusive
    one; an UNSAFE or inconclusive verdict is returned unchanged (so this
    never undoes enforce_hard_limits, which runs first). The one exception
    is a failed assessment (status "error"): it has no verdict to cap, but
    it becomes a rule-based inconclusive "partial" result so its raw
    error_message — which reporting.py shows to the user — stays in the
    trace only.
    """
    if not missing or risk is None:
        return risk
    failed = risk.status == "error"
    if risk.safe_to_go is not True and not failed:
        return risk

    reasons = "; ".join(missing)
    capped = RiskAssessment(
        status="partial",
        safe_to_go=None,
        confidence=None,
        explanation=f"Not enough data to call it safe: {reasons}.",
        verdict_source="rules",
    )
    input_summary = f"verdict: {_describe(risk)}"
    if risk.error_message:
        input_summary += f", error={risk.error_message}"
    trace.append(
        TraceStep(
            agent_name="safety_rules",
            input_summary=input_summary[:500],
            output_summary=f"CAPPED -> inconclusive (rule-based): {reasons}"[:500],
            timestamp=now_iso(),
        )
    )
    return capped
