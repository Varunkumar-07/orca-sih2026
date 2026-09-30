"""
Risk Assessment Agent — Team Claude, Tier A (LLM reasoning agent).

Reasons over an EvidenceBundle's marine + weather data to produce a
RiskAssessment. Never raises across the agent boundary — any failure or
incomplete input degrades to a partial/error result instead of a fabricated
confident answer.

Whatever path produced the verdict (LLM, degraded, or failed), the weather
hard limits in safety_limits.py are then enforced on it in code: a breached
limit makes it UNSAFE, never the reverse.
"""

import json

from backend.agents.deterministic.safety_limits import (
    WAVE_HEIGHT_LIMIT_M,
    WIND_LIMIT_KMH,
    enforce_hard_limits,
    weather_limit_breaches,
)
from backend.agents.reasoning._groq_client import call_groq_json
from backend.agents.reasoning._trace import record_trace
from backend.schemas.contracts import (
    EvidenceBundle,
    MarineDataResult,
    RiskAssessment,
    TraceStep,
    WeatherDataResult,
)

_SYSTEM_PROMPT = f"""You are the risk assessment agent for ORCA, a marine \
intelligence assistant for fishermen and coastal operators. You are given \
marine data (sea surface temperature, chlorophyll, potential fishing zones) \
and weather data (wind, wave height, cyclone/lightning alerts) for a \
location. Reason over both to judge whether it is safe to go out to sea. \
Respond with ONLY a JSON object, no other text, in the form:
{{"safe_to_go": true | false, "confidence": <float 0.0-1.0>, "explanation": "<one or two sentences>"}}

Be conservative: any active cyclone or lightning alert, or wave height above \
{WAVE_HEIGHT_LIMIT_M:g} meters, or wind above {WIND_LIMIT_KMH:g} km/h should push safe_to_go to false with high \
confidence. Calm conditions with no alerts should push safe_to_go to true."""


async def _reason_over_data(marine: MarineDataResult, weather: WeatherDataResult) -> dict:
    user_content = json.dumps(
        {
            "marine": {
                "sst_celsius": marine.sst_celsius,
                "chlorophyll_mg_m3": marine.chlorophyll_mg_m3,
                "pfz_zones": marine.pfz_zones,
            },
            "weather": {
                "wind_kmh": weather.wind_kmh,
                "wave_height_m": weather.wave_height_m,
                "cyclone_alert": weather.cyclone_alert,
                "lightning_alert": weather.lightning_alert,
            },
        }
    )

    parsed = await call_groq_json(_SYSTEM_PROMPT, user_content, max_tokens=200)

    safe_to_go = parsed["safe_to_go"]
    confidence = float(parsed["confidence"])
    explanation = str(parsed["explanation"])
    if not isinstance(safe_to_go, bool):
        raise TypeError(f"unexpected safe_to_go value: {safe_to_go!r}")

    return {"safe_to_go": safe_to_go, "confidence": confidence, "explanation": explanation}


async def run_risk_assessment_agent(
    bundle: EvidenceBundle,
    trace: list[TraceStep],
) -> RiskAssessment:
    marine = bundle.marine
    weather = bundle.weather
    input_summary = (
        f"marine.status={marine.status if marine else None}, "
        f"weather.status={weather.status if weather else None}"
    )

    try:
        missing = []
        if marine is None or marine.status == "error":
            missing.append(f"marine data {marine.status if marine else 'missing'}")
        if weather is None or weather.status == "error":
            missing.append(f"weather data {weather.status if weather else 'missing'}")

        if missing:
            # Genuinely unusable input (missing entirely, or a hard
            # "error" with no real fields at all) — degrade gracefully
            # rather than asking the model to reason over data that
            # isn't there.
            result = RiskAssessment(
                status="partial",
                safe_to_go=None,
                confidence=0.4,
                explanation=(
                    f"Incomplete data ({', '.join(missing)}); "
                    "risk assessment is not fully reliable."
                ),
            )
            output_summary = f"degraded to partial: {', '.join(missing)}"
        else:
            # marine/weather may individually be "partial" (e.g. a coastal
            # wave-height grid gap, or chlorophyll unavailable without
            # Copernicus credentials) — that's still real, usable data to
            # judge safety from, not a reason to skip reasoning. The
            # system prompt's conditions (cyclone/lightning alert, wind,
            # wave) are independent ORs, and _reason_over_data's payload
            # already sends whatever fields are actually populated (null
            # for the rest), so a missing wave reading alone doesn't
            # prevent judging wind/cyclone/lightning.
            assessment = await _reason_over_data(marine, weather)
            degraded = marine.status == "partial" or weather.status == "partial"
            result = RiskAssessment(
                status="partial" if degraded else "ok",
                safe_to_go=assessment["safe_to_go"],
                confidence=assessment["confidence"],
                explanation=assessment["explanation"],
            )
            output_summary = (
                f"safe_to_go={result.safe_to_go}, confidence={result.confidence}"
                + (" (partial input)" if degraded else "")
            )

    except Exception as exc:  # noqa: BLE001 - must never raise across the boundary
        result = RiskAssessment(
            status="error",
            safe_to_go=None,
            confidence=0.0,
            explanation="Risk assessment could not be completed.",
            error_message=str(exc),
        )
        output_summary = f"error: {exc}"

    record_trace(trace, "risk_assessment_agent", input_summary, output_summary)
    return enforce_hard_limits(result, weather_limit_breaches(weather), trace)
