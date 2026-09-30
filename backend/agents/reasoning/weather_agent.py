"""
Weather Intelligence Agent — Team Claude, Tier A (LLM reasoning agent).

Interprets what weather/hazard data a query needs and returns a
WeatherDataResult. Wind, wave height, and a thunderstorm flag are fetched
live from Open-Meteo (forecast + marine APIs, both free/keyless). Cyclone
alert is derived from live wind speed against IMD's cyclonic-storm
threshold, since no free feed publishes an explicit cyclone-alert flag.
Tide timing has no free/keyless source and is left as None (genuinely
unavailable) rather than faked. Never raises across the agent boundary —
any failure (network, parsing, missing data) degrades to status="error".
"""

import asyncio
import logging

from backend.agents.deterministic.safety_limits import (
    CYCLONE_WIND_THRESHOLD_KMH,
    THUNDERSTORM_WMO_CODES,
)
from backend.agents.reasoning._groq_client import call_groq_json
from backend.agents.reasoning._trace import record_trace
from backend.error_utils import describe_exception
from backend.schemas.contracts import GeoPoint, TraceStep, WeatherDataResult
from backend.services import open_meteo
from backend.time_utils import now_iso as _now_iso

logger = logging.getLogger("orca.weather_agent")

_FORECAST_URL = open_meteo.FORECAST_URL
_MARINE_URL = open_meteo.MARINE_URL
# Defined once in safety_limits.py (with the go/no-go limits); these names
# are what weather_service.py and alerts_service.py import from here.
_CYCLONE_WIND_THRESHOLD_KMH = CYCLONE_WIND_THRESHOLD_KMH
_THUNDERSTORM_CODES = THUNDERSTORM_WMO_CODES

_INTENT_SYSTEM_PROMPT = """You are a weather/hazard intent classifier for a \
fisheries assistant. Given a user's query, classify which single weather \
parameter is being asked about. Respond with ONLY a JSON object, no other \
text, in the form:
{"parameter": "wind" | "wave" | "cyclone" | "lightning" | "tide" | "general"}

"wind" = asking about wind speed/conditions.
"wave" = asking about wave height/sea state.
"cyclone" = asking about cyclone/storm alerts.
"lightning" = asking about lightning/thunderstorm alerts.
"tide" = asking about tide timing/levels.
"general" = anything else weather-related that doesn't fit the above, \
including broad "is it safe to go out" style queries."""


async def _fetch_live_weather(location: GeoPoint) -> dict:
    """Live wind/wave/thunderstorm snapshot for the query location, via
    Open-Meteo's free, keyless forecast and marine APIs — through the
    shared cached/rate-limited client (services/open_meteo.py), so a chat
    turn and a Weather-page load for the same point share one upstream
    call. Raises on any network/parsing failure — the caller turns that
    into status="error" rather than ever substituting a fabricated number.
    """
    forecast, marine_current = await asyncio.gather(
        open_meteo.fetch_current_forecast(location.lat, location.lon),
        open_meteo.fetch_current_marine(location.lat, location.lon),
    )
    forecast_current = forecast["current"]

    wind_raw = forecast_current["wind_speed_10m"]
    if wind_raw is None:
        raise ValueError("Open-Meteo returned no wind reading for this location")
    wind_kmh = float(wind_raw)
    weather_code = int(forecast_current["weather_code"])

    # The marine API's grid has gaps right at the coastline — a point can
    # legitimately have no wave-height cell nearby. That's a partial-data
    # case, not a failure: wind/lightning are still real.
    wave_raw = marine_current.get("wave_height")
    wave_height_m = float(wave_raw) if wave_raw is not None else None

    return {
        "wind_kmh": wind_kmh,
        "wave_height_m": wave_height_m,
        "cyclone_alert": wind_kmh >= _CYCLONE_WIND_THRESHOLD_KMH,
        "lightning_alert": weather_code in _THUNDERSTORM_CODES,
        # No free/keyless tide API exists — left honestly unavailable
        # rather than faked (reporting.py already skips a falsy tide_info).
        "tide_info": None,
    }


def _user_facing_error(exc: Exception) -> str:
    if isinstance(exc, open_meteo.OpenMeteoError):
        return exc.user_message
    if type(exc) is ValueError:
        # Our own hand-written messages (e.g. "query_location is required
        # for a weather lookup", "Open-Meteo returned no wind reading ...")
        # — exact type only, so subclasses like pydantic's ValidationError
        # or JSONDecodeError never leak through here.
        return str(exc)
    return "live weather data could not be retrieved — please try again shortly"


async def _classify_intent(query_text: str) -> str:
    parsed = await call_groq_json(_INTENT_SYSTEM_PROMPT, query_text, max_tokens=200)
    parameter = parsed.get("parameter", "general")
    if parameter not in {"wind", "wave", "cyclone", "lightning", "tide", "general"}:
        parameter = "general"
    return parameter


async def run_weather_agent(
    query_text: str,
    query_location: GeoPoint | None,
    trace: list[TraceStep],
) -> WeatherDataResult:
    input_summary = f"query='{query_text}', location={query_location}"

    try:
        if query_location is None:
            raise ValueError("query_location is required for a weather lookup")

        # Intent is cosmetic here — only used in the trace label below.
        # cyclone_alert comes from real wind speed, not this call, so a
        # Groq classification hiccup (observed live: empty/invalid JSON
        # completion) shouldn't throw away perfectly good live weather
        # data or falsely trip the Tier-2 demo fallback. Run it
        # concurrently with the real fetch below (previously awaited
        # sequentially in front of it, paying a full extra Groq round-trip
        # of latency for zero functional benefit).
        async def _classify_intent_safe() -> str:
            try:
                return await _classify_intent(query_text)
            except Exception:
                return "general"

        # The classified parameter is surfaced in the trace; the live
        # snapshot itself always reflects actual current conditions at
        # query_location regardless of which parameter was asked about.
        intent, snapshot = await asyncio.gather(_classify_intent_safe(), _fetch_live_weather(query_location))

        result = WeatherDataResult(
            status="ok" if snapshot["wave_height_m"] is not None else "partial",
            wind_kmh=snapshot["wind_kmh"],
            wave_height_m=snapshot["wave_height_m"],
            cyclone_alert=snapshot["cyclone_alert"],
            lightning_alert=snapshot["lightning_alert"],
            tide_info=snapshot["tide_info"],
            source_timestamp=_now_iso(),
        )
        wave_display = f"{result.wave_height_m}m" if result.wave_height_m is not None else "unavailable"
        output_summary = (
            f"intent={intent}, wind={result.wind_kmh}km/h, "
            f"wave={wave_display}, cyclone_alert={result.cyclone_alert}"
        )

    except Exception as exc:  # noqa: BLE001 - must never raise across the boundary
        # error_message flows straight into the user-facing chat answer
        # (reporting.py) — it must never carry a raw upstream exception
        # (previously a full Open-Meteo 429 URL plus an MDN docs link).
        # The real detail goes to the log; the trace (shown in the UI's
        # reasoning panel) gets the exception type only.
        logger.warning("Weather lookup failed for %s: %s", query_location, describe_exception(exc))
        result = WeatherDataResult(
            status="error",
            wind_kmh=None,
            wave_height_m=None,
            cyclone_alert=False,
            lightning_alert=False,
            tide_info=None,
            source_timestamp=_now_iso(),
            error_message=_user_facing_error(exc),
        )
        output_summary = f"error: {type(exc).__name__}"

    record_trace(trace, "weather_agent", input_summary, output_summary)
    return result
