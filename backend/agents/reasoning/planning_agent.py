"""
Planning Agent (Orchestrator) — Team Claude, Tier A (LLM reasoning agent).

Parses a raw user query, extracts a location, and routes to the reasoning
agents in sequence: Marine Data Discovery -> Weather Intelligence -> Risk
Assessment. Builds and returns the EvidenceBundle.

Also resolves ambiguous follow-up queries ("what about Friday?") against a
session's prior location via the User Interaction Agent's session store.
"""

import asyncio
import difflib
import re

import httpx

from backend.agents.reasoning._groq_client import call_groq_json
from backend.agents.reasoning._groq_errors import (
    is_groq_api_failure as _is_groq_api_failure,
)
from backend.agents.reasoning._trace import record_trace
from backend.agents.reasoning.marine_data_agent import run_marine_data_agent
from backend.agents.reasoning.risk_assessment_agent import run_risk_assessment_agent
from backend.agents.reasoning.user_interaction_agent import (
    SessionTurn,
    create_session,
    get_session,
    update_session,
)
from backend.agents.reasoning.weather_agent import run_weather_agent
from backend.schemas.contracts import (
    EvidenceBundle,
    GeoPoint,
    MarineDataResult,
    RiskAssessment,
    TraceStep,
    WeatherDataResult,
)
from backend.time_utils import now_iso as _now_iso

# Rough coastal reference points — the fast path for "geocoding": the LLM
# is asked to match the query against this list first, falling back to
# explicit lat/lon if present in the query text, then to fuzzy matching and
# live geocoding (_geocode_place, Nominatim) if neither hits. Covers the
# major Indian coastal/port cities plus common short forms and alternate
# spellings (e.g. "vizag", "mangalore") as separate keys onto the same point.
_KNOWN_LOCATIONS: dict[str, GeoPoint] = {
    # Gujarat / northwest coast
    "kandla": GeoPoint(lat=23.0333, lon=70.2167),
    "porbandar": GeoPoint(lat=21.6417, lon=69.6293),
    "dwarka": GeoPoint(lat=22.2394, lon=68.9678),
    "bhavnagar": GeoPoint(lat=21.7645, lon=72.1519),
    "surat": GeoPoint(lat=21.1702, lon=72.8311),
    "jamnagar": GeoPoint(lat=22.4707, lon=70.0577),
    "veraval": GeoPoint(lat=20.9159, lon=70.3629),
    "okha": GeoPoint(lat=22.4707, lon=69.0729),
    "mandvi": GeoPoint(lat=22.8312, lon=69.3541),
    "navlakhi": GeoPoint(lat=22.9667, lon=70.4667),
    "valsad": GeoPoint(lat=20.5992, lon=72.9342),
    "daman": GeoPoint(lat=20.3974, lon=72.8328),
    "diu": GeoPoint(lat=20.7141, lon=70.9822),
    # Maharashtra / Goa
    "mumbai": GeoPoint(lat=19.0760, lon=72.8777),
    "ratnagiri": GeoPoint(lat=16.9902, lon=73.3120),
    "alibaug": GeoPoint(lat=18.6414, lon=72.8722),
    "murud": GeoPoint(lat=18.3298, lon=72.9633),
    "dahanu": GeoPoint(lat=19.9770, lon=72.7370),
    "vasai": GeoPoint(lat=19.4700, lon=72.8000),
    "malvan": GeoPoint(lat=16.0667, lon=73.4667),
    "devgad": GeoPoint(lat=16.3789, lon=73.3872),
    "goa": GeoPoint(lat=15.2993, lon=74.1240),
    "panaji": GeoPoint(lat=15.2993, lon=74.1240),
    "margao": GeoPoint(lat=15.2832, lon=73.9862),
    "vasco da gama": GeoPoint(lat=15.3980, lon=73.8121),
    "mormugao": GeoPoint(lat=15.4033, lon=73.8064),
    # Karnataka
    "mangaluru": GeoPoint(lat=12.9141, lon=74.8560),
    "mangalore": GeoPoint(lat=12.9141, lon=74.8560),
    "udupi": GeoPoint(lat=13.3409, lon=74.7421),
    "karwar": GeoPoint(lat=14.8137, lon=74.1291),
    "bhatkal": GeoPoint(lat=13.9847, lon=74.5550),
    "kundapura": GeoPoint(lat=13.6229, lon=74.6910),
    "malpe": GeoPoint(lat=13.3494, lon=74.7031),
    # Kerala
    "kochi": GeoPoint(lat=9.9312, lon=76.2673),
    "cochin": GeoPoint(lat=9.9312, lon=76.2673),
    "kozhikode": GeoPoint(lat=11.2588, lon=75.7804),
    "calicut": GeoPoint(lat=11.2588, lon=75.7804),
    "kollam": GeoPoint(lat=8.8932, lon=76.6141),
    "thiruvananthapuram": GeoPoint(lat=8.5241, lon=76.9366),
    "trivandrum": GeoPoint(lat=8.5241, lon=76.9366),
    "kanyakumari": GeoPoint(lat=8.0883, lon=77.5385),
    "alappuzha": GeoPoint(lat=9.4981, lon=76.3388),
    "alleppey": GeoPoint(lat=9.4981, lon=76.3388),
    "kannur": GeoPoint(lat=11.8745, lon=75.3704),
    "kasaragod": GeoPoint(lat=12.4996, lon=74.9869),
    "ponnani": GeoPoint(lat=10.7672, lon=75.9264),
    "neendakara": GeoPoint(lat=8.9333, lon=76.5667),
    "vizhinjam": GeoPoint(lat=8.3833, lon=76.9833),
    "chavakkad": GeoPoint(lat=10.6833, lon=75.9833),
    # Tamil Nadu
    "chennai": GeoPoint(lat=13.0827, lon=80.2707),
    "madras": GeoPoint(lat=13.0827, lon=80.2707),
    "puducherry": GeoPoint(lat=11.9416, lon=79.8083),
    "pondicherry": GeoPoint(lat=11.9416, lon=79.8083),
    "tuticorin": GeoPoint(lat=8.7642, lon=78.1348),
    "thoothukudi": GeoPoint(lat=8.7642, lon=78.1348),
    "rameswaram": GeoPoint(lat=9.2876, lon=79.3129),
    "gulf of mannar": GeoPoint(lat=9.1500, lon=79.1500),
    # The classifier model consistently (5/5 at temperature=0) mis-
    # transcribes "Mannar" as "Mannard" for this exact phrasing — no
    # explicit alias needed: _resolve_known_location's own fuzzy-match
    # fallback (cutoff=0.75) already resolves "gulf of mannard" to this
    # entry at ratio ~0.97, the same mechanism that already handles any
    # other close misspelling.
    "cuddalore": GeoPoint(lat=11.7480, lon=79.7714),
    "nagapattinam": GeoPoint(lat=10.7672, lon=79.8449),
    "karaikal": GeoPoint(lat=10.9254, lon=79.8380),
    "mandapam": GeoPoint(lat=9.2833, lon=79.1333),
    "tiruchendur": GeoPoint(lat=8.4956, lon=78.1189),
    "velankanni": GeoPoint(lat=10.6819, lon=79.8422),
    "pamban": GeoPoint(lat=9.2833, lon=79.2167),
    "mahe": GeoPoint(lat=11.7000, lon=75.5333),
    # Andhra Pradesh
    "nellore": GeoPoint(lat=14.4426, lon=79.9865),
    "visakhapatnam": GeoPoint(lat=17.6868, lon=83.2185),
    "vizag": GeoPoint(lat=17.6868, lon=83.2185),
    "kakinada": GeoPoint(lat=16.9891, lon=82.2475),
    "machilipatnam": GeoPoint(lat=16.1875, lon=81.1389),
    "bheemunipatnam": GeoPoint(lat=17.8888, lon=83.4514),
    "srikakulam": GeoPoint(lat=18.2949, lon=83.8938),
    "nizampatnam": GeoPoint(lat=15.9081, lon=80.6683),
    "krishnapatnam": GeoPoint(lat=14.2500, lon=80.1167),
    "yanam": GeoPoint(lat=16.7333, lon=82.2167),
    # Odisha
    "paradip": GeoPoint(lat=20.3167, lon=86.6167),
    "puri": GeoPoint(lat=19.8135, lon=85.8312),
    "gopalpur": GeoPoint(lat=19.2647, lon=84.9089),
    "chandipur": GeoPoint(lat=21.4667, lon=87.0167),
    "balasore": GeoPoint(lat=21.4942, lon=86.9317),
    "konark": GeoPoint(lat=19.8876, lon=86.0945),
    # West Bengal
    "kolkata": GeoPoint(lat=22.5726, lon=88.3639),
    "digha": GeoPoint(lat=21.6274, lon=87.5085),
    "haldia": GeoPoint(lat=22.0667, lon=88.0698),
    "diamond harbour": GeoPoint(lat=22.1833, lon=88.1833),
    "sagar island": GeoPoint(lat=21.6500, lon=88.0500),
    "kakdwip": GeoPoint(lat=21.8833, lon=88.1833),
    # Islands
    "port blair": GeoPoint(lat=11.6234, lon=92.7265),
    "havelock": GeoPoint(lat=12.0159, lon=92.9840),
    "diglipur": GeoPoint(lat=13.2500, lon=92.9667),
    "kavaratti": GeoPoint(lat=10.5669, lon=72.6420),
}

_LATLON_PATTERN = re.compile(
    r"(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)"
)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent on every
# request (no default/no UA is grounds for a block).
_NOMINATIM_HEADERS = {"User-Agent": "ORCA-SIH2026-Hackathon/1.0 (marine safety assistant)"}


async def _geocode_place(place_name: str) -> GeoPoint | None:
    """Last-resort geocoding for a place named in the query that isn't in
    _KNOWN_LOCATIONS, even fuzzily — covers the long tail of smaller
    fishing harbors/villages our fixed ~90-entry table was never going to
    include. Uses OpenStreetMap's free Nominatim service.

    Deliberately restricted to countrycodes=in: an unrestricted search for
    a short/misspelled name can match a place anywhere in the world (e.g.
    "Vizog" unrestricted resolves to a hamlet in Brittany, France) — for a
    system that explicitly refuses to guess a location rather than risk
    silently answering the wrong city, a wrong *country* would be far
    worse. Returns None (never raises) on no match or any failure, so the
    caller's existing "couldn't resolve" degradation still applies.
    """
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                _NOMINATIM_URL,
                params={"q": place_name, "format": "json", "limit": 1, "countrycodes": "in"},
                headers=_NOMINATIM_HEADERS,
            )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        return GeoPoint(lat=float(results[0]["lat"]), lon=float(results[0]["lon"]))
    except Exception:
        return None

# Fast local pre-check for the most common greetings/small-talk — matched
# before any LLM call, so "hi" gets a friendly reply even if Groq is down
# or rate-limited. Anchored to the whole message, so it only catches simple
# standalone greetings, not "hi, is it safe near chennai" (that correctly
# falls through to the LLM classifier below).
_GREETING_PATTERN = re.compile(
    r"^\s*(hi+|hello+|hey+|yo+|sup|greetings|"
    r"good\s*(morning|afternoon|evening)|"
    r"what\s+can\s+you\s+do|who\s+are\s+you|what\s+are\s+you|"
    r"tell\s+me\s+about\s+yourself|what\s+is\s+this|what'?s\s+this|help)"
    r"\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "Hi! I'm ORCA, a marine safety assistant. Ask me things like "
    "\"is it safe to go out tomorrow near Chennai?\" or "
    "\"can I fish near Gulf of Mannar?\""
)

_INTENT_SYSTEM_PROMPT = """You are the planning/orchestrator agent for ORCA, a \
marine intelligence assistant for fishermen and coastal operators. Given a raw \
user query, do two things:

1. Classify "intent" as exactly one of:
   - "marine_query": a genuine question about marine safety, weather, fishing \
     zones, or hazards at a location — OR a short follow-up inside an ongoing \
     marine-safety conversation that implicitly refers to a previously \
     mentioned location (e.g. "what about tomorrow?", "what about Friday?", \
     "and near Kochi?").
   - "greeting_or_offtopic": a greeting, small talk, or a question about the \
     assistant itself, not a marine-safety question (e.g. "hi", "hello", \
     "what can you do", "tell me about yourself", "who are you").

2. Extract the place name being referred to, if any.

Respond with ONLY a JSON object, no other text, in the form:
{"intent": "marine_query" | "greeting_or_offtopic", "place_name": "<lowercase place name mentioned in the query, or null if none>"}"""


def _location_from_latlon(query_text: str) -> GeoPoint | None:
    match = _LATLON_PATTERN.search(query_text)
    if not match:
        return None
    return GeoPoint(lat=float(match.group(1)), lon=float(match.group(2)))


_SAFE_CLASSIFICATION_DEFAULT = {"intent": "marine_query", "place_name": None}


async def _classify_query(query_text: str) -> dict:
    """Single combined LLM call: classify intent AND extract any place name,
    so a normal marine query still costs only one Groq round-trip.

    Never raises. Very short/ambiguous phrasing (e.g. "is it safe?") can push
    this reasoning model into hundreds of hidden "thinking" tokens before it
    emits the actual JSON answer, occasionally exceeding the token budget and
    failing Groq's JSON-mode validation entirely (confirmed via direct
    testing: reasoning_effort="low" keeps this well within budget for every
    phrasing tried, including the failing case). As a second line of
    defense, if classification still fails for any reason, fall back to
    "marine_query" with no place name — the caller then makes a genuine
    pipeline attempt with an unresolved location, which already degrades
    cleanly via the existing "no location" path, instead of a raw provider
    error ever reaching the user.
    """
    try:
        parsed = await call_groq_json(_INTENT_SYSTEM_PROMPT, query_text, max_tokens=400)

        intent = parsed.get("intent")
        if intent not in {"marine_query", "greeting_or_offtopic"}:
            # Fail safe toward attempting the real pipeline, never toward
            # silently treating a real question as a greeting.
            intent = "marine_query"

        place_name = parsed.get("place_name")
        place_name = place_name.lower().strip() if place_name else None

        return {"intent": intent, "place_name": place_name, "api_failed": False}
    except Exception:  # noqa: BLE001 - classification must never break the pipeline
        return {**_SAFE_CLASSIFICATION_DEFAULT, "api_failed": True}


def _resolve_known_location(
    query_text: str, place_name: str | None
) -> tuple[GeoPoint | None, str | None]:
    """Resolve a location given the query text and an already-extracted
    place name (from _classify_query — no LLM call happens here).

    Returns (location, unrecognized_place_name):
    - (GeoPoint, None): resolved successfully (explicit lat/lon, an exact
      known place name, or a close misspelling of one — see below)
    - (None, None): no location was mentioned in the query at all — the caller
      may safely fall back to session context (this is a genuine follow-up,
      e.g. "what about tomorrow?").
    - (None, "<name>"): a place WAS named but isn't in the known-location
      table, even fuzzily — the caller must NOT fall back to a different
      session location, since that would silently answer a different
      city's question.
    """
    # Explicit lat/lon in the query wins over name-based lookup.
    direct = _location_from_latlon(query_text)
    if direct is not None:
        return direct, None

    if place_name is None:
        return None, None

    resolved = _KNOWN_LOCATIONS.get(place_name)
    if resolved is not None:
        return resolved, None

    # A common misspelling/mis-transcription of a known place (e.g.
    # "vizog" for "vizag") shouldn't fail closed the way a genuinely
    # unrecognized place should — cutoff=0.75 catches typical typos
    # (one-two edits) without collapsing genuinely different city names
    # into each other.
    close = difflib.get_close_matches(place_name, _KNOWN_LOCATIONS.keys(), n=1, cutoff=0.75)
    if close:
        return _KNOWN_LOCATIONS[close[0]], None

    return None, place_name


def _scan_known_locations_in_text(query_text: str) -> str | None:
    """Local backstop for a real observed failure mode: the classifier's
    Groq call succeeds (no exception, so api_failed stays False) but the
    model itself returns place_name=null even though the query plainly
    names an exact _KNOWN_LOCATIONS entry (e.g. "is it safe near Vizag?").
    Without this, that miss is indistinguishable from "no location
    mentioned at all" and silently falls back to the session's last-known
    (different) location — a confident, wrong-city answer with no visible
    error. See run_planning_agent's call site for how this backstop
    plugs in only when the classifier itself found nothing.

    Whole-word/phrase matching only (word boundaries), longest key first,
    so a short key never fires on a substring inside an unrelated word
    (e.g. "goa" inside "goal") and a multi-word key like "gulf of mannar"
    isn't preempted by a shorter overlapping key. Deliberately not a
    substitute for the classifier or for _resolve_known_location's own
    fuzzy matching — this only catches an exact known-table name that's
    literally present in the text.
    """
    text = query_text.lower()
    for key in sorted(_KNOWN_LOCATIONS.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", text):
            return key
    return None


def _greeting_bundle(
    query_text: str, session_id: str, trace: list[TraceStep], reason: str
) -> EvidenceBundle:
    record_trace(
        trace,
        "planning_agent",
        f"query='{query_text}', session_id={session_id}",
        f"intent=greeting_or_offtopic ({reason}); no location resolution "
        "attempted, marine/weather/risk agents not invoked",
    )
    return EvidenceBundle(
        query_text=query_text,
        query_location=None,
        marine=None,
        weather=None,
        risk=None,
        trace=trace,
    )


async def run_planning_agent(
    query_text: str,
    session_id: str | None = None,
    user_location: GeoPoint | None = None,
) -> tuple[EvidenceBundle, str, str | None, bool, SessionTurn]:
    """Returns (bundle, session_id, greeting_response, api_failure_detected, turn).

    user_location is the caller's live browser-geolocation coordinate, when
    available (see chat.py's QueryRequest.user_location). Phase 2: accepted
    and threaded through only — not yet consulted as a location fallback
    (that's Phase 3, the actual place-name -> GPS -> session priority logic).

    greeting_response is non-None only when the query was classified as a
    greeting/off-topic message — in that case marine/weather/risk were never
    invoked (bundle.marine/weather/risk are all None, not degraded results),
    and the caller should show greeting_response directly instead of running
    it through the normal reporting pipeline.

    api_failure_detected is True when a genuine Groq/API-level failure (rate
    limit, timeout, auth, model error) was hit anywhere in this turn — as
    opposed to a normal, clean "no location" degradation, which is not a
    failure at all. This is the signal main.py's demo-reliability fallback
    (Tier 2) uses to decide whether to serve a cached snapshot instead.

    turn is the SessionTurn object this call's own internal update_session
    just appended — the caller passes it straight to
    set_last_turn_language, so that patch always lands on this exact turn
    (not whichever turn happens to be last by the time it runs).
    """
    trace: list[TraceStep] = []
    api_failure_detected = False

    # Resolve the session up front: reuse it if the caller gave a valid,
    # unexpired session_id; otherwise start a fresh one. A fresh query with
    # no session_id behaves exactly as before — session is just empty.
    session = get_session(session_id) if session_id else None
    if session is None:
        session_id = create_session()
        session = get_session(session_id)

    input_summary = f"query='{query_text}', session_id={session_id}"
    session_context_used = False

    try:
        # Fast local pre-check — no LLM call, works even if Groq is down.
        if _GREETING_PATTERN.match(query_text.strip()):
            bundle = _greeting_bundle(query_text, session_id, trace, "matched fast local pattern")
            turn = update_session(session_id, query_text, bundle)
            return bundle, session_id, _GREETING_RESPONSE, False, turn

        classification = await _classify_query(query_text)
        if classification.get("api_failed"):
            api_failure_detected = True

        if classification["intent"] == "greeting_or_offtopic":
            bundle = _greeting_bundle(query_text, session_id, trace, "classified by intent model")
            turn = update_session(session_id, query_text, bundle)
            return bundle, session_id, _GREETING_RESPONSE, api_failure_detected, turn

        # Backstop for a classifier miss: if the model returned no place
        # name, check the raw text for an exact known-location match before
        # treating this as "no location mentioned" (which falls back to
        # GPS/session below) — see _scan_known_locations_in_text's docstring.
        place_name = classification["place_name"]
        if place_name is None:
            place_name = _scan_known_locations_in_text(query_text)

        query_location, unrecognized_place = _resolve_known_location(query_text, place_name)

        if unrecognized_place is not None:
            # Not in our fixed table, even fuzzily — try live geocoding
            # before giving up (covers real coastal villages/harbors the
            # table was never going to enumerate). Restricted to India;
            # see _geocode_place's docstring for why that matters.
            geocoded = await _geocode_place(unrecognized_place)
            if geocoded is not None:
                query_location = geocoded
                unrecognized_place = None

        if unrecognized_place is not None:
            # A location WAS named but isn't in our lookup table. Never fall
            # back to the session's (different) prior location here — that
            # would silently answer a different city's question with
            # confident-looking but wrong data. Degrade clearly instead.
            location_error = (
                f"Location '{unrecognized_place}' was mentioned but isn't in the "
                f"known-location table — cannot resolve coordinates for it."
            )
            marine_result = MarineDataResult(
                status="error",
                pfz_zones=[],
                sst_celsius=None,
                chlorophyll_mg_m3=None,
                source_timestamp=_now_iso(),
                error_message=location_error,
            )
            weather_result = WeatherDataResult(
                status="error",
                wind_kmh=None,
                wave_height_m=None,
                cyclone_alert=False,
                lightning_alert=False,
                tide_info=None,
                source_timestamp=_now_iso(),
                error_message=location_error,
            )
            interim_bundle = EvidenceBundle(
                query_text=query_text,
                query_location=None,
                marine=marine_result,
                weather=weather_result,
                risk=None,
                trace=trace,
            )
            risk_result = await run_risk_assessment_agent(interim_bundle, trace)
            query_location = None
            if _is_groq_api_failure(getattr(risk_result, "error_message", None)):
                api_failure_detected = True

            output_summary = (
                f"location_unrecognized='{unrecognized_place}', session_context_used=False "
                f"(refused to fall back to a different session location), "
                f"marine.status=error, weather.status=error, risk.status={risk_result.status}"
            )
        else:
            # No location mentioned at all. Priority: live GPS (the user's
            # actual current position, when the frontend captured one) over
            # the session's last-known location — GPS is more likely to be
            # correct for a device that's moved since the last turn (e.g. a
            # boat that has since left port). Session fallback only kicks in
            # when no GPS coordinate was sent (denied/unavailable/timed out).
            # location_source is trace-only bookkeeping (never affects
            # behavior) so the reasoning trace panel stays honest about
            # where query_location actually came from on this turn.
            location_source = "text" if query_location is not None else None
            if query_location is None and user_location is not None:
                query_location = user_location
                location_source = "gps"
            elif query_location is None and session is not None:
                fallback_location = session.last_known_location()
                if fallback_location is not None:
                    query_location = fallback_location
                    session_context_used = True
                    location_source = "session"

            # Independent of each other (both take only query_text/
            # query_location) — run concurrently rather than paying the
            # sum of both agents' latency on every live query. Both
            # append to the same `trace` list, which is safe here: list
            # .append() is atomic under asyncio's single-threaded
            # cooperative concurrency.
            marine_result, weather_result = await asyncio.gather(
                run_marine_data_agent(query_text, query_location, trace),
                run_weather_agent(query_text, query_location, trace),
            )
            if _is_groq_api_failure(getattr(marine_result, "error_message", None)):
                api_failure_detected = True
            if _is_groq_api_failure(getattr(weather_result, "error_message", None)):
                api_failure_detected = True

            interim_bundle = EvidenceBundle(
                query_text=query_text,
                query_location=query_location,
                marine=marine_result,
                weather=weather_result,
                risk=None,
                trace=trace,
            )
            risk_result = await run_risk_assessment_agent(interim_bundle, trace)
            if _is_groq_api_failure(getattr(risk_result, "error_message", None)):
                api_failure_detected = True

            output_summary = (
                f"location={query_location}, location_source={location_source}, "
                f"session_context_used={session_context_used}, "
                f"routed marine_data_agent -> weather_agent -> risk_assessment_agent, "
                f"marine.status={marine_result.status}, weather.status={weather_result.status}, "
                f"risk.status={risk_result.status}"
            )
    except Exception as exc:  # noqa: BLE001 - must never raise across the boundary
        # _classify_query never raises (see above), so this is now a true
        # last-resort path for genuinely unexpected failures. NEVER put the
        # raw exception/provider payload where it can reach the user-facing
        # answer (marine/weather error_message flow directly into
        # reporting.py's answer_text) — keep the real detail only in the
        # trace step below, which is the internal/debug surface.
        api_failure_detected = True
        query_location = None
        friendly_error = (
            "I couldn't fully process that — try asking with a specific "
            "location, like 'is it safe near Chennai?'"
        )
        marine_result = MarineDataResult(
            status="error",
            pfz_zones=[],
            sst_celsius=None,
            chlorophyll_mg_m3=None,
            source_timestamp=_now_iso(),
            error_message=friendly_error,
        )
        weather_result = WeatherDataResult(
            status="error",
            wind_kmh=None,
            wave_height_m=None,
            cyclone_alert=False,
            lightning_alert=False,
            tide_info=None,
            source_timestamp=_now_iso(),
            error_message=friendly_error,
        )
        risk_result = RiskAssessment(
            status="error",
            safe_to_go=None,
            confidence=0.0,
            explanation=friendly_error,
            error_message=None,
        )
        output_summary = f"error: {exc}"

    record_trace(trace, "planning_agent", input_summary, output_summary)

    bundle = EvidenceBundle(
        query_text=query_text,
        query_location=query_location,
        marine=marine_result,
        weather=weather_result,
        risk=risk_result,
        trace=trace,
    )

    turn = update_session(session_id, query_text, bundle)

    return bundle, session_id, None, api_failure_detected, turn
