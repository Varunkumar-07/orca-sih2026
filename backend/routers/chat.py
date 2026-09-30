"""Chat/query pipeline — POST /query, /query/full, /query/demo, plus
GET /fixtures (the same golden fixtures /query/demo and the demo-mode
fallback in /query/full read from).

Both /query and /query/full accept an optional session_id in the request
body for multi-turn conversations (see user_interaction_agent.py) and echo
the active session_id back via the X-Session-Id response header — the JSON
body shape is unchanged so existing clients aren't affected.

/query/full also has a demo-reliability safety net (Tier 2): if a live Groq
call hard-fails (rate limit, timeout, API error — not a normal "no location"
degradation, which already handles itself cleanly), it serves the closest
matching genuine captured snapshot from demo_snapshot.py instead of
surfacing the failure. See _bundle_has_api_failure / _closest_demo_snapshot.

Split out of main.py (P4 codebase-audit cleanup) purely for file size; no
behavior change.
"""
import asyncio
import logging
import os

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from backend.agents.deterministic.analytics import run_ocean_analytics
from backend.agents.deterministic.geospatial import (
    get_active_restricted_areas,
    haversine_km,
    run_geospatial,
)
from backend.agents.deterministic.reporting import run_reporting
from backend.agents.deterministic.safety_limits import (
    enforce_hard_limits,
    missing_safety_readings,
    prohibited_area_breaches,
    weather_limit_breaches,
)
from backend.agents.deterministic.visualization import run_visualization
from backend.agents.reasoning._groq_errors import (
    is_groq_api_failure as _is_groq_api_failure,
)
from backend.agents.reasoning.language_agent import (
    translate_input_to_english,
    translate_output_from_english,
)
from backend.agents.reasoning.navigation_agent import plan_route
from backend.agents.reasoning.planning_agent import (
    _KNOWN_LOCATIONS,
    _location_from_latlon,
    _scan_known_locations_in_text,
    run_planning_agent,
)
from backend.agents.reasoning.user_interaction_agent import (
    create_session,
    get_session,
    set_last_turn_language,
)
from backend.schemas.contracts import (
    EvidenceBundle,
    FinalResponse,
    GeoPoint,
    MapPayload,
    TraceStep,
)
from backend.schemas.demo_places import DEMO_SAMPLE_PFZ_BY_ANCHOR
from backend.schemas.demo_snapshot import DEMO_SNAPSHOTS
from backend.schemas.test_fixtures import (
    FIXTURE_1_HAPPY_PATH,
    FIXTURE_2_HAZARD_PATH,
    FIXTURE_3_PARTIAL_FAILURE,
    FIXTURE_4_RESTRICTED_ZONE,
)
from backend.time_utils import now_iso as _now_iso

logger = logging.getLogger("orca.demo_fallback")

router = APIRouter()


class QueryRequest(BaseModel):
    # Bounded so an unauthenticated caller can't forward unlimited-size
    # text into translation/LLM calls (resource/cost exhaustion) — 2000
    # chars comfortably covers any real free-text marine query.
    query: str = Field(max_length=2000)
    session_id: str | None = None
    # Phase 5.3: the destination PFZ candidate's center, once the user has
    # picked one from the multi-zone list a prior turn returned (frontend
    # click-to-select, Phase 4.3). Sending coordinates directly — rather
    # than a zone_id the backend would have to re-resolve — sidesteps the
    # fact that a fresh planning_agent call each turn can return a
    # different candidate list than the one the user actually saw and
    # clicked on. Presence of this field is what triggers Navigation Agent
    # routing (Phase 5.2); its absence means "no destination chosen yet,"
    # matching the PDF's own described flow of routing only to a chosen
    # zone, never a blanket always-route-somewhere.
    selected_destination: GeoPoint | None = None
    # Phase 5.4: live browser geolocation (Team Gemini's Phase 5.1 frontend
    # capture), sent up alongside the query so Navigation Agent can route
    # from where the user actually is instead of the session's
    # text-resolved query_location. Optional — absence just falls back to
    # the prior behavior (see _run_deterministic_pipeline below).
    user_location: GeoPoint | None = None


def _run_deterministic_pipeline(
    bundle: EvidenceBundle,
    selected_destination: GeoPoint | None = None,
    user_location: GeoPoint | None = None,
) -> FinalResponse:
    """Run Phase 1 deterministic modules + Phase 2 reporting/visualization,
    plus Phase 5.3's Navigation Agent stage when a destination was chosen."""
    # Phase 5.4: prefer the live device geolocation (when the frontend
    # captured and sent one) over the session's text-resolved location for
    # every deterministic module below — not just the Navigation Agent.
    # Without this, run_geospatial's restricted-zone containment check and
    # visualization.py's "Query location" map pin would silently keep
    # testing/showing wherever the query text resolved to, even when we
    # know exactly where the user actually is right now (e.g. a boat
    # currently drifting inside an MPA while asking a text query about a
    # different named place).
    if user_location is not None:
        bundle.query_location = user_location

    # If bundle came from fixture (no trace), add a synthetic planning step so the
    # judged trace panel shows the full decomposition: Planning → Marine/Weather …
    if not bundle.trace:
        bundle.trace.append(
            TraceStep(
                agent_name="planning_agent",
                input_summary=f"query='{bundle.query_text[:60]}'",
                output_summary=f"demo routing: location={bundle.query_location} (fixture-based)",
                timestamp=_now_iso(),
            )
        )
        # Also add synthetic marine/weather/risk steps if they exist, to show full Tier A flow
        if bundle.marine:
            bundle.trace.append(
                TraceStep(
                    agent_name="marine_data_agent",
                    input_summary=f"location={bundle.query_location}",
                    output_summary=f"status={bundle.marine.status}, pfz={len(bundle.marine.pfz_zones)}",
                    timestamp=_now_iso(),
                )
            )
        if bundle.weather:
            bundle.trace.append(
                TraceStep(
                    agent_name="weather_agent",
                    input_summary=f"location={bundle.query_location}",
                    output_summary=f"status={bundle.weather.status}, wind={bundle.weather.wind_kmh}",
                    timestamp=_now_iso(),
                )
            )
        if bundle.risk:
            bundle.trace.append(
                TraceStep(
                    agent_name="risk_assessment_agent",
                    input_summary=f"marine={bundle.marine.status if bundle.marine else None}, weather={bundle.weather.status if bundle.weather else None}",
                    output_summary=f"safe_to_go={bundle.risk.safe_to_go}, conf={bundle.risk.confidence}",
                    timestamp=_now_iso(),
                )
            )

    # Phase 1 deterministic (Team Gemini)
    try:
        run_ocean_analytics(bundle)
    except Exception as exc:
        logger.warning("[DETERMINISTIC PIPELINE] analytics stage failed: %s", exc)
    try:
        run_geospatial(bundle)
    except Exception as exc:
        logger.warning("[DETERMINISTIC PIPELINE] geospatial stage failed: %s", exc)
    # PROHIBITED is a hard limit too (safety_limits.py): inside a protected
    # area the verdict is UNSAFE whatever the risk agent said — including a
    # demo fixture's. The weather limits were already enforced inside the
    # risk agent itself, which has no geospatial result to check this with.
    try:
        bundle.risk = enforce_hard_limits(bundle.risk, prohibited_area_breaches(bundle.geospatial), bundle.trace)
    except Exception as exc:
        logger.warning("[DETERMINISTIC PIPELINE] hard-limit stage failed: %s", exc)
    # Phase 2
    final = run_reporting(bundle)
    try:
        payload = run_visualization(final, bundle)
        final.map_payload = payload
        # visualization appends to bundle.trace but final.reasoning_trace was snapshotted
        # inside reporting — update it so the judged trace panel includes visualization
        final.reasoning_trace = list(bundle.trace)
    except Exception as exc:
        logger.warning("[DETERMINISTIC PIPELINE] visualization stage failed: %s", exc)

    # Phase 5.3/5.4: Navigation Agent — only runs once a destination has
    # actually been chosen (selected_destination present) and there's a
    # start point to route from. bundle.query_location already prefers
    # live geolocation over the session-resolved location (see the
    # overwrite above), so it's the correct start point here too.
    start_point = bundle.query_location
    if selected_destination is not None and start_point is not None:
        try:
            start = {"lat": start_point.lat, "lon": start_point.lon}
            destination = {"lat": selected_destination.lat, "lon": selected_destination.lon}
            # Same restricted-zone set geospatial.py's own containment check
            # already used for this bundle — the route can never drift from
            # what determined inside_restricted_area.
            restricted_zones = get_active_restricted_areas()
            result = plan_route(start, destination, restricted_zones)

            now = _now_iso()
            if result.route is not None:
                final.map_payload.route = [GeoPoint(lat=p["lat"], lon=p["lon"]) for p in result.route]
                nav_summary = f"route found, {len(result.route)} waypoints"
                if result.start_offset_km:
                    nav_summary += f"; starts at open water {result.start_offset_km} km from your position"
            else:
                # Graceful degradation: leave route as None (its default) —
                # never break the response over an unreachable destination.
                nav_summary = f"no route found — {result.reason}"

            bundle.trace.append(
                TraceStep(
                    agent_name="navigation_agent",
                    input_summary=f"start={start}, destination={destination}",
                    output_summary=nav_summary,
                    timestamp=now,
                )
            )
            final.reasoning_trace = list(bundle.trace)
        except Exception as exc:
            logger.warning("[DETERMINISTIC PIPELINE] navigation stage failed: %s", exc)

    return final


# A demo question naming a place farther than this from the chosen
# fixture's own location is moved to that place (see _fixture_for_query).
_DEMO_RELOCATE_KM = 25.0
# Sample PFZ zones are only borrowed from an anchor city this close to the
# asked-about place; farther than that, the demo shows no fishing zone
# rather than one that isn't near the question.
_DEMO_SAMPLE_PFZ_MAX_KM = 150.0


def _demo_place(query: str) -> tuple[str | None, GeoPoint | None]:
    """(place name, point) the question names — explicit lat/lon or a
    known coastal place, matched locally (demo mode makes no LLM call)."""
    point = _location_from_latlon(query)
    if point is not None:
        return None, point
    place = _scan_known_locations_in_text(query)
    return (place, _KNOWN_LOCATIONS[place]) if place else (None, None)


def _demo_sample_zones(point: GeoPoint) -> list[dict]:
    """The captured sample PFZ zones of the nearest anchor city (see
    demo_places.py), if one is within _DEMO_SAMPLE_PFZ_MAX_KM."""
    best: tuple[float, str] | None = None
    for anchor in DEMO_SAMPLE_PFZ_BY_ANCHOR:
        anchor_point = _KNOWN_LOCATIONS.get(anchor)
        if anchor_point is None:
            continue
        km = haversine_km(point.lat, point.lon, anchor_point.lat, anchor_point.lon)
        if best is None or km < best[0]:
            best = (km, anchor)
    if best is None or best[0] > _DEMO_SAMPLE_PFZ_MAX_KM:
        return []
    return [dict(zone, center=dict(zone["center"])) for zone in DEMO_SAMPLE_PFZ_BY_ANCHOR[best[1]]]


def _fixture_for_query(query: str) -> EvidenceBundle:
    """Offline/demo answer: a scenario fixture picked by keyword (restricted
    area / hazard / partial data / happy path), then made to match the
    question actually asked — its text is echoed, and if it names a place
    other than the fixture's own (the fixtures are set at Chennai and the
    Gulf of Mannar), the scenario is moved there with that area's sample
    fishing zones. The weather/risk scenario stays the sample one."""
    q = query.lower()
    if "gulf of mannar" in q or "mannar" in q:
        base = FIXTURE_4_RESTRICTED_ZONE.model_copy(deep=True)
    elif "cyclone" in q or "hazard" in q or "storm" in q:
        base = FIXTURE_2_HAZARD_PATH.model_copy(deep=True)
    elif "imd" in q or "partial" in q or "unreachable" in q:
        base = FIXTURE_3_PARTIAL_FAILURE.model_copy(deep=True)
    else:
        base = FIXTURE_1_HAPPY_PATH.model_copy(deep=True)
    base.query_text = query

    _place, point = _demo_place(query)
    fixture_point = base.query_location
    if point is not None and (
        fixture_point is None
        or haversine_km(point.lat, point.lon, fixture_point.lat, fixture_point.lon) > _DEMO_RELOCATE_KM
    ):
        base.query_location = point
    # The fixtures share one set of Chennai sample zones — swap in the
    # asked-about area's own whenever those aren't anywhere near it.
    loc = base.query_location
    if loc is not None and base.marine is not None:
        zones_nearby = any(
            haversine_km(loc.lat, loc.lon, z["center"]["lat"], z["center"]["lon"]) <= _DEMO_SAMPLE_PFZ_MAX_KM
            for z in base.marine.pfz_zones
        )
        if not zones_nearby:
            base.marine.pfz_zones = _demo_sample_zones(loc)
    return base


async def _resolve_input_language(query: str, session_id: str | None) -> tuple[str, str]:
    """Phase 6.2: Language Agent, input side. Detects `query`'s language and
    translates it to English (no-op if it's already English), returning
    (english_query, effective_language).

    language_agent's translate functions are synchronous (urllib-based, not
    httpx) — run via asyncio.to_thread rather than called directly, so a
    slow/retried Bhashini call blocks only this request's own coroutine,
    not FastAPI's single-threaded event loop (and therefore every other
    concurrent request) for the duration of the call.

    A short follow-up query with no script of its own (e.g. "yes", a bare
    coordinate pair — a handful of words at most) always detects as "en"
    from the local heuristic alone — in that case, prefer the session's
    last known language (if any) so a reply doesn't silently switch back to
    English mid-conversation. A longer, clearly-formed sentence is trusted
    as deliberately English instead of being overridden — the heuristic
    reading it as "en" isn't ambiguous the way a bare "yes" is, so treating
    it the same way this fallback treats short replies would fight a user
    who's genuinely switching the conversation back to English. This only
    affects which language the OUTPUT gets translated into; the query text
    itself is never re-translated on the strength of session history, since
    a query the heuristic itself read as plain English/ASCII genuinely has
    nothing to translate.
    """
    _AMBIGUOUS_MAX_WORDS = 4
    english_query, detected_language = await asyncio.to_thread(translate_input_to_english, query)
    if detected_language == "en" and session_id and len(query.split()) <= _AMBIGUOUS_MAX_WORDS:
        session = get_session(session_id)
        if session is not None:
            prior_language = session.last_known_language()
            if prior_language:
                detected_language = prior_language
    return english_query, detected_language


async def _apply_output_language(final: FinalResponse, effective_language: str) -> FinalResponse:
    """Phase 6.2: Language Agent, output side. Translates final.answer_text
    back into effective_language (no-op for "en") and populates
    detected_language/response_language. response_language reflects what the
    text actually ended up in — translate_output_from_english reports "en"
    if the Bhashini call failed, so this never claims a translation that
    didn't happen.

    Run via asyncio.to_thread for the same reason as _resolve_input_language
    above — the underlying call is blocking, synchronous I/O.
    """
    translated_text, response_language = await asyncio.to_thread(
        translate_output_from_english, final.answer_text, effective_language
    )
    final.answer_text = translated_text
    final.detected_language = effective_language
    final.response_language = response_language
    return final


def _bundle_has_api_failure(bundle: EvidenceBundle) -> bool:
    for result in (bundle.marine, bundle.weather, bundle.risk):
        if (
            result is not None
            and getattr(result, "status", None) == "error"
            and _is_groq_api_failure(getattr(result, "error_message", None))
        ):
            return True
    return False


def _closest_demo_snapshot(query: str) -> FinalResponse:
    """Keyword match against the four rehearsed scenarios — same simple,
    proven style as _fixture_for_query above, applied to genuine captured
    snapshots instead of hand-authored fixtures."""
    q = query.lower()
    if "mannar" in q:
        return DEMO_SNAPSHOTS["gulf_of_mannar_restricted"]
    if "cyclone" in q or "hazard" in q or "storm" in q or "vizag" in q or "visakhapatnam" in q:
        return DEMO_SNAPSHOTS["hazard_visakhapatnam"]
    if len(q.strip(" ?!.")) <= 12:
        # short/bare phrasing like "is it safe?" — matches the ambiguous scenario
        return DEMO_SNAPSHOTS["bare_ambiguous"]
    return DEMO_SNAPSHOTS["chennai_happy_path"]


def _serve_demo_fallback(query: str) -> FinalResponse:
    """Return a copy of the closest cached snapshot with a fresh TraceStep
    noting the fallback (Step 2), and log clearly server-side (Step 3) so
    testing/rehearsal can always tell "genuinely worked" from "fell back"."""
    snapshot = _closest_demo_snapshot(query)
    final = snapshot.model_copy(deep=True)

    now = _now_iso()
    final.reasoning_trace = list(final.reasoning_trace) + [
        TraceStep(
            agent_name="demo_fallback",
            input_summary=f"query='{query}'",
            output_summary="live Groq call failed, served cached snapshot",
            timestamp=now,
        )
    ]

    logger.warning(
        "[DEMO FALLBACK] live Groq call failed for query=%r — served cached snapshot",
        query[:200],
    )
    return final


@router.post("/query", response_model=EvidenceBundle)
async def query(request: QueryRequest, response: Response) -> EvidenceBundle:
    bundle, session_id, greeting_response, _api_failure, _turn = await run_planning_agent(
        request.query, request.session_id, request.user_location
    )
    response.headers["X-Session-Id"] = session_id
    if greeting_response is not None:
        response.headers["X-Assistant-Message"] = greeting_response
    return bundle


@router.post("/query/full", response_model=FinalResponse)
async def query_full(request: QueryRequest, response: Response) -> FinalResponse:
    """Full pipeline: Language Agent (input) + planning (LLM) + deterministic
    + reporting + visualization + Language Agent (output)."""
    # Phase 6.2: Language Agent, input side — every reasoning agent below
    # keeps operating on English text, unchanged, regardless of what
    # language the conversation is conducted in.
    english_query, effective_language = await _resolve_input_language(request.query, request.session_id)

    # If no API key configured (or placeholder), fall back to fixture-based demo so
    # frontend works offline and judge demo never depends on live APIs.
    # Case-insensitive and checks both hyphen/underscore placeholder spellings
    # — the previous case-sensitive, single-prefix check would silently
    # attempt a live call against an unusual placeholder value instead of
    # degrading to fixtures.
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    _normalized_groq_key = groq_key.lower()
    if not groq_key or _normalized_groq_key in ("your-groq-key-here", "your_groq_key_here") or _normalized_groq_key.startswith(("your-", "your_")):
        bundle = _fixture_for_query(english_query)
        bundle.trace = []  # reset trace for clean demo run
        # Same X-Session-Id contract as the live-Groq branch below and /query
        # (see this file's own docstring) — reuse the caller's session_id if
        # it's still valid, otherwise mint one, so a multi-turn conversation
        # started in offline/demo mode still has something to send back.
        session_id = request.session_id if request.session_id and get_session(request.session_id) else create_session()
        response.headers["X-Session-Id"] = session_id
        final = await asyncio.to_thread(
            _run_deterministic_pipeline, bundle, request.selected_destination, request.user_location
        )
        return await _apply_output_language(final, effective_language)

    bundle, session_id, greeting_response, api_failure_detected, turn = await run_planning_agent(
        english_query, request.session_id, request.user_location
    )
    response.headers["X-Session-Id"] = session_id
    # run_planning_agent already recorded this turn via its own internal
    # update_session call (without a language — it has no knowledge of the
    # Language Agent); patch the language onto that same turn object
    # directly (returned above) rather than touching planning_agent.py's
    # internals, appending a duplicate turn, or re-deriving "the last
    # turn" by session_id — which could race with a concurrent request
    # sharing the same session_id.
    set_last_turn_language(turn, effective_language)
    if greeting_response is not None:
        # Greeting/off-topic: marine/weather/risk were never invoked (all
        # None on the bundle, not degraded). Skip the deterministic
        # pipeline entirely too — there's no marine data to analyze — and
        # return the friendly response directly instead of a "data
        # unavailable" report.
        final = FinalResponse(
            answer_text=greeting_response,
            reasoning_trace=bundle.trace,
            map_payload=MapPayload(pins=[], overlays=[]),
        )
        return await _apply_output_language(final, effective_language)

    api_failed = api_failure_detected or _bundle_has_api_failure(bundle)
    # A cached snapshot could say "safe" for some other place and time, so
    # it's only served when live data could itself have supported a verdict.
    # When live weather breaches a hard limit (UNSAFE by rule) or lacks a
    # wave or wind reading (at most inconclusive by rule), the risk agent
    # has already set a rule-based verdict with no raw error text — see
    # safety_limits.py — so answer from the live evidence below instead.
    live_reasons = (
        weather_limit_breaches(bundle.weather) + missing_safety_readings(bundle.weather) if api_failed else []
    )
    if live_reasons:
        logger.warning(
            "[DEMO FALLBACK] skipped for query=%r — live data decides by rule: %s",
            english_query[:200], "; ".join(live_reasons),
        )
    elif api_failed:
        # run_planning_agent already flags this explicitly (it's the only
        # place that knows whether ITS OWN classify call hit a Groq failure
        # before marine/weather ever got a location to work with); the
        # bundle scan is a cheap extra safety net on top. Either way: a real
        # attempt was made and Groq itself hard-failed (rate limit, timeout,
        # API error) — not a normal "no location" degradation, which already
        # produces a clean partial report on its own. Serve the closest
        # genuine cached snapshot instead of a visible failure.
        final = _serve_demo_fallback(english_query)
        return await _apply_output_language(final, effective_language)

    final = await asyncio.to_thread(
            _run_deterministic_pipeline, bundle, request.selected_destination, request.user_location
        )
    return await _apply_output_language(final, effective_language)


@router.post("/query/demo", response_model=FinalResponse)
async def query_demo(request: QueryRequest) -> FinalResponse:
    """Deterministic-only demo endpoint — always uses fixtures, no LLM call.
    Still wrapped by the Language Agent so demo mode is multilingual too:
    input-side translation lets _fixture_for_query's English keyword
    matching (e.g. "mannar", "cyclone") work against a translated query."""
    english_query, effective_language = await _resolve_input_language(request.query, request.session_id)
    bundle = _fixture_for_query(english_query)
    bundle.trace = []
    final = await asyncio.to_thread(
            _run_deterministic_pipeline, bundle, request.selected_destination, request.user_location
        )
    return await _apply_output_language(final, effective_language)


@router.get("/fixtures")
async def fixtures() -> dict:
    return {
        "fixtures": [
            FIXTURE_1_HAPPY_PATH.model_dump(),
            FIXTURE_2_HAZARD_PATH.model_dump(),
            FIXTURE_3_PARTIAL_FAILURE.model_dump(),
            FIXTURE_4_RESTRICTED_ZONE.model_dump(),
        ]
    }
