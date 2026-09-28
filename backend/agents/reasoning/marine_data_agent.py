"""
Marine Data Discovery Agent — Team Claude, Tier A (LLM reasoning agent).

Interprets what ocean data a query needs and returns a MarineDataResult.
Sea surface temperature is fetched live from Open-Meteo's free, keyless
marine API. Chlorophyll is fetched live from Copernicus Marine Service's
free (registration-required) ocean-colour product. Never raises across
the agent boundary — any failure degrades to status="error"; a missing
chlorophyll reading alone degrades to status="partial" (SST/PFZ zones are
still returned).

PFZ zone *locations*: chat and Zones Explorer share the same live source
of truth (Phase 4 of the Chat-PFZ Live-Data Integration plan) — neither
one is a separate, unrelated code path.
- Chat (this agent, run_marine_data_agent -> _live_pfz_candidate_zones):
  reads the same shared live cache Zones Explorer does (see
  backend/services/pfz_service.get_nearest_anchor_zones), resolving the
  query location to its nearest anchor region and recomputing distance_km
  from the real query point. Returns an empty list on any failure,
  timeout, or cold/unavailable cache — see _live_pfz_candidate_zones.
  There is no mock/sample fallback: a failed live lookup means no PFZ
  data is returned rather than fabricated data presented as real.
- Zones Explorer (GET /zones, via list_live_pfz_zones): real zones
  computed from live Copernicus SST + chlorophyll satellite grids — see
  backend/services/pfz_service.py for the front-detection method. An
  anchor that fails (credentials unset, a fetch failure, or Copernicus's
  real concurrency limits under load — see _MAX_CONCURRENT_PFZ_ANCHORS)
  simply contributes zero zones rather than a mock fallback.

Since chat now reads the SAME anchor-keyed cache Zones Explorer populates
(pfz_service.get_cached_zones), a chat query far from every anchor region
(sparse coastal coverage — 11 anchors, ~277-1166km apart from their
nearest neighbor) still resolves to the nearest one rather than erroring,
but is flagged via NearestAnchorZones.is_distant so the response can say
so honestly instead of presenting a 100km+-away zone as a normal nearby
result — see _live_pfz_candidate_zones and reporting.py's
_format_geospatial.
"""

import asyncio
import logging
import os

import copernicusmarine
import httpx
import numpy as np

from backend.agents.deterministic.geospatial import haversine_km
from backend.agents.reasoning._groq_client import call_groq_json
from backend.agents.reasoning._trace import record_trace
from backend.schemas.contracts import GeoPoint, MarineDataResult, TraceStep
from backend.services.pfz_service import (
    fetch_environmental_grid,
    find_pfz_candidates,
    get_nearest_anchor_zones,
)
from backend.time_utils import now_iso as _now_iso

logger = logging.getLogger("orca.marine_data")

_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
_HTTP_TIMEOUT = 6.0
# Global, daily, 4km, gap-filled (cloud gaps already interpolated) near-real-
# time chlorophyll-a product — covers Indian coastal waters, no separate
# regional product needed. Variable "CHL" is already in mg/m^3, matching
# MarineDataResult.chlorophyll_mg_m3 with no unit conversion.
_CHL_DATASET_ID = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"
# open_dataset() has no timeout parameter of its own — bounded externally
# via asyncio.wait_for so a slow/broken Copernicus auth server can't stall
# a request indefinitely. A genuinely successful call was observed live to
# take ~10s (their auth + lazy Zarr-store open, not just network latency),
# so this needs real headroom above that, not just above a typical request.
_CHL_TIMEOUT = 15.0
# One retry on top of the first attempt — Copernicus's auth server is
# intermittently flaky (see _fetch_live_chlorophyll below), and a second
# attempt often succeeds where the first hit a transient auth hiccup.
_CHL_MAX_ATTEMPTS = 2
# Despite the "gapfree" name, two real gaps were observed in production for
# Indian coastal points (e.g. right off Chennai): the newest day is often
# entirely unpopulated yet (near-real-time publication lag — the dataset's
# own metadata warns of this), and the single nearest grid cell to a
# near-shore query point can be permanently cloud/land-masked even while
# nearby open-water cells have real readings on every other day. Both are
# worked around below rather than surfacing as "unavailable".
_CHL_SPATIAL_RADIUS_DEG = 0.5
_CHL_LOOKBACK_DAYS = 5

_INTENT_SYSTEM_PROMPT = """You are a marine data intent classifier for a fisheries \
assistant. Given a user's query, classify which single ocean data parameter is \
being asked about. Respond with ONLY a JSON object, no other text, in the form:
{"parameter": "pfz" | "sst" | "chlorophyll" | "general"}

"pfz" = asking about fishing zones / where to fish / potential fishing zone advisories.
"sst" = asking specifically about sea surface temperature.
"chlorophyll" = asking specifically about chlorophyll concentration.
"general" = anything else ocean-data-related that doesn't fit the above."""


# Chat's live PFZ lookup budget (Phase 4). A warm cache (see
# pfz_service.get_cached_zones's startup pre-warming in main.py) returns
# near-instantly; a cold cache triggers a full recompute across all 11
# anchors that takes ~55s (bounded concurrency — see
# _MAX_CONCURRENT_PFZ_ANCHORS below), which is not an acceptable wait for
# a live chat response. Startup pre-warming is what's supposed to keep the
# cache warm in normal operation; this timeout is only the safety net for
# when it hasn't (a fresh deploy, pre-warm itself failing) or the live
# lookup genuinely hangs — either way chat degrades to the mock generator
# rather than blocking the user.
_LIVE_PFZ_LOOKUP_TIMEOUT = 10.0


async def _live_pfz_candidate_zones(location: GeoPoint) -> list[dict]:
    """Chat's PFZ zone source (Phase 4): the same shared live cache Zones
    Explorer/_route_/alerts/etc. already read (see
    pfz_service.get_nearest_anchor_zones). Returns an empty list on any
    failure, timeout, or cold/unavailable cache — no mock/sample fallback;
    a failed live lookup means no PFZ data rather than fabricated data
    presented as real.

    Each returned zone carries an "is_distant" flag (mirroring
    NearestAnchorZones.is_distant — see pfz_service.py) so downstream
    reporting can tell a genuinely nearby live match from a distant
    regional one.

    get_nearest_anchor_zones() returns zones shaped like GET /zones's own
    catalog entries ("id"/"name"/"coordinates" — see
    pfz_service.get_cached_zones), not the canonical
    MarineDataResult.pfz_zones shape ("zone_id"/"center") every other
    consumer here (geospatial.py, reporting.py, visualization.py, this
    module's own trace formatting below) expects — those are two
    different schemas for the same data, not a typo in one place. Remap
    here, once, at the boundary between the two, rather than teaching
    every downstream consumer both shapes.
    """
    try:
        nearest = await asyncio.wait_for(
            get_nearest_anchor_zones(location.lat, location.lon), timeout=_LIVE_PFZ_LOOKUP_TIMEOUT
        )
        if nearest:
            zones = [
                {
                    "zone_id": zone["id"],
                    "center": zone["coordinates"],
                    "distance_km": zone["distance_km"],
                    "sst_celsius": zone["sst_celsius"],
                    "chlorophyll_mg_m3": zone["chlorophyll_mg_m3"],
                    "advisory": zone["advisory"],
                    "is_distant": nearest.is_distant,
                }
                for zone in nearest
            ]
            # get_nearest_anchor_zones returns zones in front-detection score
            # order, not distance order — sort here so callers that assume
            # "nearest first" (e.g. this function's own caller,
            # run_marine_data_agent's trace summary) hold.
            zones.sort(key=lambda z: z["distance_km"])
            return zones
    except Exception as exc:
        logger.warning(
            "Live PFZ lookup failed for chat query (%.4f,%.4f): %s",
            location.lat, location.lon, exc,
        )
    return []


async def _fetch_live_sst(location: GeoPoint) -> float | None:
    """Live sea surface temperature at the query location via Open-Meteo's
    marine API. Returns None (rather than raising) on a coastal marine-grid
    gap OR a network/HTTP failure — SST unavailability alone shouldn't fail
    the whole marine lookup (run inside asyncio.gather alongside the PFZ
    zone search and chlorophyll fetch) when candidate zones and
    chlorophyll are still available.
    """
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(
                _MARINE_URL,
                params={
                    "latitude": location.lat,
                    "longitude": location.lon,
                    "current": "sea_surface_temperature",
                },
            )
        resp.raise_for_status()
        raw = resp.json()["current"].get("sea_surface_temperature")
        return float(raw) if raw is not None else None
    except Exception:
        return None


def _fetch_chlorophyll_sync(location: GeoPoint, username: str, password: str) -> float | None:
    """Blocking Copernicus Marine call — run via asyncio.to_thread so it
    doesn't block the event loop. Opens the dataset lazily (no file
    download), then reads the newest day (within _CHL_LOOKBACK_DAYS) that
    has a valid reading anywhere within _CHL_SPATIAL_RADIUS_DEG of the
    query point, taking the spatially-nearest valid cell on that day.

    A plain nearest-point-latest-day lookup (the original approach) returns
    NaN far more often than the "gapfree" dataset name suggests: the newest
    day is frequently unpopulated dataset-wide yet, and a near-shore query
    point can land on a permanently masked coastal pixel while open-water
    cells a few km away have real data on every other day.
    """
    ds = copernicusmarine.open_dataset(
        dataset_id=_CHL_DATASET_ID,
        variables=["CHL"],
        minimum_longitude=location.lon - _CHL_SPATIAL_RADIUS_DEG,
        maximum_longitude=location.lon + _CHL_SPATIAL_RADIUS_DEG,
        minimum_latitude=location.lat - _CHL_SPATIAL_RADIUS_DEG,
        maximum_latitude=location.lat + _CHL_SPATIAL_RADIUS_DEG,
        username=username,
        password=password,
    )
    lat_grid, lon_grid = np.meshgrid(ds["latitude"].values, ds["longitude"].values, indexing="ij")
    dist_deg = np.hypot(lat_grid - location.lat, lon_grid - location.lon)

    num_days = ds.sizes.get("time", 1)
    for offset in range(min(_CHL_LOOKBACK_DAYS, num_days)):
        day_values = ds["CHL"].isel(time=num_days - 1 - offset).values
        valid = ~np.isnan(day_values)
        if not valid.any():
            continue  # this whole day is unpopulated (e.g. publication lag) — try the day before
        nearest_idx = np.unravel_index(np.argmin(np.where(valid, dist_deg, np.inf)), dist_deg.shape)
        return float(day_values[nearest_idx])
    return None  # every day in the lookback window was empty near this point


async def _fetch_live_chlorophyll(location: GeoPoint) -> float | None:
    """Live chlorophyll-a concentration near the query location via
    Copernicus Marine Service (free, registration-required). Returns None
    (rather than raising) on missing credentials or any failure —
    chlorophyll is supplementary; a missing reading alone shouldn't fail
    the whole marine lookup when PFZ zones and SST are still available.
    """
    username = os.getenv("COPERNICUSMARINE_USERNAME")
    password = os.getenv("COPERNICUSMARINE_PASSWORD")
    if not username or not password:
        return None
    last_exc: Exception | None = None
    for attempt in range(1, _CHL_MAX_ATTEMPTS + 1):
        try:
            # open_dataset() has no built-in timeout and can hang well past a
            # normal request budget if Copernicus's auth server is slow/broken
            # (observed: their prod auth endpoint intermittently misbehaves) —
            # bound it explicitly so a broken upstream never stalls the whole
            # marine lookup. The background thread itself isn't killed, just
            # no longer awaited; it dies on its own once the call eventually
            # errors or returns.
            return await asyncio.wait_for(
                asyncio.to_thread(_fetch_chlorophyll_sync, location, username, password),
                timeout=_CHL_TIMEOUT,
            )
        except Exception as exc:
            last_exc = exc
    logger.warning(
        "Chlorophyll fetch failed for (%.2f,%.2f) after %d attempt(s): %s",
        location.lat,
        location.lon,
        _CHL_MAX_ATTEMPTS,
        last_exc,
    )
    return None


async def _classify_intent(query_text: str) -> str:
    parsed = await call_groq_json(_INTENT_SYSTEM_PROMPT, query_text, max_tokens=200)
    parameter = parsed.get("parameter", "general")
    if parameter not in {"pfz", "sst", "chlorophyll", "general"}:
        parameter = "general"
    return parameter


async def run_marine_data_agent(
    query_text: str,
    query_location: GeoPoint | None,
    trace: list[TraceStep],
) -> MarineDataResult:
    input_summary = f"query='{query_text}', location={query_location}"

    try:
        if query_location is None:
            raise ValueError("query_location is required for a PFZ lookup")

        # Intent is cosmetic here — only used in the trace label below,
        # never to branch the actual data fetch. Groq occasionally returns
        # an empty/invalid completion for this call (observed live: a 400
        # "Failed to validate JSON" with an empty failed_generation);
        # letting that exception propagate used to throw away perfectly
        # good SST/PFZ data over a classification hiccup that has zero
        # bearing on it, which also falsely tripped the Tier-2 demo
        # fallback. Degrade this one piece quietly instead — and run it
        # concurrently with the real fetches below (previously awaited
        # sequentially in front of them, paying a full extra Groq
        # round-trip of latency for zero functional benefit).
        async def _classify_intent_safe() -> str:
            try:
                return await _classify_intent(query_text)
            except Exception:
                return "general"

        # PFZ zone locations come from the shared live cache (Phase 4 — see
        # module docstring / _live_pfz_candidate_zones); no mock/sample
        # fallback — an empty result here means live data genuinely isn't
        # available, not that it was silently swapped for fabricated data.
        # SST and chlorophyll are both live, independently of PFZ zones.
        intent, candidate_zones, sst_celsius, chlorophyll_mg_m3 = await asyncio.gather(
            _classify_intent_safe(),
            _live_pfz_candidate_zones(query_location),
            _fetch_live_sst(query_location),
            _fetch_live_chlorophyll(query_location),
        )
        if not candidate_zones:
            raise ValueError("no live PFZ zones available for this location")

        result = MarineDataResult(
            status="ok" if sst_celsius is not None and chlorophyll_mg_m3 is not None else "partial",
            pfz_zones=candidate_zones,
            sst_celsius=sst_celsius,
            chlorophyll_mg_m3=chlorophyll_mg_m3,
            source_timestamp=_now_iso(),
        )
        nearest = candidate_zones[0]
        sst_display = f"{sst_celsius}°C" if sst_celsius is not None else "unavailable"
        chl_display = f"{chlorophyll_mg_m3}mg/m³" if chlorophyll_mg_m3 is not None else "unavailable"
        distant_note = ", distant anchor match" if nearest.get("is_distant") else ""
        output_summary = (
            f"intent={intent}, {len(candidate_zones)} candidate zones (live{distant_note}), "
            f"nearest={nearest['zone_id']} at {nearest['distance_km']}km, "
            f"sst={sst_display}, chlorophyll={chl_display}"
        )

    except Exception as exc:  # noqa: BLE001 - must never raise across the boundary
        result = MarineDataResult(
            status="error",
            pfz_zones=[],
            sst_celsius=None,
            chlorophyll_mg_m3=None,
            source_timestamp=_now_iso(),
            error_message=str(exc),
        )
        output_summary = f"error: {exc}"

    record_trace(trace, "marine_data_agent", input_summary, output_summary)
    return result


# Half-width (degrees) of the region box scanned around each anchor city
# for real front detection — matches the box size validated live in
# pfz_service.py's own development (Phase 1-3): wide enough to contain
# real SST/chlorophyll fronts, narrow enough to keep the grid fetch fast.
_PFZ_REGION_BOX_DEG = 0.75

# Max anchors processed concurrently (each opens 2 Copernicus connections —
# SST + chlorophyll — so this bounds simultaneous connections to 2x this).
# Measured live against the real service: 4 anchors (8 connections)
# completed successfully in ~20s; 6 anchors (12 connections) failed
# outright — every single one timed out and fell back to mock. Copernicus
# can't sustain naive full concurrency across all 11 anchors at once (22
# connections), so this caps it well under the observed failure point
# rather than riding right at the edge of it.
_MAX_CONCURRENT_PFZ_ANCHORS = 4


def _pfz_advisory(sst: float, chl: float, satellite_date: str) -> str:
    """Honest, live-data advisory text — states this is ORCA's own
    front-detection estimate from real satellite data, not an official
    INCOIS bulletin (which has no queryable API — see module docstring),
    and surfaces the satellite pass date so "live" doesn't silently imply
    real-time-this-second."""
    return (
        f"Potential fishing zone — SST/chlorophyll front detected in live satellite data "
        f"(pass date {satellite_date}). SST {sst:.1f}°C, chlorophyll {chl:.2f} mg/m³. "
        f"ORCA-computed estimate, not an official INCOIS advisory."
    )


async def _live_zones_for_anchor(location: GeoPoint, semaphore: asyncio.Semaphore) -> list[dict]:
    """Real front-detected zones for one anchor city. Returns an empty
    list — never mock/sample data — when live data isn't available for it:
    Copernicus credentials unset, a fetch failure/timeout, genuinely no
    front detected in that box that day, OR any unexpected exception in
    the scoring step (find_pfz_candidates can raise on a malformed grid;
    that must degrade this one anchor, not take down the other 10 via
    asyncio.gather).

    `semaphore` bounds the live Copernicus attempt — see
    _MAX_CONCURRENT_PFZ_ANCHORS for why naive full concurrency doesn't
    work.
    """
    try:
        async with semaphore:
            grid = await fetch_environmental_grid(
                location.lon - _PFZ_REGION_BOX_DEG,
                location.lon + _PFZ_REGION_BOX_DEG,
                location.lat - _PFZ_REGION_BOX_DEG,
                location.lat + _PFZ_REGION_BOX_DEG,
            )
        if grid is not None:
            candidates = find_pfz_candidates(grid, max_zones=3, min_separation_km=15.0)
            if candidates:
                return [
                    {
                        "zone_id": f"PFZ-{i + 1:03d}",
                        "center": {"lat": c.lat, "lon": c.lon},
                        "distance_km": round(haversine_km(location.lat, location.lon, c.lat, c.lon), 1),
                        "sst_celsius": round(c.sst_celsius, 1),
                        "chlorophyll_mg_m3": round(c.chlorophyll_mg_m3, 2),
                        "advisory": _pfz_advisory(c.sst_celsius, c.chlorophyll_mg_m3, grid.satellite_date),
                    }
                    for i, c in enumerate(candidates)
                ]
    except Exception as exc:
        logger.warning("PFZ live detection failed for anchor (%.2f,%.2f): %s", location.lat, location.lon, exc)
    return []


async def list_live_pfz_zones(anchors: dict[str, GeoPoint]) -> list[dict]:
    """Real front-detected PFZ zones per anchor city (see
    backend/services/pfz_service.py), one region box per anchor.
    Concurrency is capped at _MAX_CONCURRENT_PFZ_ANCHORS (measured against
    the real service — see its docstring) rather than firing all 11
    anchors at once, which overwhelms Copernicus and makes every single
    one fail. An anchor with no live data contributes zero zones — never
    mock/sample data (see _live_zones_for_anchor).

    Called from pfz_service.get_cached_zones() to build the one shared
    live-zone cache — GET /zones reads it directly, and (since Phase 4)
    so does chat via get_nearest_anchor_zones/_live_pfz_candidate_zones
    above, plus /route, /alerts, /weather/forecast, /export via their own
    calls into GET /zones's own catalog. Not called directly by any of
    them; this function only builds the shared cache they all read.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_PFZ_ANCHORS)
    results = await asyncio.gather(*(_live_zones_for_anchor(loc, semaphore) for loc in anchors.values()))
    zones: list[dict] = []
    for anchor_name, candidates in zip(anchors.keys(), results):
        for zone in candidates:
            zones.append({**zone, "near": anchor_name})
    return zones
