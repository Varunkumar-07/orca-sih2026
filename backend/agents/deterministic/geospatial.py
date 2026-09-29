"""Geospatial Reasoning deterministic module.

Distance, nearest-zone lookup, and MPA boundary checks via Shapely /
GeoPandas patterns. (An EEZ boundary check was prototyped here and later
removed — it never had a real boundary cache to read from — so this module
covers MPA restricted zones only, not EEZ/maritime-jurisdiction boundaries.)

Restricted-area boundaries are loaded from a local cache of real WDPA
(World Database on Protected Areas) polygons fetched from the Protected
Planet API — see backend/scripts/fetch_mpa_boundaries.py. That cache is
read once per process and never fetched live in the request path, so
query latency and reliability are unaffected by Protected Planet's
uptime. If the cache is missing, empty, or fails to parse, this module
falls back to a small set of hardcoded boundaries (Gulf of Mannar / Palk
Bay) so restricted-zone checks degrade gracefully instead of going dark.

Two distances, kept separate on purpose:
- An area's *extent* decides PROHIBITED: the WDPA polygon itself, or —
  for a site WDPA records only as a Point — a circle of the site's reported
  area (WDPA's own convention for point records; flagged as approximate).
- MPA_PROXIMITY_KM around that extent is only a proximity warning. It used
  to be baked into the extent itself (a 15km buffer), which made every
  query from Mumbai "PROHIBITED" — the city is 4.5km outside Thane Creek.
Route planning adds its own, much smaller clearance (navigation_agent.py).

Never raises across the boundary.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, shape

from backend.schemas.contracts import EvidenceBundle, GeospatialResult, TraceStep
from backend.time_utils import now_iso as _iso_now

logger = logging.getLogger(__name__)

# Local cache written by backend/scripts/fetch_mpa_boundaries.py — resolved
# relative to this file so it's independent of the process's cwd.
MPA_CACHE_PATH = Path(__file__).resolve().parent / "data" / "mpa_boundaries.geojson"
# Natural Earth land clipped to the seas around India (~140KB), written by
# backend/scripts/build_land_mask.py — what route planning treats as land.
LAND_CACHE_PATH = Path(__file__).resolve().parent / "data" / "land_india.geojson"

# Within this distance of a protected area's extent (but not inside it) a
# query gets a proximity warning — never a ban.
MPA_PROXIMITY_KM = 15.0
# Flat approximation (1 deg latitude ~= 111km; longitude scaled by
# cos(lat)) — fine across India's 8-22N coastal range, not at high latitudes.
KM_PER_DEG_LAT = 111.0

# ---------------------------------------------------------------------------
# Hardcoded fallback — used only when the real WDPA cache above is missing,
# empty, or malformed. (lon, lat) order for Shapely — x=lon, y=lat.
# Gulf of Mannar Marine National Park — 21 islands between Rameswaram
# and Tuticorin. Approx bounding polygon that contains Fixture 4
# (9.05, 78.85) but excludes Chennai (13.08, 80.27).
# Coordinates chosen to match published extent ~8.2–9.6N, 78.0–79.6E.
# ---------------------------------------------------------------------------
_FALLBACK_RESTRICTED_AREAS: list[dict] = [
    {
        "name": "Gulf of Mannar Marine National Park",
        "polygon": Polygon(
            [
                (78.0, 8.2),
                (79.6, 8.2),
                (79.6, 9.6),
                (78.0, 9.6),
            ]
        ),
        "approximate": True,
    },
    # Example additional restricted area (Palk Bay portion sometimes restricted)
    # Keep narrow so it doesn't interfere with existing fixtures
    {
        "name": "Palk Bay Protected Area",
        "polygon": Polygon(
            [
                (79.0, 9.7),
                (80.0, 9.7),
                (80.0, 10.2),
                (79.0, 10.2),
            ]
        ),
        "approximate": True,
    },
]

# Lazily loaded, memoized for the lifetime of the process. None = not yet
# attempted; a list (possibly the fallback) once loading has been tried.
_restricted_areas_cache: list[dict] | None = None

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance via haversine (km). Uses numpy for vectorization."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(EARTH_RADIUS_KM * c)


def extract_pfz_center(
    zone: dict, log: logging.Logger, action: str, missing_id_label: str = "unknown"
):
    """Validate + extract (lat, lon, zone_id) from a pfz_zone's canonical
    {"zone_id": ..., "center": {"lat": ..., "lon": ...}} shape (see
    MarineDataResult.pfz_zones). Returns None — after an ERROR-level log
    naming the offending zone — on any malformed shape rather than raising,
    same convention every live-fetch path in this codebase follows.

    This exact validation was independently duplicated at 4 call sites
    (this module, visualization.py's pin/overlay builders, reporting.py's
    fallback) before being consolidated here — see the zone_id/KeyError
    incident this logging was hardened after, and
    tests/test_pfz_zone_shape_logging.py, which asserts on both the exact
    log text and on log records coming from each CALLER's own logger name
    (via caplog's per-logger filtering) — which is why `log` is the
    caller's own module logger, not this module's, and callers keep doing
    their own float()/exception handling on the returned raw values rather
    than this helper doing it for them.
    """
    center = zone.get("center")
    if not isinstance(center, dict):
        log.error(
            "pfz_zone missing expected canonical 'center' dict — %s. zone=%r",
            action,
            zone,
        )
        return None
    zlat, zlon = center.get("lat"), center.get("lon")
    zname = zone.get("zone_id", missing_id_label)
    if zlat is None or zlon is None:
        log.error(
            "pfz_zone %r has a 'center' dict but missing lat/lon — %s. center=%r",
            zname,
            action,
            center,
        )
        return None
    return zlat, zlon, zname


def _circle_km(lon: float, lat: float, radius_km: float) -> Polygon:
    """A radius_km circle around (lon, lat) in degree coordinates."""
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(lat))
    unit = Point(0.0, 0.0).buffer(1.0, quad_segs=16)
    return Polygon([(lon + x * radius_km / km_per_deg_lon, lat + y * radius_km / KM_PER_DEG_LAT) for x, y in unit.exterior.coords])


def _area_extent(name: str, geom, reported_area_km2: float | None) -> dict | None:
    """{"name", "polygon", "approximate", "reported_area_km2"} for one WDPA
    site. A polygon is used as-is. A Point has no boundary in WDPA, only a
    reported area, so its extent is a circle of that area — flagged
    approximate. A Point with no reported area has no usable extent (it
    can't make anything PROHIBITED) and is skipped with a warning."""
    if geom.geom_type in ("Point", "MultiPoint"):
        if not reported_area_km2 or reported_area_km2 <= 0:
            logger.warning("MPA '%s' is a bare point with no reported area — skipped (no extent to test against)", name)
            return None
        radius_km = math.sqrt(reported_area_km2 / math.pi)
        center = geom.centroid
        return {"name": name, "polygon": _circle_km(center.x, center.y, radius_km), "approximate": True, "reported_area_km2": reported_area_km2}
    return {"name": name, "polygon": geom, "approximate": False, "reported_area_km2": reported_area_km2}


def _load_mpa_cache() -> list[dict] | None:
    """Load restricted-area polygons from the local WDPA cache file.

    Returns None (triggering the hardcoded fallback) if the cache is
    missing, empty, or malformed. Never raises.
    """
    try:
        if not MPA_CACHE_PATH.exists():
            logger.warning(
                "MPA cache not found at %s — falling back to hardcoded restricted-area boundaries",
                MPA_CACHE_PATH,
            )
            return None

        raw = MPA_CACHE_PATH.read_text()
        if not raw.strip():
            logger.warning(
                "MPA cache at %s is empty — falling back to hardcoded restricted-area boundaries",
                MPA_CACHE_PATH,
            )
            return None

        data = json.loads(raw)
        features = data.get("features") or []

        areas: list[dict] = []
        for feature in features:
            geometry = feature.get("geometry")
            name = (feature.get("properties") or {}).get("name", "Unnamed MPA")
            if not geometry:
                continue
            try:
                area = _area_extent(name, shape(geometry), (feature.get("properties") or {}).get("reported_area_km2"))
            except Exception:
                logger.warning("Skipping malformed geometry for '%s' in MPA cache", name)
                continue
            if area is not None:
                areas.append(area)

        if not areas:
            logger.warning(
                "MPA cache at %s has no usable geometries — falling back to hardcoded restricted-area boundaries",
                MPA_CACHE_PATH,
            )
            return None

        logger.info("Loaded %d real MPA boundaries from cache (%s)", len(areas), MPA_CACHE_PATH)
        return areas

    except Exception as exc:
        logger.warning(
            "Failed to load MPA cache at %s (%s) — falling back to hardcoded restricted-area boundaries",
            MPA_CACHE_PATH,
            exc,
        )
        return None


def _get_restricted_areas() -> list[dict]:
    """Return the active restricted-area set, loading and memoizing it on first use."""
    global _restricted_areas_cache
    if _restricted_areas_cache is None:
        _restricted_areas_cache = _load_mpa_cache() or _FALLBACK_RESTRICTED_AREAS
    return _restricted_areas_cache


# Lazily loaded, memoized: a prepared shapely geometry, or False once
# loading has failed (so a missing file is logged once, not per route).
_land_cache = None


def get_land_mask():
    """Land as one prepared shapely geometry (lon/lat), loaded once per
    process from LAND_CACHE_PATH — or None if that file is missing or
    unreadable, logged as an error (routes can then not be checked against
    land, which callers must surface rather than hide)."""
    global _land_cache
    if _land_cache is None:
        try:
            import shapely

            features = json.loads(LAND_CACHE_PATH.read_text())["features"]
            land = shapely.union_all([shape(f["geometry"]) for f in features])
            shapely.prepare(land)
            _land_cache = land
            logger.info("Loaded land mask: %d polygons from %s", len(features), LAND_CACHE_PATH)
        except Exception as exc:
            logger.error("Land mask unavailable at %s (%s) — routes cannot be checked against land", LAND_CACHE_PATH, exc)
            _land_cache = False
    return _land_cache or None


def get_active_restricted_areas() -> list[dict]:
    """Public accessor for the restricted-area set currently in effect.

    Used by visualization.py to draw the same boundaries (real WDPA data,
    or the hardcoded fallback if the cache is unavailable) that this
    module used for the containment check — so the map overlay never
    drifts from what actually determined inside_restricted_area.
    """
    return _get_restricted_areas()


def _restricted_area_at(lat: float, lon: float) -> dict | None:
    """The restricted area whose extent contains (lat, lon), if any.
    Boundary-inclusive ('covers'); Shapely expects (x=lon, y=lat)."""
    pt = Point(lon, lat)
    for area in _get_restricted_areas():
        poly: Polygon = area["polygon"]
        try:
            if poly.covers(pt):
                return area
        except Exception:
            if poly.intersects(pt):
                return area
    return None


def restricted_area_mask(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Boolean (lat x lon) mask of the grid points that lie inside a
    restricted area's extent — the same boundary-inclusive test as
    _restricted_area_at (a point there gets a PROHIBITED verdict and route
    planning refuses it), vectorized over a whole grid. Areas whose bounds
    miss the grid are skipped before any per-point test."""
    import shapely

    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    if mask.size == 0:
        return mask
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_lo, lat_hi = float(np.min(lats)), float(np.max(lats))
    lon_lo, lon_hi = float(np.min(lons)), float(np.max(lons))
    for area in _get_restricted_areas():
        poly = area["polygon"]
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        if max_lon < lon_lo or min_lon > lon_hi or max_lat < lat_lo or min_lat > lat_hi:
            continue
        # intersects_xy is 'covers' for a point: boundary points count.
        mask |= shapely.intersects_xy(poly, lon_grid, lat_grid)
    return mask


def _check_restricted(lat: float, lon: float) -> tuple[bool, str | None]:
    """Return (inside, area_name) for restricted-area containment."""
    area = _restricted_area_at(lat, lon)
    return (True, area["name"]) if area is not None else (False, None)


def _distance_to_area_km(lat: float, lon: float, poly) -> float:
    """Approximate km from (lat, lon) to the nearest edge of poly: the
    nearest boundary point is found in degree space (fine at this range),
    then measured with haversine."""
    from shapely.ops import nearest_points

    nearest = nearest_points(poly, Point(lon, lat))[0]
    return haversine_km(lat, lon, nearest.y, nearest.x)


def _nearby_restricted_area(lat: float, lon: float) -> tuple[str, float] | None:
    """(name, km) of the closest restricted area within MPA_PROXIMITY_KM of
    (lat, lon) — for a point already known to be outside every extent."""
    best: tuple[str, float] | None = None
    for area in _get_restricted_areas():
        try:
            km = _distance_to_area_km(lat, lon, area["polygon"])
        except Exception:
            continue
        if km <= MPA_PROXIMITY_KM and (best is None or km < best[1]):
            best = (area["name"], km)
    return best


def run_geospatial(bundle: EvidenceBundle) -> GeospatialResult:
    """Distance to nearest PFZ, nearest-zone lookup, and restricted-area check.

    Appends a TraceStep to bundle.trace before returning.
    """
    nearest_zone_name: str | None = None
    distance_km: float | None = None
    inside_restricted = False
    restricted_area_name: str | None = None
    restricted_area_approximate = False
    near_name: str | None = None
    near_km: float | None = None
    input_summary = ""
    error_flag = False

    try:
        loc = bundle.query_location
        marine = bundle.marine

        # Input summary (truncate later)
        pfz_count = len(marine.pfz_zones) if marine and marine.pfz_zones else 0
        if loc is None:
            input_summary = f"location=None, pfz_zones={pfz_count}"
        else:
            input_summary = f"location=({loc.lat:.4f},{loc.lon:.4f}), pfz_zones={pfz_count}"

        # --- Restricted area check (depends only on query location) ---
        if loc is not None:
            area = _restricted_area_at(loc.lat, loc.lon)
            inside_restricted = area is not None
            restricted_area_name = area["name"] if area else None
            restricted_area_approximate = bool(area and area.get("approximate"))
            if area is None:
                nearby = _nearby_restricted_area(loc.lat, loc.lon)
                if nearby is not None:
                    near_name, near_km = nearby[0], round(nearby[1], 1)
        else:
            inside_restricted = False
            restricted_area_name = None

        # --- Nearest zone lookup ---
        if loc is None:
            # No location => cannot compute distance
            nearest_zone_name = None
            distance_km = None
        elif marine is None or not marine.pfz_zones:
            nearest_zone_name = None
            distance_km = None
        else:
            best_dist = math.inf
            best_name: str | None = None
            for zone in marine.pfz_zones:
                extracted = extract_pfz_center(zone, logger, "skipping")
                if extracted is None:
                    continue
                zlat, zlon, zname = extracted
                try:
                    d = haversine_km(loc.lat, loc.lon, float(zlat), float(zlon))
                except Exception:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_name = str(zname)

            if best_name is not None:
                nearest_zone_name = best_name
                distance_km = round(float(best_dist), 2)
            else:
                nearest_zone_name = None
                distance_km = None

    except Exception as exc:
        error_flag = True
        input_summary = input_summary or f"error capturing inputs: {exc}"
        # Return safe defaults on error
        nearest_zone_name = None
        distance_km = None
        inside_restricted = False
        restricted_area_name = None
        restricted_area_approximate = False
        near_name = near_km = None

    result = GeospatialResult(
        nearest_zone_name=nearest_zone_name,
        distance_km=distance_km,
        inside_restricted_area=inside_restricted,
        restricted_area_name=restricted_area_name,
        restricted_area_approximate=restricted_area_approximate,
        near_restricted_area_name=near_name,
        near_restricted_area_km=near_km,
    )

    # Write back to bundle
    try:
        bundle.geospatial = result
    except Exception:
        pass

    if error_flag:
        output_summary = f"error — inside_restricted={inside_restricted}, nearest={nearest_zone_name}"
    else:
        output_summary = (
            f"nearest={nearest_zone_name}, dist={distance_km}km, "
            f"inside_restricted={inside_restricted} ({restricted_area_name})"
        )
        if near_name:
            output_summary += f", near_restricted={near_name} ({near_km}km)"

    trace = TraceStep(
        agent_name="geospatial",
        input_summary=input_summary[:500],
        output_summary=output_summary[:500],
        timestamp=_iso_now(),
    )
    bundle.trace.append(trace)

    return result
