"""
PFZ (Potential Fishing Zone) front-detection service.

Identifies candidate fishing-zone locations by detecting real SST and
chlorophyll "fronts" (sharp gradients) in live satellite grids — the same
class of remote-sensing technique behind INCOIS's own PFZ advisories.
INCOIS itself has no queryable API (see marine_data_agent.py's module
docstring), which is why the chat/zones flow historically sampled live
values at a handful of heuristic points instead of computing real zone
*locations*. This service fetches real gridded data over a coastal region
so the locations themselves come from real data too.

fetch_environmental_grid / find_pfz_candidates are the front-detection
primitives (grid fetch + resolution alignment, gradient scoring, greedy
non-max suppression into distinct zones); marine_data_agent.py calls them
per anchor city. get_cached_zones() (below) is the shared, TTL-cached
result of that per-anchor work — GET /zones (main.py) reads it directly,
and pre-warms it in the background at app startup; chat reads it too via
get_nearest_anchor_zones() (also below), which resolves an arbitrary
query location to its nearest anchor's cached zones with distance_km
recomputed from the real query point. /route, /alerts, /weather/forecast,
and /export all resolve zone ids through GET /zones's own catalog, so
they inherit this same live data transitively.
"""
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import copernicusmarine
import numpy as np
import xarray as xr
from shapely.geometry import mapping as _shapely_mapping

from backend.agents.deterministic.analytics import (
    CHL_BLOOM_THRESHOLD,
    CHL_LOW_THRESHOLD,
    CHL_OPTIMAL_MAX,
    CHL_OPTIMAL_MIN,
    SST_ACCEPTABLE_MAX,
    SST_ACCEPTABLE_MIN,
    SST_OPTIMAL_MAX,
    SST_OPTIMAL_MIN,
)
from backend.agents.deterministic.geospatial import (
    EARTH_RADIUS_KM,
    get_active_restricted_areas,
    haversine_km,
)
from backend.error_utils import describe_exception
from backend.services import heavy_work

logger = logging.getLogger("orca.pfz")

# Same class of dataset as marine_data_agent.py's chlorophyll fetch —
# global, daily, gap-filled (L4) near-real-time analyses, so front
# detection isn't fighting cloud-gap noise on top of real fronts.
# ~5km native resolution (vs chlorophyll's ~4km) — see align step in
# fetch_environmental_grid() below.
_SST_DATASET_ID = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
_CHL_DATASET_ID = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"

# Same headroom rationale as marine_data_agent._CHL_TIMEOUT: open_dataset()
# has no timeout of its own, and a real successful grid fetch was observed
# to take ~9-11s (Copernicus auth + lazy Zarr-store open, not just network
# latency) — this bounds the sync call, not just the network round trip.
_GRID_FETCH_TIMEOUT = 25.0
# One retry on top of the first attempt — a transient Copernicus hiccup
# (auth server flakiness, a momentary connection cap) shouldn't drop this
# whole anchor's zones when a second attempt often succeeds.
_GRID_FETCH_MAX_ATTEMPTS = 2

# Despite the "gapfree" name, the newest day in this dataset is frequently
# entirely unpopulated (near-real-time publication lag — the dataset's own
# metadata warns of this; confirmed live: the newest day 100% NaN over a
# real coastal box, the day before only ~44% NaN). marine_data_agent's own
# single-point chlorophyll fetch already works around this (see its
# _CHL_LOOKBACK_DAYS) — this is the same workaround for the grid fetch.
_CHL_LOOKBACK_DAYS = 5
# Same publication-lag problem as chlorophyll (see _CHL_LOOKBACK_DAYS) can
# hit the SST dataset too — the newest day can be unpublished/all-NaN over
# a given box. SST fetch used to blindly take the newest day with no
# fallback, which silently starves front detection of any SST signal.
_SST_LOOKBACK_DAYS = 5


@dataclass
class EnvironmentalGrid:
    """SST + chlorophyll aligned onto the same lat/lon grid, ready for
    front detection. `sst_celsius` and `chlorophyll_mg_m3` are 2D arrays
    (lat x lon) with NaN over land/no-data cells, matching the 1D
    `lats`/`lons` coordinate arrays."""

    lats: np.ndarray
    lons: np.ndarray
    sst_celsius: np.ndarray
    chlorophyll_mg_m3: np.ndarray
    satellite_date: str


def _is_unset_or_placeholder(value: str) -> bool:
    """Same rule as routers/chat.py's GROQ_API_KEY check: an unedited
    .env.example value ("your-...-here") counts as unset, so it never
    reaches Copernicus as a real login attempt."""
    normalized = value.strip().lower()
    return not normalized or normalized.startswith(("your-", "your_"))


def copernicus_credentials() -> tuple[str, str] | None:
    """COPERNICUSMARINE_USERNAME/PASSWORD, or None when either is unset or
    still the .env.example placeholder. Shared by every Copernicus caller
    (this module, marine_data_agent.py, analytics_service.py)."""
    username = os.getenv("COPERNICUSMARINE_USERNAME", "")
    password = os.getenv("COPERNICUSMARINE_PASSWORD", "")
    if _is_unset_or_placeholder(username) or _is_unset_or_placeholder(password):
        return None
    return username.strip(), password.strip()


# Plain lazy (numpy-backed) arrays, no dask. open_dataset's default (-1)
# only turns dask on once a request spans >50 zarr chunks, which a single
# anchor box never does but SharedGridSources' all-anchor box does — and
# with dask on, reading one anchor's one day materializes whole ~80MB
# (985-day) SST blocks on dask's own thread pool (measured: >1.4GB peak).
# 0 is falsy, which is what copernicusmarine checks to skip dask chunking.
_NO_DASK_CHUNKING = 0


def _open_sst_dataset_sync(min_lon: float, max_lon: float, min_lat: float, max_lat: float, username: str, password: str) -> xr.Dataset:
    """Blocking, lazy (no data read yet) open of the SST dataset over a box."""
    return copernicusmarine.open_dataset(
        dataset_id=_SST_DATASET_ID,
        variables=["analysed_sst"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        username=username,
        password=password,
        chunk_size_limit=_NO_DASK_CHUNKING,
    )


def _open_chlorophyll_dataset_sync(min_lon: float, max_lon: float, min_lat: float, max_lat: float, username: str, password: str) -> xr.Dataset:
    """Blocking, lazy (no data read yet) open of the chlorophyll dataset over a box."""
    return copernicusmarine.open_dataset(
        dataset_id=_CHL_DATASET_ID,
        variables=["CHL"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        username=username,
        password=password,
        chunk_size_limit=_NO_DASK_CHUNKING,
    )


def _select_box(ds: xr.Dataset, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> xr.Dataset:
    """Cut one box out of an already-open (wider) dataset — the same cells
    open_dataset() itself would have returned for that box: its default
    "inside" coordinate selection is a plain inclusive .sel(slice(...)).
    Both datasets' coordinates are ascending."""
    return ds.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))


def _fetch_sst_grid_sync(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, username: str, password: str,
    source: xr.Dataset | None = None,
) -> xr.DataArray:
    """Blocking Copernicus Marine call — run via heavy_work.run_in_thread. Opens
    the dataset lazily (no file download), or — when `source` is an
    already-open wider dataset (see SharedGridSources) — just cuts this box
    out of it.

    Walks backward from the newest day (within _SST_LOOKBACK_DAYS) to the
    first one that isn't entirely NaN over this region — same publication-
    lag fallback as _fetch_chlorophyll_grid_sync, for the same reason:
    blindly taking the newest day silently starves front detection of any
    SST signal on a day it happens to be unpublished for this box.
    """
    if source is not None:
        ds = _select_box(source, min_lon, max_lon, min_lat, max_lat)
    else:
        ds = _open_sst_dataset_sync(min_lon, max_lon, min_lat, max_lat, username, password)
    num_days = ds.sizes.get("time", 1)
    for offset in range(min(_SST_LOOKBACK_DAYS, num_days)):
        da = ds["analysed_sst"].isel(time=num_days - 1 - offset) - 273.15  # Kelvin -> Celsius
        da.load()
        if np.isfinite(da.values).any():
            return da
    raise ValueError(f"SST grid empty for the last {_SST_LOOKBACK_DAYS} day(s) over this region")


def _fetch_chlorophyll_grid_sync(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, username: str, password: str,
    source: xr.Dataset | None = None,
) -> xr.DataArray:
    """Blocking Copernicus Marine call — run via heavy_work.run_in_thread. Same
    dataset as marine_data_agent._fetch_chlorophyll_sync, but keeps the
    full grid instead of collapsing to one nearest point. `source`: same
    as _fetch_sst_grid_sync's.

    Walks backward from the newest day (within _CHL_LOOKBACK_DAYS) to the
    first one that isn't entirely NaN over this region — the newest day is
    frequently unpopulated dataset-wide yet (see _CHL_LOOKBACK_DAYS).
    Blindly taking the newest day silently starves front detection of any
    chlorophyll signal at all, so every anchor looks like "no front found"
    even though the live fetch itself succeeded.
    """
    if source is not None:
        ds = _select_box(source, min_lon, max_lon, min_lat, max_lat)
    else:
        ds = _open_chlorophyll_dataset_sync(min_lon, max_lon, min_lat, max_lat, username, password)
    num_days = ds.sizes.get("time", 1)
    for offset in range(min(_CHL_LOOKBACK_DAYS, num_days)):
        da = ds["CHL"].isel(time=num_days - 1 - offset)
        da.load()
        if np.isfinite(da.values).any():
            return da
    raise ValueError(f"chlorophyll grid empty for the last {_CHL_LOOKBACK_DAYS} day(s) over this region")


class SharedGridSources:
    """SST + chlorophyll datasets opened once over a box covering every
    anchor of one zone refresh, so each anchor just slices its own box out
    of them instead of doing its own open_dataset().

    Memory, not speed, is why this exists: every open_dataset() call builds
    its own Copernicus auth session plus three fresh boto3 S3 clients
    (~12MB each), so the old one-open-per-anchor-per-variable approach (22
    opens per refresh, up to 8 in flight) pushed the process past Render
    free tier's 512MB and crash-looped it. The datasets are lazy — only the
    chunks each anchor's box touches are ever read — and are dropped with
    this object when the refresh ends.

    Opened on first use (not in __init__), so nothing touches the network
    unless a real grid fetch actually runs. If an open fails, that
    variable's source stays None and each anchor falls back to opening its
    own box, exactly as before.
    """

    def __init__(self, min_lon: float, max_lon: float, min_lat: float, max_lat: float):
        self._box = (min_lon, max_lon, min_lat, max_lat)
        self._lock = asyncio.Lock()
        self._opened = False
        self.sst: xr.Dataset | None = None
        self.chl: xr.Dataset | None = None

    async def get(self, username: str, password: str) -> tuple[xr.Dataset | None, xr.Dataset | None]:
        async with self._lock:
            if not self._opened:
                self._opened = True
                self.sst, self.chl = await asyncio.gather(
                    self._open(_open_sst_dataset_sync, username, password),
                    self._open(_open_chlorophyll_dataset_sync, username, password),
                )
        return self.sst, self.chl

    async def _open(self, opener, username: str, password: str) -> xr.Dataset | None:
        last_exc: Exception | None = None
        for _attempt in range(_GRID_FETCH_MAX_ATTEMPTS):
            try:
                return await asyncio.wait_for(
                    heavy_work.run_in_thread(opener, *self._box, username, password), timeout=_GRID_FETCH_TIMEOUT
                )
            except Exception as exc:
                last_exc = exc
        logger.warning(
            "PFZ shared dataset open failed (%s), falling back to per-anchor opens: %s",
            opener.__name__, describe_exception(last_exc, timeout=_GRID_FETCH_TIMEOUT),
        )
        return None


async def fetch_environmental_grid(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, sources: SharedGridSources | None = None
) -> EnvironmentalGrid | None:
    """Fetch real SST + chlorophyll grids over a region and align them
    onto one common lat/lon grid. Returns None if credentials are missing
    or either fetch fails/times out — callers must degrade gracefully per
    region (a later phase), not crash the whole PFZ list over one bad box.
    `sources`, when given, supplies already-open wider datasets to slice
    this box from (see SharedGridSources).
    """
    creds = copernicus_credentials()
    if creds is None:
        return None
    username, password = creds

    sst_source = chl_source = None
    if sources is not None:
        sst_source, chl_source = await sources.get(username, password)

    sst_da = chl_da = None
    last_exc: Exception | None = None
    for attempt in range(1, _GRID_FETCH_MAX_ATTEMPTS + 1):
        try:
            sst_da, chl_da = await asyncio.gather(
                asyncio.wait_for(
                    heavy_work.run_in_thread(_fetch_sst_grid_sync, min_lon, max_lon, min_lat, max_lat, username, password, sst_source),
                    timeout=_GRID_FETCH_TIMEOUT,
                ),
                asyncio.wait_for(
                    heavy_work.run_in_thread(_fetch_chlorophyll_grid_sync, min_lon, max_lon, min_lat, max_lat, username, password, chl_source),
                    timeout=_GRID_FETCH_TIMEOUT,
                ),
            )
            break
        except Exception as exc:
            last_exc = exc
    if sst_da is None or chl_da is None:
        logger.warning(
            "PFZ grid fetch failed for box (%.2f,%.2f)-(%.2f,%.2f) after %d attempt(s): %s",
            min_lon, min_lat, max_lon, max_lat, _GRID_FETCH_MAX_ATTEMPTS,
            describe_exception(last_exc, timeout=_GRID_FETCH_TIMEOUT),
        )
        return None

    # Chlorophyll is the finer-resolution grid (~4km vs SST's ~5km) —
    # interpolate SST onto it rather than downsampling chlorophyll, so
    # front detection keeps the higher-detail grid's precision.
    sst_aligned = sst_da.interp(latitude=chl_da.latitude, longitude=chl_da.longitude)

    satellite_date = str(chl_da.time.values)[:10]

    return EnvironmentalGrid(
        lats=chl_da.latitude.values,
        lons=chl_da.longitude.values,
        sst_celsius=sst_aligned.values,
        chlorophyll_mg_m3=chl_da.values,
        satellite_date=satellite_date,
    )


@dataclass
class PFZCandidate:
    """One detected candidate zone — a real grid cell, not a heuristic
    offset point. `score` is relative (0-1ish), only meaningful for
    ranking candidates within the same region/call, not across regions."""

    lat: float
    lon: float
    sst_celsius: float
    chlorophyll_mg_m3: float
    score: float


def _sst_favorability(value: np.ndarray) -> np.ndarray:
    """1.0 inside the optimal band, 0.5 inside the wider acceptable band,
    0.0 outside both (or NaN/land) — reuses analytics.py's existing
    favorable-SST thresholds instead of inventing new ones."""
    out = np.zeros_like(value, dtype=float)
    out[(value >= SST_ACCEPTABLE_MIN) & (value <= SST_ACCEPTABLE_MAX)] = 0.5
    out[(value >= SST_OPTIMAL_MIN) & (value <= SST_OPTIMAL_MAX)] = 1.0
    return out


def _chl_favorability(value: np.ndarray) -> np.ndarray:
    """Same idea as _sst_favorability but for chlorophyll's own thresholds
    (a wider "reasonable" band bounded by LOW/BLOOM, an "optimal" band
    inside that) — both already used elsewhere in analytics.py."""
    out = np.zeros_like(value, dtype=float)
    out[(value >= CHL_LOW_THRESHOLD) & (value <= CHL_BLOOM_THRESHOLD)] = 0.5
    out[(value >= CHL_OPTIMAL_MIN) & (value <= CHL_OPTIMAL_MAX)] = 1.0
    return out


def _normalize(a: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1] so SST-front-strength (°C/km, small
    numbers) and chlorophyll-front-strength (mg/m³/km, can span orders of
    magnitude) are comparable before combining — otherwise chlorophyll's
    larger raw units would dominate the combined score regardless of
    which signal is actually stronger relative to its own region."""
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _grid_distance_km(lats: np.ndarray, lons: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    """Vectorized haversine distance from one point to every grid cell.
    geospatial.haversine_km is scalar-only (it does `float(...)` on the
    result), which can't take an array target — this is the same formula,
    just kept elementwise for the suppression mask below."""
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (center_lat, center_lon, lat_grid, lon_grid))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def find_pfz_candidates(grid: EnvironmentalGrid, max_zones: int = 3, min_separation_km: float = 15.0) -> list[PFZCandidate]:
    """Detect real candidate fishing zones from a real SST+chlorophyll
    grid: score every cell by (SST front strength) x (chlorophyll front
    strength) x (SST favorability) x (chlorophyll favorability), so a cell
    only ranks highly if a real front AND biologically favorable absolute
    conditions co-occur there — the same two-signal logic INCOIS's own PFZ
    methodology is built on, just computed here instead of consumed from
    their (nonexistent) API.

    Greedy iterative non-max suppression: take the single highest-scoring
    cell, record it, zero out every cell within min_separation_km (so a
    single strong front doesn't produce 3 near-duplicate picks along its
    length), repeat until max_zones or nothing scores above zero.

    NaN (land / no satellite data) cells are never selected — nanargmax
    skips them by construction — which also means grid cells adjacent to
    land carry NaN gradients (central-difference touches its neighbor) and
    naturally get excluded too, without needing a separate elevation/land
    check.
    """
    if grid.sst_celsius.shape != grid.chlorophyll_mg_m3.shape:
        raise ValueError("sst/chlorophyll grids must already be aligned to the same shape")

    lat_spacing_km = abs(float(grid.lats[1] - grid.lats[0])) * 111.0 if len(grid.lats) > 1 else 1.0
    mean_lat_rad = np.radians(float(np.mean(grid.lats)))
    lon_spacing_km = (
        abs(float(grid.lons[1] - grid.lons[0])) * 111.0 * max(np.cos(mean_lat_rad), 0.01) if len(grid.lons) > 1 else 1.0
    )

    sst_grad_lat, sst_grad_lon = np.gradient(grid.sst_celsius, lat_spacing_km, lon_spacing_km)
    chl_grad_lat, chl_grad_lon = np.gradient(grid.chlorophyll_mg_m3, lat_spacing_km, lon_spacing_km)
    sst_front = np.hypot(sst_grad_lat, sst_grad_lon)
    chl_front = np.hypot(chl_grad_lat, chl_grad_lon)

    score = _normalize(sst_front) * _normalize(chl_front) * _sst_favorability(grid.sst_celsius) * _chl_favorability(grid.chlorophyll_mg_m3)
    score = np.where(np.isnan(grid.sst_celsius) | np.isnan(grid.chlorophyll_mg_m3), np.nan, score)

    candidates: list[PFZCandidate] = []
    working = score.copy()
    for _ in range(max_zones):
        if not np.any(np.isfinite(working)) or np.nanmax(working) <= 0:
            break
        row, col = np.unravel_index(int(np.nanargmax(working)), working.shape)
        lat, lon = float(grid.lats[row]), float(grid.lons[col])
        candidates.append(
            PFZCandidate(
                lat=lat,
                lon=lon,
                sst_celsius=float(grid.sst_celsius[row, col]),
                chlorophyll_mg_m3=float(grid.chlorophyll_mg_m3[row, col]),
                score=float(working[row, col]),
            )
        )
        working[_grid_distance_km(grid.lats, grid.lons, lat, lon) <= min_separation_km] = np.nan

    return candidates


# --- Shared live-zone cache (Phase 1: Chat-PFZ Live-Data Integration) ---
#
# Extracted out of main.py's private global so GET /zones and (starting
# Phase 4) the chat flow read the same cached live data instead of each
# maintaining — or in chat's case, never populating — their own copy.
# Behavior/return shape is unchanged from the original main.py version:
# still {"zones": [...], "generated_at": ...}, same TTL, same anchors.

# Phase 2 (Zones Explorer): a fixed spread of major coastal anchor points —
# reused directly from planning_agent's own _KNOWN_LOCATIONS gazetteer, not
# new coordinates — used to generate a location-independent, browsable PFZ
# catalog for GET /zones (chat's PFZ generation is otherwise always relative
# to a single query location).
_ZONE_EXPLORER_ANCHORS = [
    "kandla", "mumbai", "goa", "mangaluru", "kochi", "chennai",
    "gulf of mannar", "visakhapatnam", "paradip", "kolkata", "port blair",
]

# TTL cache, not a process-lifetime one: unlike the old mock catalog (pure
# function of the fixed anchor list, safe to compute once ever), live PFZ
# detection reads real satellite data that actually changes — but only
# about once a day (see this module's own ~1-day satellite-pass lag), and
# a full recompute across all 11 anchors takes ~55s (bounded concurrency,
# see marine_data_agent._MAX_CONCURRENT_PFZ_ANCHORS). A few hours between
# refreshes stays well ahead of the data's own freshness and avoids
# hammering Copernicus on every request.
_ZONES_CACHE_TTL_SECONDS = 3 * 60 * 60
# If Copernicus is briefly overloaded right when a refresh happens (e.g.
# too many concurrent anchor connections timing out at once), most/all
# anchors come back with zero zones for that refresh. Caching that
# degraded (mostly-empty) snapshot for the full 3-hour TTL would mean
# serving a near-empty zone list for hours after Copernicus recovers.
# Instead, a refresh where a significant fraction of anchors returned no
# zones is cached for a much shorter TTL, so the next refresh is retried
# soon rather than being stuck.
_ZONES_CACHE_DEGRADED_TTL_SECONDS = 5 * 60
_ZONES_CACHE_DEGRADED_EMPTY_ANCHOR_FRACTION = 0.25
_zones_cache: dict | None = None
_zones_cache_at: datetime | None = None
_zones_cache_ttl_seconds: float = _ZONES_CACHE_TTL_SECONDS

# Single-flight lock (audit backlog item, priority #2 after the
# shape-mismatch findings): without this, N requests arriving concurrently
# right when the TTL expires each independently see a stale/absent cache
# and each kick off their own full ~55s, 11-anchor Copernicus recompute —
# exactly the "6+ anchors = every single one times out and returns zero
# zones" failure mode _MAX_CONCURRENT_PFZ_ANCHORS was already sized around
# for a SINGLE recompute, not several stacked on top of each other. A
# plain asyncio.Lock (not per-key, there's only one cache here) makes every
# concurrent caller past the fast path wait for the SAME in-flight
# recompute instead of starting their own; the cheapest way to demo-proof
# a burst of judges hitting the app right after a TTL rollover.
_zones_cache_lock = asyncio.Lock()

# Copernicus reads a zone refresh may have running at once (shared dataset
# opens, per-anchor grid slices, any per-anchor fallback opens), counting
# reads whose caller already timed out until they actually finish — see
# heavy_work.exclusive for why. Measured in a 512MB Linux container: at
# 0.1 CPU (Render free tier) a refresh peaked at 340MB anon with 2 vs
# 371MB with 4; at 2 CPUs, 2 made the refresh ~4s slower (17.6s -> 21.8s).
_MAX_REFRESH_THREADS = 2


async def get_cached_zones() -> dict:
    """Shared accessor for the live PFZ + restricted-area zone catalog —
    the same data GET /zones has always returned (combines real
    front-detected PFZ zones per anchor city — an anchor with no live data
    contributes zero zones, never mock/sample data — with the current
    restricted-area boundaries). Serves the cached copy when
    still fresh (see _ZONES_CACHE_TTL_SECONDS), otherwise recomputes and
    caches. Single source of truth: /route, /alerts, /weather/forecast,
    and /export all resolve zone ids through GET /zones already, so they
    pick this up automatically; the chat flow reads it too starting
    Phase 4.

    Single-flight: concurrent callers that all miss the fast-path cache
    check serialize on _zones_cache_lock rather than each recomputing
    independently. The lock is re-checked-after-acquire (double-checked
    locking) so only the first caller through actually pays for the
    recompute — everyone else who was waiting on the lock gets the
    now-fresh cache the first caller just populated, without redoing the
    work themselves.
    """
    global _zones_cache, _zones_cache_at, _zones_cache_ttl_seconds

    def _cache_is_fresh(now: datetime) -> bool:
        return (
            _zones_cache is not None
            and _zones_cache_at is not None
            and (now - _zones_cache_at).total_seconds() < _zones_cache_ttl_seconds
        )

    if _cache_is_fresh(datetime.now(timezone.utc)):
        return _zones_cache

    async with _zones_cache_lock:
        # Re-check: another caller may have already refreshed the cache
        # while this one was waiting for the lock — that caller's result
        # is exactly as fresh as one this call would produce itself.
        now = datetime.now(timezone.utc)
        if _cache_is_fresh(now):
            return _zones_cache

        # Local imports to avoid a circular import at module load time:
        # marine_data_agent imports fetch_environmental_grid/find_pfz_candidates
        # from this module, and planning_agent is only needed for its
        # _KNOWN_LOCATIONS gazetteer — neither is needed until this function
        # actually runs.
        from backend.agents.reasoning.marine_data_agent import list_live_pfz_zones
        from backend.agents.reasoning.planning_agent import _KNOWN_LOCATIONS

        anchors = {name: _KNOWN_LOCATIONS[name] for name in _ZONE_EXPLORER_ANCHORS if name in _KNOWN_LOCATIONS}
        # The whole refresh is one heavy operation: it waits for any
        # in-flight Copernicus read to finish, and no read starts until
        # it (and every grid-read thread it started) is done — see
        # heavy_work.py for why (Render free tier's 512MB).
        async with heavy_work.exclusive("PFZ zone refresh", max_threads=_MAX_REFRESH_THREADS):
            pfz_candidates = await list_live_pfz_zones(anchors)

        combined: list[dict] = []
        for zone in pfz_candidates:
            near = zone.get("near", "")
            zone_id = zone.get("zone_id", "PFZ")
            combined.append(
                {
                    "id": f"{near.upper().replace(' ', '-')}-{zone_id}",
                    # A bare zone_id (e.g. "PFZ-1") isn't a human-readable
                    # name on its own — these are ORCA's own algorithmically
                    # detected zones, not officially named places, so pair
                    # it with the anchor it's near for a readable label.
                    "name": f"{near.title()} — {zone_id}" if near else zone_id,
                    "type": "pfz",
                    "near": near.title(),
                    "coordinates": zone.get("center"),
                    "distance_km": zone.get("distance_km"),
                    "sst_celsius": zone.get("sst_celsius"),
                    "chlorophyll_mg_m3": zone.get("chlorophyll_mg_m3"),
                    "advisory": zone.get("advisory"),
                }
            )

        for area in get_active_restricted_areas():
            name = area["name"]
            try:
                geometry = _shapely_mapping(area["polygon"])
            except Exception:
                continue
            combined.append(
                {
                    "id": f"restricted-{name.lower().replace(' ', '-')}",
                    "name": name,
                    "type": "restricted",
                    "geometry": geometry,
                }
            )

        pfz_zones = [z for z in combined if z.get("type") == "pfz"]
        anchors_with_zones = {str(z.get("near", "")).lower() for z in pfz_zones}
        empty_anchor_count = sum(1 for name in anchors if name.lower() not in anchors_with_zones)
        empty_anchor_fraction = (empty_anchor_count / len(anchors)) if anchors else 0.0
        if empty_anchor_fraction > _ZONES_CACHE_DEGRADED_EMPTY_ANCHOR_FRACTION:
            logger.warning(
                "PFZ zone refresh degraded: %d/%d anchors returned no zones (%.0f%%) — caching for only %ds instead of %ds",
                empty_anchor_count, len(anchors), empty_anchor_fraction * 100, _ZONES_CACHE_DEGRADED_TTL_SECONDS, _ZONES_CACHE_TTL_SECONDS,
            )
            _zones_cache_ttl_seconds = _ZONES_CACHE_DEGRADED_TTL_SECONDS
        else:
            _zones_cache_ttl_seconds = _ZONES_CACHE_TTL_SECONDS

        _zones_cache = {
            "zones": combined,
            "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _zones_cache_at = now
        return _zones_cache


# --- Nearest-anchor lookup (Phase 2: Chat-PFZ Live-Data Integration) ---
#
# Chat queries arrive with an arbitrary location, not one of the fixed
# anchor regions get_cached_zones() is keyed on. This resolves an
# arbitrary lat/lon to its nearest anchor's cached live zones, with
# distance_km recomputed from the real query point — the zone itself is
# still a cached regional result (only recomputed on the anchor's own TTL,
# see get_cached_zones), but the displayed distance must reflect where the
# user actually asked from, not the anchor city's coordinates.

# Phase 3: distance honesty flag, chosen over anchor densification (a
# denser anchor grid would mean more ~5s-per-anchor cold-cache computation
# and more Copernicus load for the same underlying sparse-coverage
# problem — see the Phase 3 scoping note). 75km is chosen, not just picked
# from the middle of the suggested 50-75km range, because it lines up with
# something concrete: _PFZ_REGION_BOX_DEG = 0.75 degrees is the actual
# half-width of the region box front-detection scans around each anchor
# (see _live_zones_for_anchor in marine_data_agent.py), which is
# ~83km of latitude. A query within ~75km of the resolved anchor is still
# plausibly within (or just outside the edge of) the very box that
# produced its zones — the "this reflects roughly where you are" claim
# stays honest. Past that, the returned zone is real, live data, but for a
# genuinely different stretch of coast; that also roughly matches a
# realistic day-trip radius for small/medium coastal fishing vessels, the
# audience this data is for. Given anchors themselves sit ~277-1166km
# apart (median ~440km — see Phase 3 scoping), most queries between
# anchors WILL cross this threshold; that's the sparse-grid reality this
# flag exists to surface honestly, not a sign the threshold is miscalibrated.
_NEARBY_ANCHOR_THRESHOLD_KM = 75.0


class NearestAnchorZones(list):
    """list[dict] of zones for the resolved nearest anchor — behaves
    exactly like the plain list Phase 2 returned (len(), indexing,
    iteration all work unchanged for existing callers) — plus an additive
    `is_distant` attribute alongside it: True when the resolved anchor is
    farther than _NEARBY_ANCHOR_THRESHOLD_KM from the query point. Kept as
    an attribute on the list itself (not a new key merged into each zone
    dict) so the zone dicts stay byte-for-byte identical to what
    get_cached_zones() produces, aside from the recomputed distance_km."""

    def __init__(self, zones, is_distant: bool):
        super().__init__(zones)
        self.is_distant = is_distant


async def get_nearest_anchor_zones(lat: float, lon: float) -> NearestAnchorZones:
    """Nearest-anchor PFZ zones for an arbitrary chat query location.

    Finds the closest of _ZONE_EXPLORER_ANCHORS to (lat, lon) by haversine
    distance, retrieves that anchor's cached live zones via
    get_cached_zones(), and returns them with distance_km recomputed from
    (lat, lon) instead of the anchor's own coordinates. Every other zone
    field (zone_id/name, coordinates, sst_celsius, chlorophyll_mg_m3,
    advisory, etc.) is passed through unchanged — only distance_km differs
    from what get_cached_zones() reports for that anchor.

    Restricted-area entries (type="restricted") are excluded — those are
    global boundaries, not per-anchor PFZ results, and have no point
    coordinates to recompute a distance from.

    Always resolves to *some* anchor (the nearest one, however far), even
    for a query location far from every anchor — never raises or returns
    an empty list because nothing was "close enough"; that judgment is
    exactly what the returned `.is_distant` flag now surfaces (Phase 3) —
    still never withheld or emptied out here, just honestly labeled. Using
    that flag to change what chat actually tells the user is Phase 4, not
    this function's job.
    """
    from backend.agents.reasoning.planning_agent import _KNOWN_LOCATIONS

    nearest_anchor_name: str | None = None
    nearest_anchor_distance: float | None = None
    for name in _ZONE_EXPLORER_ANCHORS:
        anchor_point = _KNOWN_LOCATIONS.get(name)
        if anchor_point is None:
            continue
        distance = haversine_km(lat, lon, anchor_point.lat, anchor_point.lon)
        if nearest_anchor_distance is None or distance < nearest_anchor_distance:
            nearest_anchor_distance = distance
            nearest_anchor_name = name

    if nearest_anchor_name is None:
        return NearestAnchorZones([], is_distant=False)

    catalog = await get_cached_zones()
    anchor_label = nearest_anchor_name.title()

    nearest_zones: list[dict] = []
    for zone in catalog["zones"]:
        if zone.get("type") != "pfz" or zone.get("near") != anchor_label:
            continue
        coordinates = zone.get("coordinates")
        recomputed = dict(zone)
        if coordinates is not None:
            recomputed["distance_km"] = round(haversine_km(lat, lon, coordinates["lat"], coordinates["lon"]), 1)
        nearest_zones.append(recomputed)

    is_distant = nearest_anchor_distance is not None and nearest_anchor_distance > _NEARBY_ANCHOR_THRESHOLD_KM

    return NearestAnchorZones(nearest_zones, is_distant=is_distant)
