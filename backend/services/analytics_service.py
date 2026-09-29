"""
Historical analytics service — Phase 6 (Analytics Dashboard) + variable-set
expansion, backing GET /analytics/historical and GET /export.

Four data sources, each confirmed directly against the live API/SDK before
writing this (see the phase report for the raw checks):

1. Open-Meteo marine archive (marine-api.open-meteo.com/v1/marine), `daily`
   params — sea surface temperature, wave height/direction/period, and
   swell wave height/direction/period all have real daily aggregates.
   Ocean current velocity/direction do NOT (requesting a daily aggregate
   for either is rejected as an invalid variable name) — only `hourly`
   data exists, so this module fetches the hourly series and derives one
   paired (velocity, direction) daily reading itself, taken at the hour of
   that day's peak velocity — real hourly data, not fabricated, and not a
   circular-mean of compass bearings (which would be meaningless).

2. Open-Meteo weather archive (archive-api.open-meteo.com/v1/archive),
   `daily` params — air temperature, wind speed/direction/gusts,
   precipitation, cloud cover, sea-level pressure, and humidity. This is a
   SEPARATE product from the marine one; the marine API's own "wind"
   fields (wind_wave_*) describe wind-driven wave statistics, not the
   actual 10m wind this module needs.

   Visibility and UV index were investigated and dropped: the archive
   endpoint accepts both as parameter names but returns 100% null values
   across every date range tested (confirmed with both recent and 2020
   dates) — a real archive-vs-forecast coverage gap, not fetched here.

   Tide height/times were investigated and dropped entirely: no reliable,
   free, India-covering tide-prediction source exists (see the phase
   report — Open-Meteo's sea_level_height_msl is a coarse ocean-model
   proxy, not a harmonic tide prediction; WorldTides needs a paid key;
   NOAA CO-OPS is confirmed US-only; INCOIS has no public API). Shipping a
   derived "tide time" here would be exactly the misleading-number risk
   this project avoids everywhere else (see weather_agent.py's own
   tide_info, which is left None rather than faked).

3. Copernicus Marine Service, three separate datasets, each following the
   exact graceful-degradation pattern marine_data_agent.py already
   established for chlorophyll (Phase 3): missing/slow data degrades that
   ONE variable's status to "partial" or "error", never raises, never
   fails the rest of the response.
   - Chlorophyll-a: same dataset marine_data_agent.py already uses
     (cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D) — a
     near-real-time satellite product with only a rolling ~2-3 week
     window of real history (confirmed live), so older dates in a longer
     range correctly come back as gaps, reflected in this variable's own
     `status`, not a fetch failure.
   - Salinity: cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m ("so"), a daily
     physics analysis-forecast product — has a depth dimension (this
     module reads the shallowest available level as sea-surface
     salinity).
   - Dissolved oxygen: cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m ("o2"), a
     coarser (0.25°) biogeochemistry product — confirmed it can return
     NaN for points very close to shore (the coarse grid cell rounds to
     "land") even though the same point works fine for salinity's finer
     0.083° grid; offshore points return real values. This is exactly
     what the per-field status is for.

4. Computed, no API at all: sunrise/sunset (a standard NOAA solar-position
   formula, verified against known Chennai sunrise/sunset times) and moon
   phase (a standard synodic-month approximation, accurate to about a
   day). All ORCA zones are Indian coastal waters, so times are reported
   in a fixed IST (UTC+5:30) offset rather than doing real per-point
   timezone lookup, which would be over-engineering for this app's actual
   scope.

Direction is circular data (wave_direction_deg, swell_wave_direction_deg,
current_direction_deg, wind_direction_deg) — DIRECTIONAL_VARIABLES below
flags this so the frontend never line-charts it. moon_phase is similarly
cyclic/categorical (not a continuous scalar) — NON_CHART_VARIABLES flags
that one too. Both sets are exposed here as the single source of truth so
Analytics and Download can't disagree on which variables get charted.

Historical data for a past date range never changes once elapsed, so
complete results are cached permanently in-process per (lat, lon,
start_date, end_date, variables). A result with a failed source is not
cached, so the next request retries it (see get_historical_analytics).
"""

import asyncio
import logging
import math
from datetime import date, timedelta
from typing import Any

from backend.error_utils import describe_exception
from backend.services import copernicus_fetch, open_meteo
from backend.services.pfz_service import copernicus_credentials

logger = logging.getLogger("orca.analytics")

_MARINE_URL = open_meteo.MARINE_URL
_ARCHIVE_URL = open_meteo.ARCHIVE_URL
# Past-date archive data doesn't change; this only bounds how long the
# shared Open-Meteo client (services/open_meteo.py) keeps a raw response.
_OPEN_METEO_TTL_SECONDS = 60 * 60
# How long a request waits for one Copernicus series. Each is its own
# open_dataset() (~1s CPU + ~8s of network round-trips on a fast machine,
# ~30s measured at Render free tier's 0.1 CPU) — the old 20s budget timed
# out every production request. A read that outlives this keeps running
# and its result is cached for the next request (see
# services/copernicus_fetch.py).
_COPERNICUS_TIMEOUT = 45.0
_COPERNICUS_CACHE_TTL_SECONDS = 6 * 60 * 60

# Same dataset marine_data_agent.py already uses for the chat/weather path.
_CHL_DATASET_ID = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"
_SALINITY_DATASET_ID = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
_OXYGEN_DATASET_ID = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"

# Fixed India offset — every ORCA zone is Indian coastal water, so a real
# per-point timezone lookup would be pure over-engineering here.
_IST_OFFSET_HOURS = 5.5

VALID_VARIABLES = {
    # Temperature
    "sst_celsius",
    "air_temp_celsius",
    # Waves
    "wave_height_m",
    "wave_direction_deg",
    "wave_period_s",
    "swell_wave_height_m",
    "swell_wave_direction_deg",
    "swell_wave_period_s",
    # Currents
    "current_velocity_kmh",
    "current_direction_deg",
    # Wind
    "wind_kmh",
    "wind_direction_deg",
    "wind_gusts_kmh",
    # Atmospheric
    "precipitation_mm",
    "cloud_cover_pct",
    "pressure_hpa",
    "humidity_pct",
    # Biological / water quality
    "chlorophyll_mg_m3",
    "salinity_psu",
    "dissolved_oxygen_mmol_m3",
    # Astronomical
    "sunrise_hour_ist",
    "sunset_hour_ist",
    "moon_phase",
}

# Circular/compass data — never render these as a line chart (see module
# docstring). Frontend imports this same set of keys conceptually via its
# own mirrored constant (charts and CSV/PDF/DOCX both need it, so it's not
# worth a new API surface just to ship one set of strings across the wire).
DIRECTIONAL_VARIABLES = {"wave_direction_deg", "swell_wave_direction_deg", "current_direction_deg", "wind_direction_deg"}

# Cyclic/categorical, also not a line-chart candidate.
NON_CHART_VARIABLES = {"moon_phase"}

CATEGORY_OF: dict[str, str] = {
    "sst_celsius": "temperature",
    "air_temp_celsius": "temperature",
    "wave_height_m": "waves",
    "wave_direction_deg": "waves",
    "wave_period_s": "waves",
    "swell_wave_height_m": "waves",
    "swell_wave_direction_deg": "waves",
    "swell_wave_period_s": "waves",
    "current_velocity_kmh": "currents",
    "current_direction_deg": "currents",
    "wind_kmh": "wind",
    "wind_direction_deg": "wind",
    "wind_gusts_kmh": "wind",
    "precipitation_mm": "atmospheric",
    "cloud_cover_pct": "atmospheric",
    "pressure_hpa": "atmospheric",
    "humidity_pct": "atmospheric",
    "chlorophyll_mg_m3": "biological",
    "salinity_psu": "biological",
    "dissolved_oxygen_mmol_m3": "biological",
    "sunrise_hour_ist": "astronomical",
    "sunset_hour_ist": "astronomical",
    "moon_phase": "astronomical",
}

_UNITS: dict[str, str] = {
    "sst_celsius": "°C",
    "air_temp_celsius": "°C",
    "wave_height_m": "m",
    "wave_direction_deg": "°",
    "wave_period_s": "s",
    "swell_wave_height_m": "m",
    "swell_wave_direction_deg": "°",
    "swell_wave_period_s": "s",
    "current_velocity_kmh": "km/h",
    "current_direction_deg": "°",
    "wind_kmh": "km/h",
    "wind_direction_deg": "°",
    "wind_gusts_kmh": "km/h",
    "precipitation_mm": "mm",
    "cloud_cover_pct": "%",
    "pressure_hpa": "hPa",
    "humidity_pct": "%",
    "chlorophyll_mg_m3": "mg/m³",
    "salinity_psu": "PSU",
    "dissolved_oxygen_mmol_m3": "mmol/m³",
    "sunrise_hour_ist": "IST",
    "sunset_hour_ist": "IST",
    "moon_phase": "",
}

# Human-readable titles — shared by every export format (CSV headers,
# PDF/DOCX report tables, and the frontend's own display) so labeling
# can't drift between them.
VARIABLE_TITLES: dict[str, str] = {
    "sst_celsius": "Sea Surface Temperature",
    "air_temp_celsius": "Air Temperature",
    "wave_height_m": "Wave Height",
    "wave_direction_deg": "Wave Direction",
    "wave_period_s": "Wave Period",
    "swell_wave_height_m": "Swell Wave Height",
    "swell_wave_direction_deg": "Swell Wave Direction",
    "swell_wave_period_s": "Swell Wave Period",
    "current_velocity_kmh": "Current Velocity",
    "current_direction_deg": "Current Direction",
    "wind_kmh": "Wind Speed",
    "wind_direction_deg": "Wind Direction",
    "wind_gusts_kmh": "Wind Gusts",
    "precipitation_mm": "Precipitation",
    "cloud_cover_pct": "Cloud Cover",
    "pressure_hpa": "Sea-Level Pressure",
    "humidity_pct": "Relative Humidity",
    "chlorophyll_mg_m3": "Chlorophyll-a",
    "salinity_psu": "Salinity",
    "dissolved_oxygen_mmol_m3": "Dissolved Oxygen",
    "sunrise_hour_ist": "Sunrise (IST)",
    "sunset_hour_ist": "Sunset (IST)",
    "moon_phase": "Moon Phase",
}

# Why a variable isn't a plain direct-from-source reading — derived
# (aggregated from real sub-daily data), computed (no live source at all),
# or coverage-limited (a real source with a shorter-than-requested
# history window). Single source of truth for every export format.
VARIABLE_CAVEATS: dict[str, str] = {
    "current_velocity_kmh": (
        "Derived: daily max computed from Open-Meteo's hourly ocean_current_velocity — "
        "no daily aggregate exists upstream for this variable (confirmed against the live "
        "API; requesting a daily aggregate for it is rejected as an invalid variable name)."
    ),
    "current_direction_deg": (
        "Derived: current direction reading taken at the same hour as that day's peak "
        "current velocity (from Open-Meteo's hourly ocean_current_direction) — no daily "
        "aggregate exists upstream, and averaging compass bearings would produce a "
        "meaningless value, so the direction is paired with the day's peak-speed reading."
    ),
    "chlorophyll_mg_m3": (
        "Coverage-limited: this is a near-real-time satellite product with only a rolling "
        "~2-3 week window of history. Dates outside that window will show no data for this "
        "variable specifically (see its own status below), not a fetch failure."
    ),
    "salinity_psu": "Sourced from Copernicus Marine's global ocean physics analysis-forecast model at the shallowest available depth level (sea-surface salinity), not a point in-situ measurement.",
    "dissolved_oxygen_mmol_m3": (
        "Sourced from Copernicus Marine's global biogeochemistry model at ~0.25° resolution "
        "— coastal points very close to shore can fall outside the model's water grid and "
        "come back with no data (see this variable's own status below), not a fetch failure."
    ),
    "sunrise_hour_ist": "Computed from a standard solar-position formula for this zone's coordinates, reported in IST (UTC+5:30) — not fetched from any live data source.",
    "sunset_hour_ist": "Computed from a standard solar-position formula for this zone's coordinates, reported in IST (UTC+5:30) — not fetched from any live data source.",
    "moon_phase": "Computed from the synodic lunar month (accurate to within about a day) — not fetched from any live data source.",
}

_cache: dict[tuple, dict] = {}


async def _empty_dict() -> dict:
    return {}


async def _tracked(source: str, coro, failed_sources: list[str]) -> dict:
    """Degrade one failed source to {} (its variables get status "error")
    and record it, so a response with a failed source is never cached
    permanently — previously one transient upstream failure was frozen
    into the in-process cache for the rest of the process's life."""
    try:
        return await coro
    except Exception as exc:
        failed_sources.append(source)
        # open_meteo / copernicus_fetch already log their own failures.
        if not isinstance(exc, (open_meteo.OpenMeteoError, TimeoutError)):
            logger.warning("Analytics %s fetch failed: %s", source, describe_exception(exc))
        return {}


def _status_for(values: list) -> str:
    if not values:
        return "error"
    present = sum(1 for v in values if v is not None)
    if present == 0:
        return "error"
    if present < len(values):
        return "partial"
    return "ok"


# ---------------------------------------------------------------------------
# Open-Meteo marine archive — daily (SST/waves/swell) and hourly (currents)
# ---------------------------------------------------------------------------

async def _fetch_marine_daily(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    data = await open_meteo.get_json(
        _MARINE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "sea_surface_temperature_max,wave_height_max,wave_direction_dominant,"
                "wave_period_max,swell_wave_height_max,swell_wave_direction_dominant,"
                "swell_wave_period_max"
            ),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        },
        label="Open-Meteo marine daily",
        ttl=_OPEN_METEO_TTL_SECONDS,
    )
    return data.get("daily", {})


async def _fetch_marine_current_hourly(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    data = await open_meteo.get_json(
        _MARINE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "ocean_current_velocity,ocean_current_direction",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        },
        label="Open-Meteo marine hourly currents",
        ttl=_OPEN_METEO_TTL_SECONDS,
    )
    return data.get("hourly", {})


def _aggregate_current_daily(hourly: dict) -> tuple[list[str], list[float], list[float]]:
    """Real per-day (velocity, direction) pair, both read at the hour of
    that day's peak velocity — see VARIABLE_CAVEATS for why direction is
    paired rather than averaged."""
    times = hourly.get("time", [])
    velocities = hourly.get("ocean_current_velocity", [])
    directions = hourly.get("ocean_current_direction", [])

    best_by_day: dict[str, tuple[float, float]] = {}
    for t, v, d in zip(times, velocities, directions):
        if v is None:
            continue
        day = t[:10]
        if day not in best_by_day or v > best_by_day[day][0]:
            best_by_day[day] = (v, d)

    days = sorted(best_by_day.keys())
    return days, [best_by_day[d][0] for d in days], [best_by_day[d][1] for d in days]


# ---------------------------------------------------------------------------
# Open-Meteo weather archive — daily (air temp/wind/atmospheric)
# ---------------------------------------------------------------------------

async def _fetch_archive_daily(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    data = await open_meteo.get_json(
        _ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "temperature_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant,"
                "wind_gusts_10m_max,precipitation_sum,cloud_cover_mean,"
                "pressure_msl_mean,relative_humidity_2m_mean"
            ),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        },
        label="Open-Meteo weather archive",
        ttl=_OPEN_METEO_TTL_SECONDS,
    )
    return data.get("daily", {})


# ---------------------------------------------------------------------------
# Copernicus Marine — chlorophyll / salinity / dissolved oxygen
# ---------------------------------------------------------------------------

def _fetch_copernicus_series_sync(
    dataset_id: str, variable: str, lat: float, lon: float, start_date: str, end_date: str, username: str, password: str
) -> tuple[list[str], list[float | None]]:
    import copernicusmarine

    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_longitude=lon - 0.1,
        maximum_longitude=lon + 0.1,
        minimum_latitude=lat - 0.1,
        maximum_latitude=lat + 0.1,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        coordinates_selection_method="nearest",
        username=username,
        password=password,
    )
    da = ds[variable].sel(latitude=lat, longitude=lon, method="nearest")
    if "depth" in da.dims:
        # Shallowest available level — sea-surface reading (see
        # VARIABLE_CAVEATS for salinity_psu).
        da = da.isel(depth=0)
    # "nearest" time selection can snap the range end onto the next day's
    # timestamp (observed: salinity returned end_date + 1 for a
    # ...T23:59:59 end), so keep only days actually inside the request.
    pairs = [
        (str(t)[:10], None if math.isnan(v) else round(float(v), 3))
        for t, v in zip(da["time"].values, da.values)
    ]
    pairs = [(d, v) for d, v in pairs if start_date <= d <= end_date]
    return [d for d, _ in pairs], [v for _, v in pairs]


async def _fetch_copernicus_series(dataset_id: str, variable: str, lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Missing credentials degrade to an empty series (the caller turns
    that into this variable's own "error" status). A failed or timed-out
    read raises, so get_historical_analytics knows not to cache it; the
    failure itself is logged, with its exception type, by
    copernicus_fetch."""
    creds = copernicus_credentials()
    if creds is None:
        return {"dates": [], "values": []}
    username, password = creds
    dates, values = await copernicus_fetch.fetch(
        ("series", dataset_id, variable, round(lat, 4), round(lon, 4), start_date, end_date),
        _fetch_copernicus_series_sync,
        (dataset_id, variable, lat, lon, start_date, end_date, username, password),
        wait_timeout=_COPERNICUS_TIMEOUT,
        ttl=_COPERNICUS_CACHE_TTL_SECONDS,
        label=f"Copernicus {variable} series ({dataset_id}) for ({lat:.2f},{lon:.2f}) {start_date}..{end_date}",
    )
    return {"dates": dates, "values": values}


# ---------------------------------------------------------------------------
# Computed astronomical — sunrise/sunset (NOAA solar formula) and moon phase
# (synodic month). No network call at all.
# ---------------------------------------------------------------------------

def _sunrise_sunset_utc_hours(lat: float, lon: float, d: date) -> tuple[float | None, float | None]:
    """Standard NOAA solar sunrise/sunset algorithm. Returns (sunrise, sunset)
    as UTC decimal hours, or None for a component that doesn't occur (polar
    day/night — not reachable for India's latitudes, kept safe regardless)."""
    from datetime import date as _date

    j2000 = _date(2000, 1, 1)
    n = (d - j2000).days
    lng_hour = lon / 15.0

    results: list[float | None] = []
    for zenith, is_rise in ((90.833, True), (90.833, False)):
        t = n + ((6 if is_rise else 18) - lng_hour) / 24.0
        m = (0.9856 * t) - 3.289
        ecliptic_lon = (m + (1.916 * math.sin(math.radians(m))) + (0.020 * math.sin(math.radians(2 * m))) + 282.634) % 360
        ra = math.degrees(math.atan(0.91764 * math.tan(math.radians(ecliptic_lon)))) % 360
        lon_quadrant = math.floor(ecliptic_lon / 90.0) * 90
        ra_quadrant = math.floor(ra / 90.0) * 90
        ra = (ra + (lon_quadrant - ra_quadrant)) / 15.0
        sin_dec = 0.39782 * math.sin(math.radians(ecliptic_lon))
        cos_dec = math.cos(math.asin(sin_dec))
        cos_h = (math.cos(math.radians(zenith)) - (sin_dec * math.sin(math.radians(lat)))) / (cos_dec * math.cos(math.radians(lat)))
        if cos_h > 1 or cos_h < -1:
            results.append(None)
            continue
        h = (360 - math.degrees(math.acos(cos_h))) / 15.0 if is_rise else math.degrees(math.acos(cos_h)) / 15.0
        local_t = h + ra - (0.06571 * t) - 6.622
        utc_hour = (local_t - lng_hour) % 24
        results.append(utc_hour)
    return results[0], results[1]


_MOON_PHASE_NAMES = [
    (0.03, "New Moon"),
    (0.22, "Waxing Crescent"),
    (0.28, "First Quarter"),
    (0.47, "Waxing Gibbous"),
    (0.53, "Full Moon"),
    (0.72, "Waning Gibbous"),
    (0.78, "Last Quarter"),
    (0.97, "Waning Crescent"),
    (1.01, "New Moon"),
]


def _moon_phase(d: date) -> tuple[float, str]:
    """Fraction (0=new moon, 0.5=full moon, wrapping back to 0) and a
    display name, via the standard synodic-month approximation."""
    known_new_moon = date(2000, 1, 6)
    synodic_month = 29.53058867
    days_since = (d - known_new_moon).days
    fraction = (days_since % synodic_month) / synodic_month
    name = next(n for threshold, n in _MOON_PHASE_NAMES if fraction < threshold)
    return round(fraction, 3), name


def _compute_astronomical(lat: float, lon: float, start_date: str, end_date: str, variables: set[str]) -> dict:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    dates: list[str] = []
    sunrise: list[float | None] = []
    sunset: list[float | None] = []
    moon_fraction: list[float] = []
    moon_name: list[str] = []

    want_sun = bool({"sunrise_hour_ist", "sunset_hour_ist"} & variables)
    want_moon = "moon_phase" in variables

    d = start
    while d <= end:
        dates.append(d.isoformat())
        if want_sun:
            sr, ss = _sunrise_sunset_utc_hours(lat, lon, d)
            sunrise.append(round((sr + _IST_OFFSET_HOURS) % 24, 2) if sr is not None else None)
            sunset.append(round((ss + _IST_OFFSET_HOURS) % 24, 2) if ss is not None else None)
        if want_moon:
            frac, name = _moon_phase(d)
            moon_fraction.append(frac)
            moon_name.append(name)
        d += timedelta(days=1)

    return {"dates": dates, "sunrise": sunrise, "sunset": sunset, "moon_fraction": moon_fraction, "moon_name": moon_name}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def get_historical_analytics(
    lat: float, lon: float, start_date: str, end_date: str, variables: list[str]
) -> dict[str, Any]:
    """Historical time series for `variables` (see VALID_VARIABLES) between
    start_date and end_date (YYYY-MM-DD) at one point. No chat/LLM
    involvement. Never raises — a failed upstream fetch degrades that
    variable to an "error" status rather than failing the whole response."""
    # Only default to every variable when the caller passed none at all
    # (matches the router's own "omitted -> all" default). A caller that
    # passed a non-empty list containing only invalid/misspelled names
    # should get back nothing for those names, not silently every
    # variable instead — that's a much larger, unrequested response with
    # no signal the input was rejected.
    variables = [v for v in variables if v in VALID_VARIABLES] if variables else sorted(VALID_VARIABLES)
    var_set = set(variables)

    cache_key = (round(lat, 4), round(lon, 4), start_date, end_date, tuple(sorted(variables)))
    if cache_key in _cache:
        return _cache[cache_key]

    want_marine_daily = bool({"sst_celsius", "wave_height_m", "wave_direction_deg", "wave_period_s", "swell_wave_height_m", "swell_wave_direction_deg", "swell_wave_period_s"} & var_set)
    want_current = bool({"current_velocity_kmh", "current_direction_deg"} & var_set)
    want_archive = bool({"air_temp_celsius", "wind_kmh", "wind_direction_deg", "wind_gusts_kmh", "precipitation_mm", "cloud_cover_pct", "pressure_hpa", "humidity_pct"} & var_set)
    want_chl = "chlorophyll_mg_m3" in var_set
    want_salinity = "salinity_psu" in var_set
    want_oxygen = "dissolved_oxygen_mmol_m3" in var_set
    want_astronomical = bool({"sunrise_hour_ist", "sunset_hour_ist", "moon_phase"} & var_set)

    failed: list[str] = []
    marine_daily, current_hourly, archive_daily, chl, salinity, oxygen = await asyncio.gather(
        _tracked("marine daily", _fetch_marine_daily(lat, lon, start_date, end_date), failed) if want_marine_daily else _empty_dict(),
        _tracked("marine currents", _fetch_marine_current_hourly(lat, lon, start_date, end_date), failed) if want_current else _empty_dict(),
        _tracked("weather archive", _fetch_archive_daily(lat, lon, start_date, end_date), failed) if want_archive else _empty_dict(),
        _tracked("chlorophyll", _fetch_copernicus_series(_CHL_DATASET_ID, "CHL", lat, lon, start_date, end_date), failed) if want_chl else _empty_dict(),
        _tracked("salinity", _fetch_copernicus_series(_SALINITY_DATASET_ID, "so", lat, lon, start_date, end_date), failed) if want_salinity else _empty_dict(),
        _tracked("dissolved oxygen", _fetch_copernicus_series(_OXYGEN_DATASET_ID, "o2", lat, lon, start_date, end_date), failed) if want_oxygen else _empty_dict(),
    )
    astronomical = _compute_astronomical(lat, lon, start_date, end_date, var_set) if want_astronomical else {}

    series: dict[str, dict] = {}

    def _add(var: str, unit: str, dates: list, values: list) -> None:
        series[var] = {"unit": unit, "dates": dates, "values": values, "status": _status_for(values)}

    if "sst_celsius" in var_set:
        _add("sst_celsius", _UNITS["sst_celsius"], marine_daily.get("time", []), marine_daily.get("sea_surface_temperature_max", []))
    if "wave_height_m" in var_set:
        _add("wave_height_m", _UNITS["wave_height_m"], marine_daily.get("time", []), marine_daily.get("wave_height_max", []))
    if "wave_direction_deg" in var_set:
        _add("wave_direction_deg", _UNITS["wave_direction_deg"], marine_daily.get("time", []), marine_daily.get("wave_direction_dominant", []))
    if "wave_period_s" in var_set:
        _add("wave_period_s", _UNITS["wave_period_s"], marine_daily.get("time", []), marine_daily.get("wave_period_max", []))
    if "swell_wave_height_m" in var_set:
        _add("swell_wave_height_m", _UNITS["swell_wave_height_m"], marine_daily.get("time", []), marine_daily.get("swell_wave_height_max", []))
    if "swell_wave_direction_deg" in var_set:
        _add("swell_wave_direction_deg", _UNITS["swell_wave_direction_deg"], marine_daily.get("time", []), marine_daily.get("swell_wave_direction_dominant", []))
    if "swell_wave_period_s" in var_set:
        _add("swell_wave_period_s", _UNITS["swell_wave_period_s"], marine_daily.get("time", []), marine_daily.get("swell_wave_period_max", []))

    if want_current:
        days, velocities, directions = _aggregate_current_daily(current_hourly)
        if "current_velocity_kmh" in var_set:
            _add("current_velocity_kmh", _UNITS["current_velocity_kmh"], days, velocities)
        if "current_direction_deg" in var_set:
            _add("current_direction_deg", _UNITS["current_direction_deg"], days, directions)

    if "air_temp_celsius" in var_set:
        _add("air_temp_celsius", _UNITS["air_temp_celsius"], archive_daily.get("time", []), archive_daily.get("temperature_2m_mean", []))
    if "wind_kmh" in var_set:
        _add("wind_kmh", _UNITS["wind_kmh"], archive_daily.get("time", []), archive_daily.get("wind_speed_10m_max", []))
    if "wind_direction_deg" in var_set:
        _add("wind_direction_deg", _UNITS["wind_direction_deg"], archive_daily.get("time", []), archive_daily.get("wind_direction_10m_dominant", []))
    if "wind_gusts_kmh" in var_set:
        _add("wind_gusts_kmh", _UNITS["wind_gusts_kmh"], archive_daily.get("time", []), archive_daily.get("wind_gusts_10m_max", []))
    if "precipitation_mm" in var_set:
        _add("precipitation_mm", _UNITS["precipitation_mm"], archive_daily.get("time", []), archive_daily.get("precipitation_sum", []))
    if "cloud_cover_pct" in var_set:
        _add("cloud_cover_pct", _UNITS["cloud_cover_pct"], archive_daily.get("time", []), archive_daily.get("cloud_cover_mean", []))
    if "pressure_hpa" in var_set:
        _add("pressure_hpa", _UNITS["pressure_hpa"], archive_daily.get("time", []), archive_daily.get("pressure_msl_mean", []))
    if "humidity_pct" in var_set:
        _add("humidity_pct", _UNITS["humidity_pct"], archive_daily.get("time", []), archive_daily.get("relative_humidity_2m_mean", []))

    if want_chl:
        _add("chlorophyll_mg_m3", _UNITS["chlorophyll_mg_m3"], chl.get("dates", []), chl.get("values", []))
    if want_salinity:
        _add("salinity_psu", _UNITS["salinity_psu"], salinity.get("dates", []), salinity.get("values", []))
    if want_oxygen:
        _add("dissolved_oxygen_mmol_m3", _UNITS["dissolved_oxygen_mmol_m3"], oxygen.get("dates", []), oxygen.get("values", []))

    if "sunrise_hour_ist" in var_set:
        _add("sunrise_hour_ist", _UNITS["sunrise_hour_ist"], astronomical.get("dates", []), astronomical.get("sunrise", []))
    if "sunset_hour_ist" in var_set:
        _add("sunset_hour_ist", _UNITS["sunset_hour_ist"], astronomical.get("dates", []), astronomical.get("sunset", []))
    if "moon_phase" in var_set:
        # values carries the numeric fraction (for status/table use); the
        # display name travels alongside for the frontend's table view.
        series["moon_phase"] = {
            "unit": _UNITS["moon_phase"],
            "dates": astronomical.get("dates", []),
            "values": astronomical.get("moon_fraction", []),
            "status": _status_for(astronomical.get("moon_fraction", [])),
            "names": astronomical.get("moon_name", []),
        }

    result = {
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end_date,
        "series": series,
    }
    # Only a complete answer is cached for good — a response with a failed
    # source is returned as-is (those variables show "error") and the next
    # request retries the failed sources. The per-source caches in
    # open_meteo / copernicus_fetch keep that retry from re-fetching the
    # sources that did succeed.
    if not failed:
        _cache[cache_key] = result
    return result


def build_export_notes(series: dict[str, dict]) -> list[str]:
    """Self-documenting note lines for an export (GET /export, all four
    formats) — states which variables (if any) are derived/computed/
    coverage-limited, AND dynamically flags any variable whose data this
    run actually came back partial or missing, so a downloaded file
    explains itself without needing the live UI. Single source of truth:
    every export format calls this same function."""
    notes: list[str] = []
    for var in sorted(series.keys()):
        title = VARIABLE_TITLES.get(var, var)
        if var in VARIABLE_CAVEATS:
            notes.append(f"{title}: {VARIABLE_CAVEATS[var]}")
        status = series[var].get("status")
        if status == "partial":
            notes.append(f"{title}: some dates in this range had no data available from the source (shown as gaps) — not a fetch failure.")
        elif status == "error":
            notes.append(f"{title}: no data was available from the source for this range/location.")
    if not notes:
        notes.append("All variables are sourced directly from their upstream APIs with full coverage for this range.")
    return notes
