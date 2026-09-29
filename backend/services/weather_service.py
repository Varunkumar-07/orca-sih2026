"""
Standalone current-weather + marine-conditions service — Phase 3 (Weather
Page), backing GET /weather.

Deliberately a NEW, agent-independent module rather than a refactor of
weather_agent.py / marine_data_agent.py: those two files keep driving the
chat pipeline exactly as before, completely untouched by this phase. This
is a thin, safer duplicate of their live-fetch logic (per the phase brief:
"a thin duplicate function is acceptable and safer than a risky
extraction"), not a shared-code extraction that chat now depends on too.

Reuses, rather than re-implements:
- The same Open-Meteo forecast + marine endpoints weather_agent.py already
  calls for the chat flow — just combined into one marine request (wave
  height + SST together) instead of the two separate ones split across
  weather_agent.py and marine_data_agent.py today.
- The same IMD cyclone wind threshold and WMO thunderstorm codes
  weather_agent.py uses, so the two paths can never silently drift apart.
- marine_data_agent's own _fetch_live_chlorophyll helper, imported
  directly (read-only — marine_data_agent.py is not modified) rather than
  duplicated: its Copernicus auth/timeout/NaN handling is non-trivial and
  re-implementing it would be the riskier path.

Never raises: any missing field degrades to None and `status` reflects how
much of the snapshot actually came back, same convention as the reasoning
agents' own MarineDataResult/WeatherDataResult.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from backend.agents.reasoning.marine_data_agent import _fetch_live_chlorophyll
from backend.agents.reasoning.weather_agent import (
    _CYCLONE_WIND_THRESHOLD_KMH,
    _THUNDERSTORM_CODES,
)
from backend.schemas.contracts import GeoPoint
from backend.services import open_meteo
from backend.services.analytics_service import _compute_astronomical
from backend.time_utils import now_iso as _now_iso

logger = logging.getLogger("orca.weather_page")

# Re-exported so tests (and readers) can see which hosts this module hits;
# the actual request/caching/quota logic lives in services/open_meteo.py.
_FORECAST_URL = open_meteo.FORECAST_URL
_FORECAST_FALLBACK_URL = open_meteo.FORECAST_FALLBACK_URL
_MARINE_URL = open_meteo.MARINE_URL
# "Today" for sunrise/sunset must be IST's today, not the host server's —
# a UTC-hosted deployment would otherwise show yesterday's astronomical
# data for the first ~5.5 hours of every IST day.
_IST = timezone(timedelta(hours=5, minutes=30))


async def _fetch_forecast(lat: float, lon: float) -> dict | None:
    """Current conditions + today's daily totals (see
    open_meteo.fetch_current_forecast, including its fallback host), or
    None when neither host could serve it — already logged there."""
    try:
        return await open_meteo.fetch_current_forecast(lat, lon)
    except open_meteo.OpenMeteoError:
        return None


async def _fetch_marine(lat: float, lon: float) -> dict | None:
    try:
        return await open_meteo.fetch_current_marine(lat, lon)
    except open_meteo.OpenMeteoError:
        return None


async def get_current_weather(lat: float, lon: float) -> dict:
    """Current weather + marine snapshot for one point. Used directly by
    GET /weather — no chat/LLM involvement anywhere in this path."""
    location = GeoPoint(lat=lat, lon=lon)
    forecast, marine, chlorophyll = await asyncio.gather(
        _fetch_forecast(lat, lon),
        _fetch_marine(lat, lon),
        _fetch_live_chlorophyll(location),
    )

    current = (forecast or {}).get("current") or {}
    daily = (forecast or {}).get("daily") or {}

    air_temp = current.get("temperature_2m")
    wind_kmh = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")
    wave_height = marine.get("wave_height") if marine else None
    sst = marine.get("sea_surface_temperature") if marine else None
    precipitation_mm = (daily.get("precipitation_sum") or [None])[0]
    wind_max_kmh = (daily.get("wind_speed_10m_max") or [None])[0]

    # Reuses analytics_service's own astronomical calculation (Phase 6/7) —
    # not recomputed here. Sync, no network call.
    today = datetime.now(_IST).date()
    astro = _compute_astronomical(lat, lon, today.isoformat(), today.isoformat(), {"sunrise_hour_ist", "sunset_hour_ist"})
    sunrise_hour_ist = (astro.get("sunrise") or [None])[0]
    sunset_hour_ist = (astro.get("sunset") or [None])[0]

    # Open-Meteo's own fields already come back at sane precision; only the
    # raw Copernicus chlorophyll reading needs rounding for display (it
    # arrives as an unrounded satellite float, e.g. 0.6354018449783325).
    if chlorophyll is not None:
        chlorophyll = round(chlorophyll, 3)

    fields = [air_temp, wind_kmh, wave_height, sst, chlorophyll]
    present = sum(1 for f in fields if f is not None)
    status = "ok" if present == len(fields) else ("partial" if present > 0 else "error")

    return {
        "lat": lat,
        "lon": lon,
        "status": status,
        "air_temperature_c": air_temp,
        "sst_celsius": sst,
        "wind_kmh": wind_kmh,
        "wave_height_m": wave_height,
        "chlorophyll_mg_m3": chlorophyll,
        "precipitation_mm": precipitation_mm,
        "wind_max_kmh": wind_max_kmh,
        "sunrise_hour_ist": sunrise_hour_ist,
        "sunset_hour_ist": sunset_hour_ist,
        "cyclone_alert": bool(wind_kmh is not None and wind_kmh >= _CYCLONE_WIND_THRESHOLD_KMH),
        "lightning_alert": bool(weather_code is not None and int(weather_code) in _THUNDERSTORM_CODES),
        "source_timestamp": _now_iso(),
    }
