"""
Lightweight alerts service — Phase 5 (Alerts/Advisories page), backing
GET /alerts.

Derives cyclone/lightning advisories per tracked PFZ zone (the same zones
GET /zones already catalogs) by reusing weather_service.py's own forecast
fetch and its exact cyclone-wind threshold / thunderstorm-code constants
(Phase 3) — not new detection logic. Only the lightweight forecast call is
reused here (wind + weather code), not the full get_current_weather (which
also does a marine call and the slow Copernicus chlorophyll fetch) — an
alert check has no use for SST/wave/chlorophyll, so fetching them per zone
would just be needless latency across 30+ zones.

Restricted areas (also in GET /zones) are boundary polygons, not points —
they carry no live weather reading of their own, so they are not an alert
source here; this module only ever looks at "pfz" zones.

Computed on demand per request rather than on a background poll loop: the
tracked zone set is exactly GET /zones's own set, and GET /weather already
fetches live per-point conditions on demand the same way — a background
poller would be a second, independent way of reaching the same live data,
with its own startup/shutdown lifecycle to manage for no real benefit here.
The only concession to request cost is a short in-memory TTL cache (same
"cache what doesn't need to be redone every hit" spirit as GET /zones's own
cache, just time-bounded here since weather — unlike the zone catalog —
genuinely changes over time).
"""

import asyncio
import time

from backend.services.weather_service import (
    _CYCLONE_WIND_THRESHOLD_KMH,
    _THUNDERSTORM_CODES,
    _fetch_forecast,
)
from backend.time_utils import now_iso as _now_iso

# Long enough that repeated page loads/polling within a few minutes don't
# re-fetch 30+ zones' forecasts from Open-Meteo every time; short enough
# that an alert never goes stale for long.
_CACHE_TTL_SECONDS = 300

_cache: dict | None = None
_cache_at: float = 0.0
# Cache key is the zone-id set the cached result was computed for — without
# this, a call with a different/updated pfz_zones list within the TTL window
# would silently get back alerts (and a checked_zones count) computed for a
# stale, different zone list.
_cache_key: tuple[str, ...] | None = None


async def _check_zone(zone: dict) -> list[dict]:
    coords = zone.get("coordinates")
    if not coords:
        return []

    forecast = await _fetch_forecast(coords["lat"], coords["lon"])
    if forecast is None:
        return []

    current = forecast.get("current") or {}
    wind_kmh = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")

    found: list[dict] = []
    if wind_kmh is not None and wind_kmh >= _CYCLONE_WIND_THRESHOLD_KMH:
        found.append(
            {
                "zone_id": zone["id"],
                "zone_name": zone["name"],
                "near": zone.get("near"),
                "alert_type": "cyclone",
                "severity": "high",
                "detail": f"Sustained wind {wind_kmh} km/h — at/above the IMD cyclonic-storm threshold ({_CYCLONE_WIND_THRESHOLD_KMH} km/h)",
            }
        )
    if weather_code is not None and int(weather_code) in _THUNDERSTORM_CODES:
        found.append(
            {
                "zone_id": zone["id"],
                "zone_name": zone["name"],
                "near": zone.get("near"),
                "alert_type": "lightning",
                "severity": "moderate",
                "detail": "Thunderstorm conditions detected at this zone (WMO weather code)",
            }
        )
    return found


async def get_active_alerts(pfz_zones: list[dict]) -> dict:
    """Active cyclone/lightning alerts across `pfz_zones` (pass GET /zones's
    own "pfz" entries). Never raises — a zone whose forecast fetch fails is
    silently skipped rather than surfaced as an error, same degrade-only
    convention every live-fetch path in this codebase already follows."""
    global _cache, _cache_at, _cache_key
    now = time.monotonic()
    key = tuple(sorted(z["id"] for z in pfz_zones))
    if _cache is not None and _cache_key == key and (now - _cache_at) < _CACHE_TTL_SECONDS:
        return _cache

    per_zone_results = await asyncio.gather(*(_check_zone(z) for z in pfz_zones))
    alerts: list[dict] = [alert for zone_alerts in per_zone_results for alert in zone_alerts]

    _cache = {
        "alerts": alerts,
        "checked_zones": len(pfz_zones),
        "generated_at": _now_iso(),
    }
    _cache_at = now
    _cache_key = key
    return _cache
