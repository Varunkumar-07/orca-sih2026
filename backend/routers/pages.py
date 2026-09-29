"""Direct data-passthrough endpoints for the 6 non-chat pages — Zones
Explorer, Weather, Route Planner, Alerts, Analytics, Download. No chat/LLM
involvement anywhere in this file, no agent orchestration beyond
navigation_agent's own A* route search.

Split out of main.py (P4 codebase-audit cleanup) purely for file size; no
behavior change. zones() is the shared live-zone catalog /route, /alerts,
/weather/forecast, and /export all resolve zone ids through — see
pfz_service.get_cached_zones for the underlying cache/TTL/pre-warm this
depends on.
"""
import csv
import io
import json
import re

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from backend.agents.deterministic.geospatial import (
    get_active_restricted_areas,
    haversine_km,
)
from backend.agents.reasoning.navigation_agent import plan_route
from backend.schemas.contracts import GeoPoint
from backend.services.alerts_service import get_active_alerts
from backend.services.analytics_service import (
    VALID_VARIABLES,
    build_export_notes,
    get_historical_analytics,
)
from backend.services.export_render import render_docx, render_pdf
from backend.services.forecast_service import get_forecast
from backend.services.pfz_service import get_cached_zones
from backend.services.weather_service import get_current_weather

router = APIRouter()

# Audit: filenames built from raw, unvalidated zone/date query params were
# interpolated directly into Content-Disposition headers below — a
# control/quote character in either could inject extra header content.
# Strip anything outside this safe set instead of trusting the input.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def _safe_filename_component(value: str) -> str:
    return _UNSAFE_FILENAME_CHARS.sub("_", value)[:64] or "_"


def _validate_variables(variables: str | None) -> list[str]:
    if not variables:
        return sorted(VALID_VARIABLES)
    var_list = [v.strip() for v in variables.split(",") if v.strip()]
    invalid = sorted(set(var_list) - VALID_VARIABLES)
    if invalid:
        raise HTTPException(status_code=400, detail=f"invalid variables: {', '.join(invalid)}")
    return var_list


@router.get("/zones")
async def zones() -> dict:
    """Direct data passthrough for the Zones Explorer page — no chat/LLM
    involvement. Combines real front-detected PFZ zones (falling back to
    the mock heuristic generator per-anchor when live data isn't
    available) and the same restricted-area boundaries the chat flow
    already uses into one flat list. Reads through the shared cache in
    pfz_service.get_cached_zones — see that module for the TTL/anchor
    details; this is the same live-zone source of truth /route, /alerts,
    /weather/forecast, and /export already depend on via this function."""
    return await get_cached_zones()


@router.get("/weather")
async def weather(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)) -> dict:
    """Direct data passthrough for the Weather page — no chat/LLM
    involvement. See backend/services/weather_service.py."""
    return await get_current_weather(lat, lon)


@router.get("/weather/forecast")
async def weather_forecast(zone: str) -> dict:
    """Model-predicted wave height / wind speed for the next 7 days at a
    zone — genuine ML output from ORCA's own trained models, clearly
    separate from GET /weather's live conditions above (no shared
    caching, no shared code path). See
    backend/services/forecast_service.py and
    backend/scripts/train_forecast_models.py."""
    catalog = await zones()
    match = next((z for z in catalog["zones"] if z["id"] == zone), None)
    if match is None or not match.get("coordinates"):
        raise HTTPException(status_code=400, detail="zone not found or has no point coordinates")
    result = await get_forecast(match["coordinates"]["lat"], match["coordinates"]["lon"])
    return {"zone": zone, **result}


class RouteRequest(BaseModel):
    start: GeoPoint
    # Exactly one of these should be set. destination_zone_id is resolved
    # against GET /zones's own catalog (only "pfz" entries carry a point
    # `coordinates` field — a restricted-area entry has a polygon
    # `geometry` instead and isn't a valid routing destination).
    destination: GeoPoint | None = None
    destination_zone_id: str | None = None


@router.post("/route")
async def route(request: RouteRequest) -> dict:
    """Direct data passthrough for the Route Planner page — no chat/LLM
    involvement, no agent orchestration. Thin wrapper around
    navigation_agent.plan_route. start_offset_km / end_offset_km say how far
    a start/destination on land was moved to reach open water (the route
    itself begins/ends there); distance_km is the sea route only."""
    destination = request.destination

    if destination is None and request.destination_zone_id:
        catalog = await zones()
        match = next((z for z in catalog["zones"] if z["id"] == request.destination_zone_id), None)
        if match is None or not match.get("coordinates"):
            return {
                "route": None,
                "distance_km": None,
                "waypoint_count": 0,
                "start_offset_km": None,
                "end_offset_km": None,
                "reason": "destination_zone_id not found or has no point coordinates",
            }
        destination = GeoPoint(lat=match["coordinates"]["lat"], lon=match["coordinates"]["lon"])

    if destination is None:
        return {
            "route": None,
            "distance_km": None,
            "waypoint_count": 0,
            "start_offset_km": None,
            "end_offset_km": None,
            "reason": "no destination provided (set destination or destination_zone_id)",
        }

    restricted_zones = get_active_restricted_areas()
    start = {"lat": request.start.lat, "lon": request.start.lon}
    dest = {"lat": destination.lat, "lon": destination.lon}
    result = plan_route(start, dest, restricted_zones)
    route_points = result.route

    if route_points is None:
        return {
            "route": None,
            "distance_km": None,
            "waypoint_count": 0,
            "start_offset_km": None,
            "end_offset_km": None,
            "reason": f"No route found — {result.reason}.",
        }

    # plan_route itself only produces waypoints (see its docstring); total
    # distance is derived here from those waypoints with the same
    # haversine_km geospatial.py already uses, rather than reported by
    # plan_route directly. No ETA is returned — that would need an assumed
    # vessel speed nowhere else in this codebase, and fabricating one would
    # break the "degrade, don't invent" convention every agent here follows
    # (e.g. tide_info is left None rather than guessed).
    distance_km = sum(
        haversine_km(route_points[i]["lat"], route_points[i]["lon"], route_points[i + 1]["lat"], route_points[i + 1]["lon"])
        for i in range(len(route_points) - 1)
    )

    return {
        "route": route_points,
        "distance_km": round(distance_km, 2),
        "waypoint_count": len(route_points),
        "start_offset_km": result.start_offset_km,
        "end_offset_km": result.end_offset_km,
        "reason": None,
    }


@router.get("/alerts")
async def alerts() -> dict:
    """Direct data passthrough for the Alerts page — no chat/LLM
    involvement. See backend/services/alerts_service.py."""
    catalog = await zones()
    pfz_zones = [z for z in catalog["zones"] if z["type"] == "pfz"]
    return await get_active_alerts(pfz_zones)


@router.get("/analytics/historical")
async def analytics_historical(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    start_date: str = Query(...),
    end_date: str = Query(...),
    variables: str | None = None,
) -> dict:
    """Direct data passthrough for the Analytics Dashboard page — no
    chat/LLM involvement. See backend/services/analytics_service.py.
    `variables` is a comma-separated subset of VALID_VARIABLES; omitted or
    empty means all of them."""
    var_list = _validate_variables(variables)
    return await get_historical_analytics(lat, lon, start_date, end_date, var_list)


def _build_export_csv(zone_id: str, data: dict) -> str:
    series = data["series"]
    present_variables = list(series.keys())
    all_dates = sorted({d for v in present_variables for d in series[v]["dates"]})
    lookup = {v: dict(zip(series[v]["dates"], series[v]["values"])) for v in present_variables}

    buf = io.StringIO()
    buf.write("# ORCA historical export\n")
    buf.write(f"# zone: {zone_id}\n")
    buf.write(f"# location: {data['lat']}, {data['lon']}\n")
    buf.write(f"# range: {data['start_date']} to {data['end_date']}\n")
    for note in build_export_notes(series):
        buf.write(f"# note: {note}\n")

    writer = csv.writer(buf)
    writer.writerow(["date"] + [f"{v} ({series[v]['unit']})" for v in present_variables])
    for d in all_dates:
        writer.writerow([d] + [lookup[v].get(d, "") for v in present_variables])
    return buf.getvalue()


@router.get("/export")
async def export_data(
    zone: str, start_date: str, end_date: str, variables: str | None = None, format: str = "csv"
) -> Response:
    """Downloadable historical export for the Download page — no chat/LLM
    involvement. Reuses analytics_service's own historical fetch (same as
    GET /analytics/historical) rather than duplicating the Open-Meteo
    calls; only turns that same data into a csv/json/pdf/docx file
    response. See backend/services/export_render.py for the pdf/docx
    rendering step."""
    catalog = await zones()
    match = next((z for z in catalog["zones"] if z["id"] == zone), None)
    if match is None or not match.get("coordinates"):
        raise HTTPException(status_code=400, detail="zone not found or has no point coordinates")

    var_list = _validate_variables(variables)
    data = await get_historical_analytics(
        match["coordinates"]["lat"], match["coordinates"]["lon"], start_date, end_date, var_list
    )

    fmt = format.strip().lower()
    filename_base = "_".join(
        ["orca", _safe_filename_component(zone), _safe_filename_component(start_date), _safe_filename_component(end_date)]
    )
    if fmt == "json":
        payload = {**data, "zone": zone, "notes": build_export_notes(data["series"])}
        return Response(
            content=json.dumps(payload, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.json"'},
        )
    if fmt == "csv":
        return Response(
            content=_build_export_csv(zone, data),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.csv"'},
        )
    if fmt == "pdf":
        return Response(
            content=render_pdf(zone, data),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.pdf"'},
        )
    if fmt == "docx":
        return Response(
            content=render_docx(zone, data),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.docx"'},
        )
    raise HTTPException(status_code=400, detail="format must be one of 'csv', 'json', 'pdf', 'docx'")
