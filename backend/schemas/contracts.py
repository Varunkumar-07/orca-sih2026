"""
ORCA shared interface contract — backend/schemas/contracts.py

Frozen and shared between Team Claude (reasoning agents) and Team Gemini
(deterministic modules + frontend). Neither team edits this unilaterally.

Conventions:
- Coordinates: (lat, lon) in that order, human order — NOT GeoJSON's (lon, lat).
  Convert to GeoJSON order only at the boundary where it's emitted (e.g. for Leaflet).
- Distance: kilometers, float.
- Temperature: Celsius, float.
- Wind / wave speed: km/h, float.
- Timestamps: ISO 8601, UTC, string (e.g. "2026-08-27T06:00:00Z").
"""

from typing import Literal

from pydantic import BaseModel


class GeoPoint(BaseModel):
    lat: float
    lon: float


class MarineDataResult(BaseModel):
    status: Literal["ok", "partial", "error"]
    # Canonical zone shape: {"zone_id": str, "center": {"lat": float, "lon": float}, ...}
    pfz_zones: list[dict]
    sst_celsius: float | None
    chlorophyll_mg_m3: float | None
    source_timestamp: str
    error_message: str | None = None


class WeatherDataResult(BaseModel):
    status: Literal["ok", "partial", "error"]
    wind_kmh: float | None
    wave_height_m: float | None
    cyclone_alert: bool
    lightning_alert: bool
    tide_info: dict | None
    source_timestamp: str
    error_message: str | None = None


class RiskAssessment(BaseModel):
    status: Literal["ok", "partial", "error"]
    safe_to_go: bool | None
    confidence: float  # 0.0-1.0
    explanation: str
    error_message: str | None = None


# ---- Team Claude consumes everything above, produces everything below ----
# ---- (analytics / geospatial are filled by Team Gemini's modules) ----


class AnalyticsResult(BaseModel):
    anomalies: list[str]
    trend_summary: str


class GeospatialResult(BaseModel):
    nearest_zone_name: str | None
    distance_km: float | None
    inside_restricted_area: bool
    restricted_area_name: str | None = None


class TraceStep(BaseModel):
    agent_name: str
    input_summary: str
    output_summary: str
    timestamp: str


class EvidenceBundle(BaseModel):
    """The one object that crosses the team boundary."""

    query_text: str
    query_location: GeoPoint | None
    marine: MarineDataResult | None
    weather: WeatherDataResult | None
    risk: RiskAssessment | None
    analytics: AnalyticsResult | None = None  # Team Gemini fills this
    geospatial: GeospatialResult | None = None  # Team Gemini fills this
    trace: list[TraceStep] = []  # both sides append here


class MapPayload(BaseModel):
    pins: list[dict]  # [{lat, lon, label, type}]
    overlays: list[dict]  # zone/boundary polygons, GeoJSON-converted at this boundary only
    # Ordered waypoints (lat, lon) from the Navigation Agent's A* route, start to
    # destination. None until that agent exists / for any query that never reaches
    # a chosen destination — always optional, never required by existing producers.
    route: list[GeoPoint] | None = None


class FinalResponse(BaseModel):
    answer_text: str
    reasoning_trace: list[TraceStep]
    map_payload: MapPayload
    # Set only by the Language Agent (input/output sides). None until that agent
    # exists — every current producer keeps answering in English, undetected.
    detected_language: str | None = None  # source language the query was detected in
    response_language: str | None = None  # language answer_text was translated back into
