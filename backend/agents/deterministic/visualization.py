"""Visualization deterministic module.

Converts structured outputs into map pins and zone/boundary polygon overlays.
Coordinate conversion (lat,lon) -> GeoJSON (lon,lat) happens ONLY at this boundary.

Never raises across the integration boundary.
"""
from __future__ import annotations

import logging

from shapely.geometry import mapping as shapely_mapping

from backend.agents.deterministic.geospatial import extract_pfz_center
from backend.schemas.contracts import (
    EvidenceBundle,
    FinalResponse,
    MapPayload,
    TraceStep,
)
from backend.time_utils import now_iso as _iso_now

logger = logging.getLogger(__name__)

# Draw whatever boundaries geospatial.py actually used for the containment
# check (real WDPA cache, or its hardcoded fallback) — call this at overlay
# build time rather than importing a static list, so the overlay can never
# drift from the restricted_area_name already reported in bundle.geospatial.
try:
    from backend.agents.deterministic.geospatial import get_active_restricted_areas
except Exception:  # pragma: no cover
    from shapely.geometry import Polygon as _Polygon

    def get_active_restricted_areas() -> list[dict]:  # type: ignore[misc]
        return [
            {
                "name": "Gulf of Mannar Marine National Park",
                "polygon": _Polygon([(78.0, 8.2), (79.6, 8.2), (79.6, 9.6), (78.0, 9.6)]),
            }
        ]


def _build_pins(bundle: EvidenceBundle) -> list[dict]:
    """Build map pins from bundle.

    Each pin is {lat, lon, label, type} in human order (lat,lon) as per MapPayload spec.
    Frontend will convert to GeoJSON if needed; overlays are already GeoJSON.

    Multi-candidate PFZ discovery: pfz_zones is a ranked candidate list
    (1..N zones). Each candidate is rendered as its own distinct selectable
    pin (type="pfz", label=zone_id) so the frontend can let users click a
    specific zone to select/inspect it. Pins are 1:1 with pfz_zones entries —
    no deduplication or merging.
    """
    pins: list[dict] = []

    # Query location pin
    if bundle.query_location is not None:
        pins.append(
            {
                "lat": bundle.query_location.lat,
                "lon": bundle.query_location.lon,
                "label": "Query location",
                "type": "query",
            }
        )

    # PFZ zone pins from marine data — multi-candidate: one distinct selectable
    # pin per entry in pfz_zones (Phase 4). Frontend uses `label` (zone_id)
    # as the selection key via onSelectZone.
    if bundle.marine is not None:
        for zone in bundle.marine.pfz_zones:
            extracted = extract_pfz_center(zone, logger, "skipping pin", missing_id_label="PFZ")
            if extracted is None:
                continue
            zlat, zlon, zname = extracted
            try:
                pin: dict = {
                    "lat": float(zlat),
                    "lon": float(zlon),
                    "label": str(zname),
                    "type": "pfz",
                    "zone_id": str(zname),
                    "center": {"lat": float(zlat), "lon": float(zlon)},
                }
                # Full PFZ schema fields — include when present on the zone
                # (live multi-candidate provides all; fixtures may provide subset)
                for key in ("distance_km", "sst_celsius", "chlorophyll_mg_m3", "advisory"):
                    val = zone.get(key)
                    if val is not None:
                        try:
                            # Cast numeric fields to float for consistent type
                            if key in ("distance_km", "sst_celsius", "chlorophyll_mg_m3"):
                                pin[key] = float(val)
                            else:
                                pin[key] = str(val)
                        except Exception:
                            pin[key] = val
                pins.append(pin)
            except Exception as exc:
                logger.error(
                    "visualization: pfz_zone %r had a 'center' dict but failed to render as a pin: %s",
                    zname, exc,
                )
                continue

    # A separate nearest-zone pin is intentionally not added here since it's already
    # covered by the marine pfz_zones pins built earlier in this function.
    # Each PFZ candidate already has its own distinct pin above.
    return pins


def _build_overlays(bundle: EvidenceBundle) -> list[dict]:
    """Build zone/boundary polygon overlays in GeoJSON format.

    Conversion (lat,lon) -> GeoJSON (lon,lat) happens here.
    """
    overlays: list[dict] = []

    # Restricted area overlays — always include if query is inside or if fixture is restricted
    # For visualization, we emit the restricted polygon as GeoJSON so frontend can render it.
    # We highlight the relevant restricted area if inside; otherwise we can optionally show all.
    inside = False
    active_name = None
    if bundle.geospatial is not None:
        inside = bundle.geospatial.inside_restricted_area
        active_name = bundle.geospatial.restricted_area_name

    for area in get_active_restricted_areas():
        name = area["name"]
        poly = area["polygon"]
        # Shapely geometries already store coords as (lon,lat) => GeoJSON
        # order directly. mapping() handles Polygon/MultiPolygon/etc
        # uniformly — real WDPA boundaries include MultiPolygons (e.g.
        # Gulf of Mannar's island clusters), unlike the old single-Polygon
        # hardcoded fallback.
        try:
            geojson_geom = shapely_mapping(poly)
            overlays.append(
                {
                    "type": "restricted_area",
                    "name": name,
                    "geojson": geojson_geom,
                    "highlighted": bool(inside and name == active_name),
                }
            )
        except Exception:
            continue

    # PFZ zone overlays — multi-candidate: one distinct overlay per pfz_zones
    # entry, each as a GeoJSON Point feature (selectable alongside its pin).
    if bundle.marine is not None:
        for zone in bundle.marine.pfz_zones:
            extracted = extract_pfz_center(zone, logger, "skipping overlay", missing_id_label="PFZ")
            if extracted is None:
                continue
            zlat, zlon, zname = extracted
            try:
                overlay: dict = {
                    "type": "pfz_zone",
                    "name": str(zname),
                    "geojson": {"type": "Point", "coordinates": [float(zlon), float(zlat)]},
                    "zone_id": str(zname),
                    "center": {"lat": float(zlat), "lon": float(zlon)},
                }
                for key in ("distance_km", "sst_celsius", "chlorophyll_mg_m3", "advisory"):
                    val = zone.get(key)
                    if val is not None:
                        try:
                            if key in ("distance_km", "sst_celsius", "chlorophyll_mg_m3"):
                                overlay[key] = float(val)
                            else:
                                overlay[key] = str(val)
                        except Exception:
                            overlay[key] = val
                overlays.append(overlay)
            except Exception as exc:
                logger.error(
                    "visualization: pfz_zone %r had a 'center' dict but failed to render as an overlay: %s",
                    zname, exc,
                )
                continue

    return overlays


def build_map_payload(bundle: EvidenceBundle) -> MapPayload:
    """Public helper used by reporting to build MapPayload without trace side-effect."""
    pins = _build_pins(bundle)
    overlays = _build_overlays(bundle)
    return MapPayload(pins=pins, overlays=overlays)


def run_visualization(final: FinalResponse, bundle: EvidenceBundle) -> MapPayload:
    """Convert structured outputs into map payload.

    Appends a TraceStep to bundle.trace before returning.
    Handles missing data gracefully — always returns valid MapPayload.

    Args:
        final: The FinalResponse from reporting (may be used for label enrichment future).
        bundle: EvidenceBundle for location/zone data.

    Returns:
        MapPayload with pins and overlays in GeoJSON-converted form.
    """
    input_summary = ""
    error_flag = False

    try:
        pin_hint = len(bundle.marine.pfz_zones) if bundle.marine and bundle.marine.pfz_zones else 0
        loc = f"({bundle.query_location.lat:.2f},{bundle.query_location.lon:.2f})" if bundle.query_location else "None"
        input_summary = f"loc={loc}, pfz_zones={pin_hint}, inside_restricted={bundle.geospatial.inside_restricted_area if bundle.geospatial else 'unknown'}"
    except Exception as exc:
        input_summary = f"error capturing inputs: {exc}"
        error_flag = True

    # Build payload crash-proof
    try:
        payload = build_map_payload(bundle)
    except Exception as exc:
        error_flag = True
        payload = MapPayload(pins=[], overlays=[])
        input_summary = input_summary or f"visualization build failed: {exc}"

    # Optionally enrich final.map_payload if provided (keep deterministic modules decoupled)
    # We do NOT mutate final in place beyond trace; caller (reporting) already set final.map_payload.

    output_summary = f"pins={len(payload.pins)}, overlays={len(payload.overlays)}"
    if error_flag:
        output_summary = f"error — {output_summary}"

    trace = TraceStep(
        agent_name="visualization",
        input_summary=input_summary[:500],
        output_summary=output_summary[:500],
        timestamp=_iso_now(),
    )
    try:
        bundle.trace.append(trace)
    except Exception:
        pass

    return payload
