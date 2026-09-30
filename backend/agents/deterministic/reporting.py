"""Reporting deterministic module.

Merges all specialist outputs (marine, weather, risk, analytics, geospatial)
into a clean FinalResponse. Gracefully degrades on partial/error states
(e.g. Fixture 3 where weather is error) without throwing uncaught exceptions.

Every invocation appends a TraceStep to bundle.trace before returning.
"""
from __future__ import annotations

import logging

from backend.agents.deterministic.geospatial import extract_pfz_center
from backend.schemas.contracts import (
    EvidenceBundle,
    FinalResponse,
    MapPayload,
    TraceStep,
)
from backend.time_utils import now_iso as _iso_now

logger = logging.getLogger(__name__)

# Import visualization helper to build MapPayload (no circular dependency: visualization does not import reporting)
from backend.agents.deterministic.visualization import build_map_payload


def _format_marine(bundle: EvidenceBundle) -> str:
    if bundle.marine is None:
        return "Marine data unavailable."
    m = bundle.marine
    if m.status == "error":
        msg = m.error_message or "unknown error"
        return f"Marine data unavailable ({msg})."
    parts: list[str] = []
    # SST / chlorophyll
    if m.sst_celsius is not None:
        parts.append(f"SST {m.sst_celsius:.1f}°C")
    else:
        parts.append("SST unavailable")
    if m.chlorophyll_mg_m3 is not None:
        parts.append(f"chlorophyll {m.chlorophyll_mg_m3:.2f} mg/m³")
    else:
        parts.append("chlorophyll unavailable")
    # PFZ zones
    if m.pfz_zones:
        zone_names: list[str] = []
        for z in m.pfz_zones:
            zone_names.append(str(z.get("zone_id", z.get("name", z.get("zone_name", "PFZ")))))
        parts.append(f"PFZ zones: {', '.join(zone_names)}")
    else:
        parts.append("No PFZ zones identified")
    # Status note
    if m.status == "partial" and m.error_message:
        parts.append(f"Note: {m.error_message}")
    return "; ".join(parts) + "."


def _format_weather(bundle: EvidenceBundle) -> str:
    if bundle.weather is None:
        return "Weather data unavailable."
    w = bundle.weather
    if w.status == "error":
        msg = w.error_message or "unknown error"
        return f"Weather data unavailable ({msg}). Assessment will proceed with marine data only."
    parts: list[str] = []
    if w.wind_kmh is not None:
        parts.append(f"wind {w.wind_kmh:.1f} km/h")
    else:
        parts.append("wind unavailable")
    if w.wave_height_m is not None:
        parts.append(f"wave height {w.wave_height_m:.1f} m")
    else:
        parts.append("wave height unavailable")
    alerts: list[str] = []
    if w.cyclone_alert:
        alerts.append("cyclone alert active")
    if w.lightning_alert:
        alerts.append("lightning alert active")
    if alerts:
        parts.append("Alerts: " + ", ".join(alerts))
    else:
        parts.append("No active cyclone/lightning alerts")
    if w.tide_info:
        parts.append(_format_tide(w.tide_info))
    if w.status == "partial" and w.error_message:
        parts.append(f"Note: {w.error_message}")
    return "; ".join(parts) + "."


def _format_tide(tide_info) -> str:
    """Tide as readable text — tide_info is a {"high_tide": "HH:MM",
    "low_tide": "HH:MM"} dict (only the demo fixtures carry one; no live
    source exists), never printed as a raw dict."""
    if isinstance(tide_info, dict):
        labels = {"high_tide": "high tide", "low_tide": "low tide"}
        items = [f"{labels.get(k, str(k).replace('_', ' '))} {v}" for k, v in tide_info.items() if v]
        if items:
            return "Tide: " + ", ".join(items)
        return "Tide: unavailable"
    return f"Tide: {tide_info}"


def _format_headline(bundle: EvidenceBundle) -> str:
    """One plain-language verdict sentence — the first thing a fisherman
    reads, mirroring the restricted-zone branch's headline-first pattern in
    _build_answer. Reuses _format_risk's existing confidence/explanation
    text, just fronted with a verdict icon so the verdict itself is
    unmistakable at a glance rather than buried after a technical dump."""
    r = bundle.risk
    if r is None or r.safe_to_go is None:
        icon = "❓"
    elif r.safe_to_go:
        icon = "✅"
    else:
        icon = "⚠️"
    return f"{icon} {_format_risk(bundle)}"


def _format_risk(bundle: EvidenceBundle) -> str:
    if bundle.risk is None:
        return "Safety assessment not available."
    r = bundle.risk
    # A rule-based verdict (a hard limit decided it — see safety_limits.py)
    # has no probability behind it, so it says so instead of showing a %.
    if r.verdict_source == "rules":
        qualifier = "limit breached"
    elif r.confidence is None:
        qualifier = None
    else:
        qualifier = f"confidence {r.confidence:.0%}"
    suffix = f" ({qualifier})" if qualifier else ""
    if r.safe_to_go is True:
        base = f"Safe to go{suffix}"
    elif r.safe_to_go is False:
        base = f"Not safe to go{suffix}"
    else:
        base = f"Safety assessment inconclusive{suffix}"
    if r.explanation:
        base += f" — {r.explanation}"
    if r.status in ("partial", "error") and r.error_message:
        base += f" [{r.status}: {r.error_message}]"
    return base


def _format_analytics(bundle: EvidenceBundle) -> str:
    if bundle.analytics is None:
        return "Ocean analytics not yet computed."
    a = bundle.analytics
    # trend_summary already incorporates the anomaly list (see analytics.py's
    # run_ocean_analytics) — don't re-list anomalies here too, or they show
    # up twice in the final answer text.
    return a.trend_summary or "No anomalies detected."


def _nearest_zone_is_distant(bundle: EvidenceBundle) -> bool:
    """Phase 4 (Chat-PFZ Live-Data Integration): whether the geospatial
    module's picked nearest zone is a live-cache match from a distant
    anchor region (see pfz_service.NearestAnchorZones.is_distant), rather
    than a genuinely nearby result. Read directly off bundle.marine.pfz_zones
    (matched by zone_id against g.nearest_zone_name) instead of threading a
    new field through GeospatialResult/contracts.py — that schema is frozen
    and shared with Team Gemini, and every zone dict already carries
    whatever extra keys its producer wants (e.g. "near", "advisory"), so an
    additive "is_distant" key here follows the same existing convention."""
    if bundle.marine is None or bundle.geospatial is None or not bundle.geospatial.nearest_zone_name:
        return False
    name = bundle.geospatial.nearest_zone_name
    return any(str(z.get("zone_id")) == name and z.get("is_distant") for z in bundle.marine.pfz_zones)


def _format_geospatial(bundle: EvidenceBundle) -> str:
    if bundle.geospatial is None:
        return "Location analysis not yet computed."
    g = bundle.geospatial
    if g.inside_restricted_area:
        name = g.restricted_area_name or "restricted area"
        text = f"Inside {name} — fishing is prohibited regardless of other conditions."
        if g.restricted_area_approximate:
            text += " (Approximate extent: WDPA records this site as a point with a reported area, not a surveyed boundary.)"
        return text
    parts: list[str] = []
    if g.nearest_zone_name:
        dist = f" ({g.distance_km} km away)" if g.distance_km is not None else ""
        if _nearest_zone_is_distant(bundle):
            parts.append(
                f"Nearest known fishing-zone data: {g.nearest_zone_name}{dist} — this is the closest live "
                "data available, but it's well outside typical local range; treat it as a regional "
                "reference, not a nearby recommendation"
            )
        else:
            parts.append(f"Nearest fishing zone: {g.nearest_zone_name}{dist}")
    else:
        parts.append("No nearby fishing zone identified")
    if g.near_restricted_area_name:
        parts.append(
            f"Outside restricted areas, but {g.near_restricted_area_km} km from {g.near_restricted_area_name} "
            "(protected area) — keep clear of its boundary"
        )
    else:
        parts.append("Outside restricted areas")
    return "; ".join(parts) + "."


def _build_answer(bundle: EvidenceBundle) -> str:
    """Compose the human-readable answer_text with graceful degradation."""
    lines: list[str] = []

    # Query echo
    lines.append(f"Query: {bundle.query_text}")

    # Location
    if bundle.query_location:
        lines.append(f"Location: {bundle.query_location.lat:.4f}, {bundle.query_location.lon:.4f}")
    else:
        lines.append("Location: not specified")

    # Restricted area takes absolute precedence — headline must scream PROHIBITED
    # so judges never see "Safe to go" alongside a restriction footnote (Team Claude feedback)
    if bundle.geospatial and bundle.geospatial.inside_restricted_area:
        name = bundle.geospatial.restricted_area_name or "restricted area"
        # Headline front and center — restriction name in headline, not buried
        lines.append(f"⛔ PROHIBITED — Fishing & Entry Restricted: Inside {name}")
        lines.append(
            f"Entry and fishing are strictly prohibited in {name}. Do not proceed — "
            f"this restriction overrides all other sea and weather conditions."
        )
        # Qualified safety — override any optimistic risk assessment. Once
        # the hard limits have applied PROHIBITED (verdict_source "rules"),
        # the LLM's original verdict is in the trace, not repeated here.
        if bundle.risk is not None and bundle.risk.verdict_source != "rules":
            orig_risk = _format_risk(bundle)
            lines.append(f"Safety: RESTRICTED — Not permitted (overrides: {orig_risk})")
        else:
            lines.append("Safety: RESTRICTED — Not permitted")
        # Still include supporting evidence but after the headline
        lines.append(f"Marine: {_format_marine(bundle)}")
        lines.append(f"Weather: {_format_weather(bundle)}")
        lines.append(f"Geospatial: {_format_geospatial(bundle)}")
        lines.append(f"Analytics: {_format_analytics(bundle)}")
        lines.append(
            "Note: Restricted-zone status takes absolute precedence — even calm seas do not permit entry. "
            "Refer to official marine park regulations."
        )
        return "\n".join(lines)

    # Normal path — fisherman-friendly: one plain-language headline verdict
    # first (mirrors the restricted-zone branch above), then the supporting
    # facts that justify it — not the full technical dump. The Marine line
    # (SST, chlorophyll, the zones found) stays: it is where "where should I
    # fish?" gets answered, and it used to appear only on PROHIBITED answers.
    # Ocean-analytics trend detail isn't repeated here; it stays in the live
    # reasoning trace panel for anyone who wants to audit the evidence.
    lines.append(_format_headline(bundle))
    lines.append(f"Marine: {_format_marine(bundle)}")
    lines.append(f"Weather: {_format_weather(bundle)}")
    lines.append(f"Geospatial: {_format_geospatial(bundle)}")

    # Degradation notice for partial/error bundles — crucial for Fixture 3
    statuses: list[str] = []
    if bundle.marine and bundle.marine.status in ("partial", "error"):
        statuses.append(f"marine:{bundle.marine.status}")
    if bundle.weather and bundle.weather.status in ("partial", "error"):
        statuses.append(f"weather:{bundle.weather.status}")
    if bundle.risk and bundle.risk.status in ("partial", "error"):
        statuses.append(f"risk:{bundle.risk.status}")
    if statuses:
        lines.append(f"Note: partial data — {', '.join(statuses)}; advice is based on available evidence only.")

    return "\n".join(lines)


def _build_map_payload_safe(bundle: EvidenceBundle) -> MapPayload:
    """Build MapPayload crash-proof, handling import fallback."""
    try:
        return build_map_payload(bundle)
    except Exception:
        pass
    # Fallback minimal implementation
    pins: list[dict] = []
    overlays: list[dict] = []
    if bundle.query_location:
        pins.append(
            {
                "lat": bundle.query_location.lat,
                "lon": bundle.query_location.lon,
                "label": "Query location",
                "type": "query",
            }
        )
    if bundle.marine:
        for z in bundle.marine.pfz_zones:
            extracted = extract_pfz_center(z, logger, "skipping", missing_id_label="PFZ")
            if extracted is None:
                continue
            zlat, zlon, zname = extracted
            try:
                pins.append({"lat": float(zlat), "lon": float(zlon), "label": str(zname), "type": "pfz"})
                overlays.append(
                    {"type": "pfz_zone", "name": str(zname), "geojson": {"type": "Point", "coordinates": [float(zlon), float(zlat)]}}
                )
            except Exception as exc:
                logger.error(
                    "reporting: pfz_zone %r had a 'center' dict but failed to render as a pin/overlay: %s",
                    zname, exc,
                )
                continue
    return MapPayload(pins=pins, overlays=overlays)


def run_reporting(bundle: EvidenceBundle) -> FinalResponse:
    """Merge all evidence into FinalResponse.

    Appends a TraceStep to bundle.trace before returning.
    Never raises — always returns a valid FinalResponse.
    """
    input_summary = ""
    error_flag = False
    answer_text = ""
    map_payload: MapPayload | None = None

    # Capture input summary early for trace
    try:
        marine_status = bundle.marine.status if bundle.marine else "None"
        weather_status = bundle.weather.status if bundle.weather else "None"
        risk_status = bundle.risk.status if bundle.risk else "None"
        analytics_info = f"{len(bundle.analytics.anomalies)} anomalies" if bundle.analytics else "None"
        geospatial_info = (
            f"inside={bundle.geospatial.inside_restricted_area}" if bundle.geospatial else "None"
        )
        input_summary = (
            f"query='{bundle.query_text[:40]}', marine:{marine_status}, weather:{weather_status}, "
            f"risk:{risk_status}, analytics:{analytics_info}, geospatial:{geospatial_info}"
        )
    except Exception as exc:
        input_summary = f"error capturing inputs: {exc}"
        error_flag = True

    try:
        answer_text = _build_answer(bundle)
        map_payload = _build_map_payload_safe(bundle)
    except Exception as exc:  # graceful degradation
        error_flag = True
        answer_text = (
            f"Partial response for '{bundle.query_text}': an error occurred while composing the report "
            f"({exc}). Available data: marine={bundle.marine is not None}, weather={bundle.weather is not None}."
        )
        try:
            map_payload = _build_map_payload_safe(bundle)
        except Exception:
            map_payload = MapPayload(pins=[], overlays=[])

    # Ensure map_payload is always set
    if map_payload is None:
        map_payload = MapPayload(pins=[], overlays=[])

    # Append trace step BEFORE constructing reasoning_trace so it is included
    output_summary = f"answer_len={len(answer_text)}, pins={len(map_payload.pins)}, overlays={len(map_payload.overlays)}"
    if error_flag:
        output_summary = f"error — {output_summary}"

    trace = TraceStep(
        agent_name="reporting",
        input_summary=input_summary[:500],
        output_summary=output_summary[:500],
        timestamp=_iso_now(),
    )
    try:
        bundle.trace.append(trace)
    except Exception:
        pass

    # reasoning_trace is a snapshot of bundle.trace (includes reporting step)
    try:
        reasoning_trace = list(bundle.trace)
    except Exception:
        reasoning_trace = [trace]

    return FinalResponse(
        answer_text=answer_text,
        reasoning_trace=reasoning_trace,
        map_payload=map_payload,
    )
