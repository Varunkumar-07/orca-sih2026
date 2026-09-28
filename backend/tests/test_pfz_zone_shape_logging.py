"""Pytest coverage for the skip-and-log behavior in geospatial.py /
reporting.py / visualization.py when a pfz_zone doesn't match the
documented canonical shape ({"zone_id": str, "center": {"lat", "lon"}}).

Added per finding #4 of the shape-mismatch audit: these three modules
already degrade gracefully on a malformed zone (skip it, keep going)
rather than crashing — that's intentional production robustness and this
suite does NOT change it. What was missing was proof that "skip" doesn't
mean "vanish without a trace": before this, one of the three skip paths
in each module had no logging at all (a center dict present but missing
lat/lon), and the ones that did log used plain warning-level text that
didn't make clear this represents a real upstream bug rather than a
normal degradation. This file confirms, for each module, that a malformed
zone is (a) excluded from the result, (b) a well-formed sibling zone
alongside it is NOT dropped too, and (c) an ERROR-level log record naming
the offending zone/shape is actually emitted — not a silent continue.

The malformed fixture used throughout is deliberately the REAL shape from
the incident this hardening followed: a catalog-shaped zone dict
("id"/"name"/"coordinates", no "zone_id"/"center") leaking through
unremapped — see marine_data_agent._live_pfz_candidate_zones and the
shape-mismatch audit's finding #12. A center dict present but missing
lat/lon is covered too, since that skip path had no logging at all before
this change.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import logging

from backend.agents.deterministic.geospatial import run_geospatial
from backend.agents.deterministic.reporting import _build_map_payload_safe
from backend.agents.deterministic.visualization import build_map_payload
from backend.schemas.contracts import EvidenceBundle, GeoPoint, MarineDataResult

_LOCATION = GeoPoint(lat=10.0, lon=76.0)

_GOOD_ZONE = {
    "zone_id": "GOOD-1",
    "center": {"lat": 10.05, "lon": 76.05},
    "distance_km": 5.0,
    "sst_celsius": 28.0,
    "chlorophyll_mg_m3": 0.5,
    "advisory": "test",
}

# The real shape from the incident this hardening followed: a catalog dict
# (id/name/coordinates) with no "center" key at all.
_CATALOG_SHAPED_ZONE = {
    "id": "BAD-CATALOG-1",
    "name": "BAD-CATALOG-1",
    "type": "pfz",
    "near": "Kochi",
    "coordinates": {"lat": 11.0, "lon": 77.0},
    "distance_km": 8.0,
    "sst_celsius": 27.0,
    "chlorophyll_mg_m3": 0.4,
    "advisory": "test",
}

# A center dict that IS present but missing lat/lon — the skip path that
# previously had no logging at all.
_ZONE_WITH_INCOMPLETE_CENTER = {
    "zone_id": "BAD-INCOMPLETE-1",
    "center": {"lat": None, "lon": 78.0},
    "distance_km": 9.0,
}


def _bundle(pfz_zones: list[dict]) -> EvidenceBundle:
    return EvidenceBundle(
        query_text="where can I fish?",
        query_location=_LOCATION,
        marine=MarineDataResult(
            status="ok",
            pfz_zones=pfz_zones,
            sst_celsius=28.0,
            chlorophyll_mg_m3=0.5,
            source_timestamp="2026-01-01T00:00:00Z",
        ),
        weather=None,
        risk=None,
    )


class TestGeospatialSkipAndLog:
    def test_malformed_catalog_shaped_zone_is_skipped_not_crashed(self, caplog):
        bundle = _bundle([_GOOD_ZONE, _CATALOG_SHAPED_ZONE])

        with caplog.at_level(logging.ERROR, logger="backend.agents.deterministic.geospatial"):
            result = run_geospatial(bundle)

        # The well-formed zone is still found — one bad sibling doesn't
        # take down the whole nearest-zone computation.
        assert result.nearest_zone_name == "GOOD-1"
        assert result.distance_km is not None

        assert any(
            "missing expected canonical 'center' dict" in r.message and "BAD-CATALOG-1" in r.message
            for r in caplog.records
        )
        assert all(r.levelno >= logging.ERROR for r in caplog.records)

    def test_zone_with_incomplete_center_is_skipped_and_logged(self, caplog):
        bundle = _bundle([_GOOD_ZONE, _ZONE_WITH_INCOMPLETE_CENTER])

        with caplog.at_level(logging.ERROR, logger="backend.agents.deterministic.geospatial"):
            result = run_geospatial(bundle)

        assert result.nearest_zone_name == "GOOD-1"
        assert any(
            "missing lat/lon" in r.message and "BAD-INCOMPLETE-1" in r.message for r in caplog.records
        )

    def test_only_malformed_zones_present_degrades_cleanly(self, caplog):
        """No well-formed zone at all — must still not raise, and
        nearest_zone_name correctly comes back None rather than crashing
        or picking a bogus fallback."""
        bundle = _bundle([_CATALOG_SHAPED_ZONE])

        with caplog.at_level(logging.ERROR, logger="backend.agents.deterministic.geospatial"):
            result = run_geospatial(bundle)

        assert result.nearest_zone_name is None
        assert result.distance_km is None
        assert len(caplog.records) >= 1


class TestVisualizationSkipAndLog:
    def test_malformed_catalog_shaped_zone_is_skipped_not_crashed(self, caplog):
        bundle = _bundle([_GOOD_ZONE, _CATALOG_SHAPED_ZONE])

        with caplog.at_level(logging.ERROR, logger="backend.agents.deterministic.visualization"):
            payload = build_map_payload(bundle)

        pin_labels = [p["label"] for p in payload.pins if p.get("type") == "pfz"]
        assert pin_labels == ["GOOD-1"]  # bad zone excluded, good zone still rendered
        overlay_names = [o["name"] for o in payload.overlays if o.get("type") == "pfz_zone"]
        assert overlay_names == ["GOOD-1"]

        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        # Both _build_pins and _build_overlays hit the same malformed zone.
        assert sum("missing expected canonical 'center' dict" in m and "BAD-CATALOG-1" in m for m in error_messages) == 2

    def test_zone_with_incomplete_center_is_skipped_and_logged(self, caplog):
        bundle = _bundle([_GOOD_ZONE, _ZONE_WITH_INCOMPLETE_CENTER])

        with caplog.at_level(logging.ERROR, logger="backend.agents.deterministic.visualization"):
            payload = build_map_payload(bundle)

        pin_labels = [p["label"] for p in payload.pins if p.get("type") == "pfz"]
        assert pin_labels == ["GOOD-1"]
        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert sum("missing lat/lon" in m and "BAD-INCOMPLETE-1" in m for m in error_messages) == 2


class TestReportingFallbackSkipAndLog:
    """_build_map_payload_safe is reporting.py's own defensive fallback
    (only used if visualization.build_map_payload itself raises) — it has
    the identical skip-a-malformed-zone logic, tested separately here."""

    def test_malformed_catalog_shaped_zone_is_skipped_not_crashed(self, caplog):
        bundle = _bundle([_GOOD_ZONE, _CATALOG_SHAPED_ZONE])

        with caplog.at_level(logging.ERROR, logger="backend.agents.deterministic.reporting"):
            payload = _build_map_payload_safe(bundle)

        pfz_pins = [p for p in payload.pins if p.get("type") == "pfz"]
        assert len(pfz_pins) == 1
        assert pfz_pins[0]["label"] == "GOOD-1"

        assert any(
            "missing expected canonical 'center' dict" in r.message and "BAD-CATALOG-1" in r.message
            for r in caplog.records
        )

    def test_zone_with_incomplete_center_is_skipped_and_logged(self, caplog):
        bundle = _bundle([_GOOD_ZONE, _ZONE_WITH_INCOMPLETE_CENTER])

        with caplog.at_level(logging.ERROR, logger="backend.agents.deterministic.reporting"):
            payload = _build_map_payload_safe(bundle)

        pfz_pins = [p for p in payload.pins if p.get("type") == "pfz"]
        assert len(pfz_pins) == 1
        assert pfz_pins[0]["label"] == "GOOD-1"
        assert any(
            "missing lat/lon" in r.message and "BAD-INCOMPLETE-1" in r.message for r in caplog.records
        )
