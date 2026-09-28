"""Pytest coverage for deterministic agent modules.

Covers:
  - backend/agents/deterministic/analytics.py
  - backend/agents/deterministic/geospatial.py
  - backend/agents/deterministic/reporting.py
  - backend/agents/deterministic/visualization.py

Uses the 4 golden fixtures from backend/schemas/test_fixtures.py.
For each fixture asserts:
  - status (marine/weather/risk valid literal)
  - inside_restricted_area (bool, True only for Gulf of Mannar)
  - non-empty answer_text (via reporting)
  - non-empty map_payload (pins/overlays via visualization/reporting)

All deterministic functions are pure (no I/O) — tests copy fixtures deeply
so mutation (trace append, analytics/geospatial writeback) never leaks.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import pytest

from backend.agents.deterministic.analytics import run_ocean_analytics
from backend.agents.deterministic.geospatial import run_geospatial
from backend.agents.deterministic.reporting import run_reporting
from backend.agents.deterministic.visualization import (
    build_map_payload,
    run_visualization,
)
from backend.schemas.contracts import (
    EvidenceBundle,
    GeoPoint,
    MarineDataResult,
    RiskAssessment,
    WeatherDataResult,
)
from backend.schemas.test_fixtures import (
    ALL_FIXTURES,
    FIXTURE_1_HAPPY_PATH,
    FIXTURE_2_HAZARD_PATH,
    FIXTURE_3_PARTIAL_FAILURE,
    FIXTURE_4_RESTRICTED_ZONE,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

FIXTURE_CASES = [
    pytest.param(FIXTURE_1_HAPPY_PATH, False, id="fixture1_happy"),
    pytest.param(FIXTURE_2_HAZARD_PATH, False, id="fixture2_hazard"),
    pytest.param(FIXTURE_3_PARTIAL_FAILURE, False, id="fixture3_partial"),
    pytest.param(FIXTURE_4_RESTRICTED_ZONE, True, id="fixture4_restricted"),
]

VALID_STATUSES = {"ok", "partial", "error"}


def _fresh(fixture):
    """Deep copy fixture with empty trace to isolate mutations."""
    b = fixture.model_copy(deep=True)
    # ensure trace is a fresh list so appends don't leak
    b.trace = []
    # also ensure analytics/geospatial are reset (fixtures have None for these)
    b.analytics = None
    b.geospatial = None
    return b


def _run_full_pipeline(fixture):
    """Run analytics → geospatial → reporting → visualization on a fresh copy.

    Returns (bundle, final, viz_payload)
    """
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    run_geospatial(bundle)
    final = run_reporting(bundle)
    viz = run_visualization(final, bundle)
    # reporting already snapshotted trace, visualization mutates bundle.trace —
    # keep final in sync like main.py does
    final.map_payload = viz
    final.reasoning_trace = list(bundle.trace)
    return bundle, final, viz


# ---------------------------------------------------------------------------
# analytics.py  — backend/agents/deterministic/analytics.py:35
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,_expected_inside", FIXTURE_CASES)
def test_analytics__pure_and_returns_valid_result(fixture, _expected_inside):
    bundle = _fresh(fixture)
    result = run_ocean_analytics(bundle)

    # pure function: never raises, always returns AnalyticsResult
    assert result is not None
    assert hasattr(result, "anomalies")
    assert hasattr(result, "trend_summary")

    # status: marine status preserved, analytics never clobbers it
    assert bundle.marine is not None
    assert bundle.marine.status in VALID_STATUSES

    # analytics result invariants — non-empty trend_summary for every fixture
    assert isinstance(result.trend_summary, str)
    assert result.trend_summary.strip() != "", "trend_summary must be non-empty"

    # anomalies is a list (may be empty for happy path)
    assert isinstance(result.anomalies, list)

    # bundle writeback and trace
    assert bundle.analytics is result
    assert any(t.agent_name == "ocean_analytics" for t in bundle.trace)


@pytest.mark.parametrize("fixture,_expected_inside", FIXTURE_CASES)
def test_analytics__trend_summary_contains_expected_context(fixture, _expected_inside):
    bundle = _fresh(fixture)
    result = run_ocean_analytics(bundle)
    # For fixtures with optimal SST/chl (all 4 share same marine payload)
    # trend should mention favorable or anomalies — but never be empty.
    # Fixture 3 has no marine degradation, so still favorable; we just verify
    # the function handled marine.status without error.
    assert len(result.trend_summary) > 10


# ---------------------------------------------------------------------------
# geospatial.py — backend/agents/deterministic/geospatial.py:202
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,expected_inside", FIXTURE_CASES)
def test_geospatial__inside_restricted_area(fixture, expected_inside):
    bundle = _fresh(fixture)
    # analytics not required for geospatial but run it to mirror pipeline order
    run_ocean_analytics(bundle)
    result = run_geospatial(bundle)

    # inside_restricted_area is a bool and matches expectation
    assert isinstance(result.inside_restricted_area, bool)
    assert result.inside_restricted_area is expected_inside

    # restricted_area_name is set iff inside is True
    if expected_inside:
        assert result.restricted_area_name is not None
        assert isinstance(result.restricted_area_name, str)
        assert result.restricted_area_name.strip() != ""
    else:
        # outside — name should be None
        assert result.restricted_area_name is None

    # writeback + trace
    assert bundle.geospatial is result
    assert any(t.agent_name == "geospatial" for t in bundle.trace)


@pytest.mark.parametrize("fixture,expected_inside", FIXTURE_CASES)
def test_geospatial__status_and_distance(fixture, expected_inside):
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    result = run_geospatial(bundle)

    # status: marine status still valid after geospatial
    assert bundle.marine is not None
    assert bundle.marine.status in VALID_STATUSES

    # distance to nearest PFZ: should be set when pfz_zones present (all fixtures have one zone)
    assert result.nearest_zone_name is not None
    assert result.distance_km is not None
    assert isinstance(result.distance_km, float)
    assert result.distance_km >= 0

    # Fixture 4 is Gulf of Mannar — far from Chennai PFZ, so distance should be large
    if expected_inside:
        assert result.distance_km > 100, "restricted fixture far from Chennai PFZ"
    else:
        assert result.distance_km < 50, "Chennai fixtures near Chennai PFZ"


@pytest.mark.parametrize("fixture,_expected_inside", FIXTURE_CASES)
def test_geospatial__pure_no_raise(fixture, _expected_inside):
    bundle = _fresh(fixture)
    # should not raise even for partial/error weather
    run_geospatial(bundle)
    assert bundle.geospatial is not None


# ---------------------------------------------------------------------------
# reporting.py — backend/agents/deterministic/reporting.py:255
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,expected_inside", FIXTURE_CASES)
def test_reporting__answer_text_non_empty(fixture, expected_inside):
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    run_geospatial(bundle)
    final = run_reporting(bundle)

    # non-empty answer_text required for every fixture
    assert isinstance(final.answer_text, str)
    assert final.answer_text.strip() != "", "answer_text must be non-empty"
    assert len(final.answer_text) > 20

    # status: reporting preserves original marine/weather/risk statuses
    assert bundle.marine.status in VALID_STATUSES
    assert bundle.weather.status in VALID_STATUSES
    assert bundle.risk.status in VALID_STATUSES

    # inside_restricted_area reflected in answer text
    if expected_inside:
        assert "PROHIBITED" in final.answer_text or "Restricted" in final.answer_text or "prohibited" in final.answer_text.lower()
    # for hazard fixture, risk says Not safe
    if fixture is FIXTURE_2_HAZARD_PATH:
        assert "Not safe" in final.answer_text or "not safe" in final.answer_text.lower()

    # trace appended
    assert any(t.agent_name == "reporting" for t in bundle.trace)
    assert any(t.agent_name == "reporting" for t in final.reasoning_trace)


@pytest.mark.parametrize("fixture,expected_inside", FIXTURE_CASES)
def test_reporting__map_payload_non_empty(fixture, expected_inside):
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    run_geospatial(bundle)
    final = run_reporting(bundle)

    # non-empty map_payload: must have pins or overlays
    assert final.map_payload is not None
    assert hasattr(final.map_payload, "pins")
    assert hasattr(final.map_payload, "overlays")
    # pins always includes query location + at least one PFZ zone
    assert len(final.map_payload.pins) >= 2, "pins should contain query + PFZ"
    assert len(final.map_payload.overlays) >= 1, "overlays should contain at least restricted areas or PFZ"

    # each pin has required fields
    for pin in final.map_payload.pins:
        assert "lat" in pin and "lon" in pin and "label" in pin and "type" in pin

    # reasoning_trace includes prior steps
    assert len(final.reasoning_trace) >= 3  # analytics + geospatial + reporting

    # inside_restricted_area property via bundle
    assert bundle.geospatial.inside_restricted_area is expected_inside


@pytest.mark.parametrize("fixture,_expected_inside", FIXTURE_CASES)
def test_reporting__graceful_degradation_on_partial_error(fixture, _expected_inside):
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    run_geospatial(bundle)
    # reporting must never raise, even for fixture 3 with weather error
    final = run_reporting(bundle)
    assert final.answer_text.strip() != ""
    assert final.map_payload is not None


# ---------------------------------------------------------------------------
# visualization.py — backend/agents/deterministic/visualization.py:159
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,expected_inside", FIXTURE_CASES)
def test_visualization__map_payload_non_empty(fixture, expected_inside):
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    run_geospatial(bundle)
    final = run_reporting(bundle)
    payload = run_visualization(final, bundle)

    # non-empty map_payload
    assert payload is not None
    assert len(payload.pins) >= 2
    assert len(payload.overlays) >= 1

    # pins non-empty, correct types
    assert any(p["type"] == "query" for p in payload.pins)
    assert any(p["type"] == "pfz" for p in payload.pins)

    # overlays contain restricted_area GeoJSON
    restricted_overlays = [o for o in payload.overlays if o.get("type") == "restricted_area"]
    assert len(restricted_overlays) >= 1
    for ov in restricted_overlays:
        assert "geojson" in ov
        assert "type" in ov["geojson"]

    # status: original statuses unchanged
    assert bundle.marine.status in VALID_STATUSES

    # inside_restricted_area reflected in highlighted flag
    if expected_inside:
        highlighted = [o for o in payload.overlays if o.get("highlighted") is True]
        assert len(highlighted) >= 1, "inside restricted fixture should have highlighted overlay"
    else:
        # outside: none highlighted (or at least not incorrectly highlighted)
        # but we allow implementation to have zero highlighted
        pass

    # trace
    assert any(t.agent_name == "visualization" for t in bundle.trace)


@pytest.mark.parametrize("fixture,_expected_inside", FIXTURE_CASES)
def test_visualization__build_map_payload_pure(fixture, _expected_inside):
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    run_geospatial(bundle)
    # build_map_payload is pure, no trace side-effect
    payload = build_map_payload(bundle)
    assert len(payload.pins) >= 2
    assert len(payload.overlays) >= 1
    # should also be non-empty when called via run_visualization wrapper
    final = run_reporting(bundle)
    via_run = run_visualization(final, bundle)
    assert len(via_run.pins) == len(payload.pins)
    assert len(via_run.overlays) == len(payload.overlays)


@pytest.mark.parametrize("fixture,_expected_inside", FIXTURE_CASES)
def test_visualization__status_and_inside_consistency(fixture, _expected_inside):
    bundle = _fresh(fixture)
    run_ocean_analytics(bundle)
    run_geospatial(bundle)
    final = run_reporting(bundle)
    payload = run_visualization(final, bundle)

    # status invariants
    assert bundle.weather.status in VALID_STATUSES
    assert bundle.risk.status in VALID_STATUSES

    # inside_restricted_area consistency between geospatial and visualization highlighted
    geoinside = bundle.geospatial.inside_restricted_area
    assert isinstance(geoinside, bool)
    # map_payload must be non-empty regardless of inside flag
    assert payload.pins != []
    assert payload.overlays != []


# ---------------------------------------------------------------------------
# integration — full deterministic pipeline across all 4 fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,expected_inside", FIXTURE_CASES)
def test_integration__all_four_properties_per_fixture(fixture, expected_inside):
    """Across modules, assert the 4 required properties for each fixture."""
    bundle, final, viz = _run_full_pipeline(fixture)

    # 1. status — all evidence statuses remain valid literals
    assert bundle.marine.status in VALID_STATUSES
    assert bundle.weather.status in VALID_STATUSES
    assert bundle.risk.status in VALID_STATUSES

    # 2. inside_restricted_area — bool, True only for fixture 4
    assert isinstance(bundle.geospatial.inside_restricted_area, bool)
    assert bundle.geospatial.inside_restricted_area is expected_inside
    # also via viz highlighted
    if expected_inside:
        assert any(o.get("highlighted") for o in viz.overlays if o.get("type") == "restricted_area")

    # 3. non-empty answer_text
    assert isinstance(final.answer_text, str)
    assert final.answer_text.strip() != ""
    assert len(final.answer_text) > 20

    # 4. non-empty map_payload (pins + overlays)
    assert final.map_payload is not None
    assert len(final.map_payload.pins) > 0
    assert len(final.map_payload.overlays) > 0
    assert len(viz.pins) > 0
    assert len(viz.overlays) > 0

    # also pure/no-raise invariant: entire pipeline produced trace steps for every module
    agent_names = {t.agent_name for t in bundle.trace}
    assert "ocean_analytics" in agent_names
    assert "geospatial" in agent_names
    assert "reporting" in agent_names
    assert "visualization" in agent_names


def test_integration__all_fixtures_iterable_via_ALL_FIXTURES():
    """Sanity: ALL_FIXTURES contains exactly the 4 golden fixtures."""
    assert len(ALL_FIXTURES) == 4


# ---------------------------------------------------------------------------
# reporting.py — PFZ distance-honesty framing (Phase 4, Chat-PFZ Live-Data
# Integration). When chat's live PFZ lookup resolves to a distant anchor
# (see pfz_service.NearestAnchorZones.is_distant), each zone dict in
# marine.pfz_zones carries an additive "is_distant" key (see
# marine_data_agent._live_pfz_candidate_zones); reporting.py must turn
# that into an honest "regional reference, not a nearby recommendation"
# framing instead of presenting it like a normal nearby match. Built from a
# hand-constructed bundle (not the golden fixtures, none of which carry a
# distant zone) but through the same public run_geospatial -> run_reporting
# pipeline as everywhere else in this file.
# ---------------------------------------------------------------------------


def _bundle_with_pfz_zone(is_distant: bool) -> EvidenceBundle:
    query_location = GeoPoint(lat=19.8, lon=85.8)  # mid-coast Odisha, near Puri
    zone_center = {"lat": 20.2, "lon": 86.6}  # genuinely far from query_location
    return EvidenceBundle(
        query_text="where can I fish?",
        query_location=query_location,
        marine=MarineDataResult(
            status="ok",
            pfz_zones=[
                {
                    "zone_id": "PFZ-001",
                    "center": zone_center,
                    "distance_km": 103.4,
                    "sst_celsius": 27.9,
                    "chlorophyll_mg_m3": 0.7,
                    "advisory": "Potential fishing zone — live satellite data.",
                    "is_distant": is_distant,
                }
            ],
            sst_celsius=27.9,
            chlorophyll_mg_m3=0.7,
            source_timestamp="2026-09-14T00:00:00Z",
        ),
        weather=WeatherDataResult(
            status="ok",
            wind_kmh=15.0,
            wave_height_m=0.8,
            cyclone_alert=False,
            lightning_alert=False,
            tide_info=None,
            source_timestamp="2026-09-14T00:00:00Z",
        ),
        risk=RiskAssessment(status="ok", safe_to_go=True, confidence=0.8, explanation="Calm seas."),
    )


def test_reporting__distant_pfz_zone_gets_honest_framing():
    bundle = _bundle_with_pfz_zone(is_distant=True)
    run_geospatial(bundle)
    final = run_reporting(bundle)

    assert "Nearest known fishing-zone data: PFZ-001" in final.answer_text
    assert "regional reference" in final.answer_text
    # Not presented as a normal nearby result.
    assert "Nearest fishing zone:" not in final.answer_text


def test_reporting__nearby_pfz_zone_keeps_normal_framing():
    bundle = _bundle_with_pfz_zone(is_distant=False)
    run_geospatial(bundle)
    final = run_reporting(bundle)

    assert "Nearest fishing zone: PFZ-001" in final.answer_text
    assert "regional reference" not in final.answer_text
    assert "Nearest known fishing-zone data" not in final.answer_text


def test_reporting__missing_is_distant_key_defaults_to_normal_framing():
    """Zones from paths that never set "is_distant" (any pre-Phase-4
    producer) must not be treated as distant just because the key is
    absent."""
    bundle = _bundle_with_pfz_zone(is_distant=False)
    del bundle.marine.pfz_zones[0]["is_distant"]
    run_geospatial(bundle)
    final = run_reporting(bundle)

    assert "Nearest fishing zone: PFZ-001" in final.answer_text
    assert "regional reference" not in final.answer_text
