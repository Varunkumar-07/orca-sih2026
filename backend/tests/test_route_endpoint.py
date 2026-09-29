"""End-to-end coverage for POST /route (backend/routers/pages.py) — closes
the one gap test_navigation_agent.py's suite doesn't cover: that file drives
navigation_agent.find_route() directly against hand-built restricted-zone
lists; nothing drives the real HTTP endpoint through the actual
geospatial.get_active_restricted_areas() cache this app serves requests
with. This file does — same pattern test_history.py already uses for
GET /zones (real FastAPI app via TestClient, real backend data, no agent
internals mocked).

The restricted-zone geometry used below (Gulf of Mannar Marine Biosphere
Reserve) is whatever get_active_restricted_areas() actually resolves to in
this environment — the real WDPA-derived cache at
backend/agents/deterministic/data/mpa_boundaries.geojson, confirmed present
in this repo (falls back to a hardcoded Gulf of Mannar rectangle only if
that cache is ever missing — see geospatial.py's own docstring). Either way
this test exercises the genuine production code path, not a substitute.

Run: PYTHONPATH=. pytest backend/tests/test_route_endpoint.py -v
"""
from __future__ import annotations

from itertools import pairwise

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Point

from backend.agents.deterministic.geospatial import (
    get_active_restricted_areas,
    get_land_mask,
    haversine_km,
)
from backend.main import app


@pytest.fixture(scope="module")
def client(offline_pfz_data):
    """offline_pfz_data (conftest.py): destination_zone_id resolution needs
    PFZ zones in GET /zones's catalog without live Copernicus access."""
    with TestClient(app) as c:
        yield c


def _route_length_km(route: list[dict]) -> float:
    return sum(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) for a, b in pairwise(route))


def test_baseline_route_with_no_restricted_zone_in_the_way(client):
    """Two open-water points, well clear of every real restricted zone in
    this environment's cache (confirmed by direct inspection — none of the
    10 currently-cached zones fall anywhere near the Lakshadweep Sea off
    Kerala) — the route should come back close to the straight-line
    distance, not detour."""
    resp = client.post(
        "/route",
        json={"start": {"lat": 10.5, "lon": 75.0}, "destination": {"lat": 11.0, "lon": 75.5}},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["reason"] is None
    assert body["route"] is not None
    assert body["waypoint_count"] == len(body["route"])

    straight_km = haversine_km(10.5, 75.0, 11.0, 75.5)
    # No obstacle in the way — the grid-discretized route should track the
    # straight line closely, not meaningfully detour.
    assert body["distance_km"] < straight_km * 1.15


def test_route_detours_around_the_real_gulf_of_mannar_restricted_zone(client):
    """Start and destination — both at sea — straddle the real Gulf of
    Mannar biosphere reserve (WDPA records it as a point with a 10,500 km2
    reported area; geospatial.py models that as a circle of that area) — the
    straight line between them passes through it, so a genuine route must
    detour around it rather than crossing it (or any land), and must end up
    longer than the blocked straight line as a result."""
    restricted_zones = get_active_restricted_areas()
    mannar = next((z for z in restricted_zones if z["name"] == "Gulf of Mannar"), None)
    assert mannar is not None, (
        "expected the real Gulf of Mannar biosphere reserve in get_active_restricted_areas() — "
        "this test's start/destination points are specifically chosen to straddle it"
    )

    start = {"lat": 8.6, "lon": 78.3}
    destination = {"lat": 8.9, "lon": 79.6}
    # Sanity check on the test's own construction: the direct line between
    # these two points must actually cross the restricted zone, or a route
    # not detouring around it would prove nothing.
    direct_line = LineString([(start["lon"], start["lat"]), (destination["lon"], destination["lat"])])
    assert mannar["polygon"].intersects(direct_line), (
        "test setup error — the direct start->destination line doesn't even cross the restricted zone"
    )

    resp = client.post("/route", json={"start": start, "destination": destination})
    assert resp.status_code == 200
    body = resp.json()

    assert body["reason"] is None
    assert body["route"] is not None, "a route should exist by detouring around the restricted zone"

    for point in body["route"]:
        assert not mannar["polygon"].covers(Point(point["lon"], point["lat"])), (
            f"waypoint {point} falls inside {mannar['name']!r} — route did not detour"
        )

    straight_km = haversine_km(start["lat"], start["lon"], destination["lat"], destination["lon"])
    assert body["distance_km"] > straight_km, (
        "a route detouring around a real restricted zone must be longer than the blocked direct line"
    )
    land = get_land_mask()
    assert not land.intersects(LineString([(p["lon"], p["lat"]) for p in body["route"]])), "route crosses land"


def test_route_resolves_a_destination_zone_id_against_the_real_zones_catalog(client):
    """destination_zone_id (rather than a raw destination point) is resolved
    against GET /zones's own live catalog — pick whatever real PFZ zone that
    catalog currently returns rather than assuming a specific id, so this
    test doesn't depend on the PFZ cache's exact contents at any given
    moment (see pfz_service.get_cached_zones)."""
    zones_resp = client.get("/zones")
    assert zones_resp.status_code == 200
    pfz_zone = next(z for z in zones_resp.json()["zones"] if z["type"] == "pfz")

    resp = client.post(
        "/route",
        json={"start": {"lat": pfz_zone["coordinates"]["lat"] - 0.2, "lon": pfz_zone["coordinates"]["lon"] - 0.2}, "destination_zone_id": pfz_zone["id"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reason"] is None
    assert body["route"] is not None
    # The route ends at the zone — or, if the zone point itself is on land
    # (conftest's synthetic zones sit on the anchor city), at the nearest
    # open water, with that distance reported as end_offset_km.
    end = body["route"][-1]
    gap_km = haversine_km(end["lat"], end["lon"], pfz_zone["coordinates"]["lat"], pfz_zone["coordinates"]["lon"])
    assert gap_km <= body["end_offset_km"] + 3.0


def test_route_from_a_city_on_land_starts_at_open_water_and_says_so(client):
    """Every city start point is on land; the route must start at sea (never
    draw a line over land) and report how far that is from the start."""
    resp = client.post("/route", json={"start": {"lat": 19.076, "lon": 72.8777}, "destination": {"lat": 18.9792, "lon": 72.7292}})
    body = resp.json()

    assert body["reason"] is None, body["reason"]
    assert body["start_offset_km"] > 0
    assert not get_land_mask().intersects(LineString([(p["lon"], p["lat"]) for p in body["route"]]))


def test_route_from_inside_a_protected_area_is_refused_with_its_name(client):
    resp = client.post("/route", json={"start": {"lat": 9.15, "lon": 79.15}, "destination": {"lat": 13.0625, "lon": 80.3542}})
    body = resp.json()

    assert body["route"] is None
    assert "inside Gulf of Mannar" in body["reason"]


def test_no_destination_provided_returns_a_clear_reason_not_an_error(client):
    resp = client.post("/route", json={"start": {"lat": 9.0, "lon": 78.0}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] is None
    assert body["reason"]
