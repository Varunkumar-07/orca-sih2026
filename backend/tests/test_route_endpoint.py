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
    """Start and destination straddle the real, currently-cached Gulf of
    Mannar Marine Biosphere Reserve polygon — the straight line between them
    passes directly through it, so a genuine route must detour around it
    rather than crossing it, and must end up longer than the blocked
    straight line as a result."""
    restricted_zones = get_active_restricted_areas()
    mannar = next((z for z in restricted_zones if "mannar" in z["name"].lower()), None)
    assert mannar is not None, (
        "expected a real Gulf-of-Mannar restricted zone in get_active_restricted_areas() — "
        "this test's start/destination points are specifically chosen to straddle it"
    )

    start = {"lat": 9.0, "lon": 77.8}
    destination = {"lat": 9.0, "lon": 79.6}
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
    assert body["route"][-1]["lat"] == pytest.approx(pfz_zone["coordinates"]["lat"], abs=0.05)
    assert body["route"][-1]["lon"] == pytest.approx(pfz_zone["coordinates"]["lon"], abs=0.05)


def test_no_destination_provided_returns_a_clear_reason_not_an_error(client):
    resp = client.post("/route", json={"start": {"lat": 9.0, "lon": 78.0}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] is None
    assert body["reason"]
