"""Integration coverage for the geolocation -> Navigation Agent wiring in
backend/main.py (QueryRequest.user_location -> _run_deterministic_pipeline
-> find_route's `start` argument).

This is the exact layer AUDIT_REPORT_V2.md (S4/S7/S12.7) flagged as
untested: navigation_agent.find_route was covered in isolation
(test_navigation_agent.py), but nothing exercised the wiring that decides
*what start point* main.py actually passes it — which is precisely where
the original geolocation bug (frontend sent user_location, QueryRequest
didn't declare it, Pydantic silently dropped it) shipped undetected.

Drives the real FastAPI app through /query/demo (deterministic-only, no
Groq key needed) via Starlette's TestClient, so this asserts against the
actual HTTP contract the frontend uses, not just the Python function.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

# Fixture 1 (happy path, the default /query/demo match) resolves
# query_location to Chennai coast: GeoPoint(lat=13.08, lon=80.27) — see
# backend/schemas/test_fixtures.py. A generic query with no fixture keyword
# ("mannar"/"cyclone"/"hazard"/"storm"/"imd"/"partial"/"unreachable") always
# lands here.
_QUERY_LOCATION = {"lat": 13.08, "lon": 80.27}
_DESTINATION = {"lat": 13.20, "lon": 80.40}
# At sea off Chennai (a phone on a boat) — a point on land would be moved
# to the nearest open water before routing, which isn't what this tests.
_USER_LOCATION = {"lat": 12.90, "lon": 80.30}


def _route_start(response_json: dict) -> dict:
    route = response_json["map_payload"]["route"]
    assert route, "expected a route in map_payload.route"
    return route[0]


def test_route_prefers_live_user_location_over_query_location():
    """When the frontend sends real device geolocation alongside a chosen
    destination, the Navigation Agent must route from that live position —
    not from the text-resolved query_location — matching the PDF's own
    §8.4 claim and main.py's Phase 5.4 comment."""
    resp = client.post(
        "/query/demo",
        json={
            "query": "is it safe to fish today",
            "selected_destination": _DESTINATION,
            "user_location": _USER_LOCATION,
        },
    )
    assert resp.status_code == 200
    start = _route_start(resp.json())

    # Snapped to the nearest grid vertex (see test_navigation_agent.py's
    # same tolerance) — should land near user_location, not query_location.
    assert abs(start["lat"] - _USER_LOCATION["lat"]) < 0.05
    assert abs(start["lon"] - _USER_LOCATION["lon"]) < 0.05
    assert abs(start["lat"] - _QUERY_LOCATION["lat"]) > 0.05


def test_route_falls_back_to_query_location_when_no_user_location_sent():
    """Backward-compatible degrade path: a client that never sends
    user_location (or denied the browser permission) still gets a route,
    starting from the session's text-resolved location as before."""
    resp = client.post(
        "/query/demo",
        json={
            "query": "is it safe to fish today",
            "selected_destination": _DESTINATION,
        },
    )
    assert resp.status_code == 200
    start = _route_start(resp.json())

    assert abs(start["lat"] - _QUERY_LOCATION["lat"]) < 0.05
    assert abs(start["lon"] - _QUERY_LOCATION["lon"]) < 0.05


def test_no_route_without_a_selected_destination():
    """Navigation Agent must never activate on a bare query — routing only
    ever starts once the user has picked a destination (Phase 4.3), even
    if a live user_location is present."""
    resp = client.post(
        "/query/demo",
        json={
            "query": "is it safe to fish today",
            "user_location": _USER_LOCATION,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["map_payload"]["route"] is None
