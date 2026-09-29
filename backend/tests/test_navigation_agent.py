"""Pytest coverage for backend/agents/reasoning/navigation_agent.py (Phase 5.2).

Covers the 3 required cases for A* route planning over a restricted-zone
obstacle grid:
  1. Clear open water — a route is found connecting start and destination.
  2. A restricted zone directly between start and destination — a route is
     still found, and it detours around the zone rather than crossing it.
  3. Destination fully enclosed by a restricted zone (a solid ring with no
     gap) — no route exists, find_route returns None instead of raising.

restricted_zones are built directly as [{"name": ..., "polygon": Shapely
Polygon}, ...] — the same shape geospatial.get_active_restricted_areas()
returns — rather than depending on the real WDPA cache, so these tests are
self-contained and don't depend on network/data-fetch state. The three
obstacle-logic cases pass land=None (their coordinates are abstract "open
water", not real geography); land handling has its own cases below, with a
synthetic land polygon plus one check against the real coastline:
  4. A land mass between start and destination is routed around.
  5. A start on land is moved to the nearest open water, reported in km.
  6. A start inside a protected area's actual extent is refused with a
     reason; one merely within the routing clearance is not.
  7. No sea route at all -> None with a plain reason, never a line on land.
  8. Real coastline: Chennai to the Gulf of Mannar never touches land.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

from itertools import pairwise

from shapely.geometry import LineString, Point, Polygon

from backend.agents.deterministic.geospatial import haversine_km
from backend.agents.reasoning.navigation_agent import find_route, plan_route


def _zone(name: str, polygon: Polygon) -> dict:
    return {"name": name, "polygon": polygon}


def _route_length_km(route: list[dict]) -> float:
    total = 0.0
    for a, b in pairwise(route):
        total += haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    return total


# ---------------------------------------------------------------------------
# Case 1 — clear open water
# ---------------------------------------------------------------------------


def test_find_route__open_water_no_obstacles():
    start = {"lat": 9.0, "lon": 78.5}
    destination = {"lat": 9.5, "lon": 79.0}

    route = find_route(start, destination, restricted_zones=[], land=None)

    assert route is not None
    assert len(route) >= 2

    # endpoints match start/destination (snapped to nearest grid vertex —
    # within a couple of grid cells, not exact floating-point equality)
    assert abs(route[0]["lat"] - start["lat"]) < 0.05
    assert abs(route[0]["lon"] - start["lon"]) < 0.05
    assert abs(route[-1]["lat"] - destination["lat"]) < 0.05
    assert abs(route[-1]["lon"] - destination["lon"]) < 0.05

    # with no obstacles, the grid path should track the straight-line
    # distance closely (8-directional grid discretization adds only a
    # small overhead versus the true great-circle distance)
    straight_km = haversine_km(start["lat"], start["lon"], destination["lat"], destination["lon"])
    route_km = _route_length_km(route)
    assert route_km < straight_km * 1.15, "open-water route should stay close to the straight-line distance"


# ---------------------------------------------------------------------------
# Case 2 — restricted zone directly between start and destination
# ---------------------------------------------------------------------------


def test_find_route__detours_around_zone_between_start_and_destination():
    start = {"lat": 9.0, "lon": 78.5}
    destination = {"lat": 9.0, "lon": 80.5}

    # Rectangle squarely blocking the direct (due-east) line between them:
    # spans the shared latitude (9.0) and sits at the midpoint longitude.
    blocking_zone = _zone(
        "Test Restricted Corridor",
        Polygon([(79.4, 8.85), (79.6, 8.85), (79.6, 9.15), (79.4, 9.15)]),  # (lon, lat) — Shapely order
    )

    route = find_route(start, destination, restricted_zones=[blocking_zone], land=None)

    assert route is not None, "a route should still exist by detouring north or south"

    # no waypoint should fall inside the blocking polygon
    poly = blocking_zone["polygon"]
    for point in route:
        assert not poly.covers(Point(point["lon"], point["lat"])), (
            f"waypoint {point} falls inside the restricted zone — route did not detour"
        )

    # a detour is strictly longer than the (now-blocked) straight line
    straight_km = haversine_km(start["lat"], start["lon"], destination["lat"], destination["lon"])
    route_km = _route_length_km(route)
    assert route_km > straight_km, "a route around an obstacle must be longer than the direct line through it"


# ---------------------------------------------------------------------------
# Case 3 — destination fully enclosed (unreachable)
# ---------------------------------------------------------------------------


def test_find_route__destination_fully_enclosed_returns_none():
    destination = {"lat": 10.0, "lon": 80.0}
    start = {"lat": 8.5, "lon": 78.0}

    # A single polygon with a hole ("square annulus") centered on the
    # destination: the destination sits inside the hole (not restricted
    # itself), but the solid ring around it is ~13km thick in every
    # direction — far thicker than the grid's ~1-2km cells — so there is no
    # gap for an 8-directional path to slip through.
    outer = [(79.85, 9.85), (80.15, 9.85), (80.15, 10.15), (79.85, 10.15)]
    inner_hole = [(79.97, 9.97), (80.03, 9.97), (80.03, 10.03), (79.97, 10.03)]
    enclosing_ring = _zone("Test Enclosing Ring", Polygon(outer, holes=[inner_hole]))

    # sanity check on the test's own construction: destination must not be
    # considered inside the ring itself (it's in the hole)
    assert not enclosing_ring["polygon"].covers(Point(destination["lon"], destination["lat"]))

    route = find_route(start, destination, restricted_zones=[enclosing_ring], land=None)

    assert route is None, "destination fully enclosed by a restricted zone must be unreachable"


# ---------------------------------------------------------------------------
# Cases 4-8 — land
# ---------------------------------------------------------------------------

# A rectangular "island" squarely between (9.0, 78.5) and (9.0, 80.5).
_ISLAND = Polygon([(79.3, 8.7), (79.7, 8.7), (79.7, 9.3), (79.3, 9.3)])


def test_plan_route__detours_around_land():
    start, destination = {"lat": 9.0, "lon": 78.5}, {"lat": 9.0, "lon": 80.5}

    result = plan_route(start, destination, restricted_zones=[], land=_ISLAND)

    assert result.route is not None
    line = LineString([(p["lon"], p["lat"]) for p in result.route])
    assert not line.intersects(_ISLAND), "route crosses land"
    assert _route_length_km(result.route) > haversine_km(9.0, 78.5, 9.0, 80.5)


def test_plan_route__start_on_land_moves_to_nearest_water_and_says_how_far():
    start = {"lat": 9.0, "lon": 79.35}  # just inside the island's west edge
    destination = {"lat": 9.0, "lon": 78.5}

    result = plan_route(start, destination, restricted_zones=[], land=_ISLAND)

    assert result.route is not None
    assert not _ISLAND.covers(Point(result.route[0]["lon"], result.route[0]["lat"]))
    assert 0 < result.start_offset_km < 15
    line = LineString([(p["lon"], p["lat"]) for p in result.route])
    assert not line.intersects(_ISLAND)


def test_plan_route__start_inside_protected_area_is_refused_but_its_clearance_is_not():
    park = _zone("Test Marine Park", Polygon([(79.4, 8.9), (79.6, 8.9), (79.6, 9.1), (79.4, 9.1)]))
    destination = {"lat": 9.0, "lon": 78.6}

    inside = plan_route({"lat": 9.0, "lon": 79.5}, destination, [park], land=None)
    assert inside.route is None
    assert "inside Test Marine Park" in inside.reason

    # ~1km outside the park's east edge: within the 2km routing clearance,
    # not inside the park — routed (from the nearest cell clear of it).
    near = plan_route({"lat": 9.0, "lon": 79.61}, destination, [park], land=None)
    assert near.route is not None
    line = LineString([(p["lon"], p["lat"]) for p in near.route])
    assert not line.intersects(park["polygon"])


def test_plan_route__no_sea_route_gives_a_reason_not_a_line_over_land():
    # Destination in a lake fully enclosed by land.
    land = Polygon(
        [(79.0, 8.5), (80.0, 8.5), (80.0, 9.5), (79.0, 9.5)],
        holes=[[(79.45, 8.95), (79.55, 8.95), (79.55, 9.05), (79.45, 9.05)]],
    )

    result = plan_route({"lat": 9.0, "lon": 78.3}, {"lat": 9.0, "lon": 79.5}, [], land=land)

    assert result.route is None
    assert "no sea route" in result.reason


def test_plan_route__real_coastline_chennai_to_gulf_of_mannar_stays_at_sea():
    from backend.agents.deterministic.geospatial import (
        get_active_restricted_areas,
        get_land_mask,
    )

    land = get_land_mask()
    assert land is not None, "committed land mask (data/land_india.geojson) failed to load"
    chennai, mannar_pfz = {"lat": 13.08, "lon": 80.27}, {"lat": 8.9375, "lon": 79.3542}

    result = plan_route(chennai, mannar_pfz, get_active_restricted_areas())

    assert result.route is not None, result.reason
    assert not land.intersects(LineString([(p["lon"], p["lat"]) for p in result.route]))
    assert result.start_offset_km > 0  # Chennai's city point is on land
