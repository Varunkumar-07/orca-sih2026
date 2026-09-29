"""PFZ zones are never placed inside a protected area's extent.

The rule is the verdict's own (geospatial._restricted_area_at: the WDPA
polygon, or the reported-area circle for point-only sites, boundary
inclusive) — so a suggested zone can never come back PROHIBITED or be
refused by route planning. Excluded cells are masked before ranking, so the
next-best permitted front backfills the slot and zone IDs stay gap-free.

Covers:
  - restricted_area_mask agrees point-for-point with _restricted_area_at
    over a real grid (Gulf of Mannar, real WDPA cache), and counts a point
    on an area's boundary as inside
  - find_pfz_candidates never picks an excluded cell, and the freed slot
    is backfilled rather than lost
  - _live_zones_for_anchor (the only live producer — /zones, chat,
    Weather, Alerts, /route all read its output via get_cached_zones)
    keeps zones out of a protected area and numbers them PFZ-001.. with
    no gaps
  - every static zone source (demo_places.py, test_fixtures.py,
    demo_snapshot.py) stays outside every protected area
"""
from __future__ import annotations

import asyncio

import numpy as np
from shapely.geometry import box

from backend.agents.deterministic import geospatial
from backend.agents.reasoning import marine_data_agent
from backend.schemas import test_fixtures
from backend.schemas.contracts import GeoPoint
from backend.schemas.demo_places import DEMO_SAMPLE_PFZ_BY_ANCHOR
from backend.schemas.demo_snapshot import DEMO_SNAPSHOTS
from backend.services import pfz_service as pfz


def _two_front_grid() -> pfz.EnvironmentalGrid:
    """Two east-west fronts: rows 9/11 (~8.31N/8.38N, around the spike at
    row 10) and rows 19/21 (~8.66N/8.72N)."""
    lats = np.linspace(8.0, 9.0, 30)
    lons = np.linspace(77.0, 78.0, 30)
    sst = np.full((30, 30), 28.0)
    chl = np.full((30, 30), 0.5)
    sst[10, :] = 29.4
    chl[10, :] = 0.9
    sst[20, :] = 26.8
    chl[20, :] = 0.25
    return pfz.EnvironmentalGrid(lats=lats, lons=lons, sst_celsius=sst, chlorophyll_mg_m3=chl, satellite_date="2026-09-28")


# Covers the whole first front (rows 9-11) and nothing of the second.
_FRONT_RESERVE = {"name": "Test Reserve", "polygon": box(76.9, 8.30, 78.1, 8.40), "approximate": False}


class TestRestrictedAreaMask:
    def test_matches_the_verdict_check_point_for_point(self):
        lats = np.linspace(8.2, 9.8, 25)
        lons = np.linspace(78.0, 79.8, 25)
        mask = geospatial.restricted_area_mask(lats, lons)
        expected = np.array([[geospatial._restricted_area_at(la, lo) is not None for lo in lons] for la in lats])
        assert expected.any() and not expected.all()  # the grid really straddles the reserve's edge
        assert np.array_equal(mask, expected)

    def test_boundary_point_counts_as_inside(self, monkeypatch):
        monkeypatch.setattr(geospatial, "_restricted_areas_cache", [_FRONT_RESERVE])
        mask = geospatial.restricted_area_mask(np.array([8.30, 8.29]), np.array([77.5]))
        assert mask[:, 0].tolist() == [True, False]
        assert geospatial._restricted_area_at(8.30, 77.5) is not None

    def test_far_away_grid_is_all_clear(self, monkeypatch):
        monkeypatch.setattr(geospatial, "_restricted_areas_cache", [_FRONT_RESERVE])
        assert not geospatial.restricted_area_mask(np.linspace(20, 21, 5), np.linspace(70, 71, 5)).any()


class TestFindPfzCandidatesExcluded:
    def test_excluded_cells_are_never_picked_and_the_slot_is_backfilled(self):
        grid = _two_front_grid()
        excluded = (grid.lats[:, None] >= 8.30) & (grid.lats[:, None] <= 8.40) & np.ones((1, len(grid.lons)), dtype=bool)

        unmasked = pfz.find_pfz_candidates(grid, max_zones=3, min_separation_km=15.0)
        masked = pfz.find_pfz_candidates(grid, max_zones=3, min_separation_km=15.0, excluded=excluded)

        assert any(8.30 <= c.lat <= 8.40 for c in unmasked)  # the first front would have been picked
        assert len(masked) == len(unmasked) == 3
        assert all(not 8.30 <= c.lat <= 8.40 for c in masked)

    def test_everything_excluded_returns_no_zones(self):
        grid = _two_front_grid()
        excluded = np.ones(grid.sst_celsius.shape, dtype=bool)
        assert pfz.find_pfz_candidates(grid, excluded=excluded) == []


class TestLiveZonesForAnchor:
    def test_live_zones_avoid_protected_areas_with_gap_free_ids(self, monkeypatch):
        monkeypatch.setattr(geospatial, "_restricted_areas_cache", [_FRONT_RESERVE])
        grid = _two_front_grid()

        async def fake_fetch(*_a, **_k):
            return grid

        monkeypatch.setattr(marine_data_agent, "fetch_environmental_grid", fake_fetch)
        zones = asyncio.run(marine_data_agent._live_zones_for_anchor(GeoPoint(lat=8.5, lon=77.5), asyncio.Semaphore(1)))

        assert [z["zone_id"] for z in zones] == ["PFZ-001", "PFZ-002", "PFZ-003"]
        for z in zones:
            assert geospatial._restricted_area_at(z["center"]["lat"], z["center"]["lon"]) is None


def _zone_points(obj):
    """Every (label, lat, lon) of a PFZ zone anywhere in a nested dump."""
    if isinstance(obj, dict):
        center = obj.get("center")
        if isinstance(center, dict) and "lat" in center and "lon" in center:
            yield obj.get("zone_id") or obj.get("label"), center["lat"], center["lon"]
        for value in obj.values():
            yield from _zone_points(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _zone_points(value)


class TestStaticZoneSources:
    def _assert_outside(self, points):
        inside = [(label, lat, lon) for label, lat, lon in points if geospatial._restricted_area_at(lat, lon) is not None]
        assert inside == []

    def test_demo_places_zones_are_outside_protected_areas(self):
        self._assert_outside(
            (z["zone_id"], z["center"]["lat"], z["center"]["lon"]) for zones in DEMO_SAMPLE_PFZ_BY_ANCHOR.values() for z in zones
        )

    def test_demo_places_ids_are_gap_free_per_anchor(self):
        for anchor, zones in DEMO_SAMPLE_PFZ_BY_ANCHOR.items():
            prefix = anchor.upper().replace(" ", "-")
            assert [z["zone_id"] for z in zones] == [f"{prefix}-PFZ-{i:03d}" for i in range(1, len(zones) + 1)]

    def test_fixture_zones_are_outside_protected_areas(self):
        fixtures = [f for f in vars(test_fixtures).values() if getattr(getattr(f, "marine", None), "pfz_zones", None)]
        assert fixtures
        self._assert_outside(p for f in fixtures for p in _zone_points(f.marine.pfz_zones))

    def test_snapshot_zones_are_outside_protected_areas(self):
        points = [p for snap in DEMO_SNAPSHOTS.values() for p in _zone_points(snap.model_dump())]
        assert points
        self._assert_outside(points)
