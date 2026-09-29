"""Pytest coverage for backend/services/pfz_service.py — grid fetch/align
and front-detection scoring, added for Phase 7 of the live-PFZ work.

Mocking strategy: _fetch_sst_grid_sync / _fetch_chlorophyll_grid_sync (the
blocking Copernicus calls run via asyncio.to_thread) are patched directly
with real, small xarray.DataArray objects — same "patch our own function,
not the vendor SDK" principle as test_marine_data_agent.py's
_fetch_chlorophyll_sync patching. Using real DataArrays (not hand-rolled
fakes) means fetch_environmental_grid's actual .interp() alignment step
is genuinely exercised, not bypassed. find_pfz_candidates is tested with
pure synthetic numpy grids — deterministic, no dataset involved at all.

Covers:
  - fetch_environmental_grid: missing credentials -> None, no fetch
    attempted; success path aligns SST onto chlorophyll's (finer) grid so
    shapes match even though native resolutions differ; one dataset
    failing -> None (graceful, not raised); a genuine exception from
    copernicusmarine -> None, not propagated
  - find_pfz_candidates: a real front beats a flat-but-favorable region
    with no front at all; NaN (land) cells are never selected even when
    they'd otherwise score highest; shape mismatch raises; a
    uniform/no-signal grid returns an empty list; returned candidates
    stay >= min_separation_km apart from each other
  - grid reads: a timed-out read is never retried (its thread is still
    running); a real error is retried once — for per-anchor fetches and
    the shared all-anchor dataset opens alike
  - get_nearest_anchor_zones (Phase 2, Chat-PFZ Live-Data Integration):
    resolves an arbitrary query point to its nearest anchor's cached
    zones with distance_km recomputed from the query point; a query
    exactly at an anchor reproduces that anchor's own anchor->zone
    distance; a query far from every anchor still resolves to the
    genuinely-nearest one (no error, no nonsense); restricted-area
    entries are excluded; every non-distance field is passed through
    unchanged from get_cached_zones()'s own output.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from backend.agents.reasoning import marine_data_agent, planning_agent
from backend.schemas.contracts import GeoPoint
from backend.services import pfz_service as pfz


def _make_da(lats: np.ndarray, lons: np.ndarray, values: np.ndarray, time: str) -> xr.DataArray:
    """A minimal real xarray.DataArray shaped like what
    copernicusmarine.open_dataset()[...] actually returns after
    .isel(time=-1): dims (latitude, longitude), a scalar time coordinate."""
    da = xr.DataArray(
        values,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
    )
    da = da.assign_coords(time=np.datetime64(time))
    return da


class TestFetchEnvironmentalGrid:
    def test_no_credentials_returns_none_without_fetching(self, monkeypatch):
        monkeypatch.delenv("COPERNICUSMARINE_USERNAME", raising=False)
        monkeypatch.delenv("COPERNICUSMARINE_PASSWORD", raising=False)

        def fail_if_called(*_a, **_k):
            raise AssertionError("should not fetch without credentials")

        monkeypatch.setattr(pfz, "_fetch_sst_grid_sync", fail_if_called)
        monkeypatch.setattr(pfz, "_fetch_chlorophyll_grid_sync", fail_if_called)

        result = asyncio.run(pfz.fetch_environmental_grid(79.0, 80.0, 9.0, 10.0))
        assert result is None

    def test_env_example_placeholders_return_none_without_fetching(self, monkeypatch):
        """An unedited copy of backend/.env.example must not turn into a
        real Copernicus login attempt with the placeholder values."""
        monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "your-copernicusmarine-username-here")
        monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "your-copernicusmarine-password-here")

        def fail_if_called(*_a, **_k):
            raise AssertionError("should not fetch with placeholder credentials")

        monkeypatch.setattr(pfz, "_fetch_sst_grid_sync", fail_if_called)
        monkeypatch.setattr(pfz, "_fetch_chlorophyll_grid_sync", fail_if_called)

        result = asyncio.run(pfz.fetch_environmental_grid(79.0, 80.0, 9.0, 10.0))
        assert result is None

    def test_aligns_different_native_resolutions(self, monkeypatch):
        monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "u")
        monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "p")

        # SST: coarse 4x4 grid. Chlorophyll: finer 8x8 grid over the same
        # bounds — mirrors the real ~5km vs ~4km resolution mismatch found
        # live in Phase 1. _fetch_sst_grid_sync does its own Kelvin->Celsius
        # conversion internally, so the patched fake (replacing that whole
        # function) returns the already-converted Celsius value directly.
        sst_lats = np.linspace(9.0, 10.0, 4)
        sst_lons = np.linspace(79.0, 80.0, 4)
        sst_da = _make_da(sst_lats, sst_lons, np.full((4, 4), 301.0 - 273.15), "2026-09-12")

        chl_lats = np.linspace(9.0, 10.0, 8)
        chl_lons = np.linspace(79.0, 80.0, 8)
        chl_da = _make_da(chl_lats, chl_lons, np.full((8, 8), 0.5), "2026-09-12")

        monkeypatch.setattr(pfz, "_fetch_sst_grid_sync", lambda *_a, **_k: sst_da)
        monkeypatch.setattr(pfz, "_fetch_chlorophyll_grid_sync", lambda *_a, **_k: chl_da)

        grid = asyncio.run(pfz.fetch_environmental_grid(79.0, 80.0, 9.0, 10.0))
        assert grid is not None
        assert grid.sst_celsius.shape == (8, 8)  # aligned onto chlorophyll's finer grid
        assert grid.chlorophyll_mg_m3.shape == (8, 8)
        assert np.allclose(grid.sst_celsius, 301.0 - 273.15)
        assert grid.satellite_date == "2026-09-12"

    def test_one_dataset_failing_returns_none(self, monkeypatch):
        monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "u")
        monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "p")

        def broken(*_a, **_k):
            raise RuntimeError("simulated Copernicus outage")

        chl_da = _make_da(np.linspace(9, 10, 4), np.linspace(79, 80, 4), np.full((4, 4), 0.5), "2026-09-12")
        monkeypatch.setattr(pfz, "_fetch_sst_grid_sync", broken)
        monkeypatch.setattr(pfz, "_fetch_chlorophyll_grid_sync", lambda *_a, **_k: chl_da)

        result = asyncio.run(pfz.fetch_environmental_grid(79.0, 80.0, 9.0, 10.0))
        assert result is None

    def test_timed_out_read_is_not_retried(self, monkeypatch):
        """A retry would start a second read next to the still-running one
        (on Render's 0.1 CPU, every read used to time out, doubling them)."""
        monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "u")
        monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "p")
        monkeypatch.setattr(pfz, "_GRID_FETCH_TIMEOUT", 0.05)
        calls = {"n": 0}

        def slow(*_a, **_k):
            calls["n"] += 1
            time.sleep(0.2)

        chl_da = _make_da(np.linspace(9, 10, 4), np.linspace(79, 80, 4), np.full((4, 4), 0.5), "2026-09-12")
        monkeypatch.setattr(pfz, "_fetch_sst_grid_sync", slow)
        monkeypatch.setattr(pfz, "_fetch_chlorophyll_grid_sync", lambda *_a, **_k: chl_da)

        assert asyncio.run(pfz.fetch_environmental_grid(79.0, 80.0, 9.0, 10.0)) is None
        assert calls["n"] == 1

    def test_real_error_is_retried_once(self, monkeypatch):
        monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "u")
        monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "p")
        lats, lons = np.linspace(9, 10, 4), np.linspace(79, 80, 4)
        sst_da = _make_da(lats, lons, np.full((4, 4), 28.0), "2026-09-12")
        chl_da = _make_da(lats, lons, np.full((4, 4), 0.5), "2026-09-12")
        outcomes = iter([ConnectionError("auth hiccup"), sst_da])

        def flaky(*_a, **_k):
            item = next(outcomes)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(pfz, "_fetch_sst_grid_sync", flaky)
        monkeypatch.setattr(pfz, "_fetch_chlorophyll_grid_sync", lambda *_a, **_k: chl_da)

        assert asyncio.run(pfz.fetch_environmental_grid(79.0, 80.0, 9.0, 10.0)) is not None


class TestSharedGridSources:
    def _opener(self, fn):
        calls = {"n": 0}

        def opener(*_a):
            calls["n"] += 1
            return fn()

        opener.__name__ = "_open_test_dataset_sync"
        return opener, calls

    def test_timed_out_open_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(pfz, "_GRID_FETCH_TIMEOUT", 0.05)
        opener, calls = self._opener(lambda: time.sleep(0.2))
        sources = pfz.SharedGridSources(70.0, 90.0, 8.0, 23.0)

        assert asyncio.run(sources._open(opener, "u", "p")) is None
        assert calls["n"] == 1

    def test_failed_open_is_retried_once(self):
        outcomes = iter([ConnectionError("auth hiccup"), "dataset"])

        def flaky():
            item = next(outcomes)
            if isinstance(item, Exception):
                raise item
            return item

        opener, calls = self._opener(flaky)
        sources = pfz.SharedGridSources(70.0, 90.0, 8.0, 23.0)

        assert asyncio.run(sources._open(opener, "u", "p")) == "dataset"
        assert calls["n"] == 2


class TestFindPfzCandidates:
    def test_real_front_beats_flat_favorable_region(self):
        """A region that's biologically favorable everywhere but has NO
        front at all must score 0 (normalize() divides by a zero range and
        returns zeros) and produce no candidates — favorable absolute
        values alone, with nothing to indicate nutrient upwelling, isn't
        enough to flag a zone."""
        lats = np.linspace(8.0, 9.0, 10)
        lons = np.linspace(77.0, 78.0, 10)
        sst = np.full((10, 10), 28.0)  # optimal everywhere, but perfectly flat
        chl = np.full((10, 10), 0.5)  # optimal everywhere, but perfectly flat

        grid = pfz.EnvironmentalGrid(lats=lats, lons=lons, sst_celsius=sst, chlorophyll_mg_m3=chl, satellite_date="t")
        candidates = pfz.find_pfz_candidates(grid, max_zones=3, min_separation_km=15.0)
        assert candidates == []

    def test_finds_a_real_front(self):
        lats = np.linspace(8.0, 9.0, 20)
        lons = np.linspace(77.0, 78.0, 20)
        sst = np.full((20, 20), 28.0)
        chl = np.full((20, 20), 0.5)
        # A sharp, favorable-on-both-sides front around row 10.
        sst[10:, :] = 29.4
        chl[10:, :] = 0.9

        grid = pfz.EnvironmentalGrid(lats=lats, lons=lons, sst_celsius=sst, chlorophyll_mg_m3=chl, satellite_date="t")
        candidates = pfz.find_pfz_candidates(grid, max_zones=1, min_separation_km=15.0)
        assert len(candidates) == 1
        c = candidates[0]
        # The picked cell sits right on the front (row 9 or 10).
        row_idx = int(np.argmin(np.abs(lats - c.lat)))
        assert row_idx in (9, 10)
        assert c.score > 0

    def test_never_selects_nan_land_cells_even_at_the_strongest_front(self):
        lats = np.linspace(8.0, 9.0, 10)
        lons = np.linspace(77.0, 78.0, 10)
        sst = np.full((10, 10), 28.0)
        chl = np.full((10, 10), 0.5)
        sst[5:, :] = 29.4
        chl[5:, :] = 0.9
        # Blank out the entire front as NaN ("land"/no-data) — every real
        # signal is there, but it must never be selected.
        sst[4:6, :] = np.nan
        chl[4:6, :] = np.nan

        grid = pfz.EnvironmentalGrid(lats=lats, lons=lons, sst_celsius=sst, chlorophyll_mg_m3=chl, satellite_date="t")
        candidates = pfz.find_pfz_candidates(grid, max_zones=5, min_separation_km=5.0)
        for c in candidates:
            row_idx = int(np.argmin(np.abs(lats - c.lat)))
            assert row_idx not in (4, 5), "selected a NaN/land row"

    def test_shape_mismatch_raises(self):
        lats = np.linspace(8.0, 9.0, 5)
        lons = np.linspace(77.0, 78.0, 5)
        grid = pfz.EnvironmentalGrid(
            lats=lats,
            lons=lons,
            sst_celsius=np.zeros((5, 5)),
            chlorophyll_mg_m3=np.zeros((4, 4)),
            satellite_date="t",
        )
        with pytest.raises(ValueError):
            pfz.find_pfz_candidates(grid)

    def test_candidates_stay_at_least_min_separation_apart(self):
        lats = np.linspace(8.0, 9.0, 30)
        lons = np.linspace(77.0, 78.0, 30)
        sst = np.full((30, 30), 28.0)
        chl = np.full((30, 30), 0.5)
        # Two independent fronts, far apart, each favorable on both sides.
        sst[10, :] = 29.4
        chl[10, :] = 0.9
        sst[20, :] = 26.8
        chl[20, :] = 0.25

        grid = pfz.EnvironmentalGrid(lats=lats, lons=lons, sst_celsius=sst, chlorophyll_mg_m3=chl, satellite_date="t")
        candidates = pfz.find_pfz_candidates(grid, max_zones=5, min_separation_km=30.0)
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                dist = pfz._grid_distance_km(np.array([b.lat]), np.array([b.lon]), a.lat, a.lon)[0, 0]
                assert dist >= 30.0 - 1e-6


class TestGetCachedZones:
    """get_cached_zones() itself (Phase 1, extracted from main.py's old
    private global): TTL cache hit/miss, and combining live-or-mock PFZ
    zones with restricted-area boundaries into the one flat catalog GET
    /zones (and, since Phase 4, chat via get_nearest_anchor_zones) both
    read. Phase 1 was previously verified only by manual TestClient
    checks in-session, never by an actual pytest test — this fills that
    gap.

    Exercises the real module-level cache, so every test resets it via
    the autouse fixture below to avoid leaking state into other tests in
    this file (TestGetNearestAnchorZones etc. all monkeypatch
    pfz.get_cached_zones itself and never touch this real cache, but
    keep this isolated regardless)."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        pfz._zones_cache = None
        pfz._zones_cache_at = None
        # Fresh lock per test too — avoids any cross-test/cross-event-loop
        # state leakage from the single-flight lock (each test's
        # asyncio.run() spins up its own event loop).
        pfz._zones_cache_lock = asyncio.Lock()
        yield
        pfz._zones_cache = None
        pfz._zones_cache_at = None
        pfz._zones_cache_lock = asyncio.Lock()

    def _patch_sources(self, monkeypatch, pfz_zones, restricted_areas=()):
        call_count = {"n": 0}

        async def fake_list_live_pfz_zones(anchors):
            call_count["n"] += 1
            return pfz_zones

        monkeypatch.setattr(marine_data_agent, "list_live_pfz_zones", fake_list_live_pfz_zones)
        monkeypatch.setattr(pfz, "get_active_restricted_areas", lambda: list(restricted_areas))
        monkeypatch.setattr(pfz, "_ZONE_EXPLORER_ANCHORS", ["kochi"])
        monkeypatch.setattr(planning_agent, "_KNOWN_LOCATIONS", {"kochi": GeoPoint(lat=9.9312, lon=76.2673)})
        return call_count

    def test_cache_hit_within_ttl_does_not_recompute(self, monkeypatch):
        pfz_zones = [{"zone_id": "PFZ-001", "center": {"lat": 10.0, "lon": 76.3}, "distance_km": 8.0,
                      "sst_celsius": 28.0, "chlorophyll_mg_m3": 0.5, "advisory": "test", "near": "kochi"}]
        call_count = self._patch_sources(monkeypatch, pfz_zones)

        first = asyncio.run(pfz.get_cached_zones())
        second = asyncio.run(pfz.get_cached_zones())

        assert call_count["n"] == 1  # second call served from cache, no recompute
        assert first is second  # literally the same cached object
        assert first["generated_at"] == second["generated_at"]

    def test_cache_miss_after_ttl_expires_recomputes(self, monkeypatch):
        pfz_zones = [{"zone_id": "PFZ-001", "center": {"lat": 10.0, "lon": 76.3}, "distance_km": 8.0,
                      "sst_celsius": 28.0, "chlorophyll_mg_m3": 0.5, "advisory": "test", "near": "kochi"}]
        call_count = self._patch_sources(monkeypatch, pfz_zones)

        first = asyncio.run(pfz.get_cached_zones())
        # Force the cached entry to look stale without waiting 3 real hours.
        pfz._zones_cache_at = pfz._zones_cache_at - timedelta(seconds=pfz._ZONES_CACHE_TTL_SECONDS + 1)

        second = asyncio.run(pfz.get_cached_zones())

        assert call_count["n"] == 2  # recomputed after expiry
        assert first is not second

    def test_concurrent_calls_on_cold_cache_trigger_only_one_recompute(self, monkeypatch):
        """Audit backlog item #2 (single-flight lock): N requests arriving
        concurrently while the cache is cold/expired must share ONE
        in-flight recompute, not each kick off their own — the exact
        "several 11-anchor Copernicus recomputes stacked on top of each
        other" scenario a burst of concurrent judge traffic right after a
        TTL rollover would otherwise trigger. An artificial delay in the
        fake fetch forces genuine overlap between callers (without it,
        callers could coincidentally run sequentially fast enough to never
        actually race, which would prove nothing)."""
        call_count = {"n": 0}
        pfz_zones = [{"zone_id": "PFZ-001", "center": {"lat": 10.0, "lon": 76.3}, "distance_km": 8.0,
                      "sst_celsius": 28.0, "chlorophyll_mg_m3": 0.5, "advisory": "test", "near": "kochi"}]

        async def slow_fake_list_live_pfz_zones(anchors):
            call_count["n"] += 1
            await asyncio.sleep(0.1)
            return pfz_zones

        monkeypatch.setattr(marine_data_agent, "list_live_pfz_zones", slow_fake_list_live_pfz_zones)
        monkeypatch.setattr(pfz, "get_active_restricted_areas", list)
        monkeypatch.setattr(pfz, "_ZONE_EXPLORER_ANCHORS", ["kochi"])
        monkeypatch.setattr(planning_agent, "_KNOWN_LOCATIONS", {"kochi": GeoPoint(lat=9.9312, lon=76.2673)})

        async def _run_ten_concurrent_callers():
            return await asyncio.gather(*(pfz.get_cached_zones() for _ in range(10)))

        results = asyncio.run(_run_ten_concurrent_callers())

        assert call_count["n"] == 1  # only the first caller actually recomputed
        assert all(r is results[0] for r in results)  # everyone else got that same result

    def test_combines_pfz_and_restricted_areas_into_one_flat_list(self, monkeypatch):
        pfz_zones = [{"zone_id": "PFZ-001", "center": {"lat": 10.0, "lon": 76.3}, "distance_km": 8.0,
                      "sst_celsius": 28.0, "chlorophyll_mg_m3": 0.5, "advisory": "test", "near": "kochi"}]
        restricted = [{"name": "Gulf of Mannar", "polygon": box(79.0, 9.0, 79.2, 9.2)}]
        self._patch_sources(monkeypatch, pfz_zones, restricted_areas=restricted)

        catalog = asyncio.run(pfz.get_cached_zones())

        types = {z["type"] for z in catalog["zones"]}
        assert types == {"pfz", "restricted"}
        pfz_entry = next(z for z in catalog["zones"] if z["type"] == "pfz")
        assert pfz_entry["id"] == "KOCHI-PFZ-001"
        assert pfz_entry["near"] == "Kochi"
        restricted_entry = next(z for z in catalog["zones"] if z["type"] == "restricted")
        assert restricted_entry["id"] == "restricted-gulf-of-mannar"
        assert restricted_entry["geometry"]["type"] == "Polygon"

    def test_malformed_restricted_area_geometry_is_skipped_not_raised(self, monkeypatch):
        """A restricted area whose polygon can't be GeoJSON-mapped must be
        dropped from the catalog, not crash the whole /zones (and,
        downstream, chat) response."""
        broken_area = {"name": "Broken Area", "polygon": object()}  # no .exterior -> shapely_mapping raises
        self._patch_sources(monkeypatch, [], restricted_areas=[broken_area])

        catalog = asyncio.run(pfz.get_cached_zones())

        assert catalog["zones"] == []


def _fake_get_cached_zones(catalog: dict):
    async def _fake(*_a, **_k):
        return catalog

    return _fake


class TestGetNearestAnchorZones:
    """Two controlled anchors (kochi, chennai) so nearest-anchor selection
    and the "far from all anchors" case are both deterministic, plus one
    test against the real, full anchor gazetteer for the genuinely-sparse
    Odisha case Phase 3 will revisit."""

    _KOCHI = GeoPoint(lat=9.9312, lon=76.2673)
    _CHENNAI = GeoPoint(lat=13.0827, lon=80.2707)
    _KOCHI_ZONE_CENTER: ClassVar[dict] = {"lat": 10.0, "lon": 76.3}
    _CHENNAI_ZONE_CENTER: ClassVar[dict] = {"lat": 13.1, "lon": 80.32}

    def _catalog(self):
        kochi_dist = pfz.haversine_km(
            self._KOCHI.lat, self._KOCHI.lon, self._KOCHI_ZONE_CENTER["lat"], self._KOCHI_ZONE_CENTER["lon"]
        )
        chennai_dist = pfz.haversine_km(
            self._CHENNAI.lat, self._CHENNAI.lon, self._CHENNAI_ZONE_CENTER["lat"], self._CHENNAI_ZONE_CENTER["lon"]
        )
        return {
            "zones": [
                {
                    "id": "KOCHI-PFZ-001",
                    "name": "PFZ-001",
                    "type": "pfz",
                    "near": "Kochi",
                    "coordinates": self._KOCHI_ZONE_CENTER,
                    "distance_km": round(kochi_dist, 1),
                    "sst_celsius": 28.5,
                    "chlorophyll_mg_m3": 0.6,
                    "advisory": "Potential fishing zone near Kochi",
                },
                {
                    "id": "CHENNAI-PFZ-001",
                    "name": "PFZ-001",
                    "type": "pfz",
                    "near": "Chennai",
                    "coordinates": self._CHENNAI_ZONE_CENTER,
                    "distance_km": round(chennai_dist, 1),
                    "sst_celsius": 29.1,
                    "chlorophyll_mg_m3": 0.4,
                    "advisory": "Potential fishing zone near Chennai",
                },
                {
                    "id": "restricted-gulf-of-mannar",
                    "name": "Gulf of Mannar",
                    "type": "restricted",
                    "geometry": {"type": "Polygon", "coordinates": [[[79.0, 9.0], [79.1, 9.0], [79.1, 9.1], [79.0, 9.0]]]},
                },
            ],
            "generated_at": "2026-09-14T00:00:00Z",
        }

    def _patch(self, monkeypatch, catalog):
        monkeypatch.setattr(pfz, "_ZONE_EXPLORER_ANCHORS", ["kochi", "chennai"])
        monkeypatch.setattr(planning_agent, "_KNOWN_LOCATIONS", {"kochi": self._KOCHI, "chennai": self._CHENNAI})
        monkeypatch.setattr(pfz, "get_cached_zones", _fake_get_cached_zones(catalog))

    def test_query_exactly_at_anchor_reproduces_anchor_to_zone_distance(self, monkeypatch):
        catalog = self._catalog()
        self._patch(monkeypatch, catalog)

        result = asyncio.run(pfz.get_nearest_anchor_zones(self._KOCHI.lat, self._KOCHI.lon))

        assert len(result) == 1
        assert result[0]["id"] == "KOCHI-PFZ-001"
        original_zone = catalog["zones"][0]
        # Query point == anchor point, so the recomputed distance should
        # reproduce the same anchor->zone distance already cached (both
        # computed with the same haversine formula from the same point).
        assert result[0]["distance_km"] == pytest.approx(original_zone["distance_km"], abs=0.1)

    def test_query_far_from_controlled_anchors_still_resolves_nearest(self, monkeypatch):
        catalog = self._catalog()
        self._patch(monkeypatch, catalog)

        # Nowhere near either kochi or chennai — Mumbai's coast, ~1000s km
        # from both controlled anchors. Must still resolve to whichever of
        # the two is genuinely nearest (chennai, per haversine) rather than
        # erroring or returning nothing.
        far_lat, far_lon = 19.0760, 72.8777
        result = asyncio.run(pfz.get_nearest_anchor_zones(far_lat, far_lon))

        assert len(result) == 1
        assert result[0]["id"] == "CHENNAI-PFZ-001"
        expected_distance = pfz.haversine_km(
            far_lat, far_lon, self._CHENNAI_ZONE_CENTER["lat"], self._CHENNAI_ZONE_CENTER["lon"]
        )
        assert result[0]["distance_km"] == pytest.approx(round(expected_distance, 1), abs=0.05)
        # Real distance from Mumbai to the recomputed zone is large — sanity
        # check this isn't accidentally returning the old anchor-relative
        # value (which would be small, ~kochi_dist/chennai_dist scale).
        assert result[0]["distance_km"] > 1000

    def test_query_far_from_every_real_anchor_still_resolves_nearest(self, monkeypatch):
        """Odisha mid-coast point (~103km from the nearest real anchor,
        paradip) against the real, unpatched anchor gazetteer — the sparse-
        anchor-grid case Phase 3 will revisit. Must not error or return
        nonsense even though no anchor is genuinely close."""
        paradip = planning_agent._KNOWN_LOCATIONS["paradip"]
        zone_center = {"lat": paradip.lat + 0.1, "lon": paradip.lon + 0.1}
        catalog = {
            "zones": [
                {
                    "id": "PARADIP-PFZ-001",
                    "name": "PFZ-001",
                    "type": "pfz",
                    "near": "Paradip",
                    "coordinates": zone_center,
                    "distance_km": round(pfz.haversine_km(paradip.lat, paradip.lon, zone_center["lat"], zone_center["lon"]), 1),
                    "sst_celsius": 27.9,
                    "chlorophyll_mg_m3": 0.7,
                    "advisory": "Potential fishing zone near Paradip",
                },
            ],
            "generated_at": "2026-09-14T00:00:00Z",
        }
        monkeypatch.setattr(pfz, "get_cached_zones", _fake_get_cached_zones(catalog))

        odisha_lat, odisha_lon = 19.8, 85.8  # mid-coast, near Puri
        result = asyncio.run(pfz.get_nearest_anchor_zones(odisha_lat, odisha_lon))

        assert len(result) == 1
        assert result[0]["id"] == "PARADIP-PFZ-001"
        expected_distance = pfz.haversine_km(odisha_lat, odisha_lon, zone_center["lat"], zone_center["lon"])
        assert result[0]["distance_km"] == pytest.approx(round(expected_distance, 1), abs=0.05)
        assert result[0]["distance_km"] > 50  # genuinely far — not a small in-town figure

    def test_restricted_areas_are_excluded_and_other_fields_pass_through_unchanged(self, monkeypatch):
        catalog = self._catalog()
        self._patch(monkeypatch, catalog)

        result = asyncio.run(pfz.get_nearest_anchor_zones(self._KOCHI.lat, self._KOCHI.lon))

        assert all(z["type"] == "pfz" for z in result)
        assert not any("restricted" in z["id"] for z in result)

        original_zone = catalog["zones"][0]
        returned_zone = result[0]
        for key in ("id", "name", "type", "near", "coordinates", "sst_celsius", "chlorophyll_mg_m3", "advisory"):
            assert returned_zone[key] == original_zone[key]
        # Only distance_km is allowed to differ (here it coincidentally
        # matches too, since query == anchor — covered by the dedicated
        # test above; this test is about every *other* field).
        assert set(returned_zone.keys()) == set(original_zone.keys())

    def test_no_configured_anchors_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(pfz, "_ZONE_EXPLORER_ANCHORS", [])
        monkeypatch.setattr(planning_agent, "_KNOWN_LOCATIONS", {})
        result = asyncio.run(pfz.get_nearest_anchor_zones(10.0, 76.0))
        assert result == []
        assert result.is_distant is False


class TestGetNearestAnchorZonesIsDistant:
    """Phase 3: distance honesty flag. Reuses the same controlled two-anchor
    fixture and the real-gazetteer Odisha fixture from
    TestGetNearestAnchorZones so the underlying resolution logic under test
    is identical to Phase 2 — only the new `.is_distant` attribute is new
    here."""

    _KOCHI = TestGetNearestAnchorZones._KOCHI
    _CHENNAI = TestGetNearestAnchorZones._CHENNAI

    def test_query_at_anchor_is_not_distant(self, monkeypatch):
        fixture = TestGetNearestAnchorZones()
        catalog = fixture._catalog()
        fixture._patch(monkeypatch, catalog)

        result = asyncio.run(pfz.get_nearest_anchor_zones(self._KOCHI.lat, self._KOCHI.lon))

        assert isinstance(result, pfz.NearestAnchorZones)
        assert result.is_distant is False
        # Existing Phase 2-style list access still works unchanged.
        assert len(result) == 1
        assert result[0]["id"] == "KOCHI-PFZ-001"

    def test_query_near_anchor_within_threshold_is_not_distant(self, monkeypatch):
        fixture = TestGetNearestAnchorZones()
        catalog = fixture._catalog()
        fixture._patch(monkeypatch, catalog)

        # ~0.3 degrees latitude off Kochi is ~33km — comfortably inside
        # _NEARBY_ANCHOR_THRESHOLD_KM (75km).
        near_lat, near_lon = self._KOCHI.lat + 0.3, self._KOCHI.lon
        distance_to_anchor = pfz.haversine_km(near_lat, near_lon, self._KOCHI.lat, self._KOCHI.lon)
        assert distance_to_anchor < pfz._NEARBY_ANCHOR_THRESHOLD_KM

        result = asyncio.run(pfz.get_nearest_anchor_zones(near_lat, near_lon))
        assert result.is_distant is False

    def test_odisha_style_far_point_is_distant(self, monkeypatch):
        """Same ~103km-from-Paradip fixture as
        TestGetNearestAnchorZones.test_query_far_from_every_real_anchor_still_resolves_nearest
        — confirms the real, unpatched anchor gazetteer's genuinely sparse
        coverage trips the flag."""
        paradip = planning_agent._KNOWN_LOCATIONS["paradip"]
        zone_center = {"lat": paradip.lat + 0.1, "lon": paradip.lon + 0.1}
        catalog = {
            "zones": [
                {
                    "id": "PARADIP-PFZ-001",
                    "name": "PFZ-001",
                    "type": "pfz",
                    "near": "Paradip",
                    "coordinates": zone_center,
                    "distance_km": round(pfz.haversine_km(paradip.lat, paradip.lon, zone_center["lat"], zone_center["lon"]), 1),
                    "sst_celsius": 27.9,
                    "chlorophyll_mg_m3": 0.7,
                    "advisory": "Potential fishing zone near Paradip",
                },
            ],
            "generated_at": "2026-09-14T00:00:00Z",
        }
        monkeypatch.setattr(pfz, "get_cached_zones", _fake_get_cached_zones(catalog))

        odisha_lat, odisha_lon = 19.8, 85.8  # mid-coast, near Puri, ~103km from paradip
        distance_to_anchor = pfz.haversine_km(odisha_lat, odisha_lon, paradip.lat, paradip.lon)
        assert distance_to_anchor > pfz._NEARBY_ANCHOR_THRESHOLD_KM

        result = asyncio.run(pfz.get_nearest_anchor_zones(odisha_lat, odisha_lon))
        assert result.is_distant is True
        # Still returns the real (recomputed-distance) zone data — the flag
        # is additive, it never empties or withholds the result.
        assert len(result) == 1
        assert result[0]["id"] == "PARADIP-PFZ-001"

    def test_query_just_inside_and_just_outside_threshold(self, monkeypatch):
        """Boundary check directly against the threshold constant, using a
        single controlled anchor at the origin so the query->anchor
        distance is simple to reason about."""
        anchor = GeoPoint(lat=10.0, lon=76.0)
        zone_center = {"lat": 10.05, "lon": 76.05}
        catalog = {
            "zones": [
                {
                    "id": "TEST-PFZ-001",
                    "name": "PFZ-001",
                    "type": "pfz",
                    "near": "Testanchor",
                    "coordinates": zone_center,
                    "distance_km": 8.0,
                    "sst_celsius": 28.0,
                    "chlorophyll_mg_m3": 0.5,
                    "advisory": "test",
                },
            ],
            "generated_at": "2026-09-14T00:00:00Z",
        }
        monkeypatch.setattr(pfz, "_ZONE_EXPLORER_ANCHORS", ["testanchor"])
        monkeypatch.setattr(planning_agent, "_KNOWN_LOCATIONS", {"testanchor": anchor})
        monkeypatch.setattr(pfz, "get_cached_zones", _fake_get_cached_zones(catalog))

        # 0.5 degrees latitude ~= 55.5km — inside the 75km threshold.
        close_result = asyncio.run(pfz.get_nearest_anchor_zones(anchor.lat + 0.5, anchor.lon))
        assert close_result.is_distant is False

        # 1.0 degrees latitude ~= 111km — outside the 75km threshold.
        far_result = asyncio.run(pfz.get_nearest_anchor_zones(anchor.lat + 1.0, anchor.lon))
        assert far_result.is_distant is True


class TestCopernicusCredentials:
    """copernicus_credentials() is the one gate every Copernicus caller
    (pfz_service, marine_data_agent chlorophyll, analytics_service) goes
    through — same placeholder rule as the GROQ_API_KEY check."""

    @pytest.mark.parametrize(
        ("username", "password"),
        [
            ("", ""),
            ("real-user", ""),
            ("", "real-pass"),
            ("your-copernicusmarine-username-here", "your-copernicusmarine-password-here"),
            ("real-user", "your-copernicusmarine-password-here"),
            ("your_copernicusmarine_username_here", "real-pass"),
            ("  YOUR-Copernicusmarine-Username-Here  ", "real-pass"),
        ],
    )
    def test_unset_or_placeholder_values_count_as_no_credentials(self, monkeypatch, username, password):
        monkeypatch.setenv("COPERNICUSMARINE_USERNAME", username)
        monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", password)
        assert pfz.copernicus_credentials() is None

    def test_real_values_are_returned_stripped(self, monkeypatch):
        monkeypatch.setenv("COPERNICUSMARINE_USERNAME", " real-user ")
        monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "real-pass\n")
        assert pfz.copernicus_credentials() == ("real-user", "real-pass")
