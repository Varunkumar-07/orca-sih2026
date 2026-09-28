"""Pytest coverage for backend/agents/reasoning/marine_data_agent.py's live
integrations (added this session — previously untested).

Mocking strategy: httpx.AsyncClient.get patched at the class level for SST
(same approach as test_weather_agent.py); _fetch_chlorophyll_sync (the
blocking Copernicus call run via asyncio.to_thread) is patched directly
rather than mocking the copernicusmarine library itself — that's the actual
boundary this module crosses into a dependency, same "patch our own
function, not the vendor SDK" principle as everywhere else. Tests are plain
sync functions driving async code via asyncio.run() (no pytest-asyncio
dependency, matching test_weather_agent.py).

Covers:
  - _fetch_live_sst: normal case, marine-grid gap -> None
  - _fetch_live_chlorophyll: missing credentials -> None, no call attempted;
    real value on success; timeout -> None without raising
  - run_marine_data_agent: status="partial" when chlorophyll/SST missing,
    and the same intent-classification decoupling fix covered for
    weather_agent — a broken _classify_intent must not discard real SST/PFZ
    data; status="error" when no live PFZ zones are available at all (no
    mock/sample fallback — see module docstring)
  - _live_zones_for_anchor / list_live_pfz_zones (Zones Explorer's live PFZ
    front detection): live path returns real "PFZ-*" ids; grid unavailable,
    scoring raising, or no candidates found -> empty list for that anchor
    (never mock/sample data) rather than propagating through asyncio.gather
    and taking down every other anchor; concurrency is actually bounded by
    _MAX_CONCURRENT_PFZ_ANCHORS, not just nominally passed a semaphore
  - _live_pfz_candidate_zones / run_marine_data_agent (Chat-PFZ Live-Data
    Integration): chat reads the shared live cache via
    get_nearest_anchor_zones — returns its zones with an "is_distant" flag
    stamped on, sorted nearest-first; returns an empty list (never mock
    data) on an exception, a timeout past _LIVE_PFZ_LOOKUP_TIMEOUT, or an
    empty live result; run_marine_data_agent surfaces that as
    status="error" rather than silently fabricating zones.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
import types

import httpx

from backend.agents.reasoning import marine_data_agent as mda
from backend.schemas.contracts import GeoPoint
from backend.services import pfz_service as pfz
from backend.services.pfz_service import PFZCandidate

_CHENNAI = GeoPoint(lat=13.08, lon=80.27)
_KOCHI = GeoPoint(lat=9.9312, lon=76.2673)


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


def _canonical_zone(zone_id="PFZ-001", distance_km=12.0):
    """Canonical MarineDataResult.pfz_zones shape ("zone_id"/"center") —
    the actual return shape of _live_pfz_candidate_zones, distinct from
    the catalog shape _live_zone() below fabricates for mocking
    get_nearest_anchor_zones directly (see _live_zone's own docstring for
    why those two shapes must never be conflated)."""
    return {
        "zone_id": zone_id,
        "center": {"lat": 13.1, "lon": 80.3},
        "distance_km": distance_km,
        "sst_celsius": 28.4,
        "chlorophyll_mg_m3": 0.5,
        "advisory": "Potential fishing zone — live satellite data.",
        "is_distant": False,
    }


# ---------------------------------------------------------------------------
# _fetch_live_sst
# ---------------------------------------------------------------------------


def test_fetch_live_sst_normal_case(monkeypatch):
    async def fake_get(self, url, params=None, headers=None):
        assert url == mda._MARINE_URL
        return _FakeResponse({"current": {"sea_surface_temperature": 30.7}})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    sst = asyncio.run(mda._fetch_live_sst(_CHENNAI))
    assert sst == 30.7


def test_fetch_live_sst_grid_gap_returns_none(monkeypatch):
    async def fake_get(self, url, params=None, headers=None):
        return _FakeResponse({"current": {"sea_surface_temperature": None}})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    sst = asyncio.run(mda._fetch_live_sst(_CHENNAI))
    assert sst is None


# ---------------------------------------------------------------------------
# _fetch_live_chlorophyll
# ---------------------------------------------------------------------------


def test_fetch_live_chlorophyll_no_credentials_skips_gracefully(monkeypatch):
    monkeypatch.delenv("COPERNICUSMARINE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_PASSWORD", raising=False)

    def fail_if_called(*_a, **_k):
        raise AssertionError("_fetch_chlorophyll_sync must not be called without credentials")

    monkeypatch.setattr(mda, "_fetch_chlorophyll_sync", fail_if_called)

    chl = asyncio.run(mda._fetch_live_chlorophyll(_CHENNAI))
    assert chl is None


def test_fetch_live_chlorophyll_success(monkeypatch):
    monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "test-user")
    monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "test-pass")
    monkeypatch.setattr(mda, "_fetch_chlorophyll_sync", lambda *_a, **_k: 0.157)

    chl = asyncio.run(mda._fetch_live_chlorophyll(_CHENNAI))
    assert chl == 0.157


def test_fetch_live_chlorophyll_times_out_without_raising(monkeypatch):
    """Copernicus's auth was observed live to occasionally hang well past a
    normal request budget — asyncio.wait_for must cut this off cleanly."""
    monkeypatch.setenv("COPERNICUSMARINE_USERNAME", "test-user")
    monkeypatch.setenv("COPERNICUSMARINE_PASSWORD", "test-pass")
    monkeypatch.setattr(mda, "_CHL_TIMEOUT", 0.05)

    def slow_sync(*_a, **_k):
        import time

        time.sleep(1.0)
        return 0.2

    monkeypatch.setattr(mda, "_fetch_chlorophyll_sync", slow_sync)

    chl = asyncio.run(mda._fetch_live_chlorophyll(_CHENNAI))
    assert chl is None


# ---------------------------------------------------------------------------
# run_marine_data_agent
# ---------------------------------------------------------------------------


def test_run_marine_data_agent_no_location_is_error():
    trace = []
    result = asyncio.run(mda.run_marine_data_agent("where can I fish?", None, trace))
    assert result.status == "error"


def test_run_marine_data_agent_partial_when_sst_and_chlorophyll_missing(monkeypatch):
    monkeypatch.setattr(mda, "_classify_intent", _async_return("general"))
    # PFZ sourcing is covered separately below — this test is about
    # SST/chlorophyll status degradation only, so stub in a fixed live zone
    # rather than exercising the live lookup + real anchor gazetteer here.
    monkeypatch.setattr(mda, "_live_pfz_candidate_zones", _async_return([_canonical_zone()]))
    monkeypatch.setattr(mda, "_fetch_live_sst", _async_return(None))
    monkeypatch.setattr(mda, "_fetch_live_chlorophyll", _async_return(None))

    trace = []
    result = asyncio.run(mda.run_marine_data_agent("where can I fish?", _CHENNAI, trace))

    assert result.status == "partial"
    assert result.sst_celsius is None
    assert result.chlorophyll_mg_m3 is None
    assert len(result.pfz_zones) == 1


def test_intent_classification_failure_does_not_discard_real_marine_data(monkeypatch):
    """Same bug class as weather_agent's equivalent test: a broken Groq
    intent call must degrade to intent="general" in the trace without
    discarding the (still real) SST/chlorophyll/candidate-zone data."""
    monkeypatch.setattr(mda, "_live_pfz_candidate_zones", _async_return([_canonical_zone()]))
    monkeypatch.setattr(mda, "_fetch_live_sst", _async_return(30.7))
    monkeypatch.setattr(mda, "_fetch_live_chlorophyll", _async_return(0.157))

    async def broken_classify(_query_text):
        raise RuntimeError("simulated Groq JSON validation failure")

    monkeypatch.setattr(mda, "_classify_intent", broken_classify)

    trace = []
    result = asyncio.run(mda.run_marine_data_agent("where can I fish?", _CHENNAI, trace))

    assert result.status == "ok"
    assert result.sst_celsius == 30.7
    assert result.chlorophyll_mg_m3 == 0.157
    assert "intent=general" in trace[-1].output_summary


def test_run_marine_data_agent_errors_when_no_live_zones_available(monkeypatch):
    """No mock/sample fallback: a live PFZ lookup that comes back empty
    must surface as status="error", never as fabricated zones."""
    monkeypatch.setattr(mda, "_live_pfz_candidate_zones", _async_return([]))
    monkeypatch.setattr(mda, "_fetch_live_sst", _async_return(30.7))
    monkeypatch.setattr(mda, "_fetch_live_chlorophyll", _async_return(0.157))

    trace = []
    result = asyncio.run(mda.run_marine_data_agent("where can I fish?", _CHENNAI, trace))

    assert result.status == "error"
    assert result.pfz_zones == []


# --- Zones Explorer live PFZ front detection --------------------------------


def test_live_zones_for_anchor_returns_real_ids_when_grid_available(monkeypatch):
    fake_grid = types.SimpleNamespace(satellite_date="2026-09-12")
    monkeypatch.setattr(mda, "fetch_environmental_grid", _async_return(fake_grid))
    candidates = [
        PFZCandidate(lat=13.2, lon=80.4, sst_celsius=28.9, chlorophyll_mg_m3=0.55, score=0.8),
        PFZCandidate(lat=13.4, lon=80.5, sst_celsius=29.1, chlorophyll_mg_m3=0.61, score=0.6),
    ]
    monkeypatch.setattr(mda, "find_pfz_candidates", lambda grid, **_k: candidates)

    zones = asyncio.run(mda._live_zones_for_anchor(_CHENNAI, asyncio.Semaphore(4)))

    assert [z["zone_id"] for z in zones] == ["PFZ-001", "PFZ-002"]
    assert zones[0]["sst_celsius"] == 28.9
    assert zones[0]["chlorophyll_mg_m3"] == 0.55
    assert "ORCA-computed estimate" in zones[0]["advisory"]


def test_live_zones_for_anchor_empty_when_grid_unavailable(monkeypatch):
    monkeypatch.setattr(mda, "fetch_environmental_grid", _async_return(None))

    zones = asyncio.run(mda._live_zones_for_anchor(_CHENNAI, asyncio.Semaphore(4)))

    assert zones == []


def test_live_zones_for_anchor_empty_when_scoring_raises(monkeypatch):
    """The exact resilience bug found and fixed in Phase 5: find_pfz_candidates
    raising must degrade this one anchor to an empty result, not propagate
    out of _live_zones_for_anchor and take down every other anchor gathered
    alongside it in list_live_pfz_zones."""
    monkeypatch.setattr(mda, "fetch_environmental_grid", _async_return(object()))

    def broken_scoring(_grid, **_k):
        raise ValueError("simulated malformed grid")

    monkeypatch.setattr(mda, "find_pfz_candidates", broken_scoring)

    zones = asyncio.run(mda._live_zones_for_anchor(_CHENNAI, asyncio.Semaphore(4)))

    assert zones == []


def test_live_zones_for_anchor_empty_when_no_candidates_found(monkeypatch):
    """A live grid with no detectable front is a legitimate "nothing today"
    result (see test_pfz_service.py's flat-region test), not an error — but
    with no mock fallback, that anchor simply contributes zero zones."""
    monkeypatch.setattr(mda, "fetch_environmental_grid", _async_return(object()))
    monkeypatch.setattr(mda, "find_pfz_candidates", lambda grid, **_k: [])

    zones = asyncio.run(mda._live_zones_for_anchor(_CHENNAI, asyncio.Semaphore(4)))

    assert zones == []


def test_list_live_pfz_zones_combines_multiple_anchors_with_near_tag(monkeypatch):
    fake_grid = types.SimpleNamespace(satellite_date="2026-09-12")
    monkeypatch.setattr(mda, "fetch_environmental_grid", _async_return(fake_grid))
    monkeypatch.setattr(
        mda,
        "find_pfz_candidates",
        lambda grid, **_k: [PFZCandidate(lat=1.0, lon=2.0, sst_celsius=28.0, chlorophyll_mg_m3=0.5, score=0.5)],
    )

    zones = asyncio.run(mda.list_live_pfz_zones({"chennai": _CHENNAI, "kochi": _KOCHI}))

    assert {z["near"] for z in zones} == {"chennai", "kochi"}
    assert len(zones) == 2


def test_list_live_pfz_zones_bounds_concurrency(monkeypatch):
    """Concurrency must actually be capped at _MAX_CONCURRENT_PFZ_ANCHORS —
    not just nominally passed a semaphore that happens to do nothing."""
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def tracked_fetch(*_a, **_k):
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        # Implicitly returns None, forcing the (fast) empty-result path.

    monkeypatch.setattr(mda, "fetch_environmental_grid", tracked_fetch)

    anchors = {f"anchor{i}": GeoPoint(lat=10.0 + i, lon=75.0 + i) for i in range(mda._MAX_CONCURRENT_PFZ_ANCHORS * 2)}
    asyncio.run(mda.list_live_pfz_zones(anchors))

    assert peak <= mda._MAX_CONCURRENT_PFZ_ANCHORS
    assert peak > 1  # sanity: this exercised real concurrency, not accidental serialization


# ---------------------------------------------------------------------------
# _live_pfz_candidate_zones / run_marine_data_agent live wiring
# ---------------------------------------------------------------------------


def _live_zone(zone_id="PFZ-001", lat=13.1, lon=80.3, distance_km=12.0):
    """Catalog-shaped zone dict — the REAL return shape of
    get_nearest_anchor_zones() (see pfz_service.get_cached_zones: "id",
    "name", "coordinates", not "zone_id"/"center"). zone_id here feeds
    the catalog's "id" field; _live_pfz_candidate_zones is responsible for
    remapping that to the canonical "zone_id" key the rest of the pipeline
    expects — this fixture must NOT pre-remap it, or a bug in that remap
    would go undetected (as one originally did: real live data crashed
    with KeyError('zone_id') because this fixture used the canonical shape
    as input, silently matching the bug's own wrong assumption instead of
    catching it)."""
    return {
        "id": zone_id,
        "name": zone_id,
        "type": "pfz",
        "near": "Chennai",
        "coordinates": {"lat": lat, "lon": lon},
        "distance_km": distance_km,
        "sst_celsius": 28.4,
        "chlorophyll_mg_m3": 0.5,
        "advisory": "Potential fishing zone — SST/chlorophyll front detected in live satellite data.",
    }


def test_live_pfz_candidate_zones_uses_live_cache(monkeypatch):
    live = pfz.NearestAnchorZones([_live_zone()], is_distant=False)
    monkeypatch.setattr(mda, "get_nearest_anchor_zones", _async_return(live))

    result = asyncio.run(mda._live_pfz_candidate_zones(_CHENNAI))

    assert len(result) == 1
    assert result[0]["zone_id"] == "PFZ-001"
    assert result[0]["is_distant"] is False


def test_live_pfz_candidate_zones_flags_is_distant(monkeypatch):
    live = pfz.NearestAnchorZones([_live_zone(zone_id="PFZ-002", distance_km=103.4)], is_distant=True)
    monkeypatch.setattr(mda, "get_nearest_anchor_zones", _async_return(live))

    result = asyncio.run(mda._live_pfz_candidate_zones(_CHENNAI))

    assert result[0]["zone_id"] == "PFZ-002"
    assert result[0]["is_distant"] is True


def test_live_pfz_candidate_zones_sorts_nearest_first(monkeypatch):
    """get_nearest_anchor_zones returns zones in front-detection score
    order, not distance order — this function must sort them so the
    "nearest first" assumption (run_marine_data_agent's trace summary
    relies on) holds."""
    unsorted = pfz.NearestAnchorZones(
        [_live_zone(zone_id="PFZ-002", distance_km=40.0), _live_zone(zone_id="PFZ-001", distance_km=8.0)],
        is_distant=False,
    )
    monkeypatch.setattr(mda, "get_nearest_anchor_zones", _async_return(unsorted))

    result = asyncio.run(mda._live_pfz_candidate_zones(_CHENNAI))

    assert [z["zone_id"] for z in result] == ["PFZ-001", "PFZ-002"]


def test_live_pfz_candidate_zones_empty_on_exception(monkeypatch):
    async def broken(*_a, **_k):
        raise RuntimeError("simulated live lookup failure")

    monkeypatch.setattr(mda, "get_nearest_anchor_zones", broken)

    result = asyncio.run(mda._live_pfz_candidate_zones(_CHENNAI))

    assert result == []


def test_live_pfz_candidate_zones_empty_on_timeout(monkeypatch):
    async def hangs(*_a, **_k):
        await asyncio.sleep(10)
        return pfz.NearestAnchorZones([_live_zone()], is_distant=False)  # pragma: no cover

    monkeypatch.setattr(mda, "get_nearest_anchor_zones", hangs)
    monkeypatch.setattr(mda, "_LIVE_PFZ_LOOKUP_TIMEOUT", 0.05)

    result = asyncio.run(mda._live_pfz_candidate_zones(_CHENNAI))

    assert result == []


def test_live_pfz_candidate_zones_empty_when_live_result_empty(monkeypatch):
    monkeypatch.setattr(mda, "get_nearest_anchor_zones", _async_return(pfz.NearestAnchorZones([], is_distant=False)))

    result = asyncio.run(mda._live_pfz_candidate_zones(_CHENNAI))

    assert result == []


def test_run_marine_data_agent_reads_live_zones_end_to_end(monkeypatch):
    live = pfz.NearestAnchorZones([_live_zone(zone_id="PFZ-001", distance_km=9.5)], is_distant=False)
    monkeypatch.setattr(mda, "get_nearest_anchor_zones", _async_return(live))
    monkeypatch.setattr(mda, "_classify_intent", _async_return("pfz"))
    monkeypatch.setattr(mda, "_fetch_live_sst", _async_return(28.4))
    monkeypatch.setattr(mda, "_fetch_live_chlorophyll", _async_return(0.5))

    trace = []
    result = asyncio.run(mda.run_marine_data_agent("where can I fish?", _CHENNAI, trace))

    assert result.status == "ok"
    assert result.pfz_zones[0]["zone_id"] == "PFZ-001"
    assert result.pfz_zones[0]["is_distant"] is False
    assert "live" in trace[-1].output_summary
    assert "distant" not in trace[-1].output_summary


def test_run_marine_data_agent_marks_distant_anchor_in_trace(monkeypatch):
    live = pfz.NearestAnchorZones([_live_zone(zone_id="PFZ-001", distance_km=103.4)], is_distant=True)
    monkeypatch.setattr(mda, "get_nearest_anchor_zones", _async_return(live))
    monkeypatch.setattr(mda, "_classify_intent", _async_return("pfz"))
    monkeypatch.setattr(mda, "_fetch_live_sst", _async_return(28.4))
    monkeypatch.setattr(mda, "_fetch_live_chlorophyll", _async_return(0.5))

    trace = []
    result = asyncio.run(mda.run_marine_data_agent("where can I fish?", _CHENNAI, trace))

    assert result.pfz_zones[0]["is_distant"] is True
    assert "distant anchor match" in trace[-1].output_summary


def test_run_marine_data_agent_errors_out_when_live_lookup_fails(monkeypatch):
    """No mock/sample fallback: a live-lookup failure must surface as
    status="error", not a silently fabricated response."""

    async def broken(*_a, **_k):
        raise RuntimeError("simulated Copernicus/cache outage")

    monkeypatch.setattr(mda, "get_nearest_anchor_zones", broken)
    monkeypatch.setattr(mda, "_classify_intent", _async_return("pfz"))
    monkeypatch.setattr(mda, "_fetch_live_sst", _async_return(28.4))
    monkeypatch.setattr(mda, "_fetch_live_chlorophyll", _async_return(0.5))

    trace = []
    result = asyncio.run(mda.run_marine_data_agent("where can I fish?", _CHENNAI, trace))

    assert result.status == "error"
    assert result.pfz_zones == []
