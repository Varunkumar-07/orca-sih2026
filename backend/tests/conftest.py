"""Points the whole test suite at the `orca_test` Postgres database (not
`orca_dev`) for the Phase 8 history layer, before any test module imports
backend.main (and, transitively, backend.history.db).

Must run before that first import: conftest.py is collected by pytest
ahead of every test module, so os.environ.setdefault() here always wins
over main.py's own load_dotenv() call (which defaults to override=False).

If DATABASE_URL is already set in the shell environment (e.g. CI without a
local Postgres), this leaves it untouched — and if no Postgres is reachable
at all, backend/history/db.py's own no-op fallback means the existing 100
tests are unaffected either way.
"""
import asyncio
import os

import numpy as np
import pytest

os.environ.setdefault(
    "DATABASE_URL",
    f"postgresql+asyncpg://{os.getenv('USER', 'postgres')}@localhost:5432/orca_test",
)

# The test suite fires hundreds of requests through one shared TestClient
# (and therefore one shared RateLimitMiddleware instance/bucket, since
# backend.main.app is a module-level singleton reused across every test
# file) well within a minute — RateLimitMiddleware itself is covered by its
# own dedicated unit tests (backend/tests/test_rate_limit.py) against an
# isolated app instance, not through this shared one.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")


# --- Offline PFZ data for endpoint tests -----------------------------------
#
# GET /zones (and /route, /export, /weather/forecast, which resolve zone ids
# through its catalog) builds PFZ zones from live Copernicus grids. CI has no
# Copernicus credentials, so without this those endpoints return zero PFZ
# zones there. Live-data coverage belongs to test_live_smoke.py
# (.github/workflows/live-smoke.yml).
#
# Only the network boundary is faked: fetch_environmental_grid returns a
# synthetic grid with a genuine SST front (along lon) crossing a genuine
# chlorophyll front (along lat) at the box center, both inside analytics.py's
# optimal bands. Everything downstream — find_pfz_candidates' real front
# scoring, list_live_pfz_zones, get_cached_zones' catalog, and the endpoints —
# runs unmodified.


def _synthetic_front_grid(min_lon, max_lon, min_lat, max_lat):
    from backend.services.pfz_service import EnvironmentalGrid

    lats = np.linspace(min_lat, max_lat, 31)
    lons = np.linspace(min_lon, max_lon, 31)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    return EnvironmentalGrid(
        lats=lats,
        lons=lons,
        sst_celsius=28.25 + 0.75 * np.tanh((lon2d - center_lon) / 0.1),
        chlorophyll_mg_m3=0.6 + 0.3 * np.tanh((lat2d - center_lat) / 0.1),
        satellite_date="2026-01-01",
    )


def _reset_zones_cache():
    from backend.services import pfz_service

    pfz_service._zones_cache = None
    pfz_service._zones_cache_at = None
    pfz_service._zones_cache_lock = asyncio.Lock()


@pytest.fixture(scope="module")
def offline_pfz_data():
    """Module-scoped so it's in place before a module-scoped TestClient
    starts — the app's startup pre-warm populates the zones cache, and it
    must see this fake rather than a live (or credential-less) fetch. The
    cache is reset on both sides so neither real nor fake zones leak across
    modules."""
    from backend.agents.reasoning import marine_data_agent

    async def fake_fetch(min_lon, max_lon, min_lat, max_lat):
        return _synthetic_front_grid(min_lon, max_lon, min_lat, max_lat)

    _reset_zones_cache()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(marine_data_agent, "fetch_environmental_grid", fake_fetch)
        yield
    _reset_zones_cache()
