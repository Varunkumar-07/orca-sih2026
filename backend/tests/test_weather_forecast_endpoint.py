"""End-to-end coverage for GET /weather/forecast (backend/routers/pages.py)
— previously untested at the HTTP/router level at all (found during a
post-selection deploy audit: backend/tests/test_forecast_service.py only
ever drove forecast_service.get_forecast() directly, never through the
actual route + real zone-id-to-coordinates resolution the frontend
depends on). Same pattern test_route_endpoint.py already uses: real
FastAPI app via TestClient, real GET /zones catalog (whatever this
environment's live PFZ cache actually holds — see that file's own
docstring for why this is the genuine production code path, not a
substitute), only the Open-Meteo network fetches mocked (same technique
and same patch target as test_forecast_service.py — the underlying
RandomForestRegressor models are real, loaded from backend/models/, not
mocked, since this is specifically testing the full real pipeline).

Run: PYTHONPATH=. pytest backend/tests/test_weather_forecast_endpoint.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import forecast_service as svc


@pytest.fixture(scope="module")
def client(offline_pfz_data):
    """offline_pfz_data (conftest.py): zone-id resolution needs PFZ zones in
    GET /zones's catalog without live Copernicus access."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _mocked_open_meteo_fetches(monkeypatch):
    """Same mocking principle as test_forecast_service.py: patch our own
    fetch functions, not the network. Sufficient, valid values for every
    required feature so the real models actually run to completion."""

    async def fake_marine(lat, lon, start_date, end_date):
        return {"wave_height_max": [1.0, 1.2], "sea_surface_temperature_max": [27.5, 28.0]}

    async def fake_archive(lat, lon, start_date, end_date):
        return {"wind_speed_10m_max": [10.0, 12.0], "temperature_2m_mean": [29.0, 29.5]}

    monkeypatch.setattr(svc, "_fetch_marine_daily", fake_marine)
    monkeypatch.setattr(svc, "_fetch_archive_daily", fake_archive)


def _first_pfz_zone_id(client: TestClient) -> str | None:
    catalog = client.get("/zones").json()
    pfz = next((z for z in catalog["zones"] if z.get("type") == "pfz"), None)
    return pfz["id"] if pfz else None


def _first_restricted_zone_id(client: TestClient) -> str | None:
    catalog = client.get("/zones").json()
    restricted = next((z for z in catalog["zones"] if z.get("type") == "restricted"), None)
    return restricted["id"] if restricted else None


def test_forecast_for_a_real_zone_returns_seven_predictions(client):
    zone_id = _first_pfz_zone_id(client)
    assert zone_id is not None, "expected a PFZ zone in GET /zones (offline_pfz_data fixture)"

    resp = client.get("/weather/forecast", params={"zone": zone_id})
    assert resp.status_code == 200

    body = resp.json()
    assert body["zone"] == zone_id
    assert body["status"] == "ok"
    assert len(body["forecast"]) == 7

    horizons_seen = set()
    for entry in body["forecast"]:
        horizons_seen.add(entry["horizon"])
        assert isinstance(entry["wave_height_m"], float)
        assert isinstance(entry["wind_kmh"], float)
        # Real MAE from forecast_metrics.json, not a placeholder — a
        # genuinely trained model reports this for every horizon.
        assert entry["wave_height_mae"] is not None
        assert entry["wind_kmh_mae"] is not None
    assert horizons_seen == {1, 2, 3, 4, 5, 6, 7}


def test_unknown_zone_id_returns_400_with_a_clear_reason(client):
    resp = client.get("/weather/forecast", params={"zone": "this-zone-does-not-exist"})
    assert resp.status_code == 400
    assert "zone not found" in resp.json()["detail"]


def test_restricted_area_zone_id_returns_400_since_it_has_no_point_coordinates(client):
    """A restricted-area entry (a polygon boundary, e.g. a marine
    biosphere reserve) has no single point coordinates to forecast at —
    the same "not found or has no point coordinates" rejection as an
    unknown id, not a crash trying to read a coordinates field that
    isn't there."""
    zone_id = _first_restricted_zone_id(client)
    if zone_id is None:
        pytest.skip("no restricted-area zone available in this environment's cache to test against")

    resp = client.get("/weather/forecast", params={"zone": zone_id})
    assert resp.status_code == 400
    assert "zone not found" in resp.json()["detail"]
