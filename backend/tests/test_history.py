"""Coverage for the Phase 8 history persistence layer: HistoryLoggingMiddleware
(backend/history/middleware.py), the read/write service functions
(backend/history/service.py), and the GET /history + GET /history/{id} API.

Run: PYTHONPATH=. pytest backend/tests/test_history.py -v
(conftest.py points DATABASE_URL at the local `orca_test` Postgres database
before backend.main — and transitively backend.history.db — is imported.)

Drives the real FastAPI app via Starlette's TestClient, same convention as
test_main_geolocation.py, so this asserts against the actual HTTP contract
the History page will use, not just the service functions in isolation.

`client` is a module-scoped fixture using `with TestClient(app) as c:`
rather than a bare `TestClient(app)` global: the async SQLAlchemy engine's
pooled asyncpg connections are bound to whichever event loop first used
them, and a bare TestClient spins up a fresh loop per call — the `with`
form keeps one loop alive for every request in this module.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.agents.reasoning import marine_data_agent as mda
from backend.history import db as history_db
from backend.history.service import list_history, log_history
from backend.main import app


@pytest.fixture(autouse=True)
def _no_live_pfz_fetch(monkeypatch):
    """This file's /zones (and /export, which calls it internally) tests
    aren't about PFZ front detection — without this, they'd trigger a real
    ~55s live Copernicus fetch across all 11 anchors on every run. Forcing
    the live fetch to report "unavailable" exercises the exact same
    mock-fallback path a real Copernicus outage would, so these tests stay
    fast, deterministic, and offline, without weakening what they check."""
    monkeypatch.setattr(mda, "fetch_environmental_grid", lambda *_a, **_k: _async_none())


async def _async_none():
    return None


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_history_db_is_enabled_for_tests():
    """Sanity check that conftest.py's DATABASE_URL actually wired up a real
    engine — if this fails, every other test below degrades to a silent
    no-op instead of genuinely exercising the history layer."""
    assert history_db.is_enabled(), "DATABASE_URL not configured — see backend/tests/conftest.py"


def test_query_demo_is_logged_as_chat_history(client):
    """POST /query/demo (chat page) should produce a `chat` history row
    without changing anything about the endpoint's own response shape."""
    resp = client.post("/query/demo", json={"query": "mannar cyclone risk"})
    assert resp.status_code == 200
    assert "answer_text" in resp.json()  # unchanged FinalResponse contract

    listing = client.get("/history", params={"page_source": "chat", "limit": 5})
    assert listing.status_code == 200
    body = listing.json()
    assert body["items"], "expected at least one chat history row"
    assert all(item["page_source"] == "chat" for item in body["items"])


def test_query_demo_session_id_roundtrips_into_history(client):
    """A request that supplies its own session_id should end up logged
    under that same session_id, not a freshly generated one."""
    resp = client.post("/query/demo", json={"query": "goa weather", "session_id": "test-session-abc"})
    assert resp.status_code == 200

    listing = client.get("/history", params={"page_source": "chat", "limit": 20})
    matches = [item for item in listing.json()["items"] if item["session_id"] == "test-session-abc"]
    assert matches, "expected a history row logged under the request's own session_id"


def test_history_list_pagination_and_filter_by_page_source(client):
    for _ in range(3):
        client.post("/query/demo", json={"query": "chennai fishing zones"})

    page1 = client.get("/history", params={"page_source": "chat", "limit": 2, "offset": 0}).json()
    page2 = client.get("/history", params={"page_source": "chat", "limit": 2, "offset": 2}).json()

    assert len(page1["items"]) == 2
    assert page1["total"] >= 5  # at least the rows from this + prior tests
    assert {item["id"] for item in page1["items"]}.isdisjoint({item["id"] for item in page2["items"]})


def test_history_list_filters_by_date_range(client):
    today = client.get("/history", params={"start_date": "2000-01-01", "end_date": "2999-12-31"}).json()
    future_only = client.get("/history", params={"start_date": "2999-12-31", "end_date": "2999-12-31"}).json()
    assert today["total"] > 0
    assert future_only["total"] == 0


def test_history_detail_returns_request_and_response_payload(client):
    client.post("/query/demo", json={"query": "kochi restricted zone check"})
    listing = client.get("/history", params={"page_source": "chat", "limit": 1}).json()
    record_id = listing["items"][0]["id"]

    detail = client.get(f"/history/{record_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == record_id
    assert "payload" in body
    assert "request" in body["payload"]
    assert "response" in body["payload"]


def test_history_detail_404_for_unknown_id(client):
    resp = client.get("/history/999999999")
    assert resp.status_code == 404


def test_history_endpoint_itself_is_not_logged(client):
    """Browsing history shouldn't create more history — GET /history and
    GET /history/{id} aren't in HistoryLoggingMiddleware's path map."""
    first_listing = client.get("/history", params={"limit": 1}).json()
    before = first_listing["total"]
    record_id = first_listing["items"][0]["id"]

    client.get("/history", params={"limit": 1})
    client.get(f"/history/{record_id}")

    after = client.get("/history", params={"limit": 1}).json()["total"]
    assert after == before


def test_export_download_page_is_logged_with_response_metadata_not_raw_bytes(client):
    """GET /export returns a non-JSON file (csv/pdf/docx) — the middleware
    must capture size/content-type metadata for the JSONB payload rather
    than dumping raw binary/text into it."""
    zones_resp = client.get("/zones")
    assert zones_resp.status_code == 200
    pfz_zone = next(z for z in zones_resp.json()["zones"] if z["type"] == "pfz")

    resp = client.get(
        "/export",
        params={"zone": pfz_zone["id"], "start_date": "2024-01-01", "end_date": "2024-01-03", "format": "csv"},
    )
    assert resp.status_code == 200

    listing = client.get("/history", params={"page_source": "download", "limit": 1}).json()
    assert listing["items"], "expected a download history row"
    detail = client.get(f"/history/{listing['items'][0]['id']}").json()
    assert detail["payload"]["response"]["content_type"].startswith("text/csv")
    assert isinstance(detail["payload"]["response"]["size_bytes"], int)


def test_zones_page_is_logged(client):
    resp = client.get("/zones")
    assert resp.status_code == 200
    listing = client.get("/history", params={"page_source": "zones", "limit": 1}).json()
    assert listing["items"]


def test_log_history_is_a_silent_noop_when_db_disabled(monkeypatch):
    """When DATABASE_URL isn't configured (or the engine failed to init),
    log_history/list_history must degrade to a no-op rather than raising —
    this is what keeps the other 8 pages working with zero DB setup."""
    monkeypatch.setattr(history_db, "_SessionLocal", None)

    asyncio.run(
        log_history(
            page_source="weather",
            query_summary="should be skipped",
            session_id="whatever",
            request_payload={"lat": 1, "lon": 2},
            response_payload={"ok": True},
        )
    )

    result = asyncio.run(list_history(page_source="weather"))
    assert result == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_log_history_swallows_a_mid_request_db_outage(monkeypatch):
    """Distinct from the test above: DATABASE_URL IS configured and
    is_enabled() is True — is_enabled() only proves the engine was
    constructed at import time, not that Postgres is actually reachable
    right now. This simulates the DB going down (connection refused, same
    as `pg_ctl stop` mid-request) exactly when log_history opens its
    session, and asserts service.py's own try/except around the commit
    (backend/history/service.py) still degrades to a warning-logged no-op
    rather than propagating — audit finding #11."""

    class _ExplodingSessionFactory:
        def __call__(self):
            raise ConnectionRefusedError("simulated Postgres outage: connection refused")

    monkeypatch.setattr(history_db, "_SessionLocal", _ExplodingSessionFactory())
    assert history_db.is_enabled(), "outage must be simulated with the DB still reporting configured/enabled"

    # Must not raise — this is the actual assertion.
    asyncio.run(
        log_history(
            page_source="chat",
            query_summary="query made during a live outage",
            session_id="outage-session",
            request_payload={"query": "is it safe?"},
            response_payload={"answer_text": "ok"},
        )
    )


def test_query_demo_still_responds_normally_through_a_mid_request_db_outage(monkeypatch, client):
    """End-to-end version of the test above, through the real ASGI stack:
    HistoryLoggingMiddleware.dispatch() itself has no try/except around its
    `await log_history(...)` call (backend/history/middleware.py) — it
    relies entirely on log_history() never raising. This confirms that
    reliance holds under a live DB outage: /query/demo must still return
    its normal 200 response, unaffected, even though the history write
    silently fails behind it."""

    class _ExplodingSessionFactory:
        def __call__(self):
            raise ConnectionRefusedError("simulated Postgres outage: connection refused")

    # service.py's `get_session_factory` name is imported directly from
    # backend.history.db (the same function object), so patching db.py's
    # module-global _SessionLocal here is enough — no separate patch of
    # backend.history.service needed.
    monkeypatch.setattr(history_db, "_SessionLocal", _ExplodingSessionFactory())

    resp = client.post("/query/demo", json={"query": "mannar cyclone risk", "session_id": "outage-e2e"})
    assert resp.status_code == 200
    assert "answer_text" in resp.json()


def test_history_filter_by_unknown_page_source_returns_empty(client):
    result = client.get("/history", params={"page_source": "not-a-real-page", "limit": 5}).json()
    assert result["items"] == []
    assert result["total"] == 0
