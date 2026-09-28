"""Coverage for backend/rate_limit.py (audit backlog item #10 — "no
request-level rate limiting on any endpoint").

Built against a small isolated FastAPI app (not backend.main.app) so these
tests can freely construct low limits/short windows and multiple distinct
client IPs (via Starlette TestClient's `client=(ip, port)` override)
without touching the shared app instance every other test file in this
suite reuses — and without needing RATE_LIMIT_ENABLED=true (see
conftest.py) to exercise the real thing end to end.

Run: PYTHONPATH=. pytest backend/tests/test_rate_limit.py -v
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.rate_limit import RateLimitMiddleware, is_enabled


def _make_app(limit: int = 3, window_seconds: float = 60.0) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit=limit, window_seconds=window_seconds)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_requests_within_the_limit_all_succeed():
    client = TestClient(_make_app(limit=3), client=("1.2.3.4", 1234))
    for _ in range(3):
        resp = client.get("/ping")
        assert resp.status_code == 200


def test_the_request_over_the_limit_gets_429_with_retry_after():
    client = TestClient(_make_app(limit=3), client=("1.2.3.4", 1234))
    for _ in range(3):
        assert client.get("/ping").status_code == 200

    resp = client.get("/ping")
    assert resp.status_code == 429
    assert resp.json()["detail"]
    assert int(resp.headers["Retry-After"]) >= 1


def test_different_client_ips_get_independent_buckets():
    app = _make_app(limit=1)
    client_a = TestClient(app, client=("1.1.1.1", 1))
    client_b = TestClient(app, client=("2.2.2.2", 2))

    assert client_a.get("/ping").status_code == 200
    # client_a is now over its own limit...
    assert client_a.get("/ping").status_code == 429
    # ...but client_b has never made a request, so it's unaffected.
    assert client_b.get("/ping").status_code == 200


def test_health_endpoint_is_exempt_even_past_the_limit():
    client = TestClient(_make_app(limit=1), client=("9.9.9.9", 9))
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429  # confirms the limit really is tripped

    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_the_window_slides_so_capacity_frees_up_over_time(monkeypatch):
    import backend.rate_limit as rate_limit_module

    fake_now = [1000.0]
    monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: fake_now[0])

    client = TestClient(_make_app(limit=1, window_seconds=10.0), client=("5.5.5.5", 5))
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429

    fake_now[0] += 10.01  # past the window — the earlier hit should age out
    assert client.get("/ping").status_code == 200


def test_is_enabled_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    assert is_enabled() is False

    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    assert is_enabled() is False

    monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
    assert is_enabled() is True  # defaults to enabled (opt-out, not opt-in)

    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    assert is_enabled() is True


def test_backend_main_disables_rate_limiting_under_the_test_suite():
    """conftest.py sets RATE_LIMIT_ENABLED=false before backend.main is ever
    imported — this is what keeps the other 260+ tests (all sharing one
    TestClient/app instance) from tripping a real limit. Guards against that
    convention silently regressing."""
    assert os.environ.get("RATE_LIMIT_ENABLED") == "false"
    assert is_enabled() is False
