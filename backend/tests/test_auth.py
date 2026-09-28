"""Coverage for backend/auth.py (previously untested at all — found during
a post-selection deploy audit while verifying ORCA_API_KEY end to end).

Built against a small isolated FastAPI app (not backend.main.app), same
convention as test_rate_limit.py: these tests need to freely toggle
ORCA_API_KEY on/off and reach into the real middleware, without touching
the shared app instance every other test file in this suite reuses (which
is always built with ORCA_API_KEY unset — see conftest.py's own docstring
on why that shared instance must stay that way).

Run: PYTHONPATH=. pytest backend/tests/test_auth.py -v
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import ApiKeyAuthMiddleware, is_enabled


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ApiKeyAuthMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_is_enabled_false_when_unset(monkeypatch):
    monkeypatch.delenv("ORCA_API_KEY", raising=False)
    assert is_enabled() is False


def test_is_enabled_false_for_placeholder_values(monkeypatch):
    for placeholder in ("", "your-api-key-here", "your_api_key_here", "YOUR-API-KEY-HERE", "  "):
        monkeypatch.setenv("ORCA_API_KEY", placeholder)
        assert is_enabled() is False, f"{placeholder!r} should not count as configured"


def test_is_enabled_true_for_a_real_value(monkeypatch):
    monkeypatch.setenv("ORCA_API_KEY", "a-real-secret-value")
    assert is_enabled() is True


def test_request_without_the_header_is_rejected(monkeypatch):
    monkeypatch.setenv("ORCA_API_KEY", "correct-secret")
    client = TestClient(_make_app())

    resp = client.get("/ping")
    assert resp.status_code == 401
    assert "X-API-Key" in resp.json()["detail"]


def test_request_with_the_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setenv("ORCA_API_KEY", "correct-secret")
    client = TestClient(_make_app())

    resp = client.get("/ping", headers={"X-API-Key": "wrong-secret"})
    assert resp.status_code == 401


def test_request_with_the_correct_key_succeeds(monkeypatch):
    monkeypatch.setenv("ORCA_API_KEY", "correct-secret")
    client = TestClient(_make_app())

    resp = client.get("/ping", headers={"X-API-Key": "correct-secret"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_health_endpoint_is_exempt_even_with_no_key_sent(monkeypatch):
    monkeypatch.setenv("ORCA_API_KEY", "correct-secret")
    client = TestClient(_make_app())

    resp = client.get("/health")
    assert resp.status_code == 200


def test_backend_main_runs_with_auth_disabled_under_the_test_suite():
    """conftest.py never sets ORCA_API_KEY, and backend/.env (loaded by
    backend.main at import time) doesn't set it for the test DB either —
    this is what keeps the other 290+ tests (all sharing one TestClient/app
    instance, hitting real endpoints with no X-API-Key header) from getting
    401'd. Guards against that convention silently regressing, same as
    test_rate_limit.py's own equivalent guard for RATE_LIMIT_ENABLED."""
    assert is_enabled() is False
