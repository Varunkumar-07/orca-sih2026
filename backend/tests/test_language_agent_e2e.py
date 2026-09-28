"""End-to-end coverage for the Language Agent as wired into the real request
flow (backend/routers/chat.py's _resolve_input_language /
_apply_output_language, wrapping POST /query/demo, /query/full, /query) —
closes the one gap the existing unit-level suite doesn't cover:
test_language_agent.py drives language_agent.py's own functions directly
(with _post_json mocked); nothing drives a genuine non-English query through
the real HTTP endpoint. This file does.

On BHASHINI_* credentials: this codebase's own convention (see
backend/.env.example, and _get_pipeline_config's short-circuit in
language_agent.py) is that translation degrades gracefully to English
whenever BHASHINI_USER_ID/BHASHINI_ULCA_API_KEY aren't configured — the same
"works with zero keys" guarantee every other agent here follows. In
practice, a developer's local backend/.env may still have real-looking
credentials set (main.py's load_dotenv() loads it before pytest ever runs —
confirmed present, non-placeholder-shaped, on this machine while writing
this file), which would make an otherwise-unmocked end-to-end test
non-deterministic: it could attempt a real live Bhashini call depending on
who/where it runs, and CI has no .env at all. So `_no_bhashini_credentials`
below forces the missing-credentials path explicitly, on every machine,
regardless of what's actually configured locally — this is what makes "the
pipeline degrades gracefully to English" a genuine assertion about behavior,
not an assumption about whoever's environment happens to run the suite.

Run: PYTHONPATH=. pytest backend/tests/test_language_agent_e2e.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.agents.reasoning import language_agent as la
from backend.main import app


@pytest.fixture(autouse=True)
def _no_bhashini_credentials(monkeypatch):
    """See module docstring. Also clears language_agent's own module-level
    pipeline-config cache, in case an earlier test in this same pytest
    process already resolved and cached a config from real credentials."""
    monkeypatch.delenv("BHASHINI_USER_ID", raising=False)
    monkeypatch.delenv("BHASHINI_ULCA_API_KEY", raising=False)
    la._pipeline_config_cache.clear()
    yield
    la._pipeline_config_cache.clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_bhashini_credentials_are_unset_for_this_test():
    """Sanity check on the fixture above's own premise, not a duplicate of
    it — confirms the env really is clear before trusting the behavioral
    assertions below."""
    import os

    assert os.getenv("BHASHINI_USER_ID") is None
    assert os.getenv("BHASHINI_ULCA_API_KEY") is None


def test_hindi_query_through_query_demo_detects_language_and_degrades_output_to_english(client):
    """Drives POST /query/demo (fixture-only, no Groq call needed — keeps
    this test independent of GROQ_API_KEY too) with a genuine Hindi
    (Devanagari-script) query. Real end-to-end path: no agent internals
    mocked, only credentials forced absent (see fixture above).

    Expected, given no Bhashini credentials:
      - detected_language == "hi": local Unicode-script detection needs no
        credentials at all, so this must succeed regardless.
      - response_language == "en": translation genuinely could not run, so
        the response must honestly report English rather than claim a
        Hindi translation that never happened (language_agent.py's own
        explicit contract — see translate_output_from_english's docstring).
      - the request still succeeds (200) with a real, non-empty
        answer_text — a failed translation must never break the response.
    """
    resp = client.post("/query/demo", json={"query": "क्या चेन्नई के पास जाना सुरक्षित है?"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["detected_language"] == "hi"
    assert body["response_language"] == "en"
    assert isinstance(body["answer_text"], str) and body["answer_text"].strip()


def test_tamil_query_through_query_demo_also_detects_correctly(client):
    """A second, distinct script (Tamil, not Devanagari) through the same
    real endpoint — confirms detection isn't accidentally overfit to one
    script family, and degrades the same way."""
    resp = client.post("/query/demo", json={"query": "சென்னைக்கு அருகில் செல்வது பாதுகாப்பானதா?"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["detected_language"] == "ta"
    assert body["response_language"] == "en"


def test_english_query_through_query_demo_is_unaffected(client):
    """Baseline: an English query through the same real endpoint reports
    both language fields as "en" — the common case costs nothing extra."""
    resp = client.post("/query/demo", json={"query": "is it safe to go out near Chennai?"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["detected_language"] == "en"
    assert body["response_language"] == "en"
