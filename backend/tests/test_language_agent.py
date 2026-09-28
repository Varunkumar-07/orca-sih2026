"""Pytest coverage for backend/agents/reasoning/language_agent.py (Phase 6).

Recovered/rebuilt suite: a real 7-test suite for this module existed on disk
as of the "Phase 5-6" commit (f7374a4) but was never `git add`ed, so the
source itself was lost — only its compiled bytecode survived in
backend/tests/__pycache__/test_language_agent.cpython-312-pytest-9.1.1.pyc.
That bytecode was decompiled (dis/marshal) to recover the test names,
fixture helpers, mock literals, and assertions below almost verbatim; this
file is a rebuild against the *current* language_agent.py, not a byte-exact
transcription of whatever the original source looked like.

Covers 7 cases:
  a. English query and English answer — pure passthrough, no Bhashini call
     made at all on either side (detection is entirely local; nothing to
     translate).
  b. Hindi query — detection + translation on the input side, and
     translation back to Hindi on the output side, via a mocked Bhashini
     Pipeline Config + Compute call sequence.
  c. Tamil query — detection alone (no mocking needed), confirming a second
     script beyond Devanagari resolves correctly.
  d. Bhashini Pipeline Config Call failure (_post_json returns None) —
     confirms graceful fallback to the original text, and — per the Q6
     finding from the earlier audit, where fallback responses were found to
     silently lose these fields — that response_language still comes back
     as a real, non-None string ("en").
  e. Bhashini Pipeline Compute Call failure (config call succeeds, compute
     call fails) — a distinct failure point from (d), same graceful
     outcome, same explicit language-field check.
  f. No BHASHINI_USER_ID/BHASHINI_ULCA_API_KEY configured —
     _get_pipeline_config's short-circuit is hit before any network call,
     confirmed via a fake _post_json that raises if it's ever called at all.

(Cases (d) and (e) each also cover requirement (g): confirming fallback-mode
responses still populate detected_language/response_language correctly,
rather than a separate 8th test — the original recovered suite has exactly
7 test functions, and both failure paths are exactly where that population
matters.)

Mocking strategy: `_post_json` is language_agent.py's own network boundary
(the one function that actually calls urllib) — patching it directly lets
these tests exercise the real config-call/compute-call orchestration logic
(cache keys, response-shape parsing, success/failure propagation) without
any real HTTP traffic or live BHASHINI credentials.

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import pytest

from backend.agents.reasoning import language_agent as la


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Every test gets its own empty pipeline-config cache and a real-looking
    (but fake) Bhashini key pair, so _get_pipeline_config's "no key configured"
    short-circuit doesn't mask what's actually being tested."""
    la._pipeline_config_cache.clear()
    monkeypatch.setenv("BHASHINI_USER_ID", "test-user-id")
    monkeypatch.setenv("BHASHINI_ULCA_API_KEY", "test-ulca-key")
    yield
    la._pipeline_config_cache.clear()


def _fake_config_response(service_id: str = "ai4bharat/indictrans--gpu-t4") -> dict:
    """Shape confirmed against bhashini.gitbook.io/bhashini-apis/pipeline-config-call
    and the community reference client this was checked against."""
    return {
        "pipelineResponseConfig": [
            {"taskType": "translation", "config": [{"serviceId": service_id}]}
        ],
        "pipelineInferenceAPIEndPoint": {
            "callbackUrl": "https://dhruva-api.bhashini.gov.in/services/inference/pipeline",
            "inferenceApiKey": {"name": "Authorization", "value": "fake-inference-key-value"},
        },
    }


def _fake_compute_response(source: str, target: str) -> dict:
    """Shape confirmed against bhashini.gitbook.io/bhashini-apis/pipeline-compute-call."""
    return {
        "pipelineResponse": [
            {"taskType": "translation", "output": [{"source": source, "target": target}]}
        ]
    }


# ---------------------------------------------------------------------------
# (a) English passthrough — no translation call made on either side
# ---------------------------------------------------------------------------


def test_english_query_passthrough_no_translation_call(monkeypatch):
    calls = []
    monkeypatch.setattr(la, "_post_json", lambda *a, **k: calls.append((a, k)))

    text = "Is it safe to go out tomorrow near Chennai?"
    english_text, detected = la.translate_input_to_english(text)

    assert detected == "en"
    assert english_text == text
    assert calls == [], "no network call should happen for an English query"


def test_english_output_passthrough_no_translation_call(monkeypatch):
    calls = []
    monkeypatch.setattr(la, "_post_json", lambda *a, **k: calls.append((a, k)))

    answer = "Safe to go (confidence 88%) — Calm seas, no active alerts."
    text, lang = la.translate_output_from_english(answer, "en")

    assert lang == "en"
    assert text == answer
    assert calls == []


# ---------------------------------------------------------------------------
# (b) Hindi — detection + translation, both directions
# ---------------------------------------------------------------------------


def test_hindi_query_detected_translated_and_reply_returned_in_hindi(monkeypatch):
    hindi_query = "क्या कल चेन्नई के पास जाना सुरक्षित है?"
    english_translation = "Is it safe to go near Chennai tomorrow?"
    english_answer = "Safe to go (confidence 88%) — Calm seas, no active alerts."
    hindi_answer = "सुरक्षित है (विश्वास 88%) - शांत समुद्र, कोई सक्रिय चेतावनी नहीं।"

    def fake_post_json(url, body, headers):
        if url == la._ULCA_CONFIG_URL:
            return _fake_config_response()
        pair = body["pipelineTasks"][0]["config"]["language"]
        source_text = body["inputData"]["input"][0]["source"]
        if pair["sourceLanguage"] == "hi" and pair["targetLanguage"] == "en":
            return _fake_compute_response(source_text, english_translation)
        if pair["sourceLanguage"] == "en" and pair["targetLanguage"] == "hi":
            return _fake_compute_response(source_text, hindi_answer)
        raise AssertionError(f"unexpected language pair in compute call: {pair}")

    monkeypatch.setattr(la, "_post_json", fake_post_json)

    english_text, detected = la.translate_input_to_english(hindi_query)
    assert detected == "hi"
    assert english_text == english_translation

    final_text, response_language = la.translate_output_from_english(english_answer, detected)
    assert response_language == "hi"
    assert final_text == hindi_answer


# ---------------------------------------------------------------------------
# (c) Tamil — detection alone
# ---------------------------------------------------------------------------


def test_tamil_query_detected_correctly():
    """Detection alone (no mocking needed) for a second script, since the
    task calls out "Hindi or Tamil" as acceptable choices."""
    tamil_query = "நாளை சென்னைக்கு அருகில் செல்வது பாதுகாப்பானதா?"

    _text, detected = la.translate_input_to_english(tamil_query)

    assert detected == "ta"


# ---------------------------------------------------------------------------
# (d)/(e)/(g) Bhashini failures — graceful fallback, language fields still real
# ---------------------------------------------------------------------------


def test_bhashini_failure_falls_back_to_english_gracefully(monkeypatch):
    """Simulates the Pipeline Config Call itself failing (network error,
    bad key, etc. — _post_json returns None, exactly as it does on any
    request exception) and confirms every layer degrades instead of raising."""
    monkeypatch.setattr(la, "_post_json", lambda *a, **k: None)

    hindi_query = "क्या कल चेन्नई के पास जाना सुरक्षित है?"

    english_text, detected = la.translate_input_to_english(hindi_query)
    assert detected == "hi"
    assert english_text == hindi_query

    english_answer = "Safe to go (confidence 88%) — Calm seas, no active alerts."
    final_text, response_language = la.translate_output_from_english(english_answer, "hi")

    # Q6 audit finding: a fallback response must never silently lose its
    # language fields — response_language has to come back as a real,
    # non-None string ("en"), not whatever a careless caller might default to.
    assert response_language is not None
    assert response_language == "en"
    assert final_text == english_answer


def test_bhashini_failure_at_compute_call_also_falls_back(monkeypatch):
    """Config call succeeds, but the compute call itself fails — a
    different failure point than the test above, same graceful outcome."""
    call_count = {"n": 0}

    def fake_post_json(url, body, headers):
        call_count["n"] += 1
        if url == la._ULCA_CONFIG_URL:
            return _fake_config_response()
        return None  # compute call fails

    monkeypatch.setattr(la, "_post_json", fake_post_json)

    text, ok = la.translate("hello", "en", "hi")
    assert ok is False
    assert text == "hello"
    assert call_count["n"] == 2

    # Same Q6 check as the config-call-failure case above, but through the
    # compute-call failure path specifically (config succeeds and is cached,
    # only the compute call itself fails).
    final_text, response_language = la.translate_output_from_english("hello", "hi")
    assert response_language is not None
    assert response_language == "en"
    assert final_text == "hello"


# ---------------------------------------------------------------------------
# (f) No API key configured — skip gracefully, never crash
# ---------------------------------------------------------------------------


def test_no_api_key_configured_skips_translation_gracefully(monkeypatch):
    """Missing/placeholder credentials — same fallback path as any other
    Bhashini failure, confirmed separately since it's the most common
    real-world "not set up yet" case (see backend/.env.example)."""
    monkeypatch.delenv("BHASHINI_USER_ID", raising=False)
    monkeypatch.delenv("BHASHINI_ULCA_API_KEY", raising=False)

    def fail_if_called(*a, **k):
        raise AssertionError("_post_json must not be called when no API key is configured")

    monkeypatch.setattr(la, "_post_json", fail_if_called)

    text, ok = la.translate("hello", "en", "hi")
    assert ok is False
    assert text == "hello"


# ---------------------------------------------------------------------------
# (h) Line-by-line batched translation preserves structure — added this
# session after confirming live that translating the whole multi-line
# answer_text as one string made Bhashini silently collapse every newline
# into a space, breaking the frontend's line-structured rendering for any
# non-English response.
# ---------------------------------------------------------------------------


def _fake_compute_response_multi(targets: list[str]) -> dict:
    return {
        "pipelineResponse": [
            {"taskType": "translation", "output": [{"target": t} for t in targets]}
        ]
    }


def test_translate_preserves_line_structure(monkeypatch):
    text = "Query: is it safe near Chennai?\nLocation: 13.08, 80.27\nSafe to go."
    translated_lines = ["ప్రశ్నః చెన్నై సమీపంలో సురక్షితమేనా?", "స్థానంః 13.08, 80.27", "సురక్షితం."]

    def fake_post_json(url, body, headers):
        if url == la._ULCA_CONFIG_URL:
            return _fake_config_response()
        sent = [item["source"] for item in body["inputData"]["input"]]
        assert sent == text.split("\n"), "each line must be sent as its own input item, in order"
        return _fake_compute_response_multi(translated_lines)

    monkeypatch.setattr(la, "_post_json", fake_post_json)

    result, ok = la.translate(text, "en", "te")

    assert ok is True
    assert result == "\n".join(translated_lines)


def test_translate_preserves_blank_lines_without_sending_them(monkeypatch):
    """A blank line must stay in place by position — never sent for
    translation (nothing to translate) and never dropped from the output."""
    text = "line one\n\nline three"

    def fake_post_json(url, body, headers):
        if url == la._ULCA_CONFIG_URL:
            return _fake_config_response()
        sent = [item["source"] for item in body["inputData"]["input"]]
        assert sent == ["line one", "line three"], "the blank line must not be sent"
        return _fake_compute_response_multi(["translated one", "translated three"])

    monkeypatch.setattr(la, "_post_json", fake_post_json)

    result, ok = la.translate(text, "en", "hi")

    assert ok is True
    assert result == "translated one\n\ntranslated three"


def test_translate_output_count_mismatch_falls_back(monkeypatch):
    """A malformed/short response (fewer outputs than lines sent) must
    degrade to the original text, never silently misalign translated
    lines onto the wrong originals."""
    text = "line one\nline two"

    def fake_post_json(url, body, headers):
        if url == la._ULCA_CONFIG_URL:
            return _fake_config_response()
        return _fake_compute_response_multi(["only one output"])  # expected 2

    monkeypatch.setattr(la, "_post_json", fake_post_json)

    result, ok = la.translate(text, "en", "hi")

    assert ok is False
    assert result == text


# ---------------------------------------------------------------------------
# (i) _post_json retries once on a transient failure before giving up
# ---------------------------------------------------------------------------


def test_post_json_retries_once_on_transient_failure(monkeypatch):
    attempts = {"n": 0}

    def flaky_once(url, body, headers):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return None  # simulated transient failure
        return {"ok": True}

    monkeypatch.setattr(la, "_post_json_once", flaky_once)

    result = la._post_json("https://example.test", {}, {})

    assert result == {"ok": True}
    assert attempts["n"] == 2


def test_post_json_gives_up_after_two_failures(monkeypatch):
    attempts = {"n": 0}

    def always_fails(url, body, headers):
        attempts["n"] += 1  # implicitly returns None, simulating a failed call

    monkeypatch.setattr(la, "_post_json_once", always_fails)

    result = la._post_json("https://example.test", {}, {})

    assert result is None
    assert attempts["n"] == 2
