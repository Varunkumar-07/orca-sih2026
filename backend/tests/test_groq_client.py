"""Pytest coverage for backend/agents/reasoning/_groq_client.py (previously
untested — added as part of the codebase audit's P2 coverage gaps).

Mocking strategy: this module IS the boundary every other reasoning-agent
test file patches around (call_groq_json itself) — so testing it properly
means going one level deeper and faking the vendor SDK class (AsyncGroq)
directly, the one place in this suite where that's the right call rather
than a shortcut.

Covers:
  - Missing GROQ_API_KEY -> raises RuntimeError before ever constructing a
    client (no network attempt at all)
  - Successful call -> parses the JSON response body correctly, and calls
    the SDK with the exact fixed params this module promises (temperature
    0, "low" reasoning effort, JSON-object response format, system+user
    messages in order)
  - A raw SDK exception (rate limit / auth / timeout) propagates
    unmodified — this module explicitly does not catch on the caller's
    behalf (see its own docstring)
  - An invalid/empty JSON completion body raises via json.loads, also
    uncaught here

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from groq import APITimeoutError, RateLimitError

from backend.agents.reasoning import _groq_client as svc


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, response=None, exc: Exception | None = None, capture: dict | None = None):
        self._response = response
        self._exc = exc
        self._capture = capture

    async def create(self, **kwargs):
        if self._capture is not None:
            self._capture.update(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeAsyncGroq:
    def __init__(self, completions: _FakeCompletions):
        self._completions = completions

    def __call__(self, api_key: str):
        # Mirrors the real AsyncGroq(api_key=...) constructor call.
        self.received_api_key = api_key
        instance = _FakeAsyncGroqInstance(self._completions)
        return instance


class _FakeAsyncGroqInstance:
    def __init__(self, completions: _FakeCompletions):
        self.chat = _FakeChat(completions)


def test_missing_api_key_raises_without_constructing_a_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def fail_if_called(*_a, **_k):
        raise AssertionError("must not construct AsyncGroq without an API key")

    monkeypatch.setattr(svc, "AsyncGroq", fail_if_called)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        asyncio.run(svc.call_groq_json("system", "user", max_tokens=100))


def test_successful_call_parses_json_response(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    response = _FakeCompletionResponse('{"safe_to_go": true, "confidence": 0.9}')
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(_FakeCompletions(response=response)))

    result = asyncio.run(svc.call_groq_json("system prompt", "user content", max_tokens=200))

    assert result == {"safe_to_go": True, "confidence": 0.9}


def test_call_uses_the_documented_fixed_params(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    captured: dict = {}
    response = _FakeCompletionResponse("{}")
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(_FakeCompletions(response=response, capture=captured)))

    asyncio.run(svc.call_groq_json("SYS", "USR", max_tokens=321))

    assert captured["model"] == svc._GROQ_MODEL
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 321
    assert captured["reasoning_effort"] == "low"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


def test_custom_model_override_is_passed_through(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    captured: dict = {}
    response = _FakeCompletionResponse("{}")
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(_FakeCompletions(response=response, capture=captured)))

    asyncio.run(svc.call_groq_json("SYS", "USR", max_tokens=100, model="some-other-model"))

    assert captured["model"] == "some-other-model"


def test_sdk_exception_propagates_uncaught(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        svc, "AsyncGroq", _FakeAsyncGroq(_FakeCompletions(exc=RuntimeError("Error code: 429 - rate limited")))
    )

    with pytest.raises(RuntimeError, match="Error code: 429"):
        asyncio.run(svc.call_groq_json("system", "user", max_tokens=100))


def test_invalid_json_completion_body_raises_uncaught(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    response = _FakeCompletionResponse("not valid json")
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(_FakeCompletions(response=response)))

    with pytest.raises(json.JSONDecodeError):
        asyncio.run(svc.call_groq_json("system", "user", max_tokens=100))


def _fake_groq_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _fake_rate_limit_error() -> RateLimitError:
    response = httpx.Response(429, request=_fake_groq_request())
    return RateLimitError("rate limited", response=response, body=None)


def _fake_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_fake_groq_request())


class _SequencedCompletions:
    """Like _FakeCompletions, but raises/returns a different outcome on each
    successive call — needed to simulate "fails N times, then recovers" for
    the retry tests below. Each entry is either an Exception instance (to
    raise) or an _FakeCompletionResponse (to return)."""

    def __init__(self, outcomes: list):
        self._outcomes = list(outcomes)
        self.call_count = 0

    async def create(self, **_kwargs):
        outcome = self._outcomes[self.call_count]
        self.call_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _no_delay(_seconds: float) -> None:
    return None


def _patch_no_sleep(monkeypatch):
    """Retries add a real asyncio.sleep backoff — patched to a no-op so
    these tests don't actually wait ~0.5-1.5s each. `svc.asyncio` IS the
    real `asyncio` module (not a copy), so the replacement must not itself
    call `asyncio.sleep` — that would recurse into the very patch being
    applied."""
    monkeypatch.setattr(svc.asyncio, "sleep", _no_delay)


def test_rate_limit_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    _patch_no_sleep(monkeypatch)
    completions = _SequencedCompletions([_fake_rate_limit_error(), _FakeCompletionResponse('{"ok": true}')])
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(completions))

    result = asyncio.run(svc.call_groq_json("system", "user", max_tokens=100))

    assert result == {"ok": True}
    assert completions.call_count == 2


def test_connection_timeout_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    _patch_no_sleep(monkeypatch)
    # APITimeoutError is a subclass of APIConnectionError — this also covers
    # the more general connection-error case.
    completions = _SequencedCompletions([_fake_timeout_error(), _FakeCompletionResponse('{"ok": true}')])
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(completions))

    result = asyncio.run(svc.call_groq_json("system", "user", max_tokens=100))

    assert result == {"ok": True}
    assert completions.call_count == 2


def test_retries_are_exhausted_after_max_attempts_then_raises(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    _patch_no_sleep(monkeypatch)
    # One more failure than _MAX_RETRIES allows for — every attempt fails.
    completions = _SequencedCompletions([_fake_rate_limit_error() for _ in range(svc._MAX_RETRIES + 1)])
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(completions))

    with pytest.raises(RateLimitError):
        asyncio.run(svc.call_groq_json("system", "user", max_tokens=100))

    assert completions.call_count == svc._MAX_RETRIES + 1


def test_non_retryable_exception_is_raised_on_the_first_attempt_only(monkeypatch):
    """A plain SDK/auth-style failure (not RateLimitError/APIConnectionError)
    must NOT be retried — same "never catches on the caller's behalf"
    contract as before, for anything outside the narrow transient set."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    _patch_no_sleep(monkeypatch)
    completions = _SequencedCompletions([RuntimeError("Error code: 401 - invalid api key")])
    monkeypatch.setattr(svc, "AsyncGroq", _FakeAsyncGroq(completions))

    with pytest.raises(RuntimeError, match="Error code: 401"):
        asyncio.run(svc.call_groq_json("system", "user", max_tokens=100))

    assert completions.call_count == 1
