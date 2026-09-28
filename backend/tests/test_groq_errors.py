"""Pytest coverage for backend/agents/reasoning/_groq_errors.py (previously
untested — added as part of the codebase audit's P2 coverage gaps).

Pure function, no mocking needed at all.

Covers:
  - is_groq_api_failure: None/empty -> False; a raw Groq SDK error shape
    ("Error code: NNN - {...}") across several real status codes -> True;
    our own hand-written business-logic messages -> False, even ones that
    happen to mention numbers/codes elsewhere in the sentence

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

from backend.agents.reasoning._groq_errors import is_groq_api_failure


def test_none_is_not_a_failure():
    assert is_groq_api_failure(None) is False


def test_empty_string_is_not_a_failure():
    assert is_groq_api_failure("") is False


def test_real_groq_error_shapes_are_detected():
    # Confirmed empirically (per the module's own docstring) across these
    # status codes during development.
    for code, body in [
        (401, "invalid_api_key"),
        (404, "model_not_found"),
        (400, "context_length_exceeded"),
        (429, "rate_limit_exceeded"),
    ]:
        message = f"Error code: {code} - {{'error': {{'message': '{body}'}}}}"
        assert is_groq_api_failure(message) is True, message


def test_our_own_business_logic_messages_are_not_flagged():
    our_messages = [
        "query_location is required for a PFZ lookup",
        "Location 'atlantis' isn't in the known-location table and geocoding found no match",
        "Incomplete data (marine data missing); risk assessment is not fully reliable.",
        "Risk assessment could not be completed.",
    ]
    for message in our_messages:
        assert is_groq_api_failure(message) is False, message


def test_message_mentioning_a_number_without_the_exact_phrase_is_not_flagged():
    """The check is deliberately the literal substring "Error code:", not
    just "does this message contain a number" — a business message that
    happens to mention a number must not be a false positive."""
    assert is_groq_api_failure("distance 404.5 km — no route found") is False
