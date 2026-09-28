"""Pytest coverage for backend/agents/reasoning/user_interaction_agent.py
(previously untested — added as part of the codebase audit's P2 coverage
gaps).

Mocking strategy: none needed for the core SessionStore logic — it's pure
in-memory bookkeeping. TTL expiry is tested by directly rewriting a
session's last_accessed timestamp into the past (same "force staleness
without a real sleep" approach used for the TTL caches in
test_alerts_service.py / test_pfz_service.py) rather than actually
sleeping.

Isolation: most tests build a fresh SessionStore() instance directly
rather than going through the module-level create_session/get_session/
update_session functions, which share one process-lifetime singleton
(_store) — using a local instance keeps tests independent of each other
and of whatever earlier tests may have already put in the singleton. A
handful of dedicated tests confirm those module-level functions actually
delegate to the singleton correctly.

Covers:
  - create_session / get_session: round-trips a fresh, empty session;
    unknown session_id -> None
  - update_session: appends a turn with the right fields; called against
    an unknown/expired session_id defensively creates one rather than
    raising
  - SessionState.last_known_location / last_risk / last_known_language:
    walk backwards, skip turns where that field is None, most recent
    non-None wins; all-None history -> None
  - TTL expiry: get_session past the TTL returns None AND evicts the
    entry; _cleanup_expired (run inside create_session) removes other
    expired sessions
  - set_last_turn_language: mutates the exact SessionTurn object passed
    in, visible through session.history (by-reference mutation, not a
    session_id + "last turn" re-lookup)
  - module-level singleton wiring: create_session/get_session/
    update_session actually share state through _store

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

from datetime import timedelta

from backend.agents.reasoning import user_interaction_agent as uia
from backend.schemas.contracts import EvidenceBundle, GeoPoint, RiskAssessment

_LOCATION_A = GeoPoint(lat=13.08, lon=80.27)
_LOCATION_B = GeoPoint(lat=9.93, lon=76.27)


def _bundle(location: GeoPoint | None = None, risk: RiskAssessment | None = None) -> EvidenceBundle:
    return EvidenceBundle(query_text="q", query_location=location, marine=None, weather=None, risk=risk)


def _risk(safe: bool) -> RiskAssessment:
    return RiskAssessment(status="ok", safe_to_go=safe, confidence=0.8, explanation="x")


# ---------------------------------------------------------------------------
# SessionStore — create / get / update
# ---------------------------------------------------------------------------


def test_create_session_returns_fresh_empty_session():
    store = uia.SessionStore()
    session_id = store.create_session()

    session = store.get_session(session_id)
    assert session is not None
    assert session.session_id == session_id
    assert session.history == []


def test_get_session_unknown_id_returns_none():
    store = uia.SessionStore()
    assert store.get_session("does-not-exist") is None


def test_update_session_appends_a_turn_with_right_fields():
    store = uia.SessionStore()
    session_id = store.create_session()

    turn = store.update_session(session_id, "is it safe?", _bundle(_LOCATION_A, _risk(True)), detected_language="en")

    assert turn.query_text == "is it safe?"
    assert turn.query_location == _LOCATION_A
    assert turn.risk.safe_to_go is True
    assert turn.detected_language == "en"

    session = store.get_session(session_id)
    assert len(session.history) == 1
    assert session.history[0] is turn


def test_update_session_on_unknown_id_defensively_creates_one():
    """Caller should have created the session first, but a missing/expired
    session_id must never fail the request over bookkeeping."""
    store = uia.SessionStore()

    turn = store.update_session("never-created", "query", _bundle(_LOCATION_A))

    session = store.get_session("never-created")
    assert session is not None
    assert session.history == [turn]


def test_update_session_appends_multiple_turns_in_order():
    store = uia.SessionStore()
    session_id = store.create_session()

    store.update_session(session_id, "first", _bundle(_LOCATION_A))
    store.update_session(session_id, "second", _bundle(_LOCATION_B))

    session = store.get_session(session_id)
    assert [t.query_text for t in session.history] == ["first", "second"]


# ---------------------------------------------------------------------------
# SessionState — last_known_location / last_risk / last_known_language
# ---------------------------------------------------------------------------


def test_last_known_location_returns_most_recent_non_none():
    store = uia.SessionStore()
    session_id = store.create_session()
    store.update_session(session_id, "with location", _bundle(_LOCATION_A))
    store.update_session(session_id, "follow-up, no location", _bundle(None))

    session = store.get_session(session_id)
    assert session.last_known_location() == _LOCATION_A


def test_last_known_location_none_when_every_turn_lacks_one():
    store = uia.SessionStore()
    session_id = store.create_session()
    store.update_session(session_id, "no location", _bundle(None))

    session = store.get_session(session_id)
    assert session.last_known_location() is None


def test_last_risk_returns_most_recent_non_none():
    store = uia.SessionStore()
    session_id = store.create_session()
    store.update_session(session_id, "first", _bundle(_LOCATION_A, _risk(True)))
    store.update_session(session_id, "second, no risk", _bundle(_LOCATION_A, None))

    session = store.get_session(session_id)
    assert session.last_risk().safe_to_go is True


def test_last_known_language_returns_most_recent_non_none():
    store = uia.SessionStore()
    session_id = store.create_session()
    store.update_session(session_id, "hindi turn", _bundle(_LOCATION_A), detected_language="hi")
    store.update_session(session_id, "no language recorded", _bundle(_LOCATION_A), detected_language=None)

    session = store.get_session(session_id)
    assert session.last_known_language() == "hi"


def test_last_known_language_handles_pre_existing_turns_with_no_field():
    """Turns recorded before detected_language existed default to None and
    must be skipped, not crash the walk-backwards lookup."""
    store = uia.SessionStore()
    session_id = store.create_session()
    store.update_session(session_id, "old turn, no language field", _bundle(_LOCATION_A))  # detected_language defaults None

    session = store.get_session(session_id)
    assert session.last_known_language() is None


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


def test_get_session_past_ttl_returns_none_and_evicts():
    from datetime import datetime, timezone

    store = uia.SessionStore(ttl_seconds=1800)
    session_id = store.create_session()

    # Force staleness without a real sleep.
    stale = (datetime.now(timezone.utc) - timedelta(seconds=1801)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store._sessions[session_id].last_accessed = stale

    assert store.get_session(session_id) is None
    assert session_id not in store._sessions  # evicted, not just hidden


def test_get_session_just_under_ttl_is_still_valid():
    from datetime import datetime, timezone

    store = uia.SessionStore(ttl_seconds=1800)
    session_id = store.create_session()
    fresh_ish = (datetime.now(timezone.utc) - timedelta(seconds=1799)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store._sessions[session_id].last_accessed = fresh_ish

    assert store.get_session(session_id) is not None


def test_get_session_refreshes_last_accessed():
    from datetime import datetime, timezone

    store = uia.SessionStore(ttl_seconds=1800)
    session_id = store.create_session()

    backdated = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store._sessions[session_id].last_accessed = backdated

    store.get_session(session_id)

    assert store._sessions[session_id].last_accessed != backdated  # touched again on access


def test_create_session_cleans_up_other_expired_sessions():
    from datetime import datetime, timezone

    store = uia.SessionStore(ttl_seconds=1800)
    expired_id = store.create_session()
    stale = (datetime.now(timezone.utc) - timedelta(seconds=1801)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store._sessions[expired_id].last_accessed = stale

    store.create_session()  # triggers _cleanup_expired internally

    assert expired_id not in store._sessions


# ---------------------------------------------------------------------------
# set_last_turn_language — mutates the exact turn object, not a re-lookup
# ---------------------------------------------------------------------------


def test_set_last_turn_language_mutates_the_exact_turn_object():
    store = uia.SessionStore()
    session_id = store.create_session()
    turn = store.update_session(session_id, "query", _bundle(_LOCATION_A))

    uia.set_last_turn_language(turn, "ta")

    assert turn.detected_language == "ta"
    session = store.get_session(session_id)
    assert session.history[-1].detected_language == "ta"  # visible via the same reference


def test_set_last_turn_language_does_not_touch_other_turns():
    store = uia.SessionStore()
    session_id = store.create_session()
    first_turn = store.update_session(session_id, "first", _bundle(_LOCATION_A), detected_language="en")
    second_turn = store.update_session(session_id, "second", _bundle(_LOCATION_A))

    uia.set_last_turn_language(second_turn, "hi")

    assert first_turn.detected_language == "en"
    assert second_turn.detected_language == "hi"


# ---------------------------------------------------------------------------
# Module-level singleton wiring
# ---------------------------------------------------------------------------


def test_module_level_functions_share_the_same_singleton_store():
    session_id = uia.create_session()
    uia.update_session(session_id, "hello", _bundle(_LOCATION_A))

    session = uia.get_session(session_id)
    assert session is not None
    assert session.history[-1].query_text == "hello"
