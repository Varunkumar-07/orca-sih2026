"""
User Interaction Agent — Team Claude, Tier B (deterministic, no LLM call).

Multi-turn session memory: stores conversation history per session_id so the
Planning Agent can resolve ambiguous follow-up queries ("what about
Friday?") against a previously extracted location.

In-memory dict for the hackathon demo. Swapping to Redis later only means
replacing the internals of SessionStore (the dict operations below) with
hset/hget calls — the public interface (create_session, get_session,
update_session) stays the same, so callers never change.
"""

import uuid
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from backend.schemas.contracts import EvidenceBundle, GeoPoint, RiskAssessment
from backend.time_utils import now_iso as _now_iso

_SESSION_TTL_SECONDS = 1800  # 30 minutes — long enough for a live demo, short enough not to leak memory


def _parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class SessionTurn(BaseModel):
    query_text: str
    timestamp: str
    query_location: GeoPoint | None
    risk: RiskAssessment | None
    # Phase 6.2: the query's detected language (Language Agent, input side),
    # e.g. "hi", "ta", or "en". None for any turn recorded before this field
    # existed — last_known_language() below already skips those correctly.
    detected_language: str | None = None


class SessionState(BaseModel):
    session_id: str
    created_at: str
    last_accessed: str
    history: list[SessionTurn] = []

    def last_known_location(self) -> GeoPoint | None:
        for turn in reversed(self.history):
            if turn.query_location is not None:
                return turn.query_location
        return None

    def last_risk(self) -> RiskAssessment | None:
        for turn in reversed(self.history):
            if turn.risk is not None:
                return turn.risk
        return None

    def last_known_language(self) -> str | None:
        """Most recently detected language across the session (mirrors
        last_known_location()'s pattern exactly) — lets a follow-up query
        with no script of its own (e.g. a bare number, or text too short for
        the script heuristic to see anything non-ASCII, which always
        detects as "en") still get its reply translated back into whatever
        language the conversation has actually been in. Whether to prefer
        this over the current turn's own fresh detection is the caller's
        call (main.py) — this just reports what was last recorded,
        including "en"."""
        for turn in reversed(self.history):
            if turn.detected_language is not None:
                return turn.detected_language
        return None


class SessionStore:
    """In-memory session store, keyed by session_id.

    The dict below is the only thing a Redis-backed version would need to
    replace (get/set/delete -> hget/hset/expire); the three public methods
    stay identical.
    """

    def __init__(self, ttl_seconds: int = _SESSION_TTL_SECONDS):
        self._sessions: dict[str, SessionState] = {}
        self._ttl_seconds = ttl_seconds

    def _is_expired(self, session: SessionState) -> bool:
        age = datetime.now(timezone.utc) - _parse_iso(session.last_accessed)
        return age > timedelta(seconds=self._ttl_seconds)

    def _cleanup_expired(self) -> None:
        expired_ids = [sid for sid, s in self._sessions.items() if self._is_expired(s)]
        for sid in expired_ids:
            del self._sessions[sid]

    def create_session(self) -> str:
        self._cleanup_expired()
        session_id = str(uuid.uuid4())
        now = _now_iso()
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            history=[],
        )
        return session_id

    def get_session(self, session_id: str) -> SessionState | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._is_expired(session):
            del self._sessions[session_id]
            return None
        session.last_accessed = _now_iso()
        return session

    def update_session(
        self,
        session_id: str,
        query: str,
        bundle: EvidenceBundle,
        detected_language: str | None = None,
    ) -> SessionTurn:
        session = self._sessions.get(session_id)
        if session is None or self._is_expired(session):
            # Defensive: caller should have created the session first, but
            # never fail a request over missing session bookkeeping.
            now = _now_iso()
            session = SessionState(session_id=session_id, created_at=now, last_accessed=now, history=[])
            self._sessions[session_id] = session

        session.last_accessed = _now_iso()
        turn = SessionTurn(
            query_text=query,
            timestamp=_now_iso(),
            query_location=bundle.query_location,
            risk=bundle.risk,
            detected_language=detected_language,
        )
        session.history.append(turn)
        return turn


# Module-level singleton, shared across requests within the process — this
# is what makes session memory persist between turns of the same conversation.
_store = SessionStore()


def create_session() -> str:
    return _store.create_session()


def get_session(session_id: str) -> SessionState | None:
    return _store.get_session(session_id)


def update_session(
    session_id: str,
    query: str,
    bundle: EvidenceBundle,
    detected_language: str | None = None,
) -> SessionTurn:
    return _store.update_session(session_id, query, bundle, detected_language)


def set_last_turn_language(turn: SessionTurn, detected_language: str) -> None:
    """Patch detected_language directly onto the SessionTurn object
    update_session just returned. Needed because run_planning_agent already
    calls update_session internally (without a language — it has no
    knowledge of the Language Agent), so by the time main.py knows the
    detected language, the turn already exists; this avoids either touching
    planning_agent.py's internals or appending a duplicate turn.

    Takes the turn object itself, not a session_id + "patch whatever is
    last" — mutating the exact SessionTurn instance already appended to
    session.history (Python holds it by reference, so this mutation is
    visible there too) is intrinsically race-free. A session_id-based
    "last turn" lookup would instead patch whichever turn happens to be
    positionally last *at the time this runs*, which is not necessarily
    the turn this call is actually for if a second request sharing the
    same session_id was handled concurrently in between."""
    turn.detected_language = detected_language
