"""
Shared Groq API failure detection — used by both the planning agent and
main.py's demo-reliability fallback to tell raw Groq/HTTP API failures apart
from our own hand-written business-logic error messages.
"""


def is_groq_api_failure(message: str | None) -> bool:
    """True if an error_message looks like a raw Groq/HTTP API failure
    (rate limit, timeout, auth, model error, JSON-mode validation failure)
    rather than one of our own hand-written business-logic messages (e.g.
    "query_location is required...", "Location 'x' isn't in the known-
    location table..."). Every raw Groq SDK exception stringifies with this
    "Error code: NNN - {...}" shape regardless of status code — confirmed
    empirically across 401/404/400/429 failures during development — while
    none of our own controlled messages ever contain it. Used to decide
    whether main.py's demo-reliability fallback should trigger (Tier 2) —
    never to decide what to show the user, which always stays clean
    regardless.
    """
    if not message:
        return False
    return "Error code:" in message
