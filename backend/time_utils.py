"""Shared UTC-ISO-timestamp helper.

Was independently reimplemented in ~11 modules across agents/reasoning,
agents/deterministic, and services (two subtly different forms — strftime
truncating to whole seconds vs isoformat().replace() preserving
microseconds), plus 3 raw inline copies in routers/chat.py. Every API
timestamp elsewhere in the codebase is documented (see
backend/schemas/contracts.py) to use the whole-second, "Z"-suffixed form,
so that's the one canonical implementation here.
"""
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
