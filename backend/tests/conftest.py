"""Points the whole test suite at the `orca_test` Postgres database (not
`orca_dev`) for the Phase 8 history layer, before any test module imports
backend.main (and, transitively, backend.history.db).

Must run before that first import: conftest.py is collected by pytest
ahead of every test module, so os.environ.setdefault() here always wins
over main.py's own load_dotenv() call (which defaults to override=False).

If DATABASE_URL is already set in the shell environment (e.g. CI without a
local Postgres), this leaves it untouched — and if no Postgres is reachable
at all, backend/history/db.py's own no-op fallback means the existing 100
tests are unaffected either way.
"""
import os

os.environ.setdefault(
    "DATABASE_URL",
    f"postgresql+asyncpg://{os.getenv('USER', 'postgres')}@localhost:5432/orca_test",
)

# The test suite fires hundreds of requests through one shared TestClient
# (and therefore one shared RateLimitMiddleware instance/bucket, since
# backend.main.app is a module-level singleton reused across every test
# file) well within a minute — RateLimitMiddleware itself is covered by its
# own dedicated unit tests (backend/tests/test_rate_limit.py) against an
# isolated app instance, not through this shared one.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
