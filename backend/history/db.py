"""Async SQLAlchemy engine/session setup for the history persistence layer.

Reads DATABASE_URL the same way the rest of the app reads its other env
vars (via python-dotenv's load_dotenv() in main.py, already called before
this module is imported). If DATABASE_URL is unset or the engine fails to
initialize, history logging degrades to a no-op — matching the rest of
this codebase's "degrade, never fail the response" convention — rather
than breaking any of the 8 pages or the existing test suite.
"""
import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

logger = logging.getLogger("orca.history")

Base = declarative_base()


def normalize_asyncpg_url(url: str) -> str:
    """Render's managed-Postgres connection string (and most providers'
    copy-paste connection strings) come as plain postgres:// / postgresql://
    — psycopg2-style, no driver. SQLAlchemy's async engine needs the
    asyncpg driver named explicitly or it fails to load a dialect at all.
    Idempotent: a URL that already names a driver (already +asyncpg, or
    deliberately some other one) passes through unchanged."""
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


DATABASE_URL = normalize_asyncpg_url(os.getenv("DATABASE_URL", "").strip())

_engine = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None

if DATABASE_URL:
    try:
        _engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
        _SessionLocal = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    except Exception as exc:  # pragma: no cover - defensive, mirrors service-layer degradation
        logger.warning("History DB engine init failed; history logging disabled: %s", exc)
        _engine = None
        _SessionLocal = None
else:
    logger.info("DATABASE_URL not set; history logging is disabled.")


def is_enabled() -> bool:
    return _SessionLocal is not None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    # A bare assert here is stripped when Python runs with -O, which would
    # turn this into a confusing AttributeError at every history call site
    # instead of a clear error — raise explicitly.
    if _SessionLocal is None:
        raise RuntimeError("history DB is not configured (DATABASE_URL unset)")
    return _SessionLocal


async def dispose_engine() -> None:
    """Close the engine's connection pool — called from main.py's lifespan
    shutdown. No-op when history logging was never enabled."""
    if _engine is not None:
        await _engine.dispose()
