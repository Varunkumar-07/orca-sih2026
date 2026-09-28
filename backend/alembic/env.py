import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# alembic/ -> backend/ -> repo root, so `backend.*` imports resolve the same
# way they do under pytest (backend/__init__.py + backend/tests/__init__.py
# make pytest insert the repo root onto sys.path automatically; the plain
# `alembic` CLI needs the same insertion done explicitly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Anchored to backend/.env regardless of cwd — same fix as backend/main.py's
# own load_dotenv() call, for the same reason: a bare load_dotenv() only
# searches upward from cwd, so running `alembic` from the project root (the
# documented way to run everything else in this repo) would otherwise never
# find backend/.env at all.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from backend.history import (
    models,  # noqa: F401 - registers HistoryRecord on Base.metadata
)
from backend.history.db import Base, normalize_asyncpg_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# DATABASE_URL from the environment takes precedence over alembic.ini's
# sqlalchemy.url — same env-var-driven convention as the rest of the app
# (see backend/history/db.py). Same asyncpg-driver normalization too, since
# this also builds an async engine (run_migrations_online below) from
# whatever scheme the URL came in as.
database_url = normalize_asyncpg_url(os.getenv("DATABASE_URL", "").strip())
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
