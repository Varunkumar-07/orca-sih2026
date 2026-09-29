"""
Process-wide limit on memory-heavy Copernicus work, so Render's free tier
(512MB RAM) never sees several of them at once.

Why this exists: every Copernicus Marine open/read (Weather/chat
chlorophyll, each Analytics series, the PFZ zone refresh) costs ~40-70MB
of transient memory — its own auth session, boto3 S3 clients, Zarr
metadata and decoded chunks. Each caller used to run on its own, so a
normal burst (Weather page + Analytics' three series + a chat turn + the
3-hourly PFZ refresh) put five or more of them in flight together:
measured ~513MB peak, which is how the service kept hitting "exceeded its
memory limit".

What this does:
- exclusive(label, max_threads=N): at most one heavy operation holds the
  slot at a time; everyone else queues (FIFO). The PFZ refresh holds it
  for its whole run, with its own internal per-anchor concurrency capped
  at max_threads threads actually running — including ones whose caller
  already timed out. Without that cap, on Render's 0.1 CPU every grid read
  outlived its 25s timeout and each retry started a fresh read next to
  the still-running one (measured: 447MB after a refresh that returned
  zero zones).
- run_in_thread(fn, *args): asyncio.to_thread for that work, but the slot
  stays held until the thread actually finishes. A caller that stops
  waiting (asyncio.wait_for timeout) can't cancel the thread, and that
  thread keeps allocating — so it must keep counting, or the next read
  would pile on top of it. Called outside exclusive(), it takes a slot of
  its own for just that call.
- When a slot is released, garbage is collected and — on Linux/glibc —
  malloc_trim(0) hands freed heap pages back to the OS, so each operation
  starts from a trimmed process instead of from the last one's high-water
  mark. A no-op elsewhere.

These threads run on their own small pool, not asyncio's default
executor: at 0.1 CPU a queue of slow (or timed-out, still-running)
Copernicus reads otherwise filled every default worker, and unrelated
to_thread work behind them — the forecast models, the chat pipeline —
waited minutes for a thread (measured: /weather/forecast 336s).

A thread still running _ABANDON_AFTER_SECONDS after its caller gave up is
logged and no longer counted — copernicusmarine's own network timeouts
should end it long before that, and a genuinely wedged thread must not
block every Copernicus read for the rest of the process's life.
"""

from __future__ import annotations

import asyncio
import contextvars
import ctypes
import gc
import logging
import sys
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, TypeVar

logger = logging.getLogger("orca.heavy_work")

T = TypeVar("T")

_ABANDON_AFTER_SECONDS = 600.0

# The gate keeps this near 1 (or the PFZ refresh's own thread cap); the
# headroom only matters for abandoned threads, which are rare.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="heavy-work")


def _load_malloc_trim() -> Callable[[int], int] | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):  # non-glibc libc (e.g. musl)
        return None


_malloc_trim = _load_malloc_trim()


def release_memory() -> None:
    """Collect garbage and, on glibc, return freed heap memory to the OS.
    Blocking (a full gc pass) — call from a worker thread."""
    gc.collect()
    if _malloc_trim is not None:
        _malloc_trim(0)


class _Slot:
    """One holder of the gate: the exclusive() block itself plus every
    thread started through run_in_thread() while it was the current slot.
    Released once the block has exited AND all those threads are done.
    Only touched from the event loop thread."""

    def __init__(self, gate: _Gate, label: str, max_threads: int | None):
        self._gate = gate
        self.label = label
        # Thread seats: taken before a thread starts, given back only when
        # it finishes — so a timed-out read keeps its seat.
        self.seats = asyncio.Semaphore(max_threads) if max_threads else None
        self._owner_active = True
        self._pending_threads = 0
        self.released = False
        self._abandon_handle: asyncio.TimerHandle | None = None
        self._acquired_at = time.monotonic()

    def thread_started(self) -> None:
        self._pending_threads += 1

    def thread_finished(self) -> None:
        self._pending_threads -= 1
        if self.seats is not None:
            self.seats.release()
        self._maybe_release()

    def owner_done(self) -> None:
        self._owner_active = False
        if self._pending_threads and not self.released:
            logger.info("%s: caller finished, waiting for %d background read(s) before the next heavy operation", self.label, self._pending_threads)
            self._abandon_handle = self._gate.loop.call_later(_ABANDON_AFTER_SECONDS, self._abandon)
        self._maybe_release()

    def _abandon(self) -> None:
        if not self.released:
            logger.warning(
                "%s: %d background read(s) still running after %.0fs — no longer holding other Copernicus work back for them",
                self.label, self._pending_threads, _ABANDON_AFTER_SECONDS,
            )
            self._release()

    def _maybe_release(self) -> None:
        if not self._owner_active and self._pending_threads <= 0 and not self.released:
            self._release()

    def _release(self) -> None:
        self.released = True
        logger.debug("%s: released the heavy-work slot after %.1fs", self.label, time.monotonic() - self._acquired_at)
        if self._abandon_handle is not None:
            self._abandon_handle.cancel()
        self._gate.release_after_trim()


class _Gate:
    """The per-event-loop semaphore (production has one loop; tests run a
    fresh asyncio.run() loop each, and asyncio primitives can't cross loops)."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.semaphore = asyncio.Semaphore(1)

    def release_after_trim(self) -> None:
        # Trim before letting the next operation in, so it starts from the
        # smallest heap this one can leave behind.
        try:
            trimmed = self.loop.run_in_executor(None, release_memory)
        except RuntimeError:  # loop/executor shutting down
            self.semaphore.release()
            return
        trimmed.add_done_callback(lambda _f: self.semaphore.release())


_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Gate] = weakref.WeakKeyDictionary()
_current_slot: contextvars.ContextVar[_Slot | None] = contextvars.ContextVar("heavy_work_slot", default=None)


def _gate() -> _Gate:
    loop = asyncio.get_running_loop()
    gate = _gates.get(loop)
    if gate is None:
        gate = _gates[loop] = _Gate(loop)
    return gate


@asynccontextmanager
async def exclusive(label: str, max_threads: int | None = None) -> AsyncIterator[None]:
    """Hold the heavy-work slot for this block (queueing behind any other
    holder), allowing at most `max_threads` run_in_thread() calls to be
    running inside it at once (unlimited when None). Re-entrant: inside an
    already-held slot it's a no-op."""
    current = _current_slot.get()
    if current is not None and not current.released:
        yield
        return
    gate = _gate()
    queued_at = time.monotonic()
    await gate.semaphore.acquire()
    logger.debug("%s: acquired the heavy-work slot after %.1fs queued", label, time.monotonic() - queued_at)
    slot = _Slot(gate, label, max_threads)
    token = _current_slot.set(slot)
    try:
        yield
    finally:
        _current_slot.reset(token)
        slot.owner_done()


async def run_in_thread(fn: Callable[..., T], *args: Any) -> T:
    """asyncio.to_thread(fn, *args), but on the heavy-work pool and counted
    against the current heavy-work slot until the thread itself finishes —
    even if this await is cancelled or times out first. Takes its own slot
    when called outside exclusive()."""
    slot = _current_slot.get()
    if slot is None or slot.released:
        async with exclusive(getattr(fn, "__name__", "heavy read")):
            return await run_in_thread(fn, *args)

    loop = asyncio.get_running_loop()
    lock = threading.Lock()
    state = {"started": False, "finished": False}

    def finish() -> None:
        # Exactly once per call, whichever of "thread ran" / "never ran" wins.
        if state["finished"]:
            return
        state["finished"] = True
        try:
            loop.call_soon_threadsafe(slot.thread_finished)
        except RuntimeError:  # loop already closed — nothing left to gate
            pass

    def call() -> T:
        with lock:
            if state["finished"]:  # abandoned before it started
                raise asyncio.CancelledError
            state["started"] = True
        try:
            return fn(*args)
        finally:
            if slot.seats is not None:
                # A multi-read slot (the PFZ refresh) runs up to ~22 reads
                # back to back; trim after each rather than only at the end.
                release_memory()
            with lock:
                finish()

    def on_done(fut: asyncio.Future) -> None:
        # A cancelled waiter whose thread never got going must still give
        # its count back — call() will see "finished" and not run fn.
        if fut.cancelled():
            with lock:
                if not state["started"]:
                    finish()

    if slot.seats is not None:
        await slot.seats.acquire()
    slot.thread_started()
    ctx = contextvars.copy_context()
    future = loop.run_in_executor(_executor, ctx.run, call)
    future.add_done_callback(on_done)
    return await future
