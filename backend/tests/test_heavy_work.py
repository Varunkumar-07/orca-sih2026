"""Pytest coverage for backend/services/heavy_work.py — the process-wide
limit that keeps Copernicus reads and the PFZ refresh from overlapping in
memory (Render free tier, 512MB).

Mocking strategy: plain blocking functions stand in for Copernicus reads,
coordinated with threading.Events; the concurrency high-water mark is
recorded from inside the worker threads themselves.

Covers:
  - reads through copernicus_fetch never run concurrently, even for
    different keys
  - a read whose caller timed out keeps holding the slot until its
    thread really finishes
  - the PFZ refresh block and a request-path read never overlap, and all
    of the refresh's own threads run inside its one slot
  - exclusive(max_threads=N) never has more than N threads running, and
    a thread whose caller timed out keeps its seat (retries can't pile a
    new read on top of a still-running one)
  - run_in_thread's timeout starts when the read starts (queueing for a
    seat doesn't count), and a timed-out read keeps its seat
  - exclusive() is re-entrant (no self-deadlock)
  - a wedged thread is abandoned after _ABANDON_AFTER_SECONDS
  - memory is released (gc + malloc_trim hook) whenever a slot is freed,
    and after every read inside a multi-read slot (the PFZ refresh)
  - heavy reads run on their own thread pool

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from backend.services import copernicus_fetch, heavy_work


class _Concurrency:
    """Records the most threads ever inside track() at once."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def track(self, fn):
        def wrapper(*args):
            with self._lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                return fn(*args)
            finally:
                with self._lock:
                    self.active -= 1

        return wrapper


def test_reads_for_different_keys_never_overlap():
    conc = _Concurrency()
    read = conc.track(lambda x: time.sleep(0.05) or x)

    async def run():
        return await asyncio.gather(*(
            copernicus_fetch.fetch((f"k{i}",), read, (i,), wait_timeout=5, ttl=60, label=f"read {i}")
            for i in range(4)
        ))

    assert asyncio.run(run()) == [0, 1, 2, 3]
    assert conc.peak == 1


def test_timed_out_read_keeps_the_slot_until_its_thread_finishes():
    release_first = threading.Event()
    second_started = threading.Event()

    def stuck_read(_):
        release_first.wait(5)
        return "first"

    def next_read(_):
        second_started.set()
        return "second"

    async def run():
        with pytest.raises(TimeoutError):
            await copernicus_fetch.fetch(("a",), stuck_read, (0,), wait_timeout=0.05, ttl=60, label="stuck")
        second = asyncio.create_task(
            copernicus_fetch.fetch(("b",), next_read, (0,), wait_timeout=5, ttl=60, label="next")
        )
        await asyncio.sleep(0.2)
        # The first caller gave up, but its thread is still reading.
        assert not second_started.is_set()
        release_first.set()
        return await second

    assert asyncio.run(run()) == "second"


def test_pfz_refresh_block_and_request_reads_never_overlap():
    conc = _Concurrency()
    grid_read = conc.track(lambda: time.sleep(0.05))
    point_read = conc.track(lambda _x: time.sleep(0.05) or "chl")

    async def refresh():
        async with heavy_work.exclusive("PFZ zone refresh"):
            # The refresh's own per-anchor concurrency stays inside its one slot.
            await asyncio.gather(*(heavy_work.run_in_thread(grid_read) for _ in range(3)))

    async def run():
        refresh_task = asyncio.create_task(refresh())
        await asyncio.sleep(0)  # let the refresh take the slot first
        chl = await copernicus_fetch.fetch(("chl",), point_read, (1,), wait_timeout=5, ttl=60, label="chl")
        await refresh_task
        return chl

    assert asyncio.run(run()) == "chl"
    assert conc.peak == 3  # the refresh's own 3 grid reads, never 3 + the point read


def test_thread_cap_counts_reads_whose_caller_timed_out():
    conc = _Concurrency()
    release = threading.Event()
    slow_read = conc.track(lambda: release.wait(5))
    fast_read = conc.track(lambda: None)

    async def run():
        async with heavy_work.exclusive("PFZ zone refresh", max_threads=2):
            for _ in range(2):  # both time out, both threads keep running
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(heavy_work.run_in_thread(slow_read), timeout=0.05)
            retry = asyncio.create_task(heavy_work.run_in_thread(fast_read))
            await asyncio.sleep(0.2)
            assert not retry.done()  # waiting for a seat, not running alongside
            release.set()
            await retry

    asyncio.run(run())
    assert conc.peak == 2


def test_timeout_starts_when_the_read_starts_not_while_queued():
    release = threading.Event()

    async def run():
        async with heavy_work.exclusive("PFZ zone refresh", max_threads=1):
            first = asyncio.create_task(heavy_work.run_in_thread(release.wait, 5))
            await asyncio.sleep(0)
            second = asyncio.create_task(heavy_work.run_in_thread(lambda: "ran", timeout=0.2))
            await asyncio.sleep(0.4)  # queued for longer than its own timeout
            release.set()
            await first
            return await second

    assert asyncio.run(run()) == "ran"


def test_timeout_raises_and_the_thread_keeps_its_seat():
    conc = _Concurrency()
    release = threading.Event()
    slow_read = conc.track(lambda: release.wait(5))
    next_read = conc.track(lambda: None)

    async def run():
        async with heavy_work.exclusive("PFZ zone refresh", max_threads=1):
            with pytest.raises(TimeoutError):
                await heavy_work.run_in_thread(slow_read, timeout=0.05)
            queued = asyncio.create_task(heavy_work.run_in_thread(next_read))
            await asyncio.sleep(0.2)
            assert not queued.done()
            release.set()
            await queued

    asyncio.run(run())
    assert conc.peak == 1


def test_exclusive_is_reentrant():
    async def run():
        async with heavy_work.exclusive("outer"), heavy_work.exclusive("inner"):
            return await heavy_work.run_in_thread(lambda: "ok")

    assert asyncio.run(asyncio.wait_for(run(), timeout=2)) == "ok"


def test_wedged_thread_is_abandoned_after_the_limit(monkeypatch):
    monkeypatch.setattr(heavy_work, "_ABANDON_AFTER_SECONDS", 0.1)
    wedged = threading.Event()

    async def run():
        with pytest.raises(TimeoutError):
            async with heavy_work.exclusive("wedged"):
                await asyncio.wait_for(heavy_work.run_in_thread(wedged.wait, 5), timeout=0.05)
        try:
            # Without the abandon limit this would wait the full 5s.
            return await asyncio.wait_for(heavy_work.run_in_thread(lambda: "next"), timeout=2)
        finally:
            wedged.set()

    assert asyncio.run(run()) == "next"


def test_memory_is_released_when_a_slot_frees(monkeypatch):
    trims: list[int] = []
    monkeypatch.setattr(heavy_work, "_malloc_trim", trims.append)

    async def run():
        await heavy_work.run_in_thread(lambda: None)
        await heavy_work.run_in_thread(lambda: None)

    asyncio.run(run())
    assert trims == [0, 0]


def test_multi_read_slot_trims_after_every_read(monkeypatch):
    trims: list[int] = []
    monkeypatch.setattr(heavy_work, "_malloc_trim", trims.append)

    async def run():
        async with heavy_work.exclusive("PFZ zone refresh", max_threads=2):
            for _ in range(3):
                await heavy_work.run_in_thread(lambda: None)

    asyncio.run(run())
    assert trims == [0, 0, 0, 0]  # one per read, plus one when the slot frees


def test_malloc_trim_is_only_loaded_on_linux(monkeypatch):
    monkeypatch.setattr(heavy_work.sys, "platform", "darwin")
    assert heavy_work._load_malloc_trim() is None


def test_heavy_reads_run_on_their_own_pool_not_the_default_executor():
    """Slow Copernicus reads must not queue unrelated to_thread work (the
    forecast models, the chat pipeline) behind them."""
    async def run():
        return await heavy_work.run_in_thread(lambda: threading.current_thread().name)

    assert asyncio.run(run()).startswith("heavy-work")
