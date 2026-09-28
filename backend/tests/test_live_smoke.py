"""Gated live end-to-end smoke test for the /query/full chat pipeline —
added per finding #12 of the shape-mismatch audit.

Why this exists: the zone_id/KeyError bug (get_nearest_anchor_zones()
returning catalog-shaped dicts that marine_data_agent bracket-accessed as
if they were canonical-shaped) shipped with 247 passing tests because
nothing in the automated suite exercises the real, live /query/full path
— every other test mocks at some boundary. This test is deliberately the
one exception: it drives the actual FastAPI app, over HTTP, through the
real planning -> marine -> weather -> risk -> reporting pipeline, with a
real Groq call. That's the only way to catch a bug in how two pieces of
our OWN code agree on a shape, as opposed to a bug in either piece alone.

Gating: skipped automatically unless GROQ_API_KEY is a real key (same
"your-...-here" placeholder check main.py's own /query/full uses to
decide whether to run live or fall back to fixtures) — so a normal
contributor or CI run without secrets configured never sees this test,
let alone gets blocked by it. Run explicitly before demo day with:
    GROQ_API_KEY=... COPERNICUSMARINE_USERNAME=... COPERNICUSMARINE_PASSWORD=... \\
        pytest backend/tests/test_live_smoke.py -v

COPERNICUSMARINE_* credentials are NOT part of the skip gate: without them
get_cached_zones() simply returns zero PFZ zones (no mock/sample fallback
— see marine_data_agent.py's module docstring), which still exercises
get_cached_zones's catalog-shape conversion path itself. Real Copernicus
creds make this a more thorough check (genuinely live front-detected data,
not just the shape contract) but aren't required for the regression itself
to be exercised.
No VCR/cassette-recording dependency exists in this repo yet, so "real or
recorded Copernicus response" per the audit's own finding is satisfied by
"real, when available" rather than adding new recording infrastructure
for a single on-demand test — a deliberate scope call, not an oversight.

Two scenarios, matching the audit's own repro case:
  - is_distant=True: "where can I fish near Puri, Odisha?" — the exact
    query that originally crashed (Puri resolves ~103km from the nearest
    anchor, Paradip — see pfz_service._NEARBY_ANCHOR_THRESHOLD_KM).
  - is_distant=False: "where is the nearest fishing zone near Chennai?"
    — Chennai IS one of the 11 anchors, so the resolved query location
    and the anchor coincide almost exactly.

The core assertion targets the bug's exact signature: run_marine_data_agent
appends `output_summary = f"error: {exc}"` to the trace on any exception,
and the original bug's exc was `KeyError('zone_id')`, which stringifies
to exactly "'zone_id'". A regression reintroducing this class of bug —
even a different missing key — will still show up as marine_data_agent's
trace step starting with "error:", which is what this test actually
checks, rather than grepping for the literal string "zone_id" (that
would only catch the exact same bug, not the same *class* of bug).

Run from project root:  PYTHONPATH=. pytest backend/tests/test_live_smoke.py -v
"""
from __future__ import annotations

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.pfz_service import get_cached_zones


def _groq_key_is_real() -> bool:
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key) and key.strip() not in ("your-groq-key-here", "your_groq_key_here", "") and not key.startswith("your-")


pytestmark = pytest.mark.skipif(
    not _groq_key_is_real(),
    reason="live smoke test needs a real GROQ_API_KEY (not the placeholder) — set it and rerun before demo day",
)


@pytest.fixture(scope="module")
def client():
    """Pre-warms the shared PFZ cache once, up front, with a generous
    timeout — so neither of the two live queries below risks hitting
    marine_data_agent._LIVE_PFZ_LOOKUP_TIMEOUT (10s) on a cold cache and
    getting an empty result *before* the get_nearest_anchor_zones() code
    path (where the bug lived) is ever exercised. Real Copernicus calls
    across up to 11 anchors can
    legitimately take up to ~90s cold — see get_cached_zones's own
    docstring — hence the generous budget here, only for this one-time
    warm-up, not per-request."""
    with TestClient(app) as c:
        asyncio.run(asyncio.wait_for(get_cached_zones(), timeout=90.0))
        yield c


def _marine_trace_step(response_json: dict) -> dict:
    trace = response_json["reasoning_trace"]
    return next(t for t in trace if t["agent_name"] == "marine_data_agent")


def _assert_no_shape_mismatch_crash(marine_step: dict) -> None:
    """The precise signature of the bug class this test exists to catch:
    run_marine_data_agent's except-clause writes output_summary =
    f"error: {exc}" on ANY exception raised while building candidate
    zones — a KeyError('zone_id') is exactly this shape, but so would a
    KeyError on any other field the two sides of this boundary disagree
    on next time. Checking the whole output_summary rather than for the
    literal substring "zone_id" is deliberate: it catches the same class
    of regression, not just an exact repeat of this one."""
    output_summary = marine_step["output_summary"]
    assert not output_summary.startswith("error:"), (
        f"marine_data_agent degraded to status=error — likely the same class of shape-mismatch bug "
        f"as the original zone_id/KeyError incident. output_summary={output_summary!r}"
    )


def test_is_distant_true_scenario_does_not_crash_and_flags_honestly(client):
    """The exact query that originally crashed with KeyError('zone_id')."""
    resp = client.post("/query/full", json={"query": "where can I fish near Puri, Odisha?"})
    assert resp.status_code == 200
    data = resp.json()

    marine_step = _marine_trace_step(data)
    _assert_no_shape_mismatch_crash(marine_step)

    # Puri is genuinely far from every anchor (~103km from the nearest,
    # Paradip) — the is_distant honesty framing must be present.
    assert "regional reference" in data["answer_text"], (
        f"expected the is_distant honesty framing in the answer; got: {data['answer_text']!r}"
    )


def test_is_distant_false_scenario_does_not_crash_and_reads_as_normal(client):
    """Chennai IS one of the 11 anchors — the resolved query location and
    the anchor coincide, so this must NOT trigger the is_distant framing."""
    resp = client.post("/query/full", json={"query": "where is the nearest fishing zone near Chennai?"})
    assert resp.status_code == 200
    data = resp.json()

    marine_step = _marine_trace_step(data)
    _assert_no_shape_mismatch_crash(marine_step)

    assert "regional reference" not in data["answer_text"]
