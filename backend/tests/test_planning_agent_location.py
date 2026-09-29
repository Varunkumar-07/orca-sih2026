"""Pytest coverage for backend/agents/reasoning/planning_agent.py's location
resolution — fuzzy-match fallback and live geocoding fallback (both added
this session — previously untested; the only prior signal on this logic
was ad-hoc manual verification during the session itself).

Mocking strategy: httpx.AsyncClient.get patched at the class level for
_geocode_place, same approach as the other new-this-session test files.
_resolve_known_location itself makes no network calls (exact match, then a
local difflib fuzzy match) so it's tested directly with no mocking at all.

Covers:
  - _resolve_known_location: exact match, explicit lat/lon in the query
    text (wins over name lookup), no place named at all (-> None, None,
    meaning "safe to fall back to session context"), a close misspelling
    resolving via fuzzy match, and a genuinely unrecognized place still
    failing closed (never silently guessing)
  - _geocode_place: a real result, no results, restricted to India (the
    countrycodes=in param actually gets sent — this is the specific thing
    that prevented "Vizog" from silently resolving to a hamlet in France,
    confirmed live during this session), and network/HTTP failure -> None

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

import httpx

from backend.agents.reasoning import planning_agent as pa


class _FakeResponse:
    status_code = 200

    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


# ---------------------------------------------------------------------------
# _resolve_known_location
# ---------------------------------------------------------------------------


def test_resolve_known_location_exact_match():
    loc, unrecognized = pa._resolve_known_location("is it safe near chennai?", "chennai")
    assert loc is not None
    assert loc.lat == pa._KNOWN_LOCATIONS["chennai"].lat
    assert unrecognized is None


def test_resolve_known_location_explicit_latlon_wins_over_name():
    loc, unrecognized = pa._resolve_known_location("is it safe at 9.05, 79.15?", "chennai")
    assert loc is not None
    assert loc.lat == 9.05
    assert loc.lon == 79.15
    assert unrecognized is None


def test_resolve_known_location_no_place_named_is_safe_followup():
    """(None, None) specifically means "no location mentioned at all" — the
    caller is allowed to fall back to session context for this case only,
    never for an unrecognized-but-named place."""
    loc, unrecognized = pa._resolve_known_location("what about tomorrow?", None)
    assert loc is None
    assert unrecognized is None


def test_resolve_known_location_fuzzy_match_typo():
    """The exact bug this session fixed: "vizog" (typo) must resolve to
    the same point as "vizag"."""
    loc, unrecognized = pa._resolve_known_location("fishing near vizog", "vizog")
    assert unrecognized is None
    assert loc is not None
    assert loc.lat == pa._KNOWN_LOCATIONS["vizag"].lat
    assert loc.lon == pa._KNOWN_LOCATIONS["vizag"].lon


def test_resolve_known_location_gulf_of_mannard_resolves_via_fuzzy_match():
    """"Gulf of Mannard" (the classifier's consistent mis-transcription of
    "Mannar") used to be a hardcoded dict alias — now relies entirely on
    the general fuzzy-match fallback, same as any other close misspelling.
    Locks in that removing the alias didn't regress this specific,
    previously-flagship-broken query."""
    loc, unrecognized = pa._resolve_known_location("fishing near gulf of mannard", "gulf of mannard")
    assert unrecognized is None
    assert loc is not None
    assert loc.lat == pa._KNOWN_LOCATIONS["gulf of mannar"].lat
    assert loc.lon == pa._KNOWN_LOCATIONS["gulf of mannar"].lon


def test_resolve_known_location_genuinely_unknown_place_fails_closed():
    """A fictional/unrelated place must never fuzzy-match onto some
    unrelated real city just because the edit distance happens to be
    small — this must come back unresolved, not a wrong guess."""
    loc, unrecognized = pa._resolve_known_location("fishing near narnia", "narnia")
    assert loc is None
    assert unrecognized == "narnia"


# ---------------------------------------------------------------------------
# _scan_known_locations_in_text
# ---------------------------------------------------------------------------


def test_scan_known_locations_finds_exact_city_name():
    """The exact bug this fixes: the classifier returns place_name=None for
    a query that plainly names a known city — this backstop must still
    find it from the raw text."""
    assert pa._scan_known_locations_in_text("is it safe to fish near Vizag tomorrow?") == "vizag"


def test_scan_known_locations_finds_multiword_phrase():
    assert pa._scan_known_locations_in_text("any restrictions near Gulf of Mannar?") == "gulf of mannar"


def test_scan_known_locations_ignores_substring_inside_unrelated_word():
    """"goa" must not fire on "goal" — whole-word matching only."""
    assert pa._scan_known_locations_in_text("what's our goal for today's catch?") is None


def test_scan_known_locations_returns_none_for_genuine_followup():
    assert pa._scan_known_locations_in_text("what about tomorrow?") is None


# ---------------------------------------------------------------------------
# _geocode_place
# ---------------------------------------------------------------------------


def test_geocode_place_returns_a_result(monkeypatch):
    async def fake_get(self, url, params=None, headers=None):
        assert url == pa._NOMINATIM_URL
        assert params["countrycodes"] == "in"
        return _FakeResponse([{"lat": "8.1752656", "lon": "77.2519232"}])

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    result = asyncio.run(pa._geocode_place("colachel"))
    assert result is not None
    assert result.lat == 8.1752656
    assert result.lon == 77.2519232


def test_geocode_place_no_results_returns_none(monkeypatch):
    async def fake_get(self, url, params=None, headers=None):
        return _FakeResponse([])

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    result = asyncio.run(pa._geocode_place("asdkjfhaskjdfh"))
    assert result is None


def test_geocode_place_is_restricted_to_india(monkeypatch):
    """The specific safety fix confirmed live this session: an
    unrestricted Nominatim search for "Vizog" resolves to a hamlet in
    France. countrycodes=in must always be sent."""
    seen_params = {}

    async def fake_get(self, url, params=None, headers=None):
        seen_params.update(params)
        return _FakeResponse([{"lat": "17.6868", "lon": "83.2185"}])

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    asyncio.run(pa._geocode_place("vizog"))
    assert seen_params["countrycodes"] == "in"


def test_geocode_place_network_failure_returns_none(monkeypatch):
    async def fake_get(self, url, params=None, headers=None):
        raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    result = asyncio.run(pa._geocode_place("colachel"))
    assert result is None
