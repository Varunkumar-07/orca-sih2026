"""Assistant Demo mode (POST /query/demo) answers — the default view.

Demo mode answers from hand-authored scenario fixtures picked by keyword;
these check the answer matches the question actually asked: its text is
echoed (not the fixture's own "...near Chennai?"), a named place moves the
scenario there with that area's sample fishing zones, and tide reads as
text rather than a raw dict.

Run from project root:  PYTHONPATH=. pytest backend/tests/test_demo_mode.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client(offline_pfz_data):
    with TestClient(app) as c:
        yield c


def _ask(client, query: str) -> dict:
    resp = client.post("/query/demo", json={"query": query})
    assert resp.status_code == 200
    return resp.json()


def test_storm_example_echoes_the_question_and_its_place(client):
    body = _ask(client, "is it safe near Kochi with cyclone alert?")
    lines = body["answer_text"].split("\n")

    assert lines[0] == "Query: is it safe near Kochi with cyclone alert?"
    assert lines[1] == "Location: 9.9312, 76.2673"
    assert lines[2].startswith("⚠️ Not safe to go")
    assert "Tide: high tide 14:32, low tide 08:10" in body["answer_text"]
    assert "{" not in body["answer_text"]
    pfz_pins = [p["label"] for p in body["map_payload"]["pins"] if p["type"] == "pfz"]
    assert pfz_pins and all(label.startswith("KOCHI-PFZ") for label in pfz_pins)


def test_restricted_example_shows_that_areas_zones_not_chennais(client):
    body = _ask(client, "can I fish near Gulf of Mannar?")

    assert body["answer_text"].split("\n")[2].startswith("⛔ PROHIBITED")
    assert "PFZ-CH-14" not in body["answer_text"]
    assert "GULF-OF-MANNAR-PFZ" in body["answer_text"]


def test_question_without_a_place_keeps_the_sample_location(client):
    body = _ask(client, "is it safe to go fishing today?")
    lines = body["answer_text"].split("\n")

    assert lines[0] == "Query: is it safe to go fishing today?"
    assert lines[1] == "Location: 13.0800, 80.2700"
