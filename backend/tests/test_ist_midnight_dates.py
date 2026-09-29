"""Date ranges between 00:00 and 05:30 IST — when India is already on the
next calendar day but UTC is not.

The pages used to build their default range from Date#toISOString() (UTC),
so in that window "today" was still yesterday and the night's own rows
were hidden. The pages now send the user's local date; these tests pin the
backend half of that contract:

  - /history reads start_date/end_date as calendar days in the caller's
    timezone (tz_offset_minutes, IST by default): a row logged at 00:30 IST
    on 30 Sep is inside "30 Sep", not "29 Sep"
  - /analytics/historical (and so /export) accepts a local "today" that is
    still tomorrow in UTC: the Open-Meteo archive, which rejects any
    end_date past the current UTC day, is asked only up to that day,
    instead of failing every archive variable

Run from project root:  PYTHONPATH=. pytest
"""
from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.history import db as history_db
from backend.history.models import HistoryRecord
from backend.history.service import day_range_bounds
from backend.main import app
from backend.services import analytics_service as svc

_IST = timezone(timedelta(hours=5, minutes=30))
# 00:30 IST on 30 Sep 2026 = 19:00 UTC on 29 Sep 2026.
_JUST_AFTER_MIDNIGHT_IST = datetime(2026, 9, 29, 19, 0, tzinfo=UTC)
# The history row below is stamped on a far-past date no other test logs
# on, so each count is that row's alone.
_ROW_AT = datetime(2020, 3, 14, 19, 0, tzinfo=UTC)  # 00:30 IST on 15 Mar 2020
_ROW_SESSION = "ist-midnight-test"


class TestDayRangeBounds:
    def test_ist_day_covers_its_own_early_hours(self):
        since, until = day_range_bounds(date(2026, 9, 30), date(2026, 9, 30), 330)
        assert since == datetime(2026, 9, 30, 0, 0, tzinfo=_IST)
        assert until == datetime(2026, 10, 1, 0, 0, tzinfo=_IST)
        assert since <= _JUST_AFTER_MIDNIGHT_IST < until

    def test_the_utc_date_the_pages_used_to_send_misses_it(self):
        _since, until = day_range_bounds(date(2026, 9, 29), date(2026, 9, 29), 330)
        assert not _JUST_AFTER_MIDNIGHT_IST < until

    def test_ist_is_the_default(self):
        assert day_range_bounds(date(2026, 9, 30), None) == (datetime(2026, 9, 30, tzinfo=_IST), None)

    def test_other_offsets_are_honoured(self):
        since, until = day_range_bounds(date(2026, 9, 29), date(2026, 9, 29), 0)
        assert (since, until) == (datetime(2026, 9, 29, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC))


@pytest.fixture(scope="module")
def client(offline_pfz_data):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def midnight_row(client):
    """One history row stamped 00:30 IST on 15 Mar 2020."""

    async def insert():
        async with history_db.get_session_factory()() as db:
            db.add(HistoryRecord(timestamp=_ROW_AT, page_source="route", query_summary="t", session_id=_ROW_SESSION))
            await db.commit()

    async def remove():
        async with history_db.get_session_factory()() as db:
            await db.execute(delete(HistoryRecord).where(HistoryRecord.session_id == _ROW_SESSION))
            await db.commit()

    client.portal.call(remove)
    client.portal.call(insert)
    yield
    client.portal.call(remove)


def _count(client, **params) -> int:
    res = client.get("/history", params={"page_source": "route", **params})
    assert res.status_code == 200
    return res.json()["total"]


class TestHistoryEndpoint:
    def test_local_today_shows_the_row(self, client, midnight_row):
        assert _count(client, start_date="2020-03-15", end_date="2020-03-15", tz_offset_minutes=330) == 1

    def test_ist_is_assumed_when_no_offset_is_sent(self, client, midnight_row):
        assert _count(client, start_date="2020-03-15", end_date="2020-03-15") == 1

    def test_default_30_day_range_ending_local_today_includes_it(self, client, midnight_row):
        assert _count(client, start_date="2020-02-14", end_date="2020-03-15", tz_offset_minutes=330) == 1

    def test_row_is_not_on_the_previous_ist_day(self, client, midnight_row):
        # What the page used to send (UTC date) — the bug this fixes.
        assert _count(client, start_date="2020-03-14", end_date="2020-03-14", tz_offset_minutes=330) == 0

    def test_a_utc_browser_sees_it_on_its_own_day(self, client, midnight_row):
        assert _count(client, start_date="2020-03-14", end_date="2020-03-14", tz_offset_minutes=0) == 1

    @pytest.mark.parametrize("offset", [-721, 841])
    def test_out_of_range_offset_is_rejected(self, client, offset):
        assert client.get("/history", params={"tz_offset_minutes": offset}).status_code == 422


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _JUST_AFTER_MIDNIGHT_IST.astimezone(tz)


class TestAnalyticsArchiveEndDate:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        svc._cache.clear()
        monkeypatch.setattr(svc, "datetime", _FrozenDatetime)
        self.calls: list[dict] = []

        async def fake_get_json(url, params, **_k):
            self.calls.append(params)
            return {"daily": {"time": [params["end_date"]], "temperature_2m_mean": [29.0]}}

        monkeypatch.setattr(svc.open_meteo, "get_json", fake_get_json)
        yield
        svc._cache.clear()

    def test_local_today_is_capped_at_the_utc_day(self):
        asyncio.run(svc._fetch_archive_daily(13.08, 80.27, "2026-09-23", "2026-09-30"))
        assert self.calls[0]["start_date"] == "2026-09-23"
        assert self.calls[0]["end_date"] == "2026-09-29"

    def test_a_range_already_inside_utc_today_is_untouched(self):
        asyncio.run(svc._fetch_archive_daily(13.08, 80.27, "2026-09-22", "2026-09-29"))
        assert self.calls[0]["end_date"] == "2026-09-29"

    def test_a_range_entirely_after_utc_today_skips_the_request(self):
        assert asyncio.run(svc._fetch_archive_daily(13.08, 80.27, "2026-09-30", "2026-09-30")) == {}
        assert self.calls == []

    def test_analytics_default_range_after_midnight_ist_has_no_archive_errors(self):
        result = asyncio.run(svc.get_historical_analytics(13.08, 80.27, "2026-09-23", "2026-09-30", ["air_temp_celsius"]))
        assert result["series"]["air_temp_celsius"]["status"] == "ok"
