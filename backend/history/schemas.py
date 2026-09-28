"""Pydantic response models for the History API (GET /history, GET /history/{id})."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer


class HistoryRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    page_source: str
    query_summary: str
    session_id: str

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        # Every other timestamp in the API is hand-built via _now_iso() as
        # "...Z" (see contracts.py's own documented convention). Pydantic's
        # default datetime encoding would instead emit "...+00:00" here,
        # making history the one inconsistent timestamp format in the API.
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")


class HistoryRecordDetail(HistoryRecordOut):
    payload: dict[str, Any] | None = None


class HistoryListResponse(BaseModel):
    items: list[HistoryRecordOut]
    total: int
    limit: int
    offset: int
