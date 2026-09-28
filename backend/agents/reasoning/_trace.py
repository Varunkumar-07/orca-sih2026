"""Shared TraceStep-append helper — the identical "append a TraceStep with
the current timestamp" boilerplate was independently repeated across
weather_agent.py, marine_data_agent.py, risk_assessment_agent.py, and
planning_agent.py.
"""
from backend.schemas.contracts import TraceStep
from backend.time_utils import now_iso


def record_trace(trace: list[TraceStep], agent_name: str, input_summary: str, output_summary: str) -> None:
    trace.append(
        TraceStep(
            agent_name=agent_name,
            input_summary=input_summary,
            output_summary=output_summary,
            timestamp=now_iso(),
        )
    )
