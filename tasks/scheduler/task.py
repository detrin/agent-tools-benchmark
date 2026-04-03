from __future__ import annotations
from typing import Any
from tasks.base import Task
from benchmark.types import Sample


class SchedulerTask(Task):
    """
    Task: find a valid meeting slot given attendee availability and constraints.

    Rules of increasing complexity:
      1. Respect each attendee's available hours (9am–6pm local time)
      2. Apply timezone offsets correctly (find UTC overlap)
      3. Respect blocked slots
      4. No back-to-back meetings (15-minute buffer between meetings)

    Deterministic sub-tasks (timezone arithmetic, overlap calculation) → tools.
    Fuzzy sub-task (choose best slot from valid candidates) → LLM judgment.
    """

    name = "scheduler"

    rules = [
        "Working hours: each attendee is available 9:00–18:00 in their local timezone. "
        "Only schedule meetings within this window for all attendees.",

        "Timezone conversion: convert all local availability to UTC before computing overlap. "
        "UTC offset examples: UTC+1 → subtract 1h to get UTC, UTC-5 → add 5h.",

        "Blocked slots: each attendee may have blocked periods. "
        "A valid slot must not overlap any attendee's blocked period.",

        "Buffer requirement: leave at least 15 minutes between consecutive meetings. "
        "A slot starting at 14:00 is invalid if the attendee has a meeting ending at 13:50.",
    ]

    @property
    def tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "convert_to_utc",
                "description": "Convert a local time to UTC given a UTC offset.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "local_time": {"type": "string", "description": "HH:MM"},
                        "utc_offset": {"type": "integer", "description": "e.g. +1, -5"},
                    },
                    "required": ["local_time", "utc_offset"],
                },
            },
            {
                "name": "find_overlap",
                "description": "Find the intersection of multiple availability windows (all in UTC).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "windows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string"},
                                    "end": {"type": "string"},
                                },
                            },
                        },
                        "duration_minutes": {"type": "integer"},
                    },
                    "required": ["windows", "duration_minutes"],
                },
            },
        ]

    def generate_samples(self, n: int, rule_count: int) -> list[Sample]:
        # TODO: implement synthetic scheduling scenarios
        raise NotImplementedError("scheduler samples not yet implemented")

    def evaluate(self, sample: Sample, predicted: Any) -> bool:
        raise NotImplementedError

    def run_tool(self, tool_name: str, tool_input: dict) -> str:
        raise NotImplementedError

    def format_input(self, inp: dict) -> str:
        raise NotImplementedError
