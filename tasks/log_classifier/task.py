from __future__ import annotations
from typing import Any
from tasks.base import Task
from benchmark.types import Sample


class LogClassifierTask(Task):
    """
    Task: classify a log line as one of: suppress / investigate / flapping / escalate.

    Rules of increasing complexity:
      1. Known error codes → suppress (expected, no action needed)
      2. Unknown error codes → investigate
      3. Same error >3× within 10 minutes → flapping (rollup, do not create separate tickets)
      4. Priority ordering: flapping takes precedence over investigate;
         suppress takes precedence over both.

    Deterministic sub-tasks (error code lookup, rate threshold check) → tools.
    Fuzzy sub-task (is this error truly novel or a known variant?) → LLM.
    """

    name = "log_classifier"

    rules = [
        "Known errors: if the log line contains one of the known error codes "
        "(CONN_RESET, TIMEOUT_EXPECTED, DEPLOY_RESTART), classify as SUPPRESS.",

        "Unknown errors: if the error code is not in the known list, classify as INVESTIGATE.",

        "Flapping detection: if the same error code appears more than 3 times within "
        "10 minutes in the provided log window, classify as FLAPPING.",

        "Priority: SUPPRESS > FLAPPING > INVESTIGATE. "
        "A known error that is also flapping is still SUPPRESS.",
    ]

    @property
    def tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "check_known_error",
                "description": "Check whether an error code is in the known/expected list.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "error_code": {"type": "string"},
                    },
                    "required": ["error_code"],
                },
            },
            {
                "name": "check_flapping",
                "description": "Check if this error code exceeds the flapping threshold.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "error_code": {"type": "string"},
                        "occurrences_last_10min": {"type": "integer"},
                    },
                    "required": ["error_code", "occurrences_last_10min"],
                },
            },
        ]

    def generate_samples(self, n: int, rule_count: int) -> list[Sample]:
        # TODO: implement synthetic log scenarios
        raise NotImplementedError("log_classifier samples not yet implemented")

    def evaluate(self, sample: Sample, predicted: Any) -> bool:
        raise NotImplementedError

    def run_tool(self, tool_name: str, tool_input: dict) -> str:
        raise NotImplementedError

    def format_input(self, inp: dict) -> str:
        raise NotImplementedError
