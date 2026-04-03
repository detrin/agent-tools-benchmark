from __future__ import annotations
from typing import Any
from tasks.base import Task
from benchmark.types import Sample


class ContractCheckerTask(Task):
    """
    Task: given a contract clause and a compliance policy, flag whether the clause
    violates the policy.

    This task is the inverse of the others — it demonstrates where tools LOSE to
    instructions. Exact string matching misses semantically equivalent clauses;
    only the LLM can judge semantic equivalence. The tool only handles the
    mechanical exact-match check.

    Rules of increasing complexity:
      1. Exact keyword match: clause must contain required phrases verbatim
      2. Synonym equivalence: certain synonyms count as equivalent
      3. Exception handling: "unless <condition>" clauses can override violations
      4. Cross-reference: a clause in section A may satisfy a requirement in section B
    """

    name = "contract_checker"

    rules = [
        "Required language: the clause must contain the exact phrase "
        "'liability is limited to direct damages'. Flag any clause missing this.",

        "Synonym equivalence: 'liability is capped at actual losses' and "
        "'responsibility is restricted to direct harm' are acceptable equivalents.",

        "Exception clause: a clause that includes 'unless gross negligence is proven' "
        "may exceed the standard limit without being flagged.",

        "Cross-reference: if the contract includes a separate definitions section that "
        "defines 'losses' as 'direct damages', then 'liability limited to losses' satisfies Rule 1.",
    ]

    @property
    def tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "exact_phrase_check",
                "description": "Check whether a clause contains a required phrase exactly.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "clause": {"type": "string"},
                        "required_phrase": {"type": "string"},
                    },
                    "required": ["clause", "required_phrase"],
                },
            },
        ]

    def generate_samples(self, n: int, rule_count: int) -> list[Sample]:
        # TODO: implement synthetic contract scenarios
        raise NotImplementedError("contract_checker samples not yet implemented")

    def evaluate(self, sample: Sample, predicted: Any) -> bool:
        raise NotImplementedError

    def run_tool(self, tool_name: str, tool_input: dict) -> str:
        raise NotImplementedError

    def format_input(self, inp: dict) -> str:
        raise NotImplementedError
