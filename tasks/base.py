from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from benchmark.types import Sample


class Task(ABC):
    """
    A benchmark task. Subclasses define:
      - rules: ordered list of rule strings (progressively added to the prompt)
      - tool_definitions: Anthropic tool_use dicts for the with_tools config
      - generate_samples(): produce labeled test cases
      - evaluate(): check agent output against ground truth
      - run_tool(): execute a tool call by name+input (used by the with_tools agent)
    """

    name: str
    rules: list[str]  # rules[0] is the simplest, rules[-1] is the most complex/edge-case

    @property
    @abstractmethod
    def tool_definitions(self) -> list[dict]:
        """Anthropic-format tool definitions for the with_tools agent config."""

    @abstractmethod
    def generate_samples(self, n: int, rule_count: int) -> list[Sample]:
        """
        Generate n labeled samples solvable with rules[:rule_count].
        Include a mix of easy cases and edge cases.
        """

    @abstractmethod
    def evaluate(self, sample: Sample, predicted: Any) -> bool:
        """Return True if predicted matches ground truth."""

    @abstractmethod
    def run_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute a tool call and return the result (as a string)."""

    @abstractmethod
    def format_input(self, input: dict[str, Any]) -> str:
        """Format sample input as a user message string."""

    def instructions_system_prompt(self, rule_count: int) -> str:
        """System prompt for instructions-only agent with first rule_count rules."""
        rules_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(self.rules[:rule_count]))
        return (
            f"You are a precise data processing agent.\n\n"
            f"Rules to apply (in order):\n{rules_text}\n\n"
            f"Apply the rules exactly as stated. Return only the final answer, "
            f"no explanation."
        )

    def tools_system_prompt(self, rule_count: int) -> str:
        """System prompt for with_tools agent — rules are encoded in tools."""
        return (
            "You are a precise data processing agent. "
            "Use the provided tools to process the input. "
            "Return only the final answer, no explanation."
        )
