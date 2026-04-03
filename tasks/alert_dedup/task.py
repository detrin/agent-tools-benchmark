from __future__ import annotations
from typing import Any
from tasks.base import Task
from benchmark.types import Sample
from .generator import generate
from .tools import normalize_source


class AlertDedupTask(Task):
    """
    Task: normalize an alert source string so that the same underlying incident
    maps to the same JIRA search term regardless of environment prefix, build
    number, or CVE-vs-build-name variation.

    Three rules applied in order:
      1. CVE normalization  (error text contains a CVE ID)
      2. Environment prefix stripping  (known env names: prod-eu, prod-us, ...)
      3. Build number suffix stripping  (source ends with #NNN)
    """

    name = "alert_dedup"

    rules = [
        # Rule 1
        "CVE normalization: if error_text contains a CVE ID (format CVE-YYYY-NNNNN) "
        "and a package name matching 'in <package> <version>', return '<package> <CVE-ID>'. "
        "If only a CVE ID is present with no package, return just the CVE ID. "
        "This rule takes priority over all others.",

        # Rule 2
        "Environment prefix stripping: if the source starts with a known environment name "
        "followed by ' - ', strip that prefix. "
        "Known environments: prod-eu, prod-us, prod-ap, staging, dev, fed. "
        "If the prefix is not in this list, leave the source unchanged.",

        # Rule 3
        "Build number stripping: if the source (after any env prefix removal) ends with "
        "' #<number>', strip that suffix. "
        "This applies even when Rule 2 already fired — if stripping the env prefix leaves "
        "a remainder that ends in ' #<number>', strip the number too.",
    ]

    @property
    def tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "normalize_source",
                "description": (
                    "Normalize an alert source string for deduplication. "
                    "Applies CVE normalization, environment prefix stripping, and "
                    "build number stripping in order. Returns the canonical search term."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Raw alert source string",
                        },
                        "error_text": {
                            "type": "string",
                            "description": "Optional error message (used for CVE extraction)",
                        },
                    },
                    "required": ["source"],
                },
            }
        ]

    def generate_samples(self, n: int, rule_count: int) -> list[Sample]:
        return generate(n=n, rule_count=rule_count)

    def evaluate(self, sample: Sample, predicted: Any) -> bool:
        return str(predicted).strip() == str(sample.ground_truth).strip()

    def run_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "normalize_source":
            return normalize_source(
                source=tool_input["source"],
                error_text=tool_input.get("error_text"),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def format_input(self, inp: dict) -> str:
        source = inp["source"]
        error_text = inp.get("error_text")
        if error_text:
            return f"Alert source: {source}\nError text: {error_text}"
        return f"Alert source: {source}"

    def instructions_system_prompt(self, rule_count: int) -> str:
        rules_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(self.rules[:rule_count]))
        return (
            "You are a precise alert normalization agent.\n\n"
            f"Apply these rules in order to normalize the alert source:\n{rules_text}\n\n"
            "Return only the normalized string. No explanation."
        )

    def tools_system_prompt(self, rule_count: int) -> str:
        return (
            "You are a precise alert normalization agent. "
            "Use the normalize_source tool to process the input. "
            "Return only the normalized string. No explanation."
        )
