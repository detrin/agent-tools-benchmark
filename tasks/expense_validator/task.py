from __future__ import annotations
from typing import Any
from tasks.base import Task
from benchmark.types import Sample
from .generator import generate
from .tools import check_limit, check_receipt_required, compute_meal_limit


class ExpenseValidatorTask(Task):
    """
    Task: validate a single expense against a travel policy.
    Output: "APPROVED" or "REJECTED: <reason>".

    Four rules of increasing complexity:
      1. Receipt required above $25
      2. Per-category spending limits (domestic)
      3. International multiplier (1.5×)
      4. Partial-day meal proration (hours/24 × limit)

    The deterministic sub-tasks (arithmetic, threshold checks) are what tools encode.
    The LLM's job is only to reason about which checks apply and combine their results.
    """

    name = "expense_validator"

    rules = [
        # Rule 1
        "Receipt requirement: a receipt is required for any expense over $25. "
        "If the expense exceeds $25 and has_receipt is false, reject it.",

        # Rule 2
        "Per-category limits (domestic): meals ≤ $75, transport ≤ $50, "
        "accommodation ≤ $200, other ≤ $30. Reject if the amount exceeds the limit.",

        # Rule 3
        "International multiplier: when is_international is true, multiply each "
        "category limit by 1.5. An international meal is approved up to $112.50.",

        # Rule 4
        "Partial-day meal proration: when the 'hours' field is set for a meal expense, "
        "the limit is prorated as: limit × (hours / 24). "
        "A 6-hour international meal limit = $112.50 × (6/24) = $28.13.",
    ]

    @property
    def tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "check_limit",
                "description": "Check if an expense amount is within the policy limit for its category.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "amount": {"type": "number"},
                        "is_international": {"type": "boolean"},
                    },
                    "required": ["category", "amount", "is_international"],
                },
            },
            {
                "name": "check_receipt_required",
                "description": "Check whether a receipt is required for this expense amount.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                    },
                    "required": ["amount"],
                },
            },
            {
                "name": "compute_meal_limit",
                "description": "Compute the prorated meal limit for a partial travel day.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hours": {"type": "number", "description": "Hours of travel"},
                        "is_international": {"type": "boolean"},
                    },
                    "required": ["hours", "is_international"],
                },
            },
        ]

    def generate_samples(self, n: int, rule_count: int) -> list[Sample]:
        return generate(n=n, rule_count=rule_count)

    def evaluate(self, sample: Sample, predicted: Any) -> bool:
        pred = str(predicted).strip().upper()
        truth = str(sample.ground_truth).strip().upper()
        # Match on APPROVED / REJECTED prefix (not exact reason text)
        return pred.startswith("APPROVED") == truth.startswith("APPROVED")

    def run_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "check_limit":
            result = check_limit(**tool_input)
            return f"{'APPROVED' if result.approved else 'REJECTED'}: {result.reason}"
        if tool_name == "check_receipt_required":
            result = check_receipt_required(**tool_input)
            return f"{'OK' if result.approved else 'REQUIRED'}: {result.reason}"
        if tool_name == "compute_meal_limit":
            limit = compute_meal_limit(**tool_input)
            return f"Prorated meal limit: ${limit:.2f}"
        raise ValueError(f"Unknown tool: {tool_name}")

    def format_input(self, inp: dict) -> str:
        lines = [f"Expense:"]
        for k, v in inp.items():
            if v is not None:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def instructions_system_prompt(self, rule_count: int) -> str:
        rules_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(self.rules[:rule_count]))
        return (
            "You are a travel expense policy checker.\n\n"
            f"Apply these rules to validate the expense:\n{rules_text}\n\n"
            "Return APPROVED or REJECTED: <reason>. No other text."
        )

    def tools_system_prompt(self, rule_count: int) -> str:
        return (
            "You are a travel expense policy checker. "
            "Use the provided tools to validate the expense. "
            "Return APPROVED or REJECTED: <reason>. No other text."
        )
