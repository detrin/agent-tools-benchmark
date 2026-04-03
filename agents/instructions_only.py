from __future__ import annotations
import anthropic
from .base import Agent
from tasks.base import Task
from benchmark.types import Sample


class InstructionsOnlyAgent(Agent):
    """Agent that receives rules as natural language in the system prompt. No tools."""

    def __init__(self, model: str = "claude-opus-4-6"):
        super().__init__(model)
        self._client = anthropic.Anthropic()

    def run(self, task: Task, sample: Sample, rule_count: int) -> str:
        system = task.instructions_system_prompt(rule_count)
        user = task.format_input(sample.input)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
