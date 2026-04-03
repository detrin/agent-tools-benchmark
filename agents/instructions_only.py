from __future__ import annotations
import anthropic
from anthropic.types import TextBlock
from .base import Agent
from tasks.base import Task
from benchmark.types import Sample


class InstructionsOnlyAgent(Agent):
    def __init__(self, model: str = "claude-opus-4-6", aws_profile: str | None = None):
        super().__init__(model)
        self._client = (
            anthropic.AnthropicBedrock(aws_profile=aws_profile)
            if aws_profile
            else anthropic.Anthropic()
        )

    def run(self, task: Task, sample: Sample, rule_count: int) -> str:
        system = task.instructions_system_prompt(rule_count)
        user = task.format_input(sample.input)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        block = response.content[0]
        return block.text.strip() if isinstance(block, TextBlock) else ""
