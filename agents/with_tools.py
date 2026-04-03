from __future__ import annotations
import json
import anthropic
from anthropic.types import ToolParam, MessageParam
from .base import Agent
from tasks.base import Task
from benchmark.types import Sample


class WithToolsAgent(Agent):
    def __init__(self, model: str = "claude-opus-4-6", aws_profile: str | None = None):
        super().__init__(model)
        self._client = (
            anthropic.AnthropicBedrock(aws_profile=aws_profile)
            if aws_profile
            else anthropic.Anthropic()
        )

    def run(self, task: Task, sample: Sample, rule_count: int) -> str:
        system = task.tools_system_prompt(rule_count)
        user = task.format_input(sample.input)
        tools: list[ToolParam] = task.tool_definitions  # type: ignore[assignment]

        messages: list[MessageParam] = [{"role": "user", "content": user}]

        # Agentic loop: keep running until no more tool calls
        while True:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                # Extract final text answer
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text.strip()
                return ""

            if response.stop_reason == "tool_use":
                # Execute all tool calls and feed results back
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = task.run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result) if not isinstance(result, str) else result,
                        })
                messages.append({"role": "user", "content": tool_results})  # type: ignore[typeddict-item]
            else:
                break

        return ""
