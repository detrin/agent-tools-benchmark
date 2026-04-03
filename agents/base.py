from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.base import Task
    from benchmark.types import Sample


class Agent(ABC):
    def __init__(self, model: str = "claude-opus-4-6"):
        self.model = model

    @abstractmethod
    def run(self, task: Task, sample: Sample, rule_count: int) -> Any:
        """Run the agent on a single sample. Returns the agent's answer."""
