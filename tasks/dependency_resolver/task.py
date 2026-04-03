from __future__ import annotations
from typing import Any
from tasks.base import Task
from benchmark.types import Sample


class DependencyResolverTask(Task):
    """
    Task: given a directed dependency graph, produce a valid topological order
    and detect any cycles.

    Rules of increasing complexity:
      1. Produce any valid topological ordering (no cycle)
      2. Detect cycles and report them
      3. Identify which nodes can be processed in parallel (same depth level)
      4. Handle transitive dependencies (A→B→C means A must come before C)

    Deterministic sub-tasks (DFS, cycle detection, level assignment) → tools.
    Fuzzy sub-task (explain the ordering to a human) → LLM.
    """

    name = "dependency_resolver"

    rules = [
        "Topological ordering: produce a valid execution order where every dependency "
        "of a node appears before that node in the list.",

        "Cycle detection: if the graph contains a cycle (A→B→C→A), report "
        "CYCLE DETECTED: <nodes in cycle> instead of an ordering.",

        "Parallel groups: group nodes that have no dependency between them at the same "
        "depth level — these can execute simultaneously.",

        "Transitive closure: if A depends on B and B depends on C, then A implicitly "
        "depends on C. All transitive dependencies must be resolved before A.",
    ]

    @property
    def tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "topological_sort",
                "description": "Sort a DAG topologically. Returns ordered list or cycle error.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                },
                            },
                            "description": "List of directed edges (from depends on to)",
                        },
                    },
                    "required": ["edges"],
                },
            },
        ]

    def generate_samples(self, n: int, rule_count: int) -> list[Sample]:
        # TODO: implement synthetic graph scenarios
        raise NotImplementedError("dependency_resolver samples not yet implemented")

    def evaluate(self, sample: Sample, predicted: Any) -> bool:
        raise NotImplementedError

    def run_tool(self, tool_name: str, tool_input: dict) -> str:
        raise NotImplementedError

    def format_input(self, inp: dict) -> str:
        raise NotImplementedError
