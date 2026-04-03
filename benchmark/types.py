from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    id: str
    input: dict[str, Any]       # raw input passed to the agent
    ground_truth: Any           # expected output
    rules_needed: int           # minimum rule level required to solve correctly
    is_edge_case: bool = False  # harder samples held out for edge-case accuracy


@dataclass
class TrialResult:
    sample_id: str
    task_name: str
    agent_config: str           # "instructions_only" | "with_tools"
    rule_count: int
    trial_num: int
    predicted: Any
    correct: bool
    is_edge_case: bool
    latency_ms: float = 0.0


@dataclass
class BenchmarkMetrics:
    task_name: str
    agent_config: str
    rule_count: int
    accuracy: float             # % correct across all samples
    edge_case_accuracy: float   # % correct on edge cases only
    consistency: float          # % of samples where all K trials agree
    n_samples: int
    n_trials: int


@dataclass
class BenchmarkConfig:
    task_names: list[str]
    agent_configs: list[str]    # which configs to run
    rule_counts: list[int]      # e.g. [1, 2, 3] — progressive rule addition
    n_samples: int = 50         # samples per (task, rule_count) cell
    n_trials: int = 3           # trials per sample (for consistency measurement)
    model: str = "claude-opus-4-6"
