from __future__ import annotations
import time
from typing import TYPE_CHECKING
from .types import BenchmarkConfig, TrialResult, BenchmarkMetrics
from .metrics import compute_metrics

if TYPE_CHECKING:
    from tasks.base import Task
    from agents.base import Agent


class BenchmarkHarness:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, tasks: list[Task], agents: dict[str, Agent]) -> list[BenchmarkMetrics]:
        results: list[TrialResult] = []

        for task in tasks:
            if task.name not in self.config.task_names:
                continue

            for rule_count in self.config.rule_counts:
                if rule_count > len(task.rules):
                    continue

                samples = task.generate_samples(
                    n=self.config.n_samples,
                    rule_count=rule_count,
                )

                for config_name in self.config.agent_configs:
                    agent = agents[config_name]
                    print(f"  [{task.name}] rules={rule_count} config={config_name} "
                          f"({len(samples)} samples × {self.config.n_trials} trials)")

                    for sample in samples:
                        for trial in range(self.config.n_trials):
                            t0 = time.monotonic()
                            predicted = agent.run(
                                task=task,
                                sample=sample,
                                rule_count=rule_count,
                            )
                            latency_ms = (time.monotonic() - t0) * 1000
                            correct = task.evaluate(sample, predicted)
                            results.append(TrialResult(
                                sample_id=sample.id,
                                task_name=task.name,
                                agent_config=config_name,
                                rule_count=rule_count,
                                trial_num=trial,
                                predicted=predicted,
                                correct=correct,
                                is_edge_case=sample.is_edge_case,
                                latency_ms=latency_ms,
                            ))

        return compute_metrics(results, model=self.config.model)
