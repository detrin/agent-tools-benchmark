from __future__ import annotations
from collections import defaultdict
from .types import TrialResult, BenchmarkMetrics


def compute_metrics(results: list[TrialResult]) -> list[BenchmarkMetrics]:
    """Aggregate trial results into per-(task, config, rule_count) metrics."""
    # Group by (task, config, rule_count)
    groups: dict[tuple, list[TrialResult]] = defaultdict(list)
    for r in results:
        groups[(r.task_name, r.agent_config, r.rule_count)].append(r)

    metrics = []
    for (task, config, rule_count), group in sorted(groups.items()):
        # Accuracy: % correct across all trials
        accuracy = sum(r.correct for r in group) / len(group)

        # Edge case accuracy
        edge = [r for r in group if r.is_edge_case]
        edge_acc = sum(r.correct for r in edge) / len(edge) if edge else float("nan")

        # Consistency: for each sample, check if all K trials produced same prediction
        by_sample: dict[str, list[TrialResult]] = defaultdict(list)
        for r in group:
            by_sample[r.sample_id].append(r)

        consistent = 0
        total_samples = 0
        for sample_trials in by_sample.values():
            if len(sample_trials) > 1:
                total_samples += 1
                predictions = [r.predicted for r in sample_trials]
                if len(set(str(p) for p in predictions)) == 1:
                    consistent += 1

        consistency = consistent / total_samples if total_samples > 0 else 1.0

        n_trials = max(len(v) for v in by_sample.values()) if by_sample else 0

        metrics.append(BenchmarkMetrics(
            task_name=task,
            agent_config=config,
            rule_count=rule_count,
            accuracy=accuracy,
            edge_case_accuracy=edge_acc,
            consistency=consistency,
            n_samples=len(by_sample),
            n_trials=n_trials,
        ))

    return metrics
