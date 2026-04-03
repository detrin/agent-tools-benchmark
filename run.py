#!/usr/bin/env python3
"""
agent-tools-benchmark: run deduplication experiments across tasks and agent configs.

Usage:
    python run.py                          # all tasks, all configs, rule counts 1-3
    python run.py --tasks alert_dedup      # single task
    python run.py --samples 20 --trials 1  # quick smoke test
    python run.py --model claude-haiku-4-5-20251001  # cheaper model
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from benchmark import BenchmarkConfig, BenchmarkHarness
from benchmark.types import BenchmarkMetrics
from agents import InstructionsOnlyAgent, WithToolsAgent
from tasks import ALL_TASKS


def print_table(metrics: list[BenchmarkMetrics]) -> None:
    try:
        from rich.table import Table
        from rich.console import Console

        table = Table(title="Benchmark Results")
        table.add_column("Task", style="cyan")
        table.add_column("Config", style="magenta")
        table.add_column("Rules")
        table.add_column("Accuracy", justify="right")
        table.add_column("Edge Acc", justify="right")
        table.add_column("Consistency", justify="right")
        table.add_column("N")

        for m in metrics:
            edge = f"{m.edge_case_accuracy:.0%}" if m.edge_case_accuracy == m.edge_case_accuracy else "—"
            table.add_row(
                m.task_name,
                m.agent_config,
                str(m.rule_count),
                f"{m.accuracy:.0%}",
                edge,
                f"{m.consistency:.0%}",
                str(m.n_samples),
            )

        Console().print(table)
    except ImportError:
        # Fallback plain-text table
        header = f"{'Task':<20} {'Config':<20} {'Rules':>5} {'Acc':>6} {'EdgeAcc':>8} {'Cons':>6} {'N':>4}"
        print(header)
        print("-" * len(header))
        for m in metrics:
            edge = f"{m.edge_case_accuracy:.0%}" if m.edge_case_accuracy == m.edge_case_accuracy else "  —"
            print(f"{m.task_name:<20} {m.agent_config:<20} {m.rule_count:>5} "
                  f"{m.accuracy:>6.0%} {edge:>8} {m.consistency:>6.0%} {m.n_samples:>4}")


def save_results(metrics: list[BenchmarkMetrics], path: Path) -> None:
    data = [
        {
            "task_name": m.task_name,
            "agent_config": m.agent_config,
            "rule_count": m.rule_count,
            "accuracy": m.accuracy,
            "edge_case_accuracy": m.edge_case_accuracy,
            "consistency": m.consistency,
            "n_samples": m.n_samples,
            "n_trials": m.n_trials,
        }
        for m in metrics
    ]
    path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="agent-tools-benchmark")
    parser.add_argument("--tasks", nargs="+", help="Task names to run (default: all implemented)")
    parser.add_argument("--configs", nargs="+", default=["instructions_only", "with_tools"],
                        help="Agent configs to run")
    parser.add_argument("--rule-counts", nargs="+", type=int, default=[1, 2, 3],
                        help="Rule counts to sweep")
    parser.add_argument("--samples", type=int, default=30, help="Samples per cell")
    parser.add_argument("--trials", type=int, default=3, help="Trials per sample (consistency)")
    parser.add_argument("--model", default="claude-opus-4-6", help="Claude model to use")
    parser.add_argument("--aws-profile", default=None, help="AWS profile for Bedrock (skips ANTHROPIC_API_KEY)")
    parser.add_argument("--output", type=Path, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    if not args.aws_profile and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY or pass --aws-profile for Bedrock", file=sys.stderr)
        sys.exit(1)

    task_map = {t.name: t for t in ALL_TASKS}
    task_names = args.tasks or list(task_map.keys())
    unknown = set(task_names) - set(task_map.keys())
    if unknown:
        print(f"Unknown tasks: {unknown}. Available: {list(task_map.keys())}", file=sys.stderr)
        sys.exit(1)

    config = BenchmarkConfig(
        task_names=task_names,
        agent_configs=args.configs,
        rule_counts=args.rule_counts,
        n_samples=args.samples,
        n_trials=args.trials,
        model=args.model,
    )

    agents = {
        "instructions_only": InstructionsOnlyAgent(model=args.model, aws_profile=args.aws_profile),
        "with_tools": WithToolsAgent(model=args.model, aws_profile=args.aws_profile),
    }

    tasks = [task_map[n] for n in task_names]

    print(f"Running: tasks={task_names} configs={args.configs} "
          f"rule_counts={args.rule_counts} samples={args.samples} trials={args.trials}\n")

    harness = BenchmarkHarness(config)
    metrics = harness.run(tasks, agents)

    print()
    print_table(metrics)

    if args.output:
        save_results(metrics, args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / f"run_{ts}.json"
        out.parent.mkdir(exist_ok=True)
        save_results(metrics, out)


if __name__ == "__main__":
    main()
