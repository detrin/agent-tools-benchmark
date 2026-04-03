# agent-tools-benchmark

Benchmark measuring Claude agent performance on decision-making tasks across two configurations:
- **instructions_only** — agent receives only a text prompt with rules
- **with_tools** — agent receives rules plus structured tools to call

The benchmark sweeps over task types, rule counts, and trials to measure accuracy, edge-case accuracy, and consistency.

## Tasks

| Task | Description |
|------|-------------|
| `alert_dedup` | Deduplicate security alerts based on configurable rules |
| `expense_validator` | Validate expense reports against policy rules |
| `log_classifier` | Classify log entries into severity categories |
| `scheduler` | Schedule meetings respecting constraints |
| `contract_checker` | Check contracts against compliance rules |
| `dependency_resolver` | Resolve package dependency conflicts |

## Quickstart

```bash
# Install dependencies
uv sync

# Set API key
export ANTHROPIC_API_KEY=sk-...

# Run all tasks (default: 30 samples, 3 trials, rule counts 1-3)
uv run python run.py

# Quick smoke test
uv run python run.py --samples 5 --trials 1

# Single task
uv run python run.py --tasks alert_dedup

# Cheaper model
uv run python run.py --model claude-haiku-4-5-20251001

# Save results to specific file
uv run python run.py --output results/my_run.json
```

## Output

Results are printed as a table and saved to `results/run_<timestamp>.json`.

| Column | Meaning |
|--------|---------|
| Accuracy | Fraction of samples with correct output |
| Edge Acc | Accuracy on edge-case samples only |
| Consistency | Fraction of samples with identical output across trials |
| N | Number of samples evaluated |

## Requirements

- Python 3.11+
- `ANTHROPIC_API_KEY` environment variable
