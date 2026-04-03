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

## Benchmark Results

Results from 30 samples × 3 trials per cell, using AWS Bedrock.

### alert_dedup

| Model | Rules | instr acc | instr cons | tools acc | tools cons |
|-------|------:|----------:|-----------:|----------:|-----------:|
| Sonnet 4.6 | 1 | 0% | 100% | **100%** | 100% |
| Sonnet 4.6 | 2 | 71% | 100% | **100%** | 100% |
| Sonnet 4.6 | 3 | 74% | 92% | **95%** | 96% |
| Haiku 4.5 | 1 | 0% | 100% | **100%** | 100% |
| Haiku 4.5 | 2 | 59% | 100% | **100%** | 100% |
| Haiku 4.5 | 3 | 77% | 100% | **100%** | 100% |

### expense_validator

| Model | Rules | instr acc | instr cons | tools acc | tools cons |
|-------|------:|----------:|-----------:|----------:|-----------:|
| Sonnet 4.6 | 1 | **100%** | 0% | 78% | 17% |
| Sonnet 4.6 | 2 | **100%** | 77% | 82% | 15% |
| Sonnet 4.6 | 3 | **100%** | 30% | 80% | 15% |
| Haiku 4.5 | 1 | **100%** | 17% | **100%** | 0% |
| Haiku 4.5 | 2 | **100%** | 77% | **100%** | 0% |
| Haiku 4.5 | 3 | 85% | 65% | **100%** | 5% |

### Key Findings

**alert_dedup** — tools provide a large, consistent accuracy lift:
- `with_tools` achieves 95–100% accuracy vs 0–77% for `instructions_only`
- Both models benefit equally; Haiku 4.5 with tools actually ties Sonnet 4.6

**expense_validator** — task favors instructions-only; tools hurt:
- `instructions_only` reaches 100% accuracy on both models (rules 1–2)
- `with_tools` drops to 78–82% accuracy on Sonnet 4.6
- Consistency is poor across the board (0–77%) — the model wraps the same correct answer in varying text, suggesting the evaluator needs to normalize output format

**Cross-model** — Haiku 4.5 matches or beats Sonnet 4.6 on structured tasks when tools are provided, at a fraction of the cost.

## Conclusion

Native function calling is not universally better than prompt-based instructions — the benefit depends on task structure.

For tasks that decompose cleanly into discrete lookups or comparisons (`alert_dedup`), tools provide a decisive accuracy lift: 0–77% → 95–100%, with perfect consistency. The structured interface removes ambiguity and forces the model to apply each rule precisely. Notably, Haiku 4.5 with tools matches Sonnet 4.6 at a fraction of the cost, making the architecture choice more impactful than the model choice.

For tasks requiring integrated reasoning over multiple conditions (`expense_validator`), instructions-only wins or ties. Wrapping the logic in tool calls fragments what should be a single holistic judgment, introducing errors on Sonnet 4.6 (−20pp accuracy). Haiku 4.5 manages to recover this with tools, but the pattern warrants caution.

The low consistency scores on `expense_validator` are likely a measurement artifact: the evaluator normalizes output before checking correctness, but the consistency metric uses exact string comparison — so `"APPROVE"` and `"approve"` count as inconsistent even when both are right. This should be fixed before drawing conclusions about decision stability on that task.

**Design principle:** match the agent architecture to the task. Use native FC when rules map 1:1 to tool calls; use instructions when rules interact and require judgment. When in doubt, benchmark both — the right choice is task-dependent, not model-dependent.

## Requirements

- Python 3.11+
- `ANTHROPIC_API_KEY` environment variable, or `--aws-profile` for AWS Bedrock
