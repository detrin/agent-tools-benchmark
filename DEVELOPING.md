# Developing

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies (including dev)
uv sync --extra dev

# Activate venv (required before running tools directly)
source .venv/bin/activate
```

## Code Style

- **No comments or docstrings** — code documents itself through naming and structure
- **DRY** — extract repeated logic immediately
- **Minimal** — shortest correct implementation, no speculative abstractions
- Line length: 100 chars (`ruff` enforced)
- Target: Python 3.11+

## Linting and Type Checking

```bash
# Lint
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type check
uv run mypy . --ignore-missing-imports
```

Both run automatically as pre-commit hooks on staged `.py` files.

## Git Hooks

Pre-commit hooks are installed in `.git/hooks/`:

| Hook | Checks |
|------|--------|
| `pre-commit` | `ruff check` + `mypy` on staged Python files |
| `commit-msg` | Author must be `Daniel Herman <daniel.herman@protonmail.com>` — no `Co-authored-by` trailers |

Hooks activate `.venv` automatically if present.

## Running Tests

```bash
uv run pytest
```

## Adding a Task

1. Create `tasks/<task_name>/` with `__init__.py` and `task.py`
2. Implement a class inheriting from `tasks.base.BaseTask`
3. Register it in `tasks/__init__.py` under `ALL_TASKS`
4. Optionally add `generator.py` for synthetic data and `tools.py` for tool definitions

## Project Structure

```
agents/          # Agent implementations (instructions_only, with_tools)
benchmark/       # Harness, metrics, types
tasks/           # Task definitions
  <task_name>/
    task.py      # Task logic and evaluation
    generator.py # Synthetic sample generation (optional)
    tools.py     # Tool definitions for with_tools agent (optional)
run.py           # CLI entrypoint
results/         # JSON output files (gitignored)
```

## Dependency Management

```bash
# Add a runtime dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Update all dependencies
uv sync --upgrade
```
