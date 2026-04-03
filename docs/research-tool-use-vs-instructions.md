# Agent Tool Use vs. Instructions-Only: SOTA Research, Deep Benchmarks, and a Production Case Study

## Overview

This document provides an in-depth review of the state of the art in benchmarking LLM agents on tool use versus instruction-following, including actual experimental results, failure mode taxonomies, and a production case study (`detections-agents`) as a real-world instantiation of the comparison.

**The central question:** Does offloading sub-tasks to tools improve accuracy, consistency, and reliability — and by how much as rule complexity grows? No existing benchmark answers this with an explicit rule_count sweep. That is the gap this project fills.

---

## 1. Foundational Work

### 1.1 Toolformer — Tool Use as a Training Problem

Schick et al. (2023) [1] trained GPT-J (6.7B) to self-supervise tool use: generate API call candidates, execute them, keep only calls that reduce cross-entropy loss on subsequent tokens. Tools: calculator, Wikipedia search, QA system, calendar, machine translation.

**Key results:**

| Task | GPT-J (no tools) | Toolformer (tools off) | Toolformer (tools on) | GPT-3 175B |
|------|-----------------|----------------------|----------------------|------------|
| ASDiv (math) | 7.5 | 14.8 | **40.4** | 14.0 |
| SVAMP (math) | 5.2 | 6.3 | **29.4** | 10.0 |
| SQuAD (factual) | 17.8 | 22.1 | **33.8** | 26.8 |
| TriviaQA | 43.9 | 46.7 | **48.8** | 65.9 |

**Critical finding:** A 6.7B model with trained tool use outperforms a 175B model (GPT-3) on math and factual lookup. The tool-use gap on math is enormous (+173% on ASDiv). Notably, even "Toolformer (tools off)" — finetuned on tool trajectories but with tools disabled — outperforms vanilla GPT-J, suggesting that training on tool-augmented data improves base reasoning.

**Implication for instructions-only vs. tool-augmented:** The gap is not just about inference-time tool access — it is amplified by whether the model has been trained to recognize when tools are needed.

---

### 1.2 ReAct — Reasoning Interleaved with Action

Yao et al. (2022) [2] augmented the action space with a "language space" for reasoning traces (thoughts) interleaved with tool calls. Evaluated on HotPotQA, FEVER, ALFWorld (text game), WebShop (product navigation) with PaLM-540B.

**Key results:**

| Method | HotPotQA (EM) | FEVER (Acc) | ALFWorld (success) | WebShop (score) |
|--------|--------------|-------------|-------------------|-----------------|
| Standard prompting | 28.7 | 57.1 | — | — |
| Chain-of-Thought (CoT) | 29.4 | 56.3 | — | — |
| Act-only | 25.7 | 58.9 | 45% | 62.3 |
| **ReAct** | 27.4 | 60.9 | **71%** | **66.6** |
| CoT+ReAct hybrid | **35.1** | **64.6** | — | — |

**Failure mode taxonomy (from 50-trajectory human analysis):**

| Failure type | ReAct | CoT |
|-------------|-------|-----|
| Hallucination (of evidence) | **0%** | **56%** |
| Reasoning error | 47% | 16% |
| Search result error | 23% | — |

ReAct reduces hallucination from 56% to 0% of failures by grounding reasoning in actual retrieved evidence. The tradeoff is more reasoning errors (47% vs. 16%) — the model now has to reason correctly about real, sometimes noisy, retrieved content.

**BFCL V3 documents analogous failure modes [3]:**
- **Implicit action failure:** model calls `fillFuelTank(amount=50)` without first checking current fuel level
- **State awareness failure:** model creates `mkdir alex` when already inside `alex` directory
- **Over-planning failure:** model re-authenticates Twitter when authentication was already complete

---

## 2. Tool-Use Benchmarks

### 2.1 τ-bench — The Canonical Reliability Benchmark

Yao, Shinn, Razavi, Narasimhan (2024) [4] introduced the most rigorous benchmark for multi-turn tool-agent-user interaction. Two domains: **airline** (50 tasks, 13 tools, 1,242-word policy document) and **retail** (more tasks, more tools). Success is measured by final database state matching the annotated goal — not text similarity.

**pass^k definition:** Probability that all k independent trials of the same task succeed. Measures reliability, not just average accuracy. A model that succeeds 50% of the time on a task achieves pass^4 ≈ 6% on that task.

**Actual pass^k scores:**

| Strategy | Model | Pass^1 | Pass^2 | Pass^3 | Pass^4 |
|----------|-------|--------|--------|--------|--------|
| **Tool-calling (TC)** | Claude-3.5-Sonnet (Oct 2024) | **0.460** | **0.326** | **0.263** | **0.225** |
| **Tool-calling (TC)** | GPT-4o | 0.420 | 0.273 | 0.220 | 0.200 |
| **Tool-calling (TC)** | GPT-4o-mini | 0.225 | 0.140 | 0.110 | 0.100 |
| **Act (text only)** | GPT-4o | 0.365 | 0.217 | 0.160 | 0.140 |
| **ReAct (text only)** | GPT-4o | 0.325 | 0.233 | 0.185 | 0.160 |

*(Airline domain. Retail domain is easier: TC Claude-3.5-Sonnet reaches 0.692 pass^1.)*

**Key findings:**
- Even the best TC agent succeeds on only 46% of airline tasks at pass^1, dropping to 22.5% at pass^4.
- TC outperforms text-only Act by ~30% at pass^1 (0.42 vs. 0.365) and ~43% at pass^2 (0.273 vs. 0.217) for GPT-4o.
- ReAct (0.325) performs worse than Act (0.365) on this benchmark — the airline policy is complex enough that forced reasoning traces introduce errors.
- Pass^8 drops below 25% for all models in retail domain (from the abstract).
- **Rule complexity effect:** The 1,242-word policy contains eligibility rules, compensation tiers, and booking constraints. The paper's analysis shows policy non-compliance is the dominant failure mode — directly motivating rule_count as a swept variable.

---

### 2.2 Berkeley Function Calling Leaderboard V4 (BFCL)

Yan et al. (2025) [3] — the most comprehensive live leaderboard for tool/function calling. V4 evaluates: web search, memory (KV/vector/recursive), multi-turn, single-turn (AST matching), hallucination measurement, and format sensitivity.

**Top results (December 2025):**

| Rank | Model | Overall Acc | Mode | Cost ($) |
|------|-------|-------------|------|----------|
| 1 | Claude-Opus-4-5 | **77.47%** | FC | 86.55 |
| 2 | Claude-Sonnet-4-5 | 73.24% | FC | 43.73 |
| 3 | Gemini-3-Pro-Preview | 72.51% | **Prompt** | 298.47 |
| 4 | GLM-4.6 (FC thinking) | 72.38% | FC | 4.64 |
| 8 | o3-2025-04-16 | 63.05% | **Prompt** | 234.64 |

**FC vs. Prompt gap:** At the very top, the gap is ~5pp (77.47 vs. 72.51). For weaker models the gap is much larger. The gap widens substantially on multi-turn and parallel tool call scenarios. For simple single-call scenarios, capable models nearly close the gap.

**BFCL V1 conclusion:** "In terms of simple function calling (without complex planning and chained function calling), finetuning an open-source model can be as effective as proprietary models." The complexity regime matters.

**Common failure modes documented by BFCL:**
1. Type errors: `int` where `float` required
2. Hallucinated parameters not in schema
3. Prompt-mode: text that looks like function calls but is not executable
4. Missing required fields (e.g., URL omitted from REST calls)

---

### 2.3 AgentBench — Tool-Augmented Environments Favor Frontier Models

Liu et al. (2023) [5] evaluated LLMs across 8 interactive environments: OS shell, database, knowledge graph, card games, text games, web shopping, and web browsing. GPT-4 scores approximately 3.6× higher than the best open-source models — the tool-augmented environment gap is far larger than on static NLP benchmarks.

---

### 2.4 ToolLLM / ToolBench — 16,000+ Real APIs

Qin et al. (2023) [6] built a dataset of 16,000+ RapidAPI tools and a depth-first search tree planner (DFSDT) for multi-step orchestration. ToolEval measures solution rate and win rate vs. ChatGPT via LLM evaluator. ToolLLaMA (7B, fine-tuned) matches ChatGPT on unseen APIs with DFSDT — establishing that tool use is a learnable capability even in small models.

---

### 2.5 MCPGAUGE — Tool Augmentation Can Degrade Performance

Song et al. (2025) [7] is the most directly comparable study to an instructions-only vs. tool-augmented sweep. Framework evaluated 6 LLMs, 30 MCP tool suites, 160-prompt compliance suite, ~20,000 API calls across knowledge comprehension, reasoning, and code generation.

**The critical negative result:**
> "Contrary to expectations, automated MCP integration results in **performance degradation** across three major task domains rather than yielding improvements."
- Average degradation: **-9.5%**
- General reasoning: **-10.2%**
- Knowledge comprehension: **-1.4%**

**Instruction following accuracy (IFA) by model (1-turn):**

| Model | IFA (1-turn) | IFA (2-turn) | Improvement |
|-------|-------------|-------------|-------------|
| GPT-4 | **0.11** | 0.35 | +227.3% |
| Qwen-2.5 | **0.06** | 0.88 | +1366.7% |
| Claude-4 | 0.54 | 0.86 | +59.3% |
| Llama-4 | **1.00** | — | — |

**Interpretation:** Most LLMs fail to follow instructions for tool use in single-turn settings. Claude-4 (0.54) and Llama-4 (1.00) are outliers. GPT-4 achieves IFA of only 0.11 — meaning 89% of its tool calls in one-turn settings are non-compliant.

**Token overhead of tool augmentation:**

| Model | Knowledge Comp (without) | With MCP | Ratio |
|-------|--------------------------|----------|-------|
| GPT-4 | 0.04M | 0.17M | 4.3× |
| Claude-4 | 0.04M | 9.46M | **236.5×** |
| DeepSeek-V3 | 0.04M | 0.98M | 24.5× |

**Nuance:** The degradation is likely because external tool data introduces noise into the model's internal reasoning. Tool augmentation helps most when the tool provides information the model cannot know; it hurts when the tool data conflicts with or overwhelms the model's existing knowledge.

---

### 2.6 Solver-Aided Policy Compliance (SMT Verification)

Winston, Winston, Just (2026) [8] evaluated formal verification of tool-use policy compliance on the τ²-bench airline domain. Translated the 1,242-word policy to SMT-LIB-2.0 Z3 constraints; checked every tool call before execution.

**Results:**

| Metric | Prompt-only (GPT-4.1) | SMT-aided (GPT-4.1 + Z3) |
|--------|----------------------|--------------------------|
| Invalid write tool calls | ~50% | **29%** |
| Consistency degradation (k=1→4) | **40%** | **26%** |
| Overall task accuracy | similar | similar |

**Key finding:** Formal verification significantly improves consistency (26% vs. 40% degradation from pass^1 to pass^4) but does not substantially change raw accuracy. The improvement is in reliability, not capability.

**Challenge:** Full automation of policy-to-SMT translation consistently failed. AWS Bedrock produced ~600 lines with ~95% coverage but still underconstrained. Manual human tuning was required for production use.

---

### 2.7 τ²-bench — Dual Control Degrades Agents Further

Stoel et al. (2025) [9] extended τ-bench to a telecommunications domain where both agent and user employ tools (Dec-POMDP model). Key finding: **>40% performance degradation** when shifting from single-agent to dual-control compared to single-agent tool use alone.

---

### 2.8 The Evolution of Tool Use Survey (2025)

Xu et al. (2025/2026) [10] — a comprehensive survey across six dimensions. Key benchmark landscape table (multi-tool environments only):

| Benchmark | # Tools | # Instances | Environment |
|-----------|---------|-------------|-------------|
| τ-bench | 28 | 165 | Human-in-loop |
| τ²-bench | 68 | 279 | Human-in-loop |
| ToolBench | 3,451 | 126,486 | Real/Interactive |
| UltraHorizon | 400+ | 168 | Interactive (400+ tool calls/trajectory) |
| AgentLongBench | Varies | Varies | Interactive (up to 4M tokens) |
| AppWorld | 457 | 750 | Interactive |

**On the central research shift:**
> "The primary research objective has transitioned from the correctness of single-point calls to the end-to-end executability and robustness of multi-tool chains in complex environments."

> "Long-horizon evaluation should not rely solely on endpoint success. The more informative question is often whether a benchmark isolates a specific failure mode and distinguishes between accidental success and genuinely robust orchestration."

---

## 3. The Gap in Existing Literature

**No existing benchmark:**
1. Explicitly sweeps **rule_count** (1→N rules) as an independent controlled variable
2. Compares instructions-only vs. tool-augmented agents on the **same decision task** as rules increase
3. Measures the **crossover point** — the rule count at which instructions-only accuracy falls below tool-augmented
4. Combines accuracy, **pass^k consistency**, and format compliance in a single sweep

τ-bench comes closest: it has a complex policy and measures pass^k. But rule complexity is fixed and implicit (1,242-word document); it is not systematically varied.

MCPGAUGE measures instruction-following accuracy for tool use, but does not compare to instructions-only baselines on the same tasks.

---

## 4. Production Case Study: `detections-agents`

`detections-agents` [11] is a production-grade autonomous alert triage system at Cisco that provides a real-world, human-evaluated instance of the comparison.

### 4.1 Architecture

Two agents running on Jenkins (~every 10 minutes):

**Triage Agent:** 6-step deterministic decision tree (recovered? flapping? history? environment? severity? → action). Outputs `[ACTION NEEDED]` / `[NO ACTION]` / `[KNOWN ISSUE]`. State tracked via Webex message edits — idempotent, crash-recoverable.

**Investigate Agent:** 5-phase RCA state machine (OBSERVE → CHANGES → CORRELATE → COMPLETE). Produces JIRA ticket + "5 Whys" chain with confidence level (HIGH / MEDIUM / LOW). Resumable across Jenkins runs.

### 4.2 Current Tool Pattern (CLI-based, not native FC)

```
Claude Code reads SKILL.md (260-line algorithm as prose)
  → shells out: datadog-cli logs search "query" --from now-1h
  → reads stdout JSON, reasons about it, shells out again
  → webex-cli edit-reply --message-id X --state "[ACTION NEEDED]"
```

All underlying clients (`DatadogClient`, `WebexClient`, `TeamCityClient`) are clean Python HTTP wrappers — the CLI layer exists because Claude Code's native execution environment is bash.

### 4.3 Evaluation: Human Feedback Loop

No automated accuracy metrics. Evaluation is qualitative:
- Humans reply to triage decisions in Webex threads
- Weekly agent scans threads, categorizes: CORRECTION / CONFIRMATION / SUGGESTION / COMPLAINT
- Corrections distilled into `lessons.md`

**Documented failure modes from `lessons.md`** reveal exactly where instructions-only reasoning breaks without tools:
- *"Don't blame a PR until you verify it's deployed to production"* → requires deployment lookup tool; without it the model hallucinates causal links
- *"Don't re-investigate already-known issues"* → requires history lookup tool; without it the model repeats work
- *"Distinguish pre-existing patterns from recent changes"* → requires time-series queries; without tools the model conflates correlation and causation

These failure modes map directly to the failure categories documented in τ-bench and BFCL V3.

### 4.4 Migrating to Native Function Calling

Feasible in ~3–4 days. The Python clients already exist and have the right interface. What needs to be built:
- Tool schema definitions (maps 1:1 from existing argparse CLI structure)
- Python agent harness with agentic loop
- System prompt translated from `triage/SKILL.md`
- Idempotency guard as an explicit tool rather than prose instruction

**Main challenge:** The SKILL.md algorithm embeds crash recovery, idempotency, and state machine logic as prompt prose. With native FC these become structural constraints in the harness — more robust, but requires careful porting.

### 4.5 Why This Matters for the Benchmark

`detections-agents` provides:
1. **Real decision data** — production alerts with human-verified labels (CORRECTION/CONFIRMATION)
2. **Known rule structure** — 6-step decision tree, each step = one rule → maps directly to `rule_count`
3. **Documented failure modes** — confirms hypothesis that instructions-only degrades on tool-dependent reasoning
4. **Live A/B test opportunity** — run CLI-based and FC-based triage in parallel on the same alert stream, measure human correction rate as the ground truth metric

---

## 5. Quantitative Summary

| Finding | Source | Number |
|---------|--------|--------|
| TC vs. text-only Act on airline pass^1 (GPT-4o) | τ-bench [4] | 0.420 vs. 0.365 (+15%) |
| TC vs. text-only Act on airline pass^2 (GPT-4o) | τ-bench [4] | 0.273 vs. 0.217 (+26%) |
| Best agent pass^4 airline (Claude TC) | τ-bench [4] | 22.5% |
| ReAct vs. CoT: hallucination in failures | ReAct [2] | 0% vs. 56% |
| ReAct vs. Act: ALFWorld success rate | ReAct [2] | 71% vs. 45% (+58%) |
| Toolformer 6.7B vs. GPT-3 175B on ASDiv | Toolformer [1] | 40.4 vs. 14.0 (6.7B wins) |
| MCP integration: avg performance change | MCPGAUGE [7] | **−9.5%** |
| GPT-4 instruction following (1-turn) | MCPGAUGE [7] | IFA = 0.11 (89% non-compliance) |
| GPT-4 IFA improvement (2-turn) | MCPGAUGE [7] | +227% |
| SMT checker: consistency degradation k=1→4 | Policy paper [8] | 26% vs. 40% for prompt-only |
| FC vs. Prompt gap at BFCL top | BFCL V4 [3] | 77.47% vs. 72.51% (~5pp) |
| Dual-control degradation vs. single-agent | τ²-bench [9] | >40% |
| Toolformer: tool-on vs. off on ASDiv | Toolformer [1] | 40.4 vs. 14.8 (+173%) |

---

## 6. Implications for `detections-agents`: Beyond the CLI Approach

The current `detections-agents` architecture routes all tool access through bash CLIs invoked by Claude Code. The SKILL.md files (triage: 260 lines, investigate: 223 lines) embed the decision algorithm as prose instructions — making `detections-agents` a real-world instance of the **prompt-based** pattern documented in the SOTA literature. The research above suggests several alternative approaches worth considering.

### 6.1 Native Function Calling (Python Agent Harness)

Replace the Claude Code + SKILL.md entrypoint with a Python script using the Anthropic SDK `tools=` parameter. The underlying `DatadogClient`, `WebexClient`, and `TeamCityClient` classes already exist and have the right interface — they just need tool schema wrappers.

```
Current:  Claude Code → reads SKILL.md prose → bash CLI → stdout JSON → reasons
Proposed: Python harness → tools=[schema] → model calls search_logs() → dict returned
```

**What SOTA research says about this:**
- τ-bench [4]: TC outperforms text-only Act by +15% at pass^1, +26% at pass^2 — and the gap widens under complex policies, which matches the triage 6-step decision tree
- BFCL V3 [3]: documents that prompt-based tool invocation introduces state-awareness failures (acting on outdated state) and over-planning failures — both failure modes that appear in the triage `lessons.md` corrections
- Policy compliance paper [8]: prompt-only enforcement of a 1,242-word policy degrades consistency by 40% (k=1→4); SMT-aided drops to 26% — the triage policy is comparable in complexity

**Tradeoff:** Native FC requires an explicit Python harness to replace the implicit orchestration that Claude Code provides. Crash recovery (currently handled via Webex message state + `[IN PROGRESS]` markers) must move into the harness as explicit checkpointing logic.

#### Repository Layout

The repo already has two architectures: the production Jenkins + Claude Code pipeline, and an event-based Python FastAPI experiment (`experiments/event_based_galactus/`). The FC implementation would live as a third entrypoint inside the existing `detections_agents/` package (currently empty) and run as an experiment in parallel with production:

```
detections_agents/
  __init__.py               (exists, empty)
  triage/
    __init__.py
    tools.py                ← Anthropic tool definitions wrapping existing clients
    agent.py                ← agentic loop
    prompt.py               ← system prompt distilled from triage/SKILL.md
```

Jenkins would change a single line per pipeline:

```groovy
// Before
stages.runClaudeCode(stageName: 'Triage', promptFile: promptFile)

// After (shadow pipeline)
sh "python -m detections_agents.triage --room ${roomConfig.room}"
```

#### `tools.py` — wrap what already exists

Every existing CLI command maps 1:1 to a tool definition. The underlying Python clients (`DatadogClient`, `WebexClient`, `TeamCityClient`) are unchanged — only the call site moves from bash to a Python function:

```python
from datadog_cli.client import DatadogClient
from datadog_cli.config import load_config

TOOLS = [
    {
        "name": "search_logs",
        "description": "Search Datadog logs. Returns up to `limit` log events matching the query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string", "description": "Datadog log search query"},
                "time_from": {"type": "string", "default": "now-1h"},
                "limit":     {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_monitor",
        "description": "Get full details of a Datadog monitor by ID, including current status.",
        "input_schema": {
            "type": "object",
            "properties": {"monitor_id": {"type": "integer"}},
            "required": ["monitor_id"],
        },
    },
    {
        "name": "list_monitors",
        "description": "Search Datadog monitors by name. Use to find monitor IDs from alert text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string"},
                "page_size": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_messages",
        "description": "Read recent messages from a Webex room. Use to scan for untriaged alerts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "room":          {"type": "string"},
                "limit":         {"type": "integer", "default": 20},
                "mentioned_me":  {"type": "boolean", "default": False},
                "since_minutes": {"type": "integer", "default": 60},
            },
            "required": ["room"],
        },
    },
    {
        "name": "get_thread",
        "description": "Read all replies in a Webex message thread. Use to check if an alert is already claimed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "room":      {"type": "string"},
                "parent_id": {"type": "string"},
            },
            "required": ["room", "parent_id"],
        },
    },
    {
        "name": "post_triage_message",
        "description": (
            "Post or edit the bot's triage reply for an alert thread. "
            "Pass message_id to edit an existing reply (idempotent update). "
            "Omit message_id to post a new reply. Returns {message_id}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "room":       {"type": "string"},
                "parent_id":  {"type": "string"},
                "text":       {"type": "string"},
                "message_id": {"type": "string", "description": "If set, edit this message instead of posting new."},
            },
            "required": ["room", "parent_id", "text"],
        },
    },
    {
        "name": "get_builds",
        "description": "Get recent TeamCity build history for a build configuration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "build_type_id": {"type": "string"},
                "limit":         {"type": "integer", "default": 10},
            },
            "required": ["build_type_id"],
        },
    },
    {
        "name": "get_build_log",
        "description": "Fetch the log for a specific TeamCity build. Returns the last N lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "build_id": {"type": "integer"},
                "tail":     {"type": "integer", "default": 50},
            },
            "required": ["build_id"],
        },
    },
]

def execute(name: str, inputs: dict) -> dict:
    if name == "search_logs":
        with DatadogClient(load_config()) as dd:
            return dd.search_logs(**inputs)
    if name == "get_monitor":
        with DatadogClient(load_config()) as dd:
            return dd.get_monitor(inputs["monitor_id"])
    if name == "list_monitors":
        with DatadogClient(load_config()) as dd:
            return {"monitors": dd.list_monitors(**inputs)}
    # webex and teamcity tools follow the same pattern
    raise ValueError(f"Unknown tool: {name}")
```

#### `agent.py` — the agentic loop

This is the full replacement for `ClaudeRunner.analyze()` + the `claude --print -p SKILL.md` invocation. The state machine (`[IN PROGRESS]` → `[ACTION NEEDED]`) is handled entirely through tool calls — the model calls `get_thread` to check existing state, calls `post_triage_message` to claim the alert, and calls it again to finalize. The message ID returned by the first `post_triage_message` call stays in the conversation context and is reused automatically:

```python
import json
import anthropic
from .tools import TOOLS, execute
from .prompt import SYSTEM_PROMPT

def run_triage(room: str, model: str = "claude-opus-4-6") -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": f"Run triage for room: {room}"}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--room", required=True)
    p.add_argument("--model", default="claude-opus-4-6")
    args = p.parse_args()
    run_triage(args.room, args.model)
```

#### `prompt.py` — SKILL.md becomes a system prompt

The SKILL.md algorithm is largely preserved, but the bash-navigation framing is replaced with tool-call framing. The key differences:

- "Use the **webex** skill to read mentioned messages" → "Call `get_messages(room=..., mentioned_me=true)`"
- "Post `[TRIAGE] [IN PROGRESS]`" → "Call `post_triage_message(...)`. Save the returned `message_id` — use it for all subsequent edits to this alert."
- Crash recovery: "Before claiming any alert, call `get_thread(parent_id=...)` to check if a bot reply starting with `[TRIAGE]` already exists. If `[IN PROGRESS]` is found, reuse its `message_id`. If a final state is found, skip."

The decision tree (Steps 1–6 in SKILL.md) moves verbatim. The Galactus quip instructions move verbatim. Output format constraints move verbatim. Total prompt length is comparable to the original SKILL.md.

#### What is eliminated

| Current | Replaced by |
|---------|-------------|
| `nix run ... #claude-code-bin` in Jenkins | `python -m detections_agents.triage` |
| `datadog-cli logs search "q" --from now-1h` bash call | `search_logs(query="q", time_from="now-1h")` tool call |
| stdout JSON parsing | dict returned directly from `DatadogClient` |
| SKILL.md bash-navigation prose | Structured tool definitions with typed schemas |
| Implicit state across bash calls | Explicit `message_id` in conversation context |

#### What is preserved

- All three Python clients (`DatadogClient`, `WebexClient`, `TeamCityClient`) — zero rewrite
- Jenkins credentials bindings — env vars unchanged
- Room config YAML files — unchanged
- `[TRIAGE] [IN PROGRESS]` → `[ACTION NEEDED]` state machine logic
- The Galactus persona and output format
- The 6-step decision tree

### 6.2 MCP Server per Integration

Expose each integration (Datadog, Webex, TeamCity, JIRA) as an MCP server and keep Claude Code as the orchestrator. The agent reads SKILL.md and invokes MCP tools instead of bash CLIs. No Python harness needed — Claude Code handles the agentic loop natively.

```
Current:  Claude Code → bash: datadog-cli logs search ...
Proposed: Claude Code → MCP: datadog_server.search_logs(query=..., from="now-1h")
```

**What SOTA research says:**
- MCPGAUGE [7]: MCP integration with Claude-4 achieves IFA of 0.54 (one-turn) vs. GPT-4's 0.11 — Claude models are significantly better at following MCP tool-use instructions
- MCPGAUGE finding: tool augmentation degrades performance by −9.5% on average across models — **but the degradation is on knowledge/reasoning tasks where the model already has the answer internally**. Triage decisions depend on external state (current monitor status, recent logs) that the model cannot know without tools — this is exactly the regime where tools help rather than hurt
- Token overhead: Claude-4 with MCP tools uses up to 236× more input tokens on knowledge tasks; on triage the tool responses are bounded (log queries return fixed-size JSON) so this is manageable

**Advantage over native FC:** SKILL.md stays as the primary instruction document, preserving the human-readable algorithm and crash recovery prose. The MCP layer is purely additive — no rewrite of the orchestration logic.

### 6.3 Structured Output + Deterministic Rule Engine

Keep Claude Code for evidence gathering (bash CLIs unchanged) but replace the LLM triage decision with a **deterministic rule engine**: the model only classifies and extracts structured fields; a Python validator applies the 6-step decision tree to those fields.

```
Current:  LLM reads evidence → LLM applies 6-step tree → LLM outputs decision
Proposed: LLM reads evidence → LLM outputs structured JSON (monitor_name, env, status, history_count)
          → Python rule engine applies decision tree → deterministic output
```

**What SOTA research says:**
- Policy compliance paper [8]: SMT-aided verification reduces invalid tool calls from ~50% to 29% and consistency degradation from 40% to 26% — without changing raw accuracy. The improvement is in reliability, not capability. A deterministic rule engine achieves the same goal with zero inference overhead.
- ReAct [2]: hallucination in success cases drops from 14% (CoT) to 6% (ReAct) when reasoning is grounded in evidence. Moving the decision logic out of the LLM entirely eliminates hallucination in the decision step completely.

**Advantage:** Removes LLM judgment from the most rule-bound step. The model's role becomes evidence extraction (what it is good at) rather than rule application (where pass^k degrades). No harness rewrite needed — add a small Python validator called after the LLM response.

### 6.4 Fine-Tuned Specialist Model (Longer Term)

Fine-tune a small model (e.g., Haiku or an open model) on triage decision trajectories using the human feedback corrections in `lessons.md` as training signal — similar to Toolformer's self-supervised filtering approach [1].

**What SOTA research says:**
- Toolformer [1]: a 6.7B model trained on tool trajectories outperforms GPT-3 175B at zero-shot math and factual lookup. For a constrained domain like alert triage, a small fine-tuned model may match or exceed a frontier model at much lower cost and latency.
- ReAct [2]: with 3,000 training examples, PaLM-8B ReAct outperforms ALL prompting methods including PaLM-540B. The triage dataset (alerts × decisions × corrections) may already be large enough.
- ToolLLM [6]: establishes that tool-use is a learnable capability from trajectory data — the DFSDT planner approach is directly applicable to multi-step investigation workflows.

**Tradeoff:** Requires curating the corrections dataset, defining the trajectory format, and managing a fine-tuning pipeline. The feedback loop already collects corrections — the main work is structuring them as training examples.

### 6.5 Recommendation

For the **near term** (lowest risk, highest reliability gain): **Option 6.3** — add a deterministic rule engine for the triage decision step. This eliminates the most fragile part of the current architecture (LLM applying a 6-step rule tree under time pressure) without changing the evidence-gathering flow or requiring a harness rewrite.

For the **medium term** (best performance ceiling): **Option 6.1 or 6.2** — native FC or MCP. Both give the model structured access to live state rather than requiring it to parse JSON from stdout. MCP (6.2) is the lower-risk path because it preserves SKILL.md and the Claude Code orchestration layer.

For the **long term** (cost and latency): **Option 6.4** — fine-tuned specialist. Once the feedback corpus is large enough, a fine-tuned Haiku-class model running the full triage pipeline would cost a fraction of Opus and likely match it on in-distribution alerts.

---

## References

[1] Schick, T. et al. **Toolformer: Language Models Can Teach Themselves to Use Tools**. arXiv:2302.04761, 2023.
https://arxiv.org/abs/2302.04761

[2] Yao, S. et al. **ReAct: Synergizing Reasoning and Acting in Language Models**. arXiv:2210.03629, 2022.
https://arxiv.org/abs/2210.03629

[3] Yan, F. et al. **Berkeley Function-Calling Leaderboard (BFCL V4)**. UC Berkeley, 2025.
https://gorilla.cs.berkeley.edu/leaderboard.html
Blog: https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
Code: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

[4] Yao, S., Shinn, N., Razavi, P., Narasimhan, K. **τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains**. arXiv:2406.12045, 2024.
https://arxiv.org/abs/2406.12045
Code & leaderboard: https://github.com/sierra-research/tau-bench

[5] Liu, X. et al. **AgentBench: Evaluating LLMs as Agents**. arXiv:2308.03688, 2023.
https://arxiv.org/abs/2308.03688

[6] Qin, Y. et al. **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**. arXiv:2307.16789, 2023.
https://arxiv.org/abs/2307.16789

[7] Song, W. et al. **Help or Hurdle? Rethinking Model Context Protocol-Augmented Large Language Models (MCPGAUGE)**. arXiv:2508.12566, 2025.
https://arxiv.org/abs/2508.12566

[8] Winston, C., Winston, C., Just, R. **Solver-Aided Verification of Policy Compliance in Tool-Augmented LLM Agents**. arXiv:2603.20449, 2026.
https://arxiv.org/abs/2603.20449

[9] Stoel, M. et al. **τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment**. arXiv:2506.07982, 2025.
https://arxiv.org/abs/2506.07982

[10] Xu, H. et al. **The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration**. arXiv:2603.22862, 2025/2026.
https://arxiv.org/abs/2603.22862

[11] Starost, R. et al. **detections-agents: Autonomous Alert Triage and Investigation for Production Monitoring**. Cisco Systems, internal repository, 2024–2025.
https://wwwin-github.cisco.com/rstarost/detections-agents
