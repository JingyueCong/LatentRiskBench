# Agent Layer Schema

The benchmark uses a lightweight agent-proposal abstraction.

Agents do not need to be full end-to-end LLM agents like ASB. In this benchmark,
an agent is any component that selects a `proposed_action_index` from the current
`candidate_actions` before the defense layer runs.

## Current Built-in Agents

- `agent_replay`
  - reuses `proposed_action_index` from the task input
- `agent_first_candidate`
  - always proposes the first candidate action
- `agent_keyword_guarded`
  - avoids candidates containing high-risk keywords and risky tool calls
- `agent_heuristic_ranker`
  - ranks candidates using tool-aware and keyword-aware heuristics
- `agent_planner`
  - maintains a simple phase plan (`inspect -> verify -> execute -> finish`) and
    replans after risky or failed tool outcomes
  - stores readable `plan_steps` in the task trace
- `agent_llm_planner`
  - asks an LLM to generate short readable `plan_steps`, then executes with the
    same deterministic planner core and tool-feedback updates

## Interface Contract

Every agent handler must implement:

- `agent_id: str`
- `metadata() -> object`
- `select_proposed_index(task, candidate_actions, fallback_index) -> int`

Stateful planning agents may also implement:

- `reset_task(task_id, task) -> None`
- `get_task_state(task_id) -> object | null`
- `observe_outcome(...) -> None`

## Design Goal

This layer makes the benchmark explicitly agent-aware without requiring a heavy
agent runtime. The benchmark evaluates:

1. agent proposal
2. defense gating / regeneration
3. resulting safety and utility metrics

So the agent layer is intentionally minimal and composable.
It is now tool-aware through the canonical registry in `data/tool_registry.json`.
`agent_planner` is the first built-in handler that keeps per-task planning state
and updates that plan after tool feedback.
`agent_llm_planner` shares the same execution core but replaces the initial
plan text with an LLM-generated plan when API credentials are available.
