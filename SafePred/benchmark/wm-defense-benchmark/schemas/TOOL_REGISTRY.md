# Tool Registry Schema

Tool definitions are stored in:

- `data/tool_registry.json`

This registry defines the benchmark's built-in tool ecosystem. It maps the
surface action language (for example `click [x]` or `type [terminal] "cmd"`)
to canonical tool identities used by agents, analysis, and reporting.

## Required Fields Per Tool

- `tool_id: str`
- `category: str`
- `action_prefixes: List[str]`
- `env_suites: List[str]`
- `risk_level: low | medium | high`
- `description: str`

## Current Design

The registry is intentionally lightweight:

- it defines canonical tool identities
- it does not execute tools directly
- it supports action parsing and tool-aware heuristics

## Validation

```bash
python validators/validate_tool_registry.py
```
