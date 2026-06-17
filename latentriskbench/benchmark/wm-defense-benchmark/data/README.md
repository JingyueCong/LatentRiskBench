## Data Layout

This directory contains the benchmark's data assets:

- `base_tasks.jsonl`: benign compositional source tasks
- `tasks.jsonl`: explicit offline benchmark tasks
- `tasks_online.jsonl`: explicit online benchmark tasks
- `policies.json`: policy definitions used by the runner
- `tool_registry.json`: canonical tool ecosystem definitions
- `attack_payloads/`: standalone payloads grouped by attack family

Runners and validators default to this directory, but all paths remain
overrideable through CLI flags.
