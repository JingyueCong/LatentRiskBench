# Experiment Section Snippet: Skill-Level Evaluation

Beyond aggregate robustness metrics, we evaluate models with a capability-oriented
view using `skill_metrics_by_skill`, a structured slice summary that groups tasks by
the safety capability they stress. Each task may be explicitly annotated with one or
more skills, and when such annotations are absent, the benchmark infers them from
task tags, attack family, environment type, tool usage, and security-relevant state
signals. This yields skill slices such as `credential_protection`,
`suspicious_link_detection`, `instruction_robustness`, `observation_integrity`,
`multi_step_resilience`, and `safe_escalation`.

For each skill slice, we report the number of covered tasks (and steps in online
settings), task success, original-goal success, attack success, violation rate,
over-refusal, necessary refusal, and average predicted risk. This turns the
benchmark from a purely task-level robustness test into a capability-level
evaluation framework: aggregate metrics quantify overall robustness, attack slices
localize adversarial failure modes, and skill slices reveal which concrete safety
competencies improve or remain brittle under different agents and defenses.

We treat `skill_metrics_by_skill` as a primary reporting axis rather than an
auxiliary breakdown. In particular, it enables comparisons that are difficult to
observe from aggregate attack success alone, such as whether a defense improves
credential protection while leaving phishing detection weak, or whether an agent
improves tool selection but increases over-refusal on safe escalation tasks. This
capability-oriented view is central to our proposed evaluation paradigm.
