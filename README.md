# WIR — Web Interface Runtime

WIR is a browser interface for AI agents. Instead of screenshots or raw DOM, the
runtime compiles the live page into a structural interface graph and exposes it
through five deterministic verbs: `find`, `read`, `act`, `navigate`, `finish`.
The runtime owns structure, identity, execution and proof; the model owns meaning.

What that buys:

- **Nothing is silently lost.** Every result states what was withheld and how to
  get the rest.
- **Every action is verified.** An `act` reports what actually changed and what
  the browser actually sent; delivery is not success.
- **Every claim is falsifiable.** A task can only be finished by citing evidence
  the runtime itself produced.

No semantic matcher, no vision, no arbitrary JavaScript on the critical path. The
agent that drives it is a small reference loop; any model can sit in the seat.

## Status

The runtime, the reference agent and the benchmark harness exist and are measured.
Source will be published here soon; until then this repository holds the results.

## WebArena-Verified

812 tasks, official WebArena-Verified evaluator, per-task environment reset, one
first attempt with Claude Haiku 4.5, Claude Opus 5 only on the tasks Haiku missed.
A task counts as solved under the official evaluator or WebArena's own LLM judge
on string-match tasks (either ruler).

| | |
|---|---|
| tasks | 812 |
| excluded (evaluator raises on its own definition, or references that no browser can satisfy) | 14 |
| solved by Claude Haiku 4.5 | 479 |
| added by Claude Opus 5 | 142 |
| **solved** | **621 / 798 = 77.82%** |

Raw per-step trajectories for every task are available to the WebArena team.
