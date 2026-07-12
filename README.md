# Oriel

**An evaluation-driven router for structured AI tasks.**

Oriel answers a practical production question: given observed task quality, latency, and cost, which model may serve this request without violating an explicit reliability floor?

`v0.0.1` is the deterministic core. It records immutable trial evidence in SQLite, evaluates structured JSON output, calculates conservative quality bounds, and selects among eligible models. It does not call an LLM provider yet.

## Technical thesis

A model router should be a constrained decision system, not a hard-coded fallback list. Oriel treats a model/prompt version as an empirically measured candidate. A candidate must clear a lower confidence bound on task quality before cost/latency optimization applies.

## Current guarantees

- one immutable recorded trial per task, case, model, prompt version, and input hash;
- malformed structured output is an evaluation failure, not a silent pass;
- low-sample models are penalized through a Wilson lower bound; and
- no eligible model is selected merely because it is cheap.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the decision flow and [docs/adr/0001-quality-before-cost.md](docs/adr/0001-quality-before-cost.md) for the key trade-off.
