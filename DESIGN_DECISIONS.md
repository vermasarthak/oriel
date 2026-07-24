# Design Decisions

This document records the key engineering decisions in Oriel and the reasoning behind each. It is written for engineers who want to understand *why* the system works the way it does, not just how.

---

## ADR-0001: Quality-before-cost routing

**Status:** Accepted

**Context.** The naive approach to model routing is to pick the cheapest model that meets a latency budget. This is dangerous: a cheap model with 60% accuracy is worse than no routing at all if users rely on the output.

**Decision.** Candidates are filtered in two hard passes before any cost/latency optimisation occurs:

1. The Wilson lower confidence bound must exceed `minimum_quality`.
2. Mean latency and mean cost must fall within the declared constraints.

Only then is Thompson sampling used to select among eligible candidates.

**Consequences.** A new model with zero evaluation evidence will always return `no_eligible_candidate`. This is the correct behaviour: cold-start models must earn the right to receive traffic.

---

## ADR-0002: Wilson score interval over simple accuracy

**Status:** Accepted

**Context.** A model with 1 evaluation trial and 1 success has 100% observed accuracy. A model with 100 trials and 93 successes has 93% observed accuracy. The naive router would prefer the first model under a 95% quality threshold.

**Decision.** We use the Wilson score interval lower bound (z=1.96, 95% CI) as the quality metric for routing eligibility. This penalises low-sample-count candidates by widening the confidence interval, making the lower bound conservative.

Formulation:

```
lb = (p̂ + z²/2n − z·√((p̂(1−p̂) + z²/4n)/n)) / (1 + z²/n)
```

Where `p̂ = successes / total` and `z = 1.96`.

**Consequences.**

- A 1/1 model has a Wilson lower bound of ~0.21, not 1.0.
- A 100/100 model has a Wilson lower bound of ~0.996.
- Models need roughly 30+ trials before the bound meaningfully converges toward observed accuracy.

**Limitations.** The bound assumes i.i.d. Bernoulli trials. Correlated evaluation cases (e.g. all refund-type queries) will inflate confidence artificially.

---

## ADR-0003: Thompson sampling for exploration within eligible candidates

**Status:** Accepted

**Context.** Among eligible candidates, pure cost/latency optimisation would always pick the same model. This eliminates the ability to discover improving models and creates brittleness if the selected model degrades.

**Decision.** For each eligible candidate, we sample a quality estimate from a Beta distribution parameterised by its empirical successes and failures. The candidate with the highest sampled quality minus a weighted cost/latency penalty is selected.

```python
sampled = rng.betavariate(successes + 1, failures + 1)
score = sampled - cost_weight * mean_cost - latency_weight * mean_latency
```

The Beta prior (α=1, β=1) is a uniform prior over quality, allowing any model to win with small probability even if its point estimate is lower.

**Consequences.** Routing is not fully deterministic — identical requests may route to different models across calls. This is intentional: it enables passive A/B evidence collection. If determinism is required, set `rng = random.Random(fixed_seed)`.

**Limitations.** Thompson sampling without a hard exploration budget can waste traffic on clearly inferior models. At small evaluation counts (<10 trials), exploration risk is high. See `LIMITATIONS.md`.

---

## ADR-0004: Immutable trial evidence

**Status:** Accepted

**Context.** Mutable evidence allows silent data corruption: a re-run evaluation could overwrite a prior failure with a success, making a bad model look reliable.

**Decision.** The `trials` table uses a composite primary key of `(tenant_id, task, case_id, model, prompt_version, input_hash)`. Any attempt to insert a duplicate raises `ValueError("duplicate immutable trial")`. Evidence can only grow; it cannot be modified.

**Consequences.** Improving a prompt requires bumping the `prompt_version`. Old evidence is preserved and queryable for comparison. Storage grows monotonically — this is expected and is a feature, not a bug.

---

## ADR-0005: Binary success labels over scalar scoring

**Status:** Accepted (with known limitations)

**Context.** Richer quality signals — BLEU scores, embedding similarity, human preference ratings — are more informative. But they require a scoring function that is either expensive (LLM-as-judge), domain-specific, or both.

**Decision.** v1 uses binary evaluation: a model output either satisfies all required JSON fields and values or it does not. This keeps the evaluation deterministic, fast, reproducible, and dependency-free.

**Consequences.** The evaluator misses partial credit. A model that gets 3/4 required fields correct scores the same as one that returns garbage. This is a known simplification documented in `LIMITATIONS.md`. Continuous scoring is a planned v2 extension.

---

## ADR-0006: Single-file SQLite evidence store

**Status:** Accepted

**Context.** The alternatives are a remote time-series database (complex ops, latency, cost), an in-memory-only store (no durability), or an embedded relational DB.

**Decision.** SQLite with WAL mode provides durable, file-backed, zero-ops storage that is fast enough for the evaluation workloads Oriel targets (<1M trials/day per tenant). The `check_same_thread=False` flag is set for API thread-pool compatibility; callers are responsible for not issuing concurrent writes (FastAPI's synchronous thread pool serialises these naturally).

**Consequences.** Horizontal scaling to multiple API processes requires a coordination strategy (e.g. a shared NFS mount, or migrating to PostgreSQL). This is out of scope for v1.

---

## ADR-0007: Fake deterministic provider for tests and demo

**Status:** Accepted

**Context.** Tests that call real providers are slow, expensive, flaky, and require credentials. CI cannot hold credentials safely.

**Decision.** The `DeterministicFakeModel` returns a fixed, configurable JSON response with controllable latency and cost values. All correctness tests use this provider. The `OpenAIProvider` is integration-tested only when `OPENAI_API_KEY` is present in the environment.

**Consequences.** The fake provider does not exercise provider error paths (timeouts, rate limits, auth failures). These are covered by a separate provider timeout test that mocks the HTTP layer.

---

## ADR-0008: No raw prompts in logs

**Status:** Accepted

**Context.** Prompts may contain PII (user queries, account details). Logging raw prompts creates a compliance and privacy risk.

**Decision.** The trial store records only `input_hash` (SHA-256 of the input text), not the raw input. The hash is sufficient for deduplication. The API layer must not log request bodies containing user data.

**Consequences.** Debugging evaluation failures requires the caller to re-hash their input to find the corresponding trial. This is an intentional trade-off.
