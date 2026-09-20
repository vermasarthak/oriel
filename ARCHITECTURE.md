# Oriel Architecture

Oriel is an evaluation-driven model router: it routes every inference request to the model-and-prompt-version pair with the strongest *statistical evidence* of quality. This document describes how data flows through the system, how routing decisions are made, why key design choices were made, and where the current limitations lie.

---

## Table of Contents

1. [High-level data flow](#1-high-level-data-flow)
2. [Storage layer — TrialStore and SQLite](#2-storage-layer--trialstore-and-sqlite)
3. [Wilson confidence lower bound](#3-wilson-confidence-lower-bound)
4. [Routing — the `choose` function](#4-routing--the-choose-function)
5. [Fail-closed design](#5-fail-closed-design)
6. [API layer](#6-api-layer)
7. [Evaluation pipeline](#7-evaluation-pipeline)
8. [Known limitations and future work](#8-known-limitations-and-future-work)

---

## 1. High-level data flow

```
caller submits evaluation cases
           │
           ▼
  POST /v1/evaluation-runs
           │
           ▼
      ModelRunner
  (run each case, record pass/fail,
   latency_ms, cost_microusd per case)
           │
           ▼
      TrialStore.record()
  (immutable SQLite row per
   tenant × task × case × model × prompt_version × input_hash)
           │
           ▼ (accumulates over many runs)
      TrialStore.aggregates()
  (GROUP BY model → successes, total, mean_latency_ms, mean_cost_microusd)
           │
           ▼
      wilson_lower_bound(successes, total)
  (statistical quality floor, 95% confidence)
           │
           ▼
      choose(candidates, constraints)
  (filter by minimum_quality / max_latency_ms / max_cost_microusd,
   then Thompson-sample the winner)
           │
           ▼
  POST /v1/route → { model, reason, quality_lower_bound }
```

Alternatively, individual trial outcomes can be submitted directly via `POST /v1/outcomes` without running the evaluation pipeline — useful for recording production observations.

---

## 2. Storage layer — TrialStore and SQLite

**File:** [`src/oriel/store.py`](src/oriel/store.py)

### Schema

```sql
CREATE TABLE IF NOT EXISTS trials (
    tenant_id        TEXT NOT NULL,
    task             TEXT NOT NULL,
    case_id          TEXT NOT NULL,
    model            TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    input_hash       TEXT NOT NULL,   -- SHA-256 of input_text
    passed           INTEGER NOT NULL CHECK (passed IN (0,1)),
    latency_ms       REAL NOT NULL CHECK (latency_ms >= 0),
    cost_microusd    REAL NOT NULL CHECK (cost_microusd >= 0),
    PRIMARY KEY (tenant_id, task, case_id, model, prompt_version, input_hash)
)
```

The composite primary key enforces **trial immutability**: the same (tenant, task, case, model, prompt_version, input) tuple can only be recorded once. Attempting to record it twice raises `ValueError("duplicate immutable trial")`. This prevents accidentally inflating success counts by replaying the same case.

`input_hash` is `SHA-256(input_text)` — the raw text is never stored, keeping the table compact regardless of prompt size.

### Aggregation

`TrialStore.aggregates(tenant_id, task, prompt_version)` runs a single `GROUP BY model` query:

```sql
SELECT model, SUM(passed), COUNT(*), AVG(latency_ms), AVG(cost_microusd)
FROM trials
WHERE tenant_id=? AND task=? AND prompt_version=?
GROUP BY model
```

This returns one `Aggregate` dataclass per model, containing the raw statistics needed by the router.

---

## 3. Wilson confidence lower bound

**File:** [`src/oriel/router.py`](src/oriel/router.py)

The router uses the **Wilson score interval** (95% confidence, one-sided lower bound) to estimate the minimum true pass rate for a model given its observed trials.

### Formula

Given:
- $n$ = total trials
- $s$ = successes
- $\hat{p} = s / n$ = observed proportion
- $z = 1.96$ (95% confidence, two-tailed z-score)

$$
\text{WilsonLower}(s, n) = \frac{\hat{p} + \dfrac{z^2}{2n} - z\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}
$$

**Implementation** (exact code):

```python
def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - spread) / denominator
```

### Why Wilson and not raw accuracy?

Raw accuracy is a point estimate with no uncertainty quantification. A model with 1 trial and 100% accuracy has raw accuracy = 1.0 but Wilson lower bound ≈ 0.21 — correctly reflecting that a single observation is not reliable evidence. Wilson prevents Oriel from routing to undertested models that happen to look good on a tiny sample.

---

## 4. Routing — the `choose` function

**File:** [`src/oriel/router.py`](src/oriel/router.py)

```python
def choose(candidates: list[Aggregate], constraints: Constraints, rng: random.Random) -> Decision:
```

### Step 1 — Hard filtering

Each candidate is checked against three hard constraints:
1. `wilson_lower_bound(successes, total) >= minimum_quality`
2. `mean_latency_ms <= max_latency_ms`
3. `mean_cost_microusd <= max_cost_microusd`

Candidates that fail any constraint are **silently excluded** from the eligible set. If *no* candidate clears all three, `choose` returns immediately with `reason="no_eligible_candidate"`.

### Step 2 — Thompson sampling among eligible candidates

Among the eligible candidates, the winner is chosen by maximising a composite score:

```python
sampled_quality = rng.betavariate(successes + 1, total - successes + 1)
score = sampled_quality - cost_weight * mean_cost_microusd - latency_weight * mean_latency_ms
```

- **Thompson sampling** (`betavariate(α=s+1, β=n-s+1)`) draws a random sample from the Beta posterior over the true pass rate. This naturally balances exploitation (high-evidence winners) with exploration (undertested candidates), without requiring explicit exploration logic.
- **Cost and latency penalties** are subtracted with configurable weights (`cost_weight=0.001`, `latency_weight=0.0001` by default), so cheaper/faster models are preferred at equal quality.

The returned `Decision` includes the selected model name, `reason="eligible"`, and the **Wilson lower bound** (not the sampled quality) as `quality_lower_bound` — because the lower bound is a deterministic, auditable figure while the sampled quality varies per call.

---

## 5. Fail-closed design

Oriel **never silently falls back** to a weak or untested model. If no candidate meets the quality floor:

```json
{ "model": null, "reason": "no_eligible_candidate", "quality_lower_bound": null }
```

The caller receives an explicit signal and must decide what to do (e.g. use a default, queue for manual review, raise an alert). The design rationale is:

- **Silent fallbacks are dangerous in AI systems.** If a model starts degrading and Oriel silently routed around it to whatever is left, callers would receive lower-quality outputs with no indication that routing had degraded.
- **Fail-closed makes degradation observable.** A spike in `no_eligible_candidate` responses is a clear operational signal that all candidates have fallen below the minimum quality bar.
- **The quality bar is caller-controlled.** Callers choose `minimum_quality` per request, so they accept the tradeoff between availability (`minimum_quality=0.0` always routes) and correctness (`minimum_quality=0.95` is strict).

---

## 6. API layer

**File:** [`src/oriel/api.py`](src/oriel/api.py)

### Authentication

All endpoints (except `/healthz` and `/metrics`) require:

```
Authorization: Bearer <token>
```

The token is compared against `ORIEL_API_KEY` (env var, defaults to `test-token` in development). The current implementation maps every valid token to a single hardcoded `tenant-1`. Multi-tenant key management is future work.

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/evaluation-runs` | ✅ | Run evaluation cases against a model, record trials |
| `POST` | `/v1/outcomes` | ✅ | Record a single pre-evaluated outcome (async, background task) |
| `POST` | `/v1/route` | ✅ | Request a routing decision for a task+prompt_version |
| `GET` | `/healthz` | ❌ | Liveness — returns `{"status":"ok"}` |
| `GET` | `/readyz` | ❌ | Readiness — executes `SELECT 1` against the database |
| `GET` | `/metrics` | ❌ | Prometheus-style metrics (currently stub) |

### Request / response shapes

#### `POST /v1/evaluation-runs`

```json
Request:
{
  "task": "intent-routing",
  "model_name": "gpt-4o-mini",
  "prompt_version": "v1",
  "cases": [
    { "id": "c1", "input": "I want a refund", "required": { "intent": "refund" } }
  ],
  "use_real_provider": false
}

Response 200:
{ "total": 1, "passed": 1 }
```

#### `POST /v1/outcomes`

```json
Request:
{
  "task": "intent-routing", "case_id": "c1", "model": "gpt-4o-mini",
  "prompt_version": "v1", "input_text": "I want a refund",
  "passed": true, "latency_ms": 320, "cost_microusd": 200
}

Response 202:
{ "status": "accepted" }
```

The write is offloaded to a FastAPI `BackgroundTask`. The API responds immediately; duplicate submissions are silently ignored.

#### `POST /v1/route`

```json
Request:
{
  "task": "intent-routing", "prompt_version": "v1",
  "minimum_quality": 0.85, "max_latency_ms": 500, "max_cost_microusd": 1000
}

Response 200:
{ "model": "gpt-4o-mini", "reason": "eligible", "quality_lower_bound": 0.891 }

— or, if no candidate is eligible —

{ "model": null, "reason": "no_eligible_candidate", "quality_lower_bound": null }
```

---

## 7. Evaluation pipeline

**Files:** [`src/oriel/runner.py`](src/oriel/runner.py), [`src/oriel/evaluator.py`](src/oriel/evaluator.py), [`src/oriel/dataset.py`](src/oriel/dataset.py)

The pipeline executes when `POST /v1/evaluation-runs` is called:

1. **`run_cases(store, tenant_id, task, model_name, prompt_version, model, cases)`**  
   Iterates over `Case` objects. For each case, calls `model.generate(input_text)` → `(output, latency_ms, cost_microusd)`.

2. **`evaluate_json_object(output, required)`**  
   Parses the model output as JSON and checks that every key in `required` is present with the exact expected value. Returns `Evaluation(passed=True/False, reason=...)`.

3. Records the trial in `TrialStore` via `store.record(...)`.

The `Model` protocol is:
```python
class Model(Protocol):
    def generate(self, input_text: str) -> tuple[str, float, float]: ...
    #                                              output  latency cost
```

This makes it easy to swap in real providers (`OpenAIProvider`) or deterministic fakes (`DeterministicFakeModel`) without changing the evaluation pipeline.

---

## 8. Known limitations and future work

### SQLite single-writer limitation

SQLite uses a **file-level write lock**. In a multi-process or multi-instance deployment, concurrent writers will serialize or fail with `database is locked` errors. This means Oriel in its current form cannot be horizontally scaled.

**Future work:** Add an optional Postgres backend (`PostgresTrialStore`) selected by `DATABASE_URL`. The same `record` / `aggregates` interface would be implemented using a connection pool (`psycopg[binary]`), allowing multiple Oriel instances to write safely. See [FIX 7](https://github.com/vermasarthak/oriel/pulls) in the roadmap.

### Single-tenant key mapping

The current auth layer maps every valid API key to `tenant-1`. A production system needs per-key tenant resolution (e.g. from a database table) and key rotation.

### Evaluation contract is binary

The evaluator checks JSON field equality — useful for structured tasks like intent classification, but not for open-ended generation. Future work includes embedding-similarity or LLM-judge based evaluation.

### No model-level versioning isolation

`prompt_version` is caller-supplied and task-scoped. There is no enforcement that the same `prompt_version` string means the same prompt template — callers must manage this discipline externally.

### Thompson sampling is unseeded per process

`rng = random.Random()` in `api.py` is initialised without a seed, giving non-deterministic routing for repeated identical requests. This is intentional for production (exploration), but makes integration tests harder without dependency injection of the RNG.

### Metrics are a stub

`GET /metrics` returns `oriel_up 1\n`. Real Prometheus metrics (route decision counts, no-eligible-candidate rate, trial counts) are future work.
