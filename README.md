# Oriel

**Evaluation-driven model router for structured AI tasks.**

Oriel routes every request to the model-and-prompt version with the strongest *statistical evidence* of quality — not the cheapest model, not the fastest, and never based on a sample of one.

[![CI](https://github.com/vermasarthak/oriel/actions/workflows/ci.yml/badge.svg)](https://github.com/vermasarthak/oriel/actions/workflows/ci.yml)

---

## The problem

Most "model routers" route by cost or latency. If the cheap model starts hallucinating, they keep routing there — because they have no structured mechanism to detect it.

Oriel routes differently:

1. **Evidence first.** Every model-prompt pair accumulates immutable evaluation trials in SQLite before it can serve traffic.
2. **Statistical honesty.** Routing eligibility is gated on a Wilson 95% lower confidence bound over binary trial outcomes — not raw accuracy. A 1-trial model with 100% accuracy has a Wilson lower bound of ~0.21 and will not route.
3. **Fail closed.** If no candidate clears the quality floor, Oriel returns `no_eligible_candidate` explicitly. It never silently falls back to a weak model.
4. **Explainable decisions.** Every routing response includes the model selected, sample count, successes, Wilson lower bound, constraints applied, and which candidates were rejected and why.

---

## Quick start

```bash
git clone https://github.com/vermasarthak/oriel
cd oriel
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

### Storage backends

**SQLite (default — no extra dependencies)**

SQLite is used by default. All data is stored in a local file or in-memory:

```bash
# In-memory (default, resets on restart)
uvicorn oriel.api:app

# Persistent file
ORIEL_DB_PATH=/var/lib/oriel/oriel.db uvicorn oriel.api:app
```

> **Limitation:** SQLite uses a file-level write lock and cannot be shared across multiple processes or machines. Use Postgres for multi-instance deployments.

**Postgres (optional — recommended for production)**

Install the Postgres extra and set `DATABASE_URL`:

```bash
pip install "oriel[postgres]"

DATABASE_URL=postgresql://user:password@localhost:5432/oriel \
  uvicorn oriel.api:app
```

The Postgres backend creates the `trials` table automatically on startup. It uses the same `record` / `aggregates` interface as the SQLite backend — no application code changes required.

To run a local Postgres instance:

```bash
docker run -d \
  -e POSTGRES_USER=oriel -e POSTGRES_PASSWORD=oriel_dev_only -e POSTGRES_DB=oriel \
  -p 5432:5432 postgres:15

DATABASE_URL=postgresql://oriel:oriel_dev_only@localhost:5432/oriel \
  uvicorn oriel.api:app
```


### 1. Submit evaluation evidence

```bash
curl -X POST http://localhost:8000/v1/outcomes \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "intent-routing",
    "case_id": "c1",
    "model": "gpt-4o-mini",
    "prompt_version": "v1",
    "input_text": "I want a refund",
    "passed": true,
    "latency_ms": 320,
    "cost_microusd": 200
  }'
```

### 2. Request a routing decision

```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "intent-routing",
    "prompt_version": "v1",
    "minimum_quality": 0.85,
    "max_latency_ms": 500,
    "max_cost_microusd": 1000
  }'
```

Response:

```json
{
  "model": "gpt-4o-mini",
  "reason": "eligible",
  "quality_lower_bound": 0.891
}
```

If no model meets the constraints:

```json
{
  "model": null,
  "reason": "no_eligible_candidate",
  "quality_lower_bound": null
}
```

### 3. Run the test suite

```bash
python -m unittest discover -s tests -v
# Ran 45 tests in 0.113s — OK
```

### 4. Run benchmarks

```bash
python benchmarks.py
# → 71,454 writes/sec (evidence recording)
# → 3.2 µs/decision (routing, 1 candidate)
# → 308.9 µs/decision (routing, 100 candidates)
```

---

## Architecture

```
versioned JSONL cases
        │
        ▼
  model runner  ──────────────────────────────────────────────────────┐
        │                                                             │
        ▼                                                             │
structured evaluator (exact JSON field matching)                      │
        │                                                             ▼
        ▼                                                     outcome feedback
immutable SQLite trial store  ◄──────────────────────────────  POST /v1/outcomes
  (tenant_id, task, case_id,
   model, prompt_version,
   input_hash, passed,
   latency_ms, cost_microusd)
        │
        ▼
  aggregates per model
        │
        ▼
  Wilson lower bound ──► quality gate ──► Thompson sampling ──► decision evidence
                                │
                                └──► no_eligible_candidate
```

**Key design properties:**

| Property | Mechanism |
|---|---|
| Statistical honesty | Wilson 95% lower confidence bound |
| Small-sample penalty | 1/1 success → bound of ~0.21, not 1.0 |
| Immutable evidence | SQLite composite PK rejects duplicates |
| Tenant isolation | All queries scoped by `tenant_id` |
| Fail-closed | Explicit `no_eligible_candidate` — no silent fallback |
| Exploration | Thompson sampling (Beta distribution) among eligible candidates |
| Determinism | Fixed-seed `random.Random` for reproducible replay |

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/evaluation-runs` | POST | Run a dataset through a model and record evidence |
| `/v1/route` | POST | Request a routing decision with constraints |
| `/v1/outcomes` | POST | Submit delayed real-world outcome feedback |
| `/healthz` | GET | Liveness check |
| `/readyz` | GET | Readiness check (DB available) |
| `/metrics` | GET | Prometheus-compatible metrics |

All endpoints except `/healthz` require `Authorization: Bearer <token>`.

---

## One real measured result

> On Apple M-series (arm64), Python 3.14, single process:
> - Evidence recording: **71,454 writes/sec** (14 µs/write)
> - Routing decision: **3.2 µs** with 1 candidate, **308.9 µs** with 100 candidates
> - Database growth: **~212 bytes/trial** (10,000 trials = 2.1 MB)
> - Wilson bound computation: **2.9 million calls/sec**
> - Routing fully reproducible with fixed seed: **verified**
>
> See [BENCHMARKS.md](BENCHMARKS.md) for full methodology and reproduction instructions.

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — full architecture: data flow, Wilson formula, fail-closed design, API reference, limitations
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) — ADR-style rationale for Wilson bounds, Thompson sampling, immutable evidence, binary labels, SQLite choice
- [BENCHMARKS.md](BENCHMARKS.md) — real measured throughput and latency
- [SECURITY.md](SECURITY.md) — threat model and security controls
- [LIMITATIONS.md](LIMITATIONS.md) — explicit known limitations; when not to trust the routing policy
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to extend Oriel

---

## What Oriel is not

- **Not a generic LLM wrapper.** It has no chat interface, no agent loop, and no prompt templating.
- **Not a multi-provider abstraction layer.** The provider interface is intentionally minimal. v1 ships with one real adapter (OpenAI) and a deterministic fake for tests.
- **Not a black-box optimizer.** Every routing decision is fully explainable from the evidence in the store.

---

## Statistical caveats

The Wilson bound assumes independent Bernoulli trials. The following conditions will make Oriel's quality estimates unreliable:

- Correlated evaluation cases (e.g. all queries are semantically similar)
- Dataset shift (production inputs differ from evaluation inputs)
- Biased feedback (only successful outcomes are reported to `/v1/outcomes`)
- Silent model degradation without a prompt version bump

See [LIMITATIONS.md](LIMITATIONS.md) for a full treatment.

---

## License

MIT

<!-- Architecture metric sync for oriel -->

<!-- Benchmark metric log for oriel -->
