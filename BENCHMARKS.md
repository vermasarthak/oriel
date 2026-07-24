# Benchmarks

All results are real measurements from the hardware listed below. No numbers have been invented or estimated.

## Hardware and environment

| Field | Value |
|---|---|
| Machine | Apple M-series (arm64) |
| Python | 3.14.7 |
| SQLite | bundled with Python stdlib |
| Concurrency | single process, no threads |
| Methodology | `time.perf_counter()` wall-clock; `gc.collect()` before each run |
| Command | `python benchmarks.py` from repo root |

---

## 1. Evaluation-recording throughput (in-memory SQLite)

Records how fast Oriel can persist trial evidence.

```
10,000 writes in 0.140s → 71,454 writes/sec
Per-write latency: 14.0 µs
```

**Methodology.** Each write calls `store.record(...)`, which hashes the input, inserts a row, and commits. In-memory SQLite; no I/O on disk.

**Limitation.** In-memory SQLite does not reflect disk-backed latency. File-backed throughput is lower and depends on fsync behaviour and storage speed.

---

## 2. Routing-decision latency

Measures how fast `choose()` selects among N eligible candidates. This is the hot path on every `/v1/route` call.

| Candidates | Latency per decision |
|---|---|
| 1 | 3.2 µs |
| 10 | 30.1 µs |
| 100 | 308.9 µs |
| 1,000 | 2,152.5 µs (2.15 ms) |

**Methodology.** 10,000 decisions per candidate count with a fixed RNG seed. All candidates pass all constraints (quality, latency, cost), so the full Beta sampling and scoring loop runs for every candidate.

**Practical implication.** At 100 candidate models, routing adds ≈0.3 ms to request latency. For most applications this is negligible. If you operate >100 models per task, consider partitioning by task sub-type.

---

## 3. Database growth per trial (file-backed SQLite)

| Trials | Total size | Bytes per trial |
|---|---|---|
| 100 | 36.0 KB | 246 bytes |
| 1,000 | 216.0 KB | 205 bytes |
| 5,000 | 1,060.0 KB | 216 bytes |
| 10,000 | 2,096.0 KB | 212 bytes |

**Methodology.** A real SQLite file on disk, no WAL tuning. Rows include the full primary key (tenant_id, task, case_id, model, prompt_version, input_hash) plus three metric columns.

**Practical implication.** At 212 bytes/trial, 1 million trials ≈ 200 MB. A typical deployment evaluating 10 models × 100 cases × 10 prompt versions = 10,000 trials ≈ 2 MB. Storage is not a concern for years of normal use.

---

## 4. Wilson lower bound computation throughput

This runs on every candidate during routing.

```
1,000,000 computations in 0.342s → 2,922,898 calls/sec
Per-call latency: 0.342 µs
```

**Methodology.** Calls `wilson_lower_bound(i % 100, 100)` in a tight loop. Pure Python + `math.sqrt`.

---

## 5. Reproducibility with fixed seed

Routing decisions must be fully reproducible when `random.Random` is initialised with a fixed seed. This is required for deterministic replay tests.

```
3 runs, seed=7, 10 decisions each:
Run 1: ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-3.5-turbo', ...]
Run 2: ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-3.5-turbo', ...]
Run 3: ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-3.5-turbo', ...]
Fully reproducible: True
```

---

## Reproducing these results

```bash
git clone https://github.com/vermasarthak/oriel
cd oriel
python3 -m venv venv && source venv/bin/activate
pip install -e ".[ui]"
python benchmarks.py
```

Results will vary with hardware. Do not compare numbers across machines without re-running the benchmarks.
