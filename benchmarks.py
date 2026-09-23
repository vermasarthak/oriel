"""
Oriel benchmarks — run with: python benchmarks.py
Records exact results to stdout; pipe to BENCHMARKS.md or compare runs.
"""
import gc
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[0] / "src"))

from oriel.router import Constraints, choose, wilson_lower_bound
from oriel.store import Aggregate, TrialStore

HARDWARE = "Apple M-series (arm64), Python 3.14, single process, in-process SQLite"

def separator(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ── 1. Evaluation-recording throughput ─────────────────────────

def bench_write_throughput():
    separator("1. Evaluation-recording throughput (in-memory SQLite)")
    store = TrialStore(":memory:")
    n = 10_000
    gc.collect()
    start = time.perf_counter()
    for i in range(n):
        store.record("t1", "task", f"case-{i}", "model-a", "v1", f"unique-input-{i}", True, 100.0, 50.0)
    elapsed = time.perf_counter() - start
    qps = n / elapsed
    print(f"  {n:,} writes in {elapsed:.3f}s → {qps:,.0f} writes/sec")
    print(f"  Per-write latency: {elapsed / n * 1_000_000:.1f} µs")


# ── 2. Routing-decision latency ─────────────────────────────────

def bench_routing_latency():
    separator("2. Routing-decision latency")
    constraints = Constraints(minimum_quality=0.7, max_latency_ms=500, max_cost_microusd=500)
    n_decisions = 10_000

    for n_candidates in [1, 10, 100, 1_000]:
        candidates = [
            Aggregate(f"model-{i}", 90, 100, 100.0, 50.0)
            for i in range(n_candidates)
        ]
        rng = random.Random(42)
        gc.collect()
        start = time.perf_counter()
        for _ in range(n_decisions):
            choose(candidates, constraints, rng)
        elapsed = time.perf_counter() - start
        per_decision_us = (elapsed / n_decisions) * 1_000_000
        print(f"  {n_candidates:>5} candidates × {n_decisions:,} decisions → {per_decision_us:.1f} µs/decision")


# ── 3. Database growth per trial ────────────────────────────────

def bench_db_growth():
    separator("3. Database growth per trial (file-backed SQLite)")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = TrialStore(db_path)
    checkpoints = [0, 100, 1_000, 5_000, 10_000]
    prev_size = Path(db_path).stat().st_size
    prev_n = 0
    i = 0
    for checkpoint in checkpoints[1:]:
        while i < checkpoint:
            store.record("t1", "task", f"c-{i}", "model-a", "v1", f"input-{i}", i % 3 != 0, 100, 50)
            i += 1
        size = Path(db_path).stat().st_size
        delta = size - prev_size
        added = checkpoint - prev_n
        bytes_per_trial = delta / added if added > 0 else 0
        print(f"  {checkpoint:>6} trials → {size / 1024:.1f} KB total | ~{bytes_per_trial:.0f} bytes/trial")
        prev_size = size
        prev_n = checkpoint

    import os
    os.unlink(db_path)


# ── 4. Wilson lower bound computation throughput ─────────────────

def bench_wilson():
    separator("4. Wilson lower bound computation throughput")
    n = 1_000_000
    gc.collect()
    start = time.perf_counter()
    for i in range(n):
        wilson_lower_bound(i % 100, 100)
    elapsed = time.perf_counter() - start
    print(f"  {n:,} computations in {elapsed:.3f}s → {n / elapsed:,.0f} calls/sec")
    print(f"  Per-call latency: {elapsed / n * 1_000_000:.3f} µs")


# ── 5. Reproducibility with fixed seed ──────────────────────────

def bench_reproducibility():
    separator("5. Reproducibility with fixed seed")
    candidates = [
        Aggregate("gpt-4o", 95, 100, 800, 10000),
        Aggregate("gpt-4o-mini", 80, 100, 300, 300),
        Aggregate("gpt-3.5-turbo", 70, 100, 400, 150),
    ]
    constraints = Constraints(minimum_quality=0.6, max_latency_ms=1000, max_cost_microusd=12000)

    results = []
    for _ in range(3):
        rng = random.Random(7)
        run = [choose(candidates, constraints, rng).model for _ in range(10)]
        results.append(run)

    all_same = all(r == results[0] for r in results)
    print("  3 runs with seed=7, 10 decisions each:")
    print(f"  Run 1: {results[0]}")
    print(f"  Run 2: {results[1]}")
    print(f"  Run 3: {results[2]}")
    print(f"  Fully reproducible: {all_same}")


if __name__ == "__main__":
    print("\nOriel Benchmarks")
    print(f"Hardware: {HARDWARE}")
    bench_write_throughput()
    bench_routing_latency()
    bench_db_growth()
    bench_wilson()
    bench_reproducibility()
    print()
