import random
import time
from oriel.store import TrialStore
from oriel.router import Constraints, choose

def run_benchmarks():
    store = TrialStore()
    start_time = time.time()
    
    # 1. Evaluation-recording throughput
    num_trials = 10000
    for i in range(num_trials):
        store.record("t1", "task1", f"case-{i}", "model-a", "v1", f"input-{i}", True, 100.0, 50.0)
    
    end_time = time.time()
    write_qps = num_trials / (end_time - start_time)
    print(f"Evaluation-recording throughput: {write_qps:.2f} writes/sec")
    
    # 2. Routing-decision latency with increasing candidate count
    rng = random.Random(42)
    constraints = Constraints(minimum_quality=0.5, max_latency_ms=500.0, max_cost_microusd=500.0)
    
    for num_candidates in [10, 100, 1000]:
        candidates = []
        for i in range(num_candidates):
            candidates.append(
                # (model, successes, total, latency, cost)
                __import__("oriel.store", fromlist=["Aggregate"]).Aggregate(f"model-{i}", 90, 100, 100.0, 50.0)
            )
        
        start_time = time.time()
        num_decisions = 1000
        for _ in range(num_decisions):
            choose(candidates, constraints, rng)
        end_time = time.time()
        
        latency_ms = ((end_time - start_time) / num_decisions) * 1000
        print(f"Routing-decision latency ({num_candidates} candidates): {latency_ms:.3f} ms/decision")

if __name__ == "__main__":
    run_benchmarks()
