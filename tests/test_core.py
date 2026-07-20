import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from oriel.dataset import load_jsonl
from oriel.evaluator import evaluate_json_object
from oriel.router import Constraints, choose, wilson_lower_bound
from oriel.runner import run_cases
from oriel.store import Aggregate, TrialStore


class CoreTests(unittest.TestCase):
    def test_invalid_json_is_a_failure(self) -> None:
        self.assertEqual(evaluate_json_object("not json", {"answer": "yes"}).reason, "invalid_json")

    def test_trials_are_immutable(self) -> None:
        store = TrialStore()
        store.record("tenant-1", "classification", "one", "model-a", "v1", "input", True, 10, 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            store.record("tenant-1", "classification", "one", "model-a", "v1", "input", False, 10, 2)

    def test_router_rejects_sparse_perfect_candidate(self) -> None:
        sparse = Aggregate("cheap", 1, 1, 1, 1)
        proven = Aggregate("reliable", 98, 100, 100, 100)
        constraints = Constraints(minimum_quality=0.9, max_latency_ms=200, max_cost_microusd=200)
        decision = choose([sparse, proven], constraints, random.Random(7))
        self.assertEqual(decision.model, "reliable")
        self.assertLess(wilson_lower_bound(1, 1), 0.9)

    def test_runner_records_versioned_evaluation_evidence(self) -> None:
        class FakeModel:
            def generate(self, input_text: str) -> tuple[str, float, float]:
                return ('{"intent":"refund","priority":"high"}', 3, 7)

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.jsonl"
            dataset.write_text('{"id":"refund","input":"refund please","required":{"intent":"refund","priority":"high"}}\n')
            store = TrialStore()
            summary = run_cases(store, "tenant-1", "intent", "fake-v1", "prompt-1", FakeModel(), load_jsonl(dataset))
        self.assertEqual(summary.passed, 1)
        self.assertEqual(store.aggregates("tenant-1", "intent", "prompt-1")[0].successes, 1)
