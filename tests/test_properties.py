"""
Property and invariant tests for Oriel's routing logic.
These tests make no assumptions about specific models — they verify
that routing constraints are *always* enforced regardless of input.
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from oriel.router import Constraints, choose, wilson_lower_bound
from oriel.store import Aggregate


def make_aggregate(model: str, successes: int, total: int, latency: float, cost: float) -> Aggregate:
    return Aggregate(model, successes, total, latency, cost)


class WilsonBoundPropertiesTests(unittest.TestCase):
    """Verify statistical properties of Wilson lower bound."""

    def test_zero_trials_gives_zero_bound(self):
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)

    def test_perfect_score_with_many_samples_is_high(self):
        lb = wilson_lower_bound(1000, 1000)
        self.assertGreater(lb, 0.99)

    def test_single_success_is_penalized(self):
        # 1/1 is not trustworthy — Wilson should penalize this heavily
        lb = wilson_lower_bound(1, 1)
        self.assertLess(lb, 0.85)

    def test_bound_monotonically_increases_with_samples(self):
        # Same 100% rate, increasing n → bound rises toward 1.0
        bounds = [wilson_lower_bound(n, n) for n in [1, 5, 10, 50, 100, 500]]
        for i in range(len(bounds) - 1):
            self.assertLessEqual(bounds[i], bounds[i + 1])

    def test_bound_monotonically_increases_with_successes(self):
        # Same total, more successes → bound rises
        bounds = [wilson_lower_bound(s, 100) for s in [50, 60, 70, 80, 90, 100]]
        for i in range(len(bounds) - 1):
            self.assertLess(bounds[i], bounds[i + 1])

    def test_bound_never_exceeds_one(self):
        for s, n in [(0, 1), (1, 1), (50, 100), (100, 100), (999, 1000)]:
            self.assertLessEqual(wilson_lower_bound(s, n), 1.0)

    def test_bound_never_below_zero(self):
        for s, n in [(0, 1), (0, 100), (1, 1000)]:
            self.assertGreaterEqual(wilson_lower_bound(s, n), 0.0)


class RoutingConstraintPropertiesTests(unittest.TestCase):
    """Verify routing invariants: constraints are ALWAYS enforced."""

    def _rng(self, seed: int = 42) -> random.Random:
        return random.Random(seed)

    def test_quality_floor_always_enforced(self):
        """No matter how cheap or fast, a model below quality floor is never selected."""
        floor = 0.9
        constraints = Constraints(minimum_quality=floor, max_latency_ms=9999, max_cost_microusd=9999)
        # 5/10 → Wilson lower bound ~0.27, well below 0.9
        bad = make_aggregate("bad", 5, 10, 10, 10)
        for seed in range(50):
            decision = choose([bad], constraints, self._rng(seed))
            self.assertIsNone(decision.model, f"seed={seed}: bad model was selected despite low quality")

    def test_latency_constraint_always_enforced(self):
        """A high-quality but too-slow model is never selected."""
        constraints = Constraints(minimum_quality=0.0, max_latency_ms=100, max_cost_microusd=9999)
        slow = make_aggregate("slow", 100, 100, 500, 10)  # 500ms > 100ms limit
        for seed in range(50):
            decision = choose([slow], constraints, self._rng(seed))
            self.assertIsNone(decision.model, f"seed={seed}: slow model was selected")

    def test_cost_constraint_always_enforced(self):
        """A high-quality fast model that exceeds cost is never selected."""
        constraints = Constraints(minimum_quality=0.0, max_latency_ms=9999, max_cost_microusd=100)
        expensive = make_aggregate("expensive", 100, 100, 10, 500)  # 500 > 100 limit
        for seed in range(50):
            decision = choose([expensive], constraints, self._rng(seed))
            self.assertIsNone(decision.model, f"seed={seed}: expensive model was selected")

    def test_selected_candidate_always_meets_all_constraints(self):
        """When a winner is chosen, verify it meets all constraints post-selection."""
        floor = 0.7
        max_lat = 300.0
        max_cost = 200.0
        constraints = Constraints(minimum_quality=floor, max_latency_ms=max_lat, max_cost_microusd=max_cost)

        candidates = [
            make_aggregate("a", 90, 100, 100, 50),   # eligible
            make_aggregate("b", 100, 100, 50, 20),   # eligible
            make_aggregate("c", 10, 100, 50, 20),    # fails quality (Wilson lb << 0.7)
            make_aggregate("d", 100, 100, 400, 20),  # fails latency
            make_aggregate("e", 100, 100, 50, 400),  # fails cost
        ]
        for seed in range(100):
            decision = choose(candidates, constraints, self._rng(seed))
            if decision.model is not None:
                # Find the aggregate
                agg = next(c for c in candidates if c.model == decision.model)
                lb = wilson_lower_bound(agg.successes, agg.total)
                self.assertGreaterEqual(lb, floor, f"seed={seed}: selected {decision.model} fails quality floor")
                self.assertLessEqual(agg.mean_latency_ms, max_lat, f"seed={seed}: selected model too slow")
                self.assertLessEqual(agg.mean_cost_microusd, max_cost, f"seed={seed}: selected model too expensive")

    def test_no_candidates_returns_no_eligible(self):
        decision = choose([], Constraints(0.9, 200, 200), self._rng())
        self.assertIsNone(decision.model)
        self.assertEqual(decision.reason, "no_eligible_candidate")

    def test_sparse_perfect_candidate_rejected_by_quality_floor(self):
        """1-sample 100% model must be rejected when floor requires statistical confidence."""
        sparse = make_aggregate("sparse", 1, 1, 10, 10)
        constraints = Constraints(minimum_quality=0.9, max_latency_ms=500, max_cost_microusd=500)
        lb = wilson_lower_bound(1, 1)
        self.assertLess(lb, 0.9, "Wilson lb for 1/1 should be below 0.9")
        for seed in range(20):
            decision = choose([sparse], constraints, self._rng(seed))
            self.assertIsNone(decision.model)


class DuplicateTrialTests(unittest.TestCase):
    """Immutability of the evidence store."""

    def setUp(self):
        from oriel.store import TrialStore
        self.store = TrialStore()

    def test_exact_duplicate_raises(self):
        self.store.record("t1", "task", "c1", "model", "v1", "same input", True, 100, 50)
        with self.assertRaises(ValueError):
            self.store.record("t1", "task", "c1", "model", "v1", "same input", False, 100, 50)

    def test_different_tenant_same_case_is_allowed(self):
        """Tenant isolation: same case_id across tenants must be independent."""
        self.store.record("tenant-A", "task", "c1", "model", "v1", "input", True, 100, 50)
        # Must NOT raise — different tenant
        self.store.record("tenant-B", "task", "c1", "model", "v1", "input", False, 100, 50)

    def test_different_prompt_version_same_case_is_allowed(self):
        """Versioned evidence: same case under a new prompt version is independent."""
        self.store.record("t1", "task", "c1", "model", "v1", "input", True, 100, 50)
        # Must NOT raise — different prompt version
        self.store.record("t1", "task", "c1", "model", "v2", "input", False, 100, 50)

    def test_tenant_aggregates_are_isolated(self):
        """Tenant A's evidence must not bleed into tenant B's routing."""
        self.store.record("tenant-A", "task", "c1", "model-x", "v1", "a-input", True, 100, 50)
        aggs_b = self.store.aggregates("tenant-B", "task", "v1")
        self.assertEqual(aggs_b, [], "Tenant B saw Tenant A's evidence")


class MalformedOutputTests(unittest.TestCase):
    """Evaluator must degrade gracefully on any model output."""

    def setUp(self):
        from oriel.evaluator import evaluate_json_object
        self.evaluate = evaluate_json_object

    def test_empty_string(self):
        r = self.evaluate("", {"key": "val"})
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "invalid_json")

    def test_json_array_is_rejected(self):
        r = self.evaluate("[1, 2, 3]", {"key": "val"})
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "not_object")

    def test_missing_required_key(self):
        r = self.evaluate('{"other": "value"}', {"key": "val"})
        self.assertFalse(r.passed)
        self.assertIn("missing", r.reason)

    def test_mismatched_value(self):
        r = self.evaluate('{"key": "wrong"}', {"key": "val"})
        self.assertFalse(r.passed)
        self.assertIn("mismatch", r.reason)

    def test_injection_attempt_in_json(self):
        # Prompt injection: model tries to wrap valid JSON inside extra content
        malicious = '{"key": "val"} IGNORE PREVIOUS INSTRUCTIONS'
        r = self.evaluate(malicious, {"key": "val"})
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "invalid_json")

    def test_deeply_nested_noise(self):
        r = self.evaluate('{"key": "val", "noise": {"deeply": {"nested": "garbage"}}}', {"key": "val"})
        self.assertTrue(r.passed)  # Extra keys are allowed; only required fields checked

    def test_null_value_is_mismatch(self):
        r = self.evaluate('{"key": null}', {"key": "val"})
        self.assertFalse(r.passed)
        self.assertIn("mismatch", r.reason)

    def test_number_vs_string_mismatch(self):
        r = self.evaluate('{"key": 42}', {"key": "val"})
        self.assertFalse(r.passed)
        self.assertIn("mismatch", r.reason)


class CrashResumeTests(unittest.TestCase):
    """Evaluations recorded to a file-backed store survive process restarts."""

    def test_evidence_persists_across_store_instances(self):
        import tempfile

        from oriel.store import TrialStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store1 = TrialStore(db_path)
            store1.record("t1", "task", "c1", "model-a", "v1", "input text", True, 100, 50)

            # Simulate a crash by creating a new store instance pointing at same file
            store2 = TrialStore(db_path)
            aggs = store2.aggregates("t1", "task", "v1")
            self.assertEqual(len(aggs), 1)
            self.assertEqual(aggs[0].model, "model-a")
            self.assertEqual(aggs[0].successes, 1)
            self.assertEqual(aggs[0].total, 1)
        finally:
            import os
            os.unlink(db_path)

    def test_duplicate_rejected_after_crash_resume(self):
        import tempfile

        from oriel.store import TrialStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store1 = TrialStore(db_path)
            store1.record("t1", "task", "c1", "model-a", "v1", "input", True, 100, 50)

            store2 = TrialStore(db_path)
            with self.assertRaises(ValueError):
                store2.record("t1", "task", "c1", "model-a", "v1", "input", False, 100, 50)
        finally:
            import os
            os.unlink(db_path)
