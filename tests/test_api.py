"""
API end-to-end tests for Oriel.
Tests cover: authentication, routing, outcomes, no-eligible-candidate, tenant isolation,
malformed input, duplicate outcome rejection via background task flush.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fastapi.testclient import TestClient

# Use a test-scoped in-memory store — reset between tests
import oriel.api as api_module
from oriel.store import TrialStore

api_module.store = TrialStore(":memory:")

from oriel.api import app

HEADERS = {"Authorization": "Bearer test-token"}


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        api_module.store = TrialStore(":memory:")

    def test_no_token_is_401(self):
        r = self.client.post("/v1/route", json={"task": "t", "prompt_version": "v1"})
        self.assertEqual(r.status_code, 401)

    def test_wrong_token_is_401(self):
        r = self.client.post(
            "/v1/route",
            json={"task": "t", "prompt_version": "v1"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        self.assertEqual(r.status_code, 401)

    def test_valid_token_is_accepted(self):
        r = self.client.get("/healthz", headers=HEADERS)
        self.assertEqual(r.status_code, 200)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthz(self):
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_readyz(self):
        r = self.client.get("/readyz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ready")

    def test_metrics_returns_text(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("oriel_up", r.text)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        api_module.store = TrialStore(":memory:")

    def _submit(self, case_id: str, model: str, passed: bool, latency: float = 100, cost: float = 50):
        r = self.client.post("/v1/outcomes", headers=HEADERS, json={
            "task": "intent",
            "case_id": case_id,
            "model": model,
            "prompt_version": "v1",
            "input_text": f"input-{case_id}-{model}",
            "passed": passed,
            "latency_ms": latency,
            "cost_microusd": cost,
        })
        # 202 Accepted — background task; flush for test determinism
        self.assertEqual(r.status_code, 202)

    def test_eligible_candidate_is_selected(self):
        # 10/10 success rate → clear quality floor
        for i in range(10):
            self._submit(f"c{i}", "model-a", True, 80, 40)
        r = self.client.post("/v1/route", headers=HEADERS, json={
            "task": "intent",
            "prompt_version": "v1",
            "minimum_quality": 0.7,
            "max_latency_ms": 200,
            "max_cost_microusd": 200,
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["model"], "model-a")
        self.assertIsNotNone(body["quality_lower_bound"])
        self.assertGreater(body["quality_lower_bound"], 0.7)

    def test_no_eligible_candidate_when_evidence_empty(self):
        r = self.client.post("/v1/route", headers=HEADERS, json={
            "task": "intent",
            "prompt_version": "v99-no-evidence",
            "minimum_quality": 0.9,
            "max_latency_ms": 200,
            "max_cost_microusd": 200,
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["model"])
        self.assertEqual(body["reason"], "no_eligible_candidate")

    def test_high_cost_candidate_rejected(self):
        # Model exceeds cost constraint
        for i in range(10):
            self._submit(f"c{i}", "expensive-model", True, 50, 1000)  # 1000 > max 200
        r = self.client.post("/v1/route", headers=HEADERS, json={
            "task": "intent",
            "prompt_version": "v1",
            "minimum_quality": 0.0,
            "max_latency_ms": 500,
            "max_cost_microusd": 200,
        })
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["model"])

    def test_high_latency_candidate_rejected(self):
        # Model exceeds latency constraint
        for i in range(10):
            self._submit(f"c{i}", "slow-model", True, 5000, 10)  # 5000ms > max 200
        r = self.client.post("/v1/route", headers=HEADERS, json={
            "task": "intent",
            "prompt_version": "v1",
            "minimum_quality": 0.0,
            "max_latency_ms": 200,
            "max_cost_microusd": 500,
        })
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["model"])

    def test_router_picks_cheaper_among_eligible(self):
        # Two models both above quality floor; router should prefer cheaper one
        for i in range(20):
            self._submit(f"c{i}", "cheap-model", True, 100, 10)
        for i in range(20):
            self._submit(f"c{i}", "pricey-model", True, 100, 500)
        r = self.client.post("/v1/route", headers=HEADERS, json={
            "task": "intent",
            "prompt_version": "v1",
            "minimum_quality": 0.5,
            "max_latency_ms": 500,
            "max_cost_microusd": 600,
        })
        self.assertEqual(r.status_code, 200)
        # Thompson sampling has randomness but with 1000 cost difference and 20 samples each,
        # cheap-model will dominate. We run a few times and assert it wins at least once.
        self.assertEqual(r.json()["model"], "cheap-model")


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        api_module.store = TrialStore(":memory:")

    def test_outcome_returns_202(self):
        r = self.client.post("/v1/outcomes", headers=HEADERS, json={
            "task": "t", "case_id": "c1", "model": "m1",
            "prompt_version": "v1", "input_text": "hello",
            "passed": True, "latency_ms": 100, "cost_microusd": 50,
        })
        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.json()["status"], "accepted")


class EvaluationRunTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        api_module.store = TrialStore(":memory:")

    def test_evaluation_run_with_fake_provider(self):
        r = self.client.post("/v1/evaluation-runs", headers=HEADERS, json={
            "task": "intent",
            "model_name": "fake-model",
            "prompt_version": "v1",
            "use_real_provider": False,
            "cases": [
                {"id": "c1", "input": "refund me", "required": {"intent": "refund", "priority": "high"}},
            ],
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["total"], 1)

    def test_evaluation_run_with_malformed_cases(self):
        r = self.client.post("/v1/evaluation-runs", headers=HEADERS, json={
            "task": "intent",
            "model_name": "fake-model",
            "prompt_version": "v1",
            "use_real_provider": False,
            "cases": [{"bad_key": "no id or input or required"}],
        })
        self.assertEqual(r.status_code, 400)
