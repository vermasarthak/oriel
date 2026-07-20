import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from oriel.api import app, store

class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test-token"}
        # Clear store for tests
        store.connection.execute("DELETE FROM trials")
        store.connection.commit()

    def test_healthz(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)

    def test_auth_required(self):
        response = self.client.post("/v1/route", json={"task": "t", "prompt_version": "v1"})
        self.assertEqual(response.status_code, 401)

    def test_outcome_and_route(self):
        # 1. Submit positive outcome
        res = self.client.post("/v1/outcomes", headers=self.headers, json={
            "task": "intent",
            "case_id": "c1",
            "model": "model-a",
            "prompt_version": "v1",
            "input_text": "hello",
            "passed": True,
            "latency_ms": 100,
            "cost_microusd": 50
        })
        self.assertEqual(res.status_code, 200)

        # 2. Submit negative outcome to make it realistic
        res = self.client.post("/v1/outcomes", headers=self.headers, json={
            "task": "intent",
            "case_id": "c2",
            "model": "model-a",
            "prompt_version": "v1",
            "input_text": "bad",
            "passed": False,
            "latency_ms": 100,
            "cost_microusd": 50
        })
        self.assertEqual(res.status_code, 200)

        # 3. Route
        res = self.client.post("/v1/route", headers=self.headers, json={
            "task": "intent",
            "prompt_version": "v1",
            "minimum_quality": 0.0,
            "max_latency_ms": 200,
            "max_cost_microusd": 200
        })
        self.assertEqual(res.status_code, 200)
        decision = res.json()
        self.assertEqual(decision["model"], "model-a")

    def test_no_eligible_candidate(self):
        res = self.client.post("/v1/route", headers=self.headers, json={
            "task": "intent",
            "prompt_version": "v2", # no evidence
            "minimum_quality": 0.9,
            "max_latency_ms": 200,
            "max_cost_microusd": 200
        })
        self.assertEqual(res.status_code, 200)
        decision = res.json()
        self.assertIsNone(decision["model"])
        self.assertEqual(decision["reason"], "no_eligible_candidate")
