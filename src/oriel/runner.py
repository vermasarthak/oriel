from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .dataset import Case
from .evaluator import evaluate_json_object
from .store import TrialStore


class Model(Protocol):
    def generate(self, input_text: str) -> tuple[str, float, float]: ...


@dataclass(frozen=True)
class RunSummary:
    total: int
    passed: int


def run_cases(store: TrialStore, tenant_id: str, task: str, model_name: str, prompt_version: str, model: Model, cases: list[Case]) -> RunSummary:
    passed = 0
    for case in cases:
        started = time.perf_counter()
        output, provider_latency_ms, cost_microusd = model.generate(case.input_text)
        observed_latency_ms = (time.perf_counter() - started) * 1_000
        result = evaluate_json_object(output, case.required)
        store.record(
            tenant_id, task, case.id, model_name, prompt_version, case.input_text, result.passed,
            max(provider_latency_ms, observed_latency_ms), cost_microusd,
        )
        passed += int(result.passed)
    return RunSummary(total=len(cases), passed=passed)
