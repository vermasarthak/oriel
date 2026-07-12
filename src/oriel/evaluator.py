from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Evaluation:
    passed: bool
    reason: str


def evaluate_json_object(output: str, required: dict[str, object]) -> Evaluation:
    """Checks parseability and exact values for a small deterministic contract."""
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return Evaluation(False, "invalid_json")
    if not isinstance(value, dict):
        return Evaluation(False, "not_object")
    for key, expected in required.items():
        if key not in value:
            return Evaluation(False, f"missing:{key}")
        if value[key] != expected:
            return Evaluation(False, f"mismatch:{key}")
    return Evaluation(True, "passed")
