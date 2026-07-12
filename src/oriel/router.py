from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .store import Aggregate


@dataclass(frozen=True)
class Constraints:
    minimum_quality: float
    max_latency_ms: float
    max_cost_microusd: float
    cost_weight: float = 0.001
    latency_weight: float = 0.0001


@dataclass(frozen=True)
class Decision:
    model: str | None
    reason: str
    quality_lower_bound: float | None


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - spread) / denominator


def choose(candidates: list[Aggregate], constraints: Constraints, rng: random.Random) -> Decision:
    eligible: list[tuple[Aggregate, float]] = []
    for candidate in candidates:
        lower = wilson_lower_bound(candidate.successes, candidate.total)
        if lower < constraints.minimum_quality:
            continue
        if candidate.mean_latency_ms > constraints.max_latency_ms:
            continue
        if candidate.mean_cost_microusd > constraints.max_cost_microusd:
            continue
        eligible.append((candidate, lower))
    if not eligible:
        return Decision(None, "no_eligible_candidate", None)
    def score(entry: tuple[Aggregate, float]) -> float:
        candidate, _ = entry
        sampled_quality = rng.betavariate(candidate.successes + 1, candidate.total - candidate.successes + 1)
        return sampled_quality - constraints.cost_weight * candidate.mean_cost_microusd - constraints.latency_weight * candidate.mean_latency_ms
    candidate, lower = max(eligible, key=score)
    return Decision(candidate.model, "eligible", lower)
