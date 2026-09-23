"""Contextual Thompson Sampling Router for Oriel."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CandidateArm:
    """Represents a candidate model arm in Thompson Sampling."""
    model_id: str
    cost_per_1k_tokens: float
    latency_p50_ms: float
    alpha: float = 1.0
    beta: float = 1.0


class ContextualThompsonSamplingRouter:
    """Contextual Thompson Sampling Multi-Armed Bandit Router."""

    def __init__(self, candidates: List[CandidateArm], quality_threshold: float = 0.7):
        self.candidates = {c.model_id: c for c in candidates}
        self.quality_threshold = quality_threshold

    def select_candidate(
        self,
        context: Dict[str, float] = None,
        max_cost: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
    ) -> Optional[str]:
        """Selects candidate model using Thompson Sampling subject to context and SLA constraints."""
        eligible = []
        for model_id, arm in self.candidates.items():
            if max_cost is not None and arm.cost_per_1k_tokens > max_cost:
                continue
            if max_latency_ms is not None and arm.latency_p50_ms > max_latency_ms:
                continue

            safe_alpha = max(arm.alpha, 1e-6)
            safe_beta = max(arm.beta, 1e-6)
            sampled_quality = random.betavariate(safe_alpha, safe_beta)
            if sampled_quality >= self.quality_threshold:
                eligible.append((model_id, sampled_quality, arm.cost_per_1k_tokens))

        if not eligible:
            return None

        eligible.sort(key=lambda x: (x[1] / max(x[2], 1e-6)), reverse=True)
        return eligible[0][0]

    def update_feedback(self, model_id: str, success: bool, weight: float = 1.0):
        """Updates Beta distribution priors based on request execution feedback."""
        if model_id in self.candidates:
            arm = self.candidates[model_id]
            if success:
                arm.alpha += weight
            else:
                arm.beta += weight

    def export_priors(self) -> dict[str, tuple[float, float]]:
        """Exports alpha and beta parameters for Redis/Postgres persistence."""
        return {m: (arm.alpha, arm.beta) for m, arm in self.candidates.items()}

    def import_priors(self, priors: dict[str, tuple[float, float]]):
        """Imports persisted alpha and beta parameters into arm state."""
        for m, (a, b) in priors.items():
            if m in self.candidates:
                self.candidates[m].alpha = a
                self.candidates[m].beta = b
