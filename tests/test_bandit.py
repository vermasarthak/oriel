"""Unit tests for Thompson Sampling Router."""

import random

from oriel.bandit import CandidateArm, ThompsonSamplingRouter


def test_thompson_sampling_routing_deterministic():
    arms = [
        CandidateArm(model_id="gpt-4o", cost_per_1k_tokens=0.005, latency_p50_ms=200, alpha=10, beta=1),
        CandidateArm(model_id="gpt-4o-mini", cost_per_1k_tokens=0.0005, latency_p50_ms=50, alpha=10, beta=1),
    ]
    rng1 = random.Random(42)
    router1 = ThompsonSamplingRouter(arms, quality_threshold=0.5, rng=rng1)

    rng2 = random.Random(42)
    router2 = ThompsonSamplingRouter(arms, quality_threshold=0.5, rng=rng2)

    assert router1.select_candidate(max_cost=0.01) == router2.select_candidate(max_cost=0.01)


def test_thompson_sampling_cost_filter():
    arms = [
        CandidateArm(model_id="expensive-model", cost_per_1k_tokens=0.05, latency_p50_ms=100, alpha=10, beta=1),
        CandidateArm(model_id="cheap-model", cost_per_1k_tokens=0.001, latency_p50_ms=100, alpha=10, beta=1),
    ]
    router = ThompsonSamplingRouter(arms, quality_threshold=0.5)

    selected = router.select_candidate(max_cost=0.01)
    assert selected == "cheap-model"


def test_thompson_sampling_latency_filter():
    arms = [
        CandidateArm(model_id="slow-model", cost_per_1k_tokens=0.001, latency_p50_ms=500, alpha=10, beta=1),
        CandidateArm(model_id="fast-model", cost_per_1k_tokens=0.002, latency_p50_ms=50, alpha=10, beta=1),
    ]
    router = ThompsonSamplingRouter(arms, quality_threshold=0.5)

    selected = router.select_candidate(max_latency_ms=100)
    assert selected == "fast-model"


def test_thompson_sampling_cold_start():
    # Cold start arms with uninformative priors (alpha=1, beta=1)
    arms = [
        CandidateArm(model_id="model-a", cost_per_1k_tokens=0.001, latency_p50_ms=50, alpha=1.0, beta=1.0),
        CandidateArm(model_id="model-b", cost_per_1k_tokens=0.001, latency_p50_ms=50, alpha=1.0, beta=1.0),
    ]
    rng = random.Random(12345)
    router = ThompsonSamplingRouter(arms, quality_threshold=0.1, rng=rng)
    selected = router.select_candidate()
    assert selected in {"model-a", "model-b"}


def test_thompson_sampling_feedback_update():
    arms = [CandidateArm(model_id="model-a", cost_per_1k_tokens=0.001, latency_p50_ms=50, alpha=1, beta=1)]
    router = ThompsonSamplingRouter(arms)

    router.update_feedback("model-a", success=True)
    assert arms[0].alpha == 2.0

    router.update_feedback("model-a", success=False)
    assert arms[0].beta == 2.0
