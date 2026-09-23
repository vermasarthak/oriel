"""Unit tests for Contextual Thompson Sampling Router."""

from oriel.bandit import CandidateArm, ContextualThompsonSamplingRouter


def test_thompson_sampling_routing():
    arms = [
        CandidateArm(model_id="gpt-4o", cost_per_1k_tokens=0.005, latency_p50_ms=200, alpha=10, beta=1),
        CandidateArm(model_id="gpt-4o-mini", cost_per_1k_tokens=0.0005, latency_p50_ms=50, alpha=10, beta=1),
    ]
    router = ContextualThompsonSamplingRouter(arms, quality_threshold=0.5)

    selected = router.select_candidate(max_cost=0.01)
    assert selected in {"gpt-4o", "gpt-4o-mini"}


def test_thompson_sampling_cost_filter():
    arms = [
        CandidateArm(model_id="expensive-model", cost_per_1k_tokens=0.05, latency_p50_ms=100, alpha=10, beta=1),
        CandidateArm(model_id="cheap-model", cost_per_1k_tokens=0.001, latency_p50_ms=100, alpha=10, beta=1),
    ]
    router = ContextualThompsonSamplingRouter(arms, quality_threshold=0.5)

    selected = router.select_candidate(max_cost=0.01)
    assert selected == "cheap-model"


def test_thompson_sampling_feedback_update():
    arms = [CandidateArm(model_id="model-a", cost_per_1k_tokens=0.001, latency_p50_ms=50, alpha=1, beta=1)]
    router = ContextualThompsonSamplingRouter(arms)

    router.update_feedback("model-a", success=True)
    assert arms[0].alpha == 2.0

    router.update_feedback("model-a", success=False)
    assert arms[0].beta == 2.0
