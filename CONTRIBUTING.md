# Contributing to Oriel

Thank you for your interest in contributing to Oriel! 

## Core Philosophy
1. **Statistical rigor**: Oriel is an evidence-based system. Do not introduce heuristics or "magic numbers" for routing. Any changes to the routing logic must be justifiable in `DESIGN_DECISIONS.md`.
2. **Deterministic execution**: All core evaluation and routing logic must be fully deterministic. Tests should never fail due to floating-point drift or random sampling.
3. **Small surface area**: Oriel is not an LLM framework. It does not provide prompt chaining, memory, or RAG components. It is strictly an evaluation-driven router.

## Development Setup

1. Clone the repository
2. Set up a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install editable with test and UI dependencies: `pip install -e ".[ui]"`
4. Run the full test suite: `python -m unittest discover -s tests -v`

## Adding a New Provider
If you want to add support for a new model provider (e.g., Anthropic, Gemini):
1. Implement the `Model` protocol in `src/oriel/providers.py`.
2. Ensure you strictly handle timeouts and return empty JSON on any provider failure.
3. Add the provider to the `EvaluateRequest` handler in `src/oriel/api.py`.
4. Do not commit API keys or credentials to the codebase.

## Submitting a PR
1. Ensure `python benchmarks.py` runs without performance regressions.
2. Add comprehensive unit and property tests in `tests/`.
3. If changing architecture, add an ADR to `DESIGN_DECISIONS.md`.
4. Ensure code passes all CI checks.
