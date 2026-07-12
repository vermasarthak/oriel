# Oriel architecture

```text
evaluation cases -> structured evaluator -> immutable SQLite trial store
                                      -> aggregates -> constrained router -> decision evidence
```

The evaluator owns correctness evidence; it never trusts provider claims. The store owns trial history and rejects duplicate evidence keys. The router first filters candidates by observed quality lower bound and request budgets, then optimizes a scored exploration/exploitation choice among the survivors.

## Initial failure semantics

- Invalid JSON is a failed trial.
- A duplicate trial identity is rejected rather than overwritten.
- If no candidate clears the reliability floor, Oriel returns no decision; it does not silently downgrade quality.
- The first version runs locally over SQLite. Multi-process writers, provider outages, and online feedback ingestion remain later milestones.
