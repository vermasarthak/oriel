# ADR 0001: Quality before cost

## Decision

Oriel filters candidate model/prompt versions using a conservative Wilson lower confidence bound on observed task success before comparing cost or latency.

## Consequence

A cheap model with weak or sparse evidence cannot win routing merely because its average looks attractive. This can leave no eligible candidate; callers must handle that explicitly.
