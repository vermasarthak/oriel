# Limitations

- **Binary success labels**: The current Wilson lower bound implementation assumes binary successes (pass/fail). It does not capture partial credit or continuous quality metrics.
- **Small-sample behavior**: Wilson bounds heavily penalize low-sample counts to prevent exploration risk. This means a new model will initially be rejected even if its true quality is perfect.
- **Dataset shift**: The evaluation dataset must be representative of production traffic. If the distribution of tasks changes, the evidence may become misleading.
- **Correlated evaluation cases**: The statistical bound assumes independent trials. Highly correlated test cases can inflate confidence artificially.
- **Offline-to-online mismatch**: The delayed feedback loop from production (`/v1/outcomes`) may have biases if only successful routes are fully evaluated by users.
- **When the routing policy should not be trusted**: If the underlying model weights change without a version bump, or if the provider degrades silently, historical evidence will be falsely confident until new failures accumulate.
