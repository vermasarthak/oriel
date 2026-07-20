# Security and Threat Model

- **API Authentication**: All endpoints are protected by `Authorization: Bearer` API keys.
- **Tenant Isolation**: Evidence is strictly partitioned by `tenant_id` to prevent data leakage or poisoned routing evidence across tenants.
- **Untrusted Outputs**: Model outputs are considered fundamentally untrusted and are aggressively validated against a strict JSON schema before any downstream usage.
- **Secret Management**: No credentials are ever stored in Git history. Providers' API keys must be provided via environment variables in production.
- **Denial of Service**: The API should be placed behind a reverse proxy (e.g. Nginx, Cloudflare) for rate limiting and request-size bounds, as the Python layer focuses on routing logic.
- **Prompt Injection**: Oriel does not inherently prevent prompt injection into models, but its strict JSON evaluator will likely fail and penalize models that succumb to injection and return malformed structure.
