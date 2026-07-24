from __future__ import annotations

import time
import os
from typing import Optional
from .runner import Model

try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class OpenAIProvider(Model):
    """
    Real provider adapter for OpenAI models.
    Enforces strict timeouts and handles structured JSON output.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None, timeout: float = 10.0):
        if not HAS_OPENAI:
            raise ImportError("openai package is required for OpenAIProvider")
        
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required to use OpenAIProvider")
            
        self.client = OpenAI(api_key=key, timeout=timeout)
        self.model_name = model_name
        
        self.pricing_microusd = {
            "gpt-3.5-turbo": 1500,
            "gpt-4o-mini": 300,
            "gpt-4-turbo": 20000,
            "gpt-4o": 10000,
        }

    def generate(self, input_text: str) -> tuple[str, float, float]:
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a structured data extractor. You MUST return ONLY valid JSON matching the requested schema. No prose."},
                    {"role": "user", "content": input_text}
                ],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            provider_latency_ms = (time.perf_counter() - start_time) * 1000
            
            output = response.choices[0].message.content or "{}"
            
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0
            rate = self.pricing_microusd.get(self.model_name, 5000)
            cost_microusd = (total_tokens / 1000.0) * rate
            
            return output, provider_latency_ms, cost_microusd
            
        except Exception:
            # Silently fail and return empty JSON so evaluator records a failure
            provider_latency_ms = (time.perf_counter() - start_time) * 1000
            return "{}", provider_latency_ms, 0.0
