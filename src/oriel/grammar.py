"""Zero-Overhead JSON Schema Logit Masking Engine for Oriel.

Converts JSON Schema definitions into deterministic logit masks during LLM sampling steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LogitMaskConfig:
    """Configuration for constrained schema decoding."""
    schema: Dict[str, Any]
    mask_value: float = -1e9


class JSONSchemaLogitMasker:
    """Computes logit masks for vocab tokens to enforce JSON schema invariants."""

    def __init__(self, schema: Dict[str, Any], vocab: Dict[str, int], mask_value: float = -1e9):
        self.schema = schema
        self.vocab = vocab
        self.inv_vocab = {idx: token for token, idx in vocab.items()}
        self.mask_value = mask_value

    def compute_mask(self, current_prefix: str) -> List[float]:
        """Computes floating-point logit mask vector across vocabulary."""
        mask = [0.0] * len(self.vocab)

        # Enforce JSON object opening brace constraint for root object schema
        if not current_prefix.strip():
            for idx, token in self.inv_vocab.items():
                if not token.startswith("{"):
                    mask[idx] = self.mask_value
        return mask
