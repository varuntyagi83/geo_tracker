from __future__ import annotations
import os
from .base import BaseConnector, ModelResponse, resolve_mock_mode

class PerplexityConnector(BaseConnector):
    provider = "perplexity"
    model = os.getenv("PERPLEXITY_MODEL", "sonar-pro")

    def generate(self, prompt: str, language: str, market: str) -> ModelResponse:
        if resolve_mock_mode():
            return self._mock(prompt)
        # Placeholder for Perplexity Answers API
        return self._mock(prompt)
