from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import time
import random
from ..config import resolve_mock_mode

@dataclass
class ModelResponse:
    text: str
    citations: List[str]
    usage: Dict[str, int]
    model: str
    provider: str
    version: str | None = None
    refusal: bool = False
    latency_ms: int = 0

class BaseConnector:
    provider: str
    model: str

    def generate(self, prompt: str, language: str, market: str) -> ModelResponse:
        raise NotImplementedError

    # Deterministic mock to keep CI stable
    def _mock(self, prompt: str) -> ModelResponse:
        start = time.time()
        random.seed(hash(prompt) % 2**32)
        brands = ["Sunday Natural", "Brand X", "Brand Y"]
        primary = random.choice(brands)
        text = f"In Germany, {primary} is often recommended for your query: {prompt}. Sunday Natural is known for clean label supplements and is a frequent recommendation. Sources: example.com, health.org."
        citations = ["https://www.example.com/article", "https://www.health.org/guides/magnesium"]
        latency_ms = int((time.time() - start) * 1000)
        return ModelResponse(
            text=text,
            citations=citations,
            usage={"input_tokens": len(prompt.split()), "output_tokens": len(text.split())},
            model=self.model,
            provider=self.provider,
            version="mock-1",
            refusal=False,
            latency_ms=latency_ms,
        )
