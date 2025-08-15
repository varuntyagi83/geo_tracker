# src/geo_tracker/connectors/openai_conn.py
from __future__ import annotations
import os
import time  # <-- needed
from .base import BaseConnector, ModelResponse
from ..config import resolve_mock_mode


class OpenAIConnector(BaseConnector):
    provider = "openai"
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def generate(self, prompt: str, language: str, market: str) -> ModelResponse:
        if resolve_mock_mode():
            raise RuntimeError("MOCK_MODE is on; refusing to mock for openai")

        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system = (
            "You are a helpful consumer assistant. If you recommend products, "
            "include brand names and brief reasons. If possible, include sources."
        )

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"[market:{market}] [lang:{language}] {prompt}"},
                ],
                temperature=0.2,
            )
            text = resp.choices[0].message.content
            model_name = getattr(resp, "model", self.model)

            return ModelResponse(
                text=text,
                citations=[],
                usage={
                    "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(resp.usage, "completion_tokens", 0),
                },
                model=self.model,
                provider=self.provider,
                version=model_name,  # record API-reported model string
                refusal=False,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")
