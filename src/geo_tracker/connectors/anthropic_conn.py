# src/geo_tracker/connectors/anthropic_conn.py
from __future__ import annotations
import os
import time
from .base import BaseConnector, ModelResponse
from ..config import resolve_mock_mode


class AnthropicConnector(BaseConnector):
    provider = "anthropic"
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    def generate(self, prompt: str, language: str, market: str) -> ModelResponse:
        if resolve_mock_mode():
            raise RuntimeError("MOCK_MODE is on; refusing to mock for anthropic")

        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        system = (
            "You are a helpful consumer assistant. If you recommend products, "
            "include brand names and brief reasons. If possible, include sources."
        )

        t0 = time.time()
        try:
            msg = client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": f"[market:{market}] [lang:{language}] {prompt}"}],
                temperature=0.2,
                max_tokens=600,
            )

            # Extract text safely from blocks
            chunks = []
            for block in getattr(msg, "content", []) or []:
                text_piece = getattr(block, "text", None)
                if text_piece:
                    chunks.append(text_piece)
            text = "".join(chunks)

            # Usage is a Pydantic model, not a dict
            usage_obj = getattr(msg, "usage", None)
            in_tokens = getattr(usage_obj, "input_tokens", 0) if usage_obj else 0
            out_tokens = getattr(usage_obj, "output_tokens", 0) if usage_obj else 0

            return ModelResponse(
                text=text,
                citations=[],
                usage={"input_tokens": in_tokens, "output_tokens": out_tokens},
                model=self.model,
                provider=self.provider,
                version=self.model,  # record model string for visibility
                refusal=False,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic call failed: {e}")
