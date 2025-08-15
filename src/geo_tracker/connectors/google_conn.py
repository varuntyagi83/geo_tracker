from __future__ import annotations
import os
import time
from .base import BaseConnector, ModelResponse
from ..config import resolve_mock_mode

class GoogleConnector(BaseConnector):
    provider = "google"
    model = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")

    def generate(self, prompt: str, language: str, market: str) -> ModelResponse:
        # Explicit mock path
        if resolve_mock_mode():
            return self._mock(prompt)

        t0 = time.time()
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            system = (
                "You are a helpful consumer assistant. If you recommend products, "
                "include brand names and brief reasons. If possible, include sources."
            )
            model = genai.GenerativeModel(self.model, system_instruction=system)

            resp = model.generate_content(
                f"[market:{market}] [lang:{language}] {prompt}"
            )

            # Extract plain text
            text = ""
            if hasattr(resp, "candidates") and resp.candidates:
                parts = getattr(resp.candidates[0].content, "parts", [])
                text = "".join(getattr(p, "text", "") for p in parts) or str(resp)

            latency_ms = int((time.time() - t0) * 1000)

            # gemini does not return exact token usage here; set rough output_tokens
            return ModelResponse(
                text=text,
                citations=[],  # Could parse from safety/grounding if enabled later
                usage={"input_tokens": 0, "output_tokens": len(text.split())},
                model=self.model,
                provider=self.provider,
                version=self.model,
                refusal=False,
                latency_ms=latency_ms,
            )
        except Exception:
            # Any failure → deterministic mock so the pipeline keeps running
            return self._mock(prompt)
