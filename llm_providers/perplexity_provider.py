import time
import re
import urllib.parse
from typing import Dict, Any, Optional, List

from openai import OpenAI
from config import PERPLEXITY_API_KEY, PERPLEXITY_DEFAULT_MODEL
from .base import LLMProvider

# Regexes for fallback URL extraction
URL_RE = re.compile(r'\bhttps?://[^\s\)\]]+', re.IGNORECASE)
MD_LINK_RE = re.compile(r'\[([^\]]{0,200})\]\((https?://[^\s\)]+)\)')


def _norm_url_key(url: str):
    try:
        p = urllib.parse.urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = p.path or "/"
        return (host, path)
    except Exception:
        return (url, "")


def _dedupe_sources_dict(sources: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for s in sources:
        url = (s.get("url") or "").strip()
        if not url:
            continue
        key = _norm_url_key(url)
        if key in seen:
            continue
        seen.add(key)
        title = (s.get("title") or "").strip() or None
        out.append({"url": url, "title": title})
    return out


def _extract_sources_from_text(text: str) -> List[dict]:
    if not text:
        return []
    found: List[dict] = []

    for m in MD_LINK_RE.finditer(text):
        title = (m.group(1) or "").strip() or None
        url = m.group(2).strip()
        found.append({"url": url, "title": title})

    seen_urls = {f["url"] for f in found}
    for m in URL_RE.finditer(text):
        url = m.group(0).strip().rstrip(").,;")
        if url not in seen_urls:
            found.append({"url": url, "title": None})

    return _dedupe_sources_dict(found)


class PerplexityProvider(LLMProvider):
    """
    Perplexity AI provider using their OpenAI-compatible API.

    Perplexity has NATIVE web search built into their models.
    The 'sonar' family of models automatically search the web.

    Models:
    - sonar: Fast, cost-effective with web search
    - sonar-pro: Advanced reasoning with web search
    - sonar-reasoning: Chain-of-thought reasoning with web search
    """
    name = "perplexity"

    def __init__(self):
        if not PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY not set")
        self.client = OpenAI(
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )

    def _extract_usage(self, resp) -> tuple:
        try:
            usage = resp.usage
            return (
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None)
            )
        except Exception:
            return None, None

    def _extract_citations(self, resp) -> List[dict]:
        """
        Extract citations from Perplexity response.
        Perplexity returns citations in the response object.
        """
        sources = []

        # Perplexity includes citations array in the response
        try:
            citations = getattr(resp, "citations", None)
            if citations and isinstance(citations, list):
                for url in citations:
                    if url and isinstance(url, str):
                        sources.append({"url": url, "title": None})
        except Exception:
            pass

        # Also check choices for any citation metadata
        try:
            for choice in resp.choices:
                msg = choice.message
                # Some API versions include citations in message
                if hasattr(msg, "citations"):
                    for url in msg.citations:
                        if url and isinstance(url, str):
                            sources.append({"url": url, "title": None})
                # Check for context array (older API format)
                if hasattr(msg, "context"):
                    for ctx in (msg.context or []):
                        url = ctx.get("url") if isinstance(ctx, dict) else None
                        if url:
                            sources.append({"url": url, "title": ctx.get("title")})
        except Exception:
            pass

        return _dedupe_sources_dict(sources)

    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal mode: Uses Perplexity without emphasizing web search.
        Note: Perplexity sonar models still have some web capability built-in.
        """
        model = model or PERPLEXITY_DEFAULT_MODEL
        start = time.time()

        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        latency_ms = int((time.time() - start) * 1000)
        text = (resp.choices[0].message.content or "").strip()
        tokens_in, tokens_out = self._extract_usage(resp)

        # Extract citations from response
        sources = self._extract_citations(resp)

        # Fallback: extract URLs from text
        if not sources:
            sources = _extract_sources_from_text(text)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }

    def generate_provider_web(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Provider web mode: Perplexity's core feature is web search.
        This explicitly requests web-grounded responses with citations.
        """
        model = model or PERPLEXITY_DEFAULT_MODEL
        start = time.time()

        # System message to encourage citation usage
        sys_msg = (
            "Search the web thoroughly to answer this question. "
            "Always cite your sources with URLs. Provide comprehensive, "
            "up-to-date information from multiple sources when available."
        )

        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        latency_ms = int((time.time() - start) * 1000)
        text = (resp.choices[0].message.content or "").strip()
        tokens_in, tokens_out = self._extract_usage(resp)

        # Extract citations
        sources = self._extract_citations(resp)

        # Fallback: extract URLs from text
        if not sources:
            sources = _extract_sources_from_text(text)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }
