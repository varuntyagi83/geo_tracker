import time
import re
import urllib.parse
from typing import Dict, Any, Optional, List

import anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_DEFAULT_MODEL
from .base import LLMProvider

# Regexes for URL extraction
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


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    IMPORTANT: Anthropic does NOT have native web search.
    - generate(): Standard Claude chat completion
    - generate_provider_web(): Uses RAG approach (external search + Claude synthesis)

    Models:
    - claude-3-haiku-20240307: Fast and affordable
    - claude-sonnet-4-20250514: Balanced performance (default)
    - claude-opus-4-20250514: Most capable, supports extended thinking

    Extended thinking is automatically enabled for Opus model to improve reasoning.
    """
    name = "anthropic"

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _extract_usage(self, resp) -> tuple:
        try:
            usage = resp.usage
            return (
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None)
            )
        except Exception:
            return None, None

    def _extract_text(self, resp) -> str:
        """Extract text from Claude response, handling extended thinking."""
        try:
            text_parts = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
            return "\n".join(text_parts).strip()
        except Exception:
            return str(resp)

    def _is_opus_model(self, model: str) -> bool:
        """Check if model supports extended thinking."""
        return "opus" in model.lower()

    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Internal mode: Standard Claude chat completion.
        Uses extended thinking for Opus models.
        """
        model = model or ANTHROPIC_DEFAULT_MODEL
        start = time.time()

        # Build request kwargs
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Enable extended thinking for Opus
        if self._is_opus_model(model):
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 5000
            }
            kwargs["max_tokens"] = 16000
            kwargs["temperature"] = 1  # Required for extended thinking

        resp = self.client.messages.create(**kwargs)

        latency_ms = int((time.time() - start) * 1000)
        text = self._extract_text(resp)
        tokens_in, tokens_out = self._extract_usage(resp)

        # Extract any URLs from the response
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
        Provider web mode: RAG approach using web search + Claude synthesis.

        Since Anthropic doesn't have native web search, we:
        1. Use DuckDuckGo/Brave to search the web
        2. Pass search results as context to Claude
        3. Claude synthesizes the answer with citations

        This provides web-grounded responses similar to other providers.
        """
        model = model or ANTHROPIC_DEFAULT_MODEL
        start = time.time()

        # Import web retrieval for search
        try:
            from retrieval.web_retrieval import build_context
        except ImportError:
            # Fallback to internal mode if retrieval not available
            return self.generate(prompt, model=model)

        # Search the web for relevant content
        try:
            context, search_sources = build_context(
                query=prompt,
                max_results=5,
                market=None,
                lang=None
            )
        except Exception as e:
            # If search fails, fall back to internal mode
            print(f"[anthropic] Web search failed: {e}. Using internal mode.")
            return self.generate(prompt, model=model)

        # Build RAG prompt with search results
        if context:
            rag_prompt = f"""Based on the following web search results, answer the user's question.
Always cite your sources using the URLs provided. Be comprehensive and accurate.

WEB SEARCH RESULTS:
{context}

USER QUESTION:
{prompt}

INSTRUCTIONS:
- Use the search results above to answer the question
- Cite sources with their URLs when referencing information
- If the search results don't contain relevant information, say so
- Provide a thorough, well-organized response"""
        else:
            # No search results, use original prompt
            rag_prompt = prompt

        # Build request kwargs
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": rag_prompt}],
        }

        # Enable extended thinking for Opus
        if self._is_opus_model(model):
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 5000
            }
            kwargs["max_tokens"] = 16000
            kwargs["temperature"] = 1

        resp = self.client.messages.create(**kwargs)

        latency_ms = int((time.time() - start) * 1000)
        text = self._extract_text(resp)
        tokens_in, tokens_out = self._extract_usage(resp)

        # Combine search sources with any URLs extracted from response
        sources = []

        # Add search sources first
        for s in search_sources:
            sources.append({
                "url": s.get("url", ""),
                "title": s.get("title")
            })

        # Add any additional URLs mentioned in the response
        text_sources = _extract_sources_from_text(text)
        sources.extend(text_sources)

        # Deduplicate
        sources = _dedupe_sources_dict(sources)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }
