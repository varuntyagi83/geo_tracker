import time, re, urllib.parse, sys
from typing import Dict, Any, Optional, List

# Using the modern, unified SDK package name (google-genai)
import google.genai as genai
from google.genai import types

from config import GOOGLE_API_KEY, GEMINI_DEFAULT_MODEL
from .base import LLMProvider

# Regexes
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


def _dedupe_sources(sources: List[dict]) -> List[dict]:
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

    # Markdown links first to capture titles
    for m in MD_LINK_RE.finditer(text):
        title = (m.group(1) or "").strip() or None
        url = m.group(2).strip()
        found.append({"url": url, "title": title})

    # Bare URLs
    seen_urls = {f["url"] for f in found}
    for m in URL_RE.finditer(text):
        url = m.group(0).strip().rstrip(").,;")
        if url not in seen_urls:
            found.append({"url": url, "title": None})

    return _dedupe_sources(found)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set")

        # FIX: Initialize the client using the new SDK structure.
        self._client = genai.Client(api_key=GOOGLE_API_KEY)

    # ---------------- helpers ----------------

    def _extract_text(self, resp) -> str:
        try:
            return resp.text or ""
        except Exception:
            try:
                # The structure for accessing text is retained for robustness
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return str(resp)

    def _extract_usage(self, resp):
        tokens_in = None
        tokens_out = None
        try:
            meta = getattr(resp, "usage_metadata", None)
            if meta:
                tokens_in = getattr(meta, "prompt_token_count", None)
                tokens_out = getattr(meta, "candidates_token_count", None)
        except Exception:
            pass
        return tokens_in, tokens_out

    def _extract_any_citations(self, resp_dict: dict) -> List[dict]:
        """
        Robust citation/grounding extraction across Gemini 1.5, 2.5, and later shapes.
        """
        found: List[dict] = []
        cands = (resp_dict or {}).get("candidates") or []
        for c in cands:
            # 1) groundingMetadata (chunks, citations, attributions)
            gm = c.get("groundingMetadata") or {}
            for ch in gm.get("groundingChunks", []) or []:
                web = ch.get("web") or {}
                url = web.get("uri") or web.get("url")
                title = web.get("title")
                if url:
                    found.append({"url": url, "title": title})
            for src in gm.get("citations", []) or []:
                url = src.get("uri") or src.get("url")
                title = src.get("title")
                if url:
                    found.append({"url": url, "title": title})
            for att in gm.get("groundingAttributions", []) or []:
                url = att.get("sourceUrl") or att.get("url") or att.get("uri")
                title = att.get("title")
                if url:
                    found.append({"url": url, "title": title})

            # 2) citationMetadata attached to content.parts
            content = c.get("content") or {}
            for part in (content.get("parts") or []):
                cm = part.get("citationMetadata") or {}
                for cs in cm.get("citationSources", []) or []:
                    url = cs.get("uri") or cs.get("url")
                    title = cs.get("title")
                    if url:
                        found.append({"url": url, "title": title})

            # 3) top-level candidate citationMetadata
            cm2 = c.get("citationMetadata") or {}
            for cs in cm2.get("citationSources", []) or []:
                url = cs.get("uri") or cs.get("url")
                title = cs.get("title")
                if url:
                    found.append({"url": url, "title": title})

        return _dedupe_sources(found)

    def _extract_grounded_sources(self, resp, fallback_text: str) -> List[dict]:
        """
        Prefer grounded/citation metadata; fallback to URLs in text.
        """
        sources: List[dict] = []
        try:
            data = resp.to_dict() if hasattr(resp, "to_dict") else None
            if data:
                sources = self._extract_any_citations(data)
        except Exception:
            pass

        if not sources and fallback_text:
            sources = _extract_sources_from_text(fallback_text)

        return _dedupe_sources(sources)

    # ---------------- public API ----------------

    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        INTERNAL mode (no live web). We still parse any links in the text for parity.
        """
        model = model or GEMINI_DEFAULT_MODEL
        start = time.time()

        # FIX: Use the client's models service generate_content method directly
        resp = self._client.models.generate_content(
            model=model,
            contents=prompt
        )

        latency_ms = int((time.time() - start) * 1000)

        text = self._extract_text(resp)
        tokens_in, tokens_out = self._extract_usage(resp)
        sources = self._extract_grounded_sources(resp, text)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }

    def generate_provider_web(self, prompt: str, model: Optional[str] = None,
                              dynamic_threshold: float = 0.0) -> Dict[str, Any]:
        """
        PROVIDER_WEB: Gemini 2.5+ built-in grounded search (GoogleSearch).

        NOTE: The standard tool is enabled and the model decides when to use it.
        The dynamic_threshold parameter is deprecated/ignored for this tool setup
        but kept for API compatibility.

        IMPORTANT: We DO NOT modify the user prompt (so --raw remains truly raw).
        We instead set a system_instruction to nudge retrieval & citations.
        """
        model = model or GEMINI_DEFAULT_MODEL
        start = time.time()

        # Nudge to use retrieval & include URLs, without touching the user prompt
        sys_instr = (
            "Use Google Search Retrieval to answer when available. "
            "Verify key claims with web results and include citations with direct URLs. "
            "Prefer recent, authoritative sources."
        )

        # FIX 1: Define tools using the SDK classes
        tools_list = [
            types.Tool(google_search=types.GoogleSearch())
        ]

        # FIX 2: Create a single GenerateContentConfig object that includes the system instruction,
        # mime type, AND the tools list. This resolves the 'unexpected keyword argument tools' error.
        config = types.GenerateContentConfig(
            system_instruction=sys_instr,
            response_mime_type="text/plain",
            tools=tools_list
        )

        # FIX 3: Use the client's models service generate_content method, passing the
        # combined config argument.
        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config  # Pass the combined config object
        )

        latency_ms = int((time.time() - start) * 1000)

        text = self._extract_text(resp)
        tokens_in, tokens_out = self._extract_usage(resp)
        sources = self._extract_grounded_sources(resp, text)

        if not sources:
            print(
                "[gemini] provider_web returned no grounding/citations; using text link fallback if present.",
                file=sys.stderr
            )

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }