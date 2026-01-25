import time, re, urllib.parse
from typing import Dict, Any, Optional, List

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_DEFAULT_MODEL
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

    # Prefer markdown links first to capture titles
    for m in MD_LINK_RE.finditer(text):
        title = (m.group(1) or "").strip() or None
        url = m.group(2).strip()
        found.append({"url": url, "title": title})

    # Then bare URLs (avoid duplicates)
    seen_urls = {f["url"] for f in found}
    for m in URL_RE.finditer(text):
        url = m.group(0).strip().rstrip(").,;")
        if url not in seen_urls:
            found.append({"url": url, "title": None})

    return _dedupe_sources_dict(found)

class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    # ---------- helpers ----------
    def _extract_usage_chat(self, resp):
        try:
            usage = resp.usage
            return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        except Exception:
            return None, None

    def _extract_usage_responses(self, resp):
        try:
            usage = getattr(resp, "usage", None)
            if usage:
                ti = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
                to = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
                return ti, to
        except Exception:
            pass
        return None, None

    def _extract_text_responses(self, resp) -> str:
        # Fast path (SDK helper)
        text = getattr(resp, "output_text", None)
        if text:
            return text
        # Robust path
        try:
            data = resp.model_dump()
            # Find first message text
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if "text" in part and part["text"]:
                            return part["text"]
            # Fallback: whole dump
            return str(data)
        except Exception:
            return str(resp)

    def _extract_citations_responses(self, resp, fallback_text: str) -> List[dict]:
        """
        Extract citations/URLs from Responses API output (native annotations first),
        then fall back to parsing links from text.
        """
        sources: List[dict] = []
        # Native annotations
        try:
            data = resp.model_dump()
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        # 'annotations' may include url citations for web_search tools
                        for ann in (part.get("annotations") or []):
                            t = ann.get("type")
                            if t and ("citation" in t or "url" in t):
                                url = ann.get("url") or ann.get("href")
                                title = ann.get("title") or ann.get("source") or None
                                if url:
                                    sources.append({"url": url, "title": title})
                        # Some SDKs expose 'references' or 'citations' arrays beside annotations
                        for ref in (part.get("references") or []):
                            url = ref.get("url") or ref.get("href")
                            title = ref.get("title")
                            if url:
                                sources.append({"url": url, "title": title})
        except Exception:
            pass

        # Fallback: parse URLs/markdown links from the final text
        if not sources and fallback_text:
            sources = _extract_sources_from_text(fallback_text)

        return _dedupe_sources_dict(sources)

    # ---------- public API ----------
    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        INTERNAL mode: standard chat completion (no live web).
        """
        model = model or OPENAI_DEFAULT_MODEL
        start = time.time()
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        latency_ms = int((time.time() - start) * 1000)
        text = (resp.choices[0].message.content or "").strip()
        tokens_in, tokens_out = self._extract_usage_chat(resp)

        # Extract any URLs/links from the text as "sources" for parity
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
        PROVIDER_WEB mode: use OpenAI Responses API with built-in web search.
        We (1) try to REQUIRE 'web_search' via tool_choice, (2) fall back to allowing
        the model to choose the tool, (3) try 'web_search_preview', and finally
        (4) degrade to internal generate() if tools are unavailable.
        If the first successful call returns no citations, we retry once with preview.
        """
        model = model or OPENAI_DEFAULT_MODEL
        start = time.time()

        sys_msg = (
            "Use the web_search tool to answer. Cite sources with URLs. "
            "Do not answer from memory if web_search is available."
        )

        def _do_call(tools, require=False):
            kwargs = dict(model=model, input=[{"role":"system","content":sys_msg},{"role":"user","content":prompt}], tools=tools)
            if require:
                # Require the named tool if the API/model supports tool_choice
                try:
                    kwargs["tool_choice"] = {"type": "tool", "name": tools[0]["type"]}
                except Exception:
                    pass
            return self.client.responses.create(**kwargs)

        # 1) Require web_search if possible
        try:
            resp = _do_call([{"type": "web_search"}], require=True)
        except Exception:
            # 2) Allow model to choose web_search
            try:
                resp = _do_call([{"type": "web_search"}], require=False)
            except Exception:
                # 3) Try preview tool
                try:
                    resp = _do_call([{"type": "web_search_preview"}], require=False)
                except Exception:
                    # 4) Last resort: degrade to internal
                    return self.generate(prompt, model=model)

        latency_ms = int((time.time() - start) * 1000)
        text = self._extract_text_responses(resp)
        tokens_in, tokens_out = self._extract_usage_responses(resp)
        sources = self._extract_citations_responses(resp, text)

        # If still no sources, try a one-shot retry with preview tool
        if not sources:
            try:
                resp2 = _do_call([{"type": "web_search_preview"}], require=False)
                text2 = self._extract_text_responses(resp2)
                s2 = self._extract_citations_responses(resp2, text2)
                if s2:
                    text, sources = text2, s2
            except Exception:
                pass

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }
