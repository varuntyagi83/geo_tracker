# retrieval/web_retrieval.py
import os
import time
import requests
from typing import List, Tuple, Dict, Optional

# If you want a DDG fallback, we try ddgs first; else quietly skip fallback.
try:
    from ddgs import DDGS  # modern package name
    _HAS_DDGS = True
except Exception:
    try:
        from duckduckgo_search import DDGS  # legacy package
        _HAS_DDGS = True
    except Exception:
        _HAS_DDGS = False

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")


def _domain(url: str) -> str:
    if not url:
        return ""
    return url.split("/")[2] if "://" in url else url


def brave_search(query: str,
                 count: int = 5,
                 country: Optional[str] = None,
                 search_lang: Optional[str] = None,
                 timeout: int = 15) -> List[Dict]:
    """
    Brave Web Search API.
    Env: BRAVE_API_KEY
    Docs: https://api.search.brave.com/app/documentation/web-search
    """
    if not BRAVE_API_KEY:
        raise RuntimeError("BRAVE_API_KEY not set")

    endpoint = "https://api.search.brave.com/res/v1/web/search"
    params = {
        "q": query,
        "count": count,
        "safesearch": "moderate",
    }
    if country:
        params["country"] = country  # e.g., "DE"
    if search_lang:
        params["search_lang"] = search_lang  # e.g., "de" or "de-DE"
        # you may also set "ui_lang": search_lang

    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
        "Accept": "application/json",
    }

    r = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
    if r.status_code == 429:
        # surface a ratelimit-like exception to upstream retry logic
        raise RuntimeError("Brave rate limit (HTTP 429)")
    r.raise_for_status()

    data = r.json()
    results = []
    web = data.get("web", {}) or {}
    for i, item in enumerate(web.get("results", [])[:count], start=1):
        results.append({
            "rank": i,
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "snippet": item.get("description") or "",
            "domain": _domain(item.get("url") or ""),
        })
    return results


_DDG_REGION_MAP = {
    # minimal useful mapping; extend as needed
    "DE": "de-de",
    "AT": "at-de",
    "CH": "ch-de",
    "FR": "fr-fr",
    "ES": "es-es",
    "IT": "it-it",
    "GB": "uk-en",
    "US": "us-en",
    "NL": "nl-nl",
    "SE": "se-sv",
}

def _ddg_region(market: Optional[str]) -> Optional[str]:
    if not market:
        return None
    market = market.upper()
    return _DDG_REGION_MAP.get(market, None)


def ddg_search(query: str,
               count: int = 5,
               market: Optional[str] = None) -> List[Dict]:
    if not _HAS_DDGS:
        return []
    region = _ddg_region(market)
    out = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, max_results=count, region=region), start=1):
            out.append({
                "rank": i,
                "title": r.get("title") or "",
                "url": r.get("href") or r.get("url") or "",
                "snippet": r.get("body") or r.get("snippet") or "",
                "domain": _domain(r.get("href") or r.get("url") or ""),
            })
    return out


def build_context(query: str,
                  max_results: int = 5,
                  market: Optional[str] = None,
                  lang: Optional[str] = None) -> Tuple[str, List[Dict]]:
    """
    Returns a (context, sources) tuple.
    - context: a compact text summary with titles/snippets for the top results
    - sources: list of {title, url, snippet, rank, domain}
    """
    # Try Brave first
    sources: List[Dict] = []
    err: Optional[str] = None
    if BRAVE_API_KEY:
        try:
            print(f"[search] Using Brave for query: {query}")
            sources = brave_search(query, count=max_results, country=market, search_lang=lang)
        except Exception as e:
            err = str(e)
            print(f"[search] Brave failed: {e}. Falling back to DDG (if available)...")

    # Fallback: DDG (no API key; light regionalization)
    if not sources:
        try:
            sources = ddg_search(query, count=max_results, market=market)
        except Exception as e:
            err = (err or "") + f" | DDG failed: {e}"

    if not sources:
        if err:
            raise RuntimeError(err)
        return ("", [])

    # Build a compact context text for the prompt (keep it short to avoid token bloat)
    lines = []
    for s in sources:
        title = s["title"].strip()
        snippet = (s["snippet"] or "").strip()
        url = s["url"]
        rank = s["rank"]
        if snippet:
            lines.append(f"[{rank}] {title} — {snippet}")
        else:
            lines.append(f"[{rank}] {title} — {url}")

    context = "\n".join(lines)
    return (context, sources)
