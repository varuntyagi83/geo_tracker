"""
Trustworthiness split:
 - trust_authority: fraction of sources that look non-sunday (authority sites)
 - trust_sunday: fraction of sources that are Sunday-owned
If no sources or not applicable, return (None, None).
"""
from urllib.parse import urlparse

def _is_sunday(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return "sunday" in host  # tweak to your canonical domain(s)
    except Exception:
        return False

def compute_trustworthiness(answer_text: str | None, provider_sources: list | None):
    if not provider_sources:
        return (None, None)
    urls = [s.get("url") for s in provider_sources if isinstance(s, dict) and s.get("url")]
    if not urls:
        return (None, None)

    sunday = sum(1 for u in urls if _is_sunday(u))
    total  = len(urls)
    other  = total - sunday

    trust_sunday = sunday / total
    trust_authority = other / total
    return (trust_authority, trust_sunday)
