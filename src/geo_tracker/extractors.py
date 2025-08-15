from __future__ import annotations
import re
import json
from typing import Dict, List, Tuple
import yaml
import tldextract
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def load_brand_dict(path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("brands", {})

def extract_mentions(text: str, brands: Dict[str, Dict[str, List[str]]]) -> List[Dict]:
    found = []
    lower = text.lower()
    for key, meta in brands.items():
        variants = meta.get("variants", []) + [meta.get("canonical", "")]
        positions = []
        for v in variants:
            if not v:
                continue
            for m in re.finditer(re.escape(v.lower()), lower):
                positions.append(m.start())
        if positions:
            positions.sort()
            found.append({
                "brand": meta.get("canonical", key),
                "first_position": positions[0],
                "mention_count": len(positions),
                "in_final_recommendation": 1 if _in_final_recommendation(lower, meta) else 0
            })
    return found

def _in_final_recommendation(lower_text: str, brand_meta: Dict) -> bool:
    # Heuristic: look near common conclusion cues
    cues = ["final recommendation", "in summary", "overall", "conclusion"]
    brand = brand_meta.get("canonical", "").lower()
    for cue in cues:
        idx = lower_text.rfind(cue)
        if idx >= 0:
            window = lower_text[idx: idx + 400]
            if brand in window:
                return True
    return False

def placement_quality(text: str, brand: str) -> float:
    lower = text.lower()
    pos = lower.find(brand.lower())
    if pos < 0:
        return 0.0
    # Score based on early appearance
    return max(0.0, 1.0 - (pos / max(1, len(lower))))

def parse_domains(citations: List[str]) -> List[str]:
    domains = []
    for url in citations or []:
        try:
            ext = tldextract.extract(url)
            domains.append(".".join([d for d in [ext.domain, ext.suffix] if d]))
        except Exception:
            continue
    return domains

def grounding_quality(citations: List[str], credible_whitelist: List[str] | None = None) -> Tuple[int, int, float]:
    if citations is None:
        citations = []
    domains = parse_domains(citations)
    total = len(domains)
    if total == 0:
        return 0, 0, 0.0
    credible_set = set([d.lower() for d in credible_whitelist or []])
    credible = sum(1 for d in domains if d.lower() in credible_set or d.endswith(".gov") or d.endswith(".edu"))
    return total, credible, credible / total if total else 0.0

def sentiment_score(text: str) -> Tuple[float, float, str]:
    vs = _analyzer.polarity_scores(text)
    # map compound from [-1,1]
    score = vs["compound"]
    # simple confidence proxy: distance from neutral
    confidence = min(1.0, abs(score))
    rationale = f"VADER compound={score:.3f} (pos={vs['pos']:.2f}, neu={vs['neu']:.2f}, neg={vs['neg']:.2f})"
    return score, confidence, rationale
