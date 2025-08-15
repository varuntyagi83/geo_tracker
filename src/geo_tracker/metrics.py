from __future__ import annotations
from typing import List, Dict, Tuple

def presence_rate(calls: List[Dict]) -> float:
    if not calls:
        return 0.0
    present = sum(1 for c in calls if c.get("sunday_present", False))
    return present / len(calls)

def share_of_voice(calls: List[Dict]) -> float:
    if not calls:
        return 0.0
    ratios = []
    for c in calls:
        sun = c.get("sunday_mentions", 0)
        comp = c.get("competitor_mentions", 0)
        denom = sun + comp
        if denom > 0:
            ratios.append(sun / denom)
    return sum(ratios) / len(ratios) if ratios else 0.0

def aggregate_sentiment(calls: List[Dict]) -> float:
    vals = [c.get("sentiment", 0.0) for c in calls if "sentiment" in c]
    return sum(vals) / len(vals) if vals else 0.0

def avg_grounding(calls: List[Dict]) -> float:
    vals = [c.get("grounding_fraction", 0.0) for c in calls]
    return sum(vals) / len(vals) if vals else 0.0

def avg_placement(calls: List[Dict]) -> float:
    vals = [c.get("placement", 0.0) for c in calls]
    return sum(vals) / len(vals) if vals else 0.0

def refusal_rate(calls: List[Dict]) -> float:
    if not calls:
        return 0.0
    return sum(1 for c in calls if c.get("refusal", False)) / len(calls)

def hallucination_rate(calls: List[Dict]) -> float:
    # placeholder rule based proxy: 0 for now
    return 0.0

def geo_score(pr: float, sov: float, sent: float, gq: float, pq: float) -> float:
    sent_nonneg = max(0.0, sent)
    return 0.35*pr + 0.25*sov + 0.2*sent_nonneg + 0.1*gq + 0.1*pq
