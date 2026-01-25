"""
Very thin wrapper to allow skipping upstream:
- If answer_text is None, return None.
- Otherwise return a heuristic sentiment score in [-1, 1].
Replace with your classifier if needed.
"""
import re

NEG_WORDS = {"bad","poor","terrible","fake","fraud","scam","not recommended","avoid"}
POS_WORDS = {"good","great","excellent","recommended","love","trust","authentic","credible"}

def compute_sentiment(answer_text: str | None, assume_neutral_if_absent: bool = True):
    if answer_text is None:
        return None
    if not answer_text.strip():
        return 0.0 if assume_neutral_if_absent else None

    text = answer_text.lower()
    neg = sum(1 for w in NEG_WORDS if w in text)
    pos = sum(1 for w in POS_WORDS if w in text)
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / max(pos + neg, 1)
