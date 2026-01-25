"""
Presence metric.
Return:
  - True/False when presence is expected (metric field is non-empty)
  - None when presence is NOT expected (metric field empty), so caller can skip sentiment/trust.
"""
import re


def compute_presence_rate(answer_text: str, metric_field: str):
    if not metric_field or not str(metric_field).strip():
        return None  # presence not expected for this prompt

    if not answer_text:
        return 0.0

    needle = str(metric_field).strip().lower()
    hay = answer_text.lower()

    # Check 1: Exact match of full brand name
    if needle in hay:
        return 1.0

    # Check 2: Check for individual significant words from the brand name
    # This handles cases like "Sunday Natural" being mentioned as just "Sunday"
    brand_words = [w for w in needle.split() if len(w) > 2]  # Skip short words like "of", "the"

    if brand_words:
        # Check if ANY significant word from the brand appears as a standalone word
        for word in brand_words:
            # Use word boundary to avoid partial matches (e.g., "sun" in "sunshine")
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, hay):
                return 1.0

    # Check 3: Domain-style variations (e.g., "sunday.de", "sundaynatural.com")
    brand_no_spaces = needle.replace(" ", "")
    domain_pattern = r'\b' + re.escape(brand_no_spaces) + r'\.(com|de|net|org|co|io)\b'
    if re.search(domain_pattern, hay):
        return 1.0

    # Also check first word + domain (e.g., "sunday.de" for "Sunday Natural")
    if brand_words:
        first_word_domain = r'\b' + re.escape(brand_words[0]) + r'\.(com|de|net|org|co|io)\b'
        if re.search(first_word_domain, hay):
            return 1.0

    return 0.0
