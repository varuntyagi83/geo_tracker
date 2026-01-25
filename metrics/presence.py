"""
Presence metric.
Return:
  - True/False when presence is expected (metric field is non-empty)
  - None when presence is NOT expected (metric field empty), so caller can skip sentiment/trust.
"""
def compute_presence_rate(answer_text: str, metric_field: str):
    if not metric_field or not str(metric_field).strip():
        return None  # presence not expected for this prompt

    if not answer_text:
        return 0.0

    needle = str(metric_field).strip().lower()
    hay = answer_text.lower()

    return 1.0 if needle in hay else 0.0
