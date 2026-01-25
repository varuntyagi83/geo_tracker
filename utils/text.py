import re
def clean_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t
