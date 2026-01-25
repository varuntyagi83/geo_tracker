from typing import List, Dict
from .templates import BASE_SYSTEM, INTERNAL_TEMPLATE, WEB_TEMPLATE

def build_internal_prompt(question: str) -> str:
    return INTERNAL_TEMPLATE.format(system=BASE_SYSTEM, question=question)

def build_web_prompt(question: str, context: str, sources: List[Dict[str,str]]) -> str:
    src_lines = []
    for i, s in enumerate(sources, start=1):
        src_lines.append(f"[{i}] {s.get('title','')} — {s.get('url','')}")
    src_block = "\n".join(src_lines) if src_lines else "No sources"
    return WEB_TEMPLATE.format(system=BASE_SYSTEM, question=question, context=context, sources=src_block)
