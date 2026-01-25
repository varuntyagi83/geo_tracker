BASE_SYSTEM = "You are a careful, honest assistant."
INTERNAL_TEMPLATE = (
    "{system}\n\n"
    "Question:\n{question}\n\n"
    "Answer with clear, verifiable statements. If uncertain, say so."
)
WEB_TEMPLATE = (
    "{system}\n\n"
    "You can use the following external web search context to answer. "
    "Cite claims with [n] where n maps to the source index.\n\n"
    "Question:\n{question}\n\n"
    "Web context:\n{context}\n\n"
    "Sources:\n{sources}\n\n"
    "Answer:"
)
