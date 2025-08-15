import sys, os; sys.path.append("src")
from geo_tracker.connectors.anthropic_conn import AnthropicConnector
r = AnthropicConnector().generate("One sentence test answer about magnesium.", "en", "DE")
print("ANTHROPIC:", r.model, r.version, r.text[:200])
