import sys, os; sys.path.append("src")
from geo_tracker.connectors.openai_conn import OpenAIConnector
r = OpenAIConnector().generate("One sentence test answer about magnesium.", "en", "DE")
print("OPENAI:", r.model, r.version, r.text[:200])
