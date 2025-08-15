import sys; sys.path.append("src")
from geo_tracker.connectors.google_conn import GoogleConnector
r = GoogleConnector().generate("One sentence test about magnesium.", "en", "DE")
print("GOOGLE:", r.model, r.version); print(r.text[:200])
