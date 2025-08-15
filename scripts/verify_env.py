# scripts/verify_env.py
import os
from geo_tracker.config import SETTINGS, resolve_mock_mode

def mask(v: str | None) -> str | None:
    if not v:
        return None
    return v[:4] + "..." + v[-4:]

def show(pkg: str):
    try:
        m = __import__(pkg)
        print(f"{pkg:16} OK      v{getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{pkg:16} MISSING {e}")

print("=== ENV (raw) ===")
print("ACTIVE_PROVIDERS:", os.getenv("ACTIVE_PROVIDERS"))
print("MOCK_MODE:", os.getenv("MOCK_MODE"))

print("\n=== Keys loaded by geo_tracker.config ===")
print(f"OPENAI_API_KEY     {'OK' if SETTINGS.openai_api_key else 'MISSING'}  {mask(SETTINGS.openai_api_key)}")
print(f"ANTHROPIC_API_KEY  {'OK' if SETTINGS.anthropic_api_key else 'MISSING'}  {mask(SETTINGS.anthropic_api_key)}")
print(f"GOOGLE_API_KEY     {'OK' if SETTINGS.google_api_key else 'MISSING'}  {mask(SETTINGS.google_api_key)}")
print(f"DATABASE_URL       {SETTINGS.database_url}")

print("\nresolve_mock_mode():", resolve_mock_mode())

print("\n=== Python packages present? ===")
show("openai")
show("anthropic")
show("google.generativeai")
