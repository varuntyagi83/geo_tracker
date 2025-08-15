from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    perplexity_api_key: str | None = os.getenv("PERPLEXITY_API_KEY")
    together_api_key: str | None = os.getenv("TOGETHER_API_KEY")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./geo_tracker.db")
    brand_name: str = "Sunday Natural"
    market_default: str = "DE"
    language_default: str = "en"
    credibility_whitelist_path: str = os.getenv("CREDIBILITY_LIST", "")
    model_timeout_s: int = int(os.getenv("MODEL_TIMEOUT_S", "60"))
    mock_mode: bool = os.getenv("MOCK_MODE", "auto").lower()  # "on", "off", or "auto"

SETTINGS = Settings()

def resolve_mock_mode() -> bool:
    if SETTINGS.mock_mode == "on":
        return True
    if SETTINGS.mock_mode == "off":
        return False
    # auto: enable mock if no keys are present
    has_any_key = any([SETTINGS.openai_api_key, SETTINGS.anthropic_api_key, SETTINGS.google_api_key, SETTINGS.perplexity_api_key, SETTINGS.together_api_key])
    return not has_any_key
