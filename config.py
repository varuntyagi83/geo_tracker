import os
from dotenv import load_dotenv
load_dotenv()

# Database path - check for Railway volume mount first
# Railway volumes are typically mounted at /data or /app/data
def get_db_path():
    """Get database path, preferring persistent volume if available."""
    # Check explicit env var first
    if os.getenv("SQLITE_PATH"):
        return os.getenv("SQLITE_PATH")
    if os.getenv("DB_PATH"):
        return os.getenv("DB_PATH")

    # Check for Railway volume mount at common locations
    volume_paths = ["/data", "/app/data", "/railway/data"]
    for vol_path in volume_paths:
        if os.path.isdir(vol_path) and os.access(vol_path, os.W_OK):
            db_path = os.path.join(vol_path, "geo_tracker.db")
            print(f"[config] Using persistent volume for database: {db_path}")
            return db_path

    # Fallback to local file (ephemeral on Railway!)
    print("[config] WARNING: No persistent volume found, database will be ephemeral!")
    return "geo_tracker.db"

SQLITE_PATH = get_db_path()

GSHEET_SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID")
GSHEET_WORKSHEET_NAME = os.getenv("GSHEET_WORKSHEET_NAME", "Prompts")
GSHEET_AS_PUBLISHED_CSV_URL = os.getenv("GSHEET_AS_PUBLISHED_CSV_URL")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# For Railway/cloud deployment - inline JSON credentials
GOOGLE_SHEETS_CREDENTIALS_JSON = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

RETRIEVAL_MAX_RESULTS = int(os.getenv("RETRIEVAL_MAX_RESULTS", "3"))
RETRIEVAL_FETCH_TIMEOUT = int(os.getenv("RETRIEVAL_FETCH_TIMEOUT", "10"))

TRUST_MODE = os.getenv("TRUST_MODE", "heuristic")

OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")

# Perplexity API
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_DEFAULT_MODEL = os.getenv("PERPLEXITY_DEFAULT_MODEL", "sonar")

# Anthropic API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_DEFAULT_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-20250514")
