# gsheets.py
import io, pandas as pd, requests, gspread, sys, os
from typing import Optional
from google.oauth2.service_account import Credentials
from config import (
    GSHEET_SPREADSHEET_ID,
    GSHEET_WORKSHEET_NAME,
    GSHEET_AS_PUBLISHED_CSV_URL,
    GOOGLE_APPLICATION_CREDENTIALS,
)

# Column overrides from .env
QUESTION_COL  = os.getenv("QUESTION_COL", "").strip() or None
METRIC_COL    = os.getenv("METRIC_COL", "").strip() or None
CATEGORY_COL  = os.getenv("CATEGORY_COL", "").strip() or None
PROMPT_ID_COL = os.getenv("PROMPT_ID_COL", "").strip() or None
KEYWORDS_COL  = os.getenv("KEYWORDS_COL", "").strip() or None  # <- your keywords source

SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

def read_prompts_dataframe() -> pd.DataFrame:
    # 1) Sheets API (preferred)
    if GOOGLE_APPLICATION_CREDENTIALS and GSHEET_SPREADSHEET_ID:
        try:
            creds = Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS, scopes=SCOPE)
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(GSHEET_SPREADSHEET_ID)
            ws = sh.worksheet(GSHEET_WORKSHEET_NAME)
            records = ws.get_all_records()
            df = pd.DataFrame(records)
            df = _normalize(df)
            print(f"[gsheets] Source: Sheets API • Rows: {len(df)}", file=sys.stderr)
            return df
        except Exception as e:
            print(f"[gsheets] Sheets API failed: {e}", file=sys.stderr)

    # 2) Published CSV (tolerant)
    if GSHEET_AS_PUBLISHED_CSV_URL:
        try:
            r = requests.get(GSHEET_AS_PUBLISHED_CSV_URL, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(
                io.StringIO(r.text),
                engine="python",
                sep=",", quotechar='"', escapechar="\\",
                on_bad_lines="skip"
            )
            df = _normalize(df)
            print(f"[gsheets] Source: Published CSV • Rows: {len(df)}", file=sys.stderr)
            return df
        except Exception as e:
            print(f"[gsheets] CSV fallback failed: {e}", file=sys.stderr)

    # 3) Local sample
    df = pd.read_csv("data/sample_prompts.csv")
    df = _normalize(df)
    print(f"[gsheets] Source: local sample • Rows: {len(df)}", file=sys.stderr)
    return df

def _pick_col(df: pd.DataFrame, target: str, override: Optional[str], candidates: list) -> Optional[str]:
    """Return actual df column name for `target`, using override first, then candidates (case-insensitive)."""
    if override:
        if override in df.columns:
            return override
        for c in df.columns:
            if c.strip().lower() == override.strip().lower():
                return c
        raise ValueError(f"Configured column '{override}' for {target} not found. Columns: {list(df.columns)}")
    lowered = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Question = Prompt_DE
    q_col  = _pick_col(df, "question", QUESTION_COL, ["prompt_de", "question", "frage"])

    # Metric = Metrik (do NOT rewrite Presence to keywords)
    m_col  = _pick_col(df, "metric",   METRIC_COL,   ["metrik", "metric"])

    # Optional: keep keywords separately (not used by runner unless you wire it)
    kw_col = _pick_col(df, "keywords", KEYWORDS_COL, ["google_flywheel_keyword_quelle", "keywords"])

    # Category = Topic (not Category)
    c_col = _pick_col(df, "category", CATEGORY_COL, ["topic", "kategorie", "category"])
    if c_col.lower() == "category" and "Topic" in df.columns:
        c_col = "Topic"  # force Topic over Category

    # Prompt IDs
    pid_col = _pick_col(df, "prompt_id", PROMPT_ID_COL, ["prompt_id", "id"])

    out = pd.DataFrame()
    out["question"] = df[q_col].astype(str).fillna("").str.strip()
    out["category"] = df[c_col].astype(str).fillna("").str.strip()

    # Keep metric exactly as in the sheet (e.g., "Presence", "Presence, Sentiment", etc.)
    out["metric"] = df[m_col].astype(str).fillna("").str.strip()

    # Preserve raw keywords (normalized) in a separate column (not returned by default)
    if kw_col and kw_col in df.columns:
        out["keywords"] = (
            df[kw_col].astype(str).fillna("")
              .str.replace(r"[;|/]+", ",", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
        )
    else:
        out["keywords"] = ""

    # Prompt IDs (use provided if present and non-empty; else autogenerate)
    if pid_col and pid_col in df.columns:
        pid = df[pid_col].astype(str).fillna("").str.strip()
        gen = [f"p{idx+1:03d}" for idx in range(len(df))]
        pid = pid.mask(pid.eq(""), gen)
        out["prompt_id"] = pid
    else:
        out["prompt_id"] = [f"p{idx+1:03d}" for idx in range(len(out))]

    # Return the original four columns so the rest of the app keeps working
    return out[["prompt_id", "category", "question", "metric"]]
