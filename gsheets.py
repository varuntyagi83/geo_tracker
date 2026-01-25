# gsheets.py
import io
import os
import sys
from typing import Optional, List

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials

from config import (
    GSHEET_SPREADSHEET_ID,
    GSHEET_WORKSHEET_NAME,
    GSHEET_AS_PUBLISHED_CSV_URL,
    GOOGLE_APPLICATION_CREDENTIALS,
)

# -----------------------------------------------------------------------------
# Optional env overrides (if set, they take precedence)
# -----------------------------------------------------------------------------
QUESTION_COL  = (os.getenv("QUESTION_COL") or "").strip() or None
CATEGORY_COL  = (os.getenv("CATEGORY_COL") or "").strip() or None
PROMPT_ID_COL = (os.getenv("PROMPT_ID_COL") or "").strip() or None
KEYWORDS_COL  = (os.getenv("KEYWORDS_COL") or "").strip() or None  # not returned

# Intentionally NO METRIC_COL anymore.

SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Canonical output columns expected by the app
CANON = ["prompt_id", "category", "question", "metric"]

# -----------------------------------------------------------------------------
# Header candidates (case/space/punct insensitive)
# -----------------------------------------------------------------------------
QUESTION_CANDIDATES = [
    # New
    "shopping intent prompts, some general vms prompts",
    # Old / fallbacks
    "prompt_de", "question", "frage", "prompt",
]

CATEGORY_CANDIDATES = [
    # New
    "geo topic",
    # Old / fallbacks
    "topic", "kategorie", "category",
]

PROMPT_ID_CANDIDATES = ["prompt_id", "id"]

# Not returned, but useful to track lineage
KEYWORDS_CANDIDATES = [
    # New
    "keyword de",
    # Old
    "google_flywheel_keyword_quelle",
    # Generic
    "keywords",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm(s: str) -> str:
    """Normalize a header key for matching (lowercase + alnum only)."""
    return "".join(ch for ch in s.strip().lower() if ch.isalnum())

def _pick_col(df: pd.DataFrame, target: str, override: Optional[str], candidates: List[str]) -> Optional[str]:
    """
    Return the actual df column name for `target`.
    Priority: explicit env override -> candidates (case-insensitive, punctuation-insensitive).
    Raises if override is provided but not found.
    """
    if df is None or df.empty:
        return None

    if override:
        if override in df.columns:
            return override
        lowered = {c.lower(): c for c in df.columns}
        if override.lower() in lowered:
            return lowered[override.lower()]
        raise ValueError(
            f"Configured column '{override}' for {target} not found. "
            f"Available columns: {list(df.columns)}"
        )

    normalized_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in normalized_map:
            return normalized_map[key]

    return None

# -----------------------------------------------------------------------------
# Normalization (NO metric mapping — metric always empty)
# -----------------------------------------------------------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw sheet dataframe to the canonical shape:
        ['prompt_id', 'category', 'question', 'metric']
    No 'metric' is mapped — it is always empty.
    We also store the chosen mapping in df.attrs['colmap'] for visibility.
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=CANON)
        out.attrs["colmap"] = {}
        return out

    # Map question & category to the new headers (with old fallbacks)
    q_col  = _pick_col(df, "question",  QUESTION_COL,  QUESTION_CANDIDATES)
    c_col  = _pick_col(df, "category",  CATEGORY_COL,  CATEGORY_CANDIDATES)
    pidcol = _pick_col(df, "prompt_id", PROMPT_ID_COL, PROMPT_ID_CANDIDATES)
    kw_col = _pick_col(df, "keywords",  KEYWORDS_COL,  KEYWORDS_CANDIDATES)  # optional, not returned

    if not q_col:
        raise ValueError(
            "Could not detect the 'question' column. "
            f"Set QUESTION_COL env var or ensure one of these headers exists: {QUESTION_CANDIDATES}. "
            f"Current columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["question"] = (
        df[q_col].astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    out["category"] = df[c_col].astype(str).fillna("").str.strip() if c_col else ""

    # IMPORTANT: metric is intentionally NOT mapped anymore
    out["metric"] = ""  # stays empty so presence/sentiment are skipped

    # prompt_id: use provided if non-empty; else synthesize p001…
    if pidcol:
        pid = df[pidcol].astype(str).fillna("").str.strip()
        gen = [f"p{idx+1:03d}" for idx in range(len(df))]
        pid = pid.mask(pid.eq(""), gen)
        out["prompt_id"] = pid
    else:
        out["prompt_id"] = [f"p{idx+1:03d}" for idx in range(len(out))]

    # Trim and order
    for c in CANON:
        out[c] = out[c].astype(str).str.strip()
    out = out[CANON]

    # Save mapping for visibility
    colmap = {
        "question_col": q_col,             # "Shopping intent prompts, some general VMS prompts" or old "Prompt_DE"
        "category_col": c_col or "(none)", # "GEO Topic" or old "Topic"
        "prompt_id_col": pidcol or "(auto)",
        "metric_col": "(none)",            # explicitly none
        "keywords_col": kw_col or "(unused)",  # "Keyword DE" or old "Google_Flywheel_Keyword_Quelle" (not returned)
        "all_columns": list(df.columns),
    }
    out.attrs["colmap"] = colmap
    return out

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def read_prompts_dataframe() -> pd.DataFrame:
    """
    Reads the configured Google Sheet (API first, CSV fallback) and returns a DataFrame with:
        ['prompt_id', 'category', 'question', 'metric']  (metric is empty)
    Prints a one-line mapping summary so you can verify the forwarded prompt column.
    """
    # 1) Sheets API (preferred)
    if GOOGLE_APPLICATION_CREDENTIALS and GSHEET_SPREADSHEET_ID:
        try:
            creds = Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS, scopes=SCOPE)
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(GSHEET_SPREADSHEET_ID)
            ws = sh.worksheet(GSHEET_WORKSHEET_NAME)
            records = ws.get_all_records()  # list[dict]
            df_raw = pd.DataFrame(records)
            df = _normalize(df_raw)
            colmap = df.attrs.get("colmap", {})
            print(
                "[gsheets] Source: Sheets API • Rows: {} • Mapping: question='{}' category='{}' prompt_id='{}' metric='{}'".format(
                    len(df),
                    colmap.get("question_col"),
                    colmap.get("category_col"),
                    colmap.get("prompt_id_col"),
                    colmap.get("metric_col"),
                ),
                file=sys.stderr
            )
            return df
        except Exception as e:
            print(f"[gsheets] Sheets API failed: {e}", file=sys.stderr)

    # 2) Published CSV (tolerant)
    if GSHEET_AS_PUBLISHED_CSV_URL:
        try:
            r = requests.get(GSHEET_AS_PUBLISHED_CSV_URL, timeout=20)
            r.raise_for_status()
            df_raw = pd.read_csv(
                io.StringIO(r.text),
                engine="python",
                sep=",",
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",
            )
            df = _normalize(df_raw)
            colmap = df.attrs.get("colmap", {})
            print(
                "[gsheets] Source: Published CSV • Rows: {} • Mapping: question='{}' category='{}' prompt_id='{}' metric='{}'".format(
                    len(df),
                    colmap.get("question_col"),
                    colmap.get("category_col"),
                    colmap.get("prompt_id_col"),
                    colmap.get("metric_col"),
                ),
                file=sys.stderr
            )
            return df
        except Exception as e:
            print(f"[gsheets] CSV fallback failed: {e}", file=sys.stderr)

    # 3) Local sample (very last resort)
    try:
        df_raw = pd.read_csv("data/sample_prompts.csv")
        df = _normalize(df_raw)
        colmap = df.attrs.get("colmap", {})
        print(
            "[gsheets] Source: local sample • Rows: {} • Mapping: question='{}' category='{}' prompt_id='{}' metric='{}'".format(
                len(df),
                colmap.get("question_col"),
                colmap.get("category_col"),
                colmap.get("prompt_id_col"),
                colmap.get("metric_col"),
            ),
            file=sys.stderr
        )
        return df
    except Exception as e:
        raise RuntimeError(f"No valid Sheets source and no local sample available. Last error: {e}") from e
