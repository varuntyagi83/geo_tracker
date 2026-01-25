"""
Service for reading user-provided Google Sheets.

Supports:
- Extracting sheet ID from various URL formats
- Validating sheet access (service account must have read permission)
- Auto-detecting column names (flexible naming, multi-language)
- Caching to avoid repeated API calls
"""
import re
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
    _HAS_GSPREAD = True
except ImportError:
    _HAS_GSPREAD = False

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_SHEETS_CREDENTIALS_JSON

# Simple in-memory cache
_sheet_cache: Dict[str, Tuple[datetime, pd.DataFrame, str]] = {}
CACHE_TTL_MINUTES = 15

SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Column detection patterns (case-insensitive, supports multiple languages)
QUESTION_PATTERNS = [
    r"question", r"prompt", r"frage", r"pregunta", r"domanda",
    r"query", r"text", r"input", r"shopping.*prompt", r"geo.*prompt",
    r"prompt[_\s]?de", r"prompt[_\s]?en", r"prompt[_\s]?fr",
]
CATEGORY_PATTERNS = [
    r"category", r"kategorie", r"topic", r"tema", r"type",
    r"geo.*topic", r"thema", r"categor[ií]a",
]


def extract_sheet_id(url_or_id: str) -> str:
    """
    Extract Google Sheet ID from various formats:
    - Full URL: https://docs.google.com/spreadsheets/d/SHEET_ID/edit
    - Sharing URL: https://docs.google.com/spreadsheets/d/SHEET_ID/...
    - Just the ID: 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms
    """
    if not url_or_id:
        raise ValueError("Sheet URL or ID is required")

    url_or_id = url_or_id.strip()

    # Try to extract from URL
    match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url_or_id)
    if match:
        return match.group(1)

    # Assume it's a raw ID if no URL pattern found
    if re.match(r'^[a-zA-Z0-9-_]+$', url_or_id) and len(url_or_id) > 20:
        return url_or_id

    raise ValueError(f"Could not extract sheet ID from: {url_or_id}")


def _detect_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Find a column matching any of the patterns."""
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for pattern in patterns:
            if re.search(pattern, col_lower):
                return col
    return None


def _cache_key(sheet_id: str, worksheet: str) -> str:
    return hashlib.md5(f"{sheet_id}:{worksheet}".encode()).hexdigest()


def fetch_sheet_prompts(
    sheet_url_or_id: str,
    worksheet_name: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Fetch prompts from a user-provided Google Sheet.

    Args:
        sheet_url_or_id: Google Sheet URL or ID
        worksheet_name: Specific worksheet name (defaults to first sheet)
        force_refresh: Bypass cache and fetch fresh data

    Returns:
        {
            "prompts": [{"prompt_id": "p001", "category": "...", "question": "..."}],
            "total_count": 150,
            "columns_detected": {"question": "Question", "category": "Topic"},
            "all_columns": ["Question", "Topic", "Keywords", ...],
            "cached": True/False,
            "sheet_title": "My Prompts Sheet"
        }
    """
    if not _HAS_GSPREAD:
        raise ValueError("gspread library not installed. Run: pip install gspread google-auth")

    # Check for credentials - support both file path and inline JSON
    if not GOOGLE_APPLICATION_CREDENTIALS and not GOOGLE_SHEETS_CREDENTIALS_JSON:
        raise ValueError(
            "Google credentials not configured. "
            "Set GOOGLE_APPLICATION_CREDENTIALS (file path) or GOOGLE_SHEETS_CREDENTIALS_JSON (inline JSON)."
        )

    sheet_id = extract_sheet_id(sheet_url_or_id)
    ws_name = worksheet_name or "Sheet1"

    # Check cache
    cache_key = _cache_key(sheet_id, ws_name)
    if not force_refresh and cache_key in _sheet_cache:
        cached_time, cached_df, cached_title = _sheet_cache[cache_key]
        if datetime.now() - cached_time < timedelta(minutes=CACHE_TTL_MINUTES):
            return _build_response(cached_df, sheet_id, sheet_title=cached_title, from_cache=True)

    # Fetch from API
    try:
        # Support both file path and inline JSON credentials
        if GOOGLE_SHEETS_CREDENTIALS_JSON:
            # Inline JSON (for Railway/cloud deployment)
            creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON)
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)
        else:
            # File path (for local development)
            creds = Credentials.from_service_account_file(
                GOOGLE_APPLICATION_CREDENTIALS, scopes=SCOPE
            )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)

        # Try to find worksheet
        try:
            if worksheet_name:
                ws = sh.worksheet(ws_name)
            else:
                ws = sh.get_worksheet(0)  # Default to first worksheet
        except gspread.WorksheetNotFound:
            # Default to first worksheet
            ws = sh.get_worksheet(0)

        records = ws.get_all_records()
        df = pd.DataFrame(records)

        # Cache the result
        _sheet_cache[cache_key] = (datetime.now(), df, sh.title)

        return _build_response(df, sheet_id, sheet_title=sh.title, from_cache=False)

    except gspread.SpreadsheetNotFound:
        raise ValueError(
            f"Sheet not found. Make sure the sheet is shared with your service account email."
        )
    except gspread.exceptions.APIError as e:
        if "403" in str(e):
            raise ValueError(
                "Access denied. Share the sheet with your service account email "
                "(found in your credentials JSON under 'client_email')."
            )
        raise ValueError(f"Google Sheets API error: {str(e)}")
    except FileNotFoundError:
        raise ValueError(
            f"Service account credentials file not found: {GOOGLE_APPLICATION_CREDENTIALS}"
        )


def _build_response(
    df: pd.DataFrame,
    sheet_id: str,
    sheet_title: str = "",
    from_cache: bool = False
) -> Dict[str, Any]:
    """Build standardized response from DataFrame."""
    if df.empty:
        return {
            "prompts": [],
            "total_count": 0,
            "columns_detected": {"question": None, "category": None},
            "all_columns": [],
            "cached": from_cache,
            "sheet_title": sheet_title,
            "sheet_id": sheet_id,
        }

    # Detect columns
    question_col = _detect_column(df, QUESTION_PATTERNS)
    category_col = _detect_column(df, CATEGORY_PATTERNS)

    if not question_col:
        # Try to find any column that might contain questions
        for col in df.columns:
            # Check if column has string content that looks like questions
            sample = df[col].dropna().head(5).tolist()
            if sample and all(isinstance(s, str) and len(s) > 10 for s in sample):
                question_col = col
                break

    if not question_col:
        raise ValueError(
            f"Could not find question/prompt column. "
            f"Available columns: {list(df.columns)}. "
            f"Please name your column with one of: question, prompt, frage, query, text"
        )

    # Build prompts list
    prompts = []
    for idx, row in df.iterrows():
        question = str(row.get(question_col, "")).strip()
        if not question:
            continue

        category = ""
        if category_col:
            category = str(row.get(category_col, "")).strip()

        prompts.append({
            "prompt_id": f"p{idx+1:03d}",
            "category": category,
            "question": question,
        })

    return {
        "prompts": prompts,
        "total_count": len(prompts),
        "columns_detected": {
            "question": question_col,
            "category": category_col,
        },
        "all_columns": list(df.columns),
        "cached": from_cache,
        "sheet_title": sheet_title,
        "sheet_id": sheet_id,
    }


def get_prompts_subset(
    prompts: List[Dict],
    count: Optional[int] = None,
    start: int = 0,
    end: Optional[int] = None
) -> List[Dict]:
    """
    Get a subset of prompts.

    Args:
        prompts: Full list of prompts
        count: Number of prompts to return (from start)
        start: Starting index
        end: Ending index (exclusive)

    Returns:
        Subset of prompts
    """
    if end is not None:
        return prompts[start:end]
    elif count is not None:
        return prompts[start:start + count]
    return prompts


def clear_cache(sheet_id: Optional[str] = None):
    """Clear the sheet cache."""
    global _sheet_cache
    if sheet_id:
        # Clear specific sheet
        keys_to_remove = [k for k in _sheet_cache if sheet_id in k]
        for k in keys_to_remove:
            del _sheet_cache[k]
    else:
        # Clear all
        _sheet_cache = {}
