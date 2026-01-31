# db.py
import os, json, sqlite3, threading
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from config import SQLITE_PATH

# Thread-local storage for connections
_DB_PATH = os.path.abspath(SQLITE_PATH)
_local = threading.local()

print(f"[db] Database path: {_DB_PATH}")

# ---------- Connection & helpers ----------

def _connect() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        con = sqlite3.connect(_DB_PATH, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        _local.conn = con
    return _local.conn

def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def _columns(cur: sqlite3.Cursor, table: str) -> List[Tuple[int,str,str,int,int,int]]:
    # rows: cid, name, type, notnull, dflt_value, pk
    cur.execute(f"PRAGMA table_info({table})")
    return cur.fetchall()

def _colnames(cur: sqlite3.Cursor, table: str) -> List[str]:
    return [c[1] for c in _columns(cur, table)]

def _ensure_fk_on(con: sqlite3.Connection):
    con.execute("PRAGMA foreign_keys=ON;")

def _ensure_fk_off(con: sqlite3.Connection):
    con.execute("PRAGMA foreign_keys=OFF;")

# ---------- Migration ----------

def _needs_migration(cur: sqlite3.Cursor) -> bool:
    # If runs missing 'id' -> migrate
    if _table_exists(cur, "runs"):
        if "id" not in _colnames(cur, "runs"):
            return True
    # If responses missing provider_sources -> migrate
    if _table_exists(cur, "responses"):
        if "provider_sources" not in _colnames(cur, "responses"):
            return True
    # If metrics missing any of the new columns -> migrate
    if _table_exists(cur, "metrics"):
        mcols = set(_colnames(cur, "metrics"))
        needed = {"presence", "sentiment", "trust_authority", "trust_sunday", "details"}
        if not needed.issubset(mcols):
            return True
    return False

def _safe_select_list(existing: List[str], wanted: List[str]) -> str:
    """
    Build a SELECT list that pulls existing columns directly and fills NULL for missing ones,
    preserving target column order.
    """
    parts = []
    ex = set(existing)
    for w in wanted:
        if w in ex:
            parts.append(w)
        else:
            parts.append(f"NULL AS {w}")
    return ", ".join(parts)

def _migrate(cur: sqlite3.Cursor):
    """
    Migrate legacy schemas to:
      runs(id PK, ..., market, lang, extra)
      responses(run_id FK, provider_sources JSON)
      metrics(presence, sentiment, trust_authority, trust_sunday, details)
    Keep run_id mapping by setting new runs.id = OLD rowid.
    """
    # Turn off FK checks while we reshape
    _ensure_fk_off(cur.connection)

    # ---- RUNS ----
    runs_exists = _table_exists(cur, "runs")
    if runs_exists and "id" not in _colnames(cur, "runs"):
        # Create new runs table
        cur.execute("""
        CREATE TABLE runs_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            prompt_id TEXT,
            category TEXT,
            mode TEXT,
            question TEXT,
            prompt_text TEXT,
            market TEXT,
            lang TEXT,
            extra TEXT
        )
        """)
        existing = _colnames(cur, "runs")
        # Source columns we may already have
        wanted = ["run_ts","provider","model","prompt_id","category","mode","question","prompt_text","market","lang","extra"]
        select_list = _safe_select_list(existing, wanted)
        # Preserve mapping: id := old rowid
        cur.execute(f"""
            INSERT INTO runs_new (id, {", ".join(wanted)})
            SELECT rowid, {select_list} FROM runs
        """)
        cur.execute("DROP TABLE runs")
        cur.execute("ALTER TABLE runs_new RENAME TO runs")

    # Ensure runs exists if it did not
    if not _table_exists(cur, "runs"):
        cur.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            prompt_id TEXT,
            category TEXT,
            mode TEXT,
            question TEXT,
            prompt_text TEXT,
            market TEXT,
            lang TEXT,
            extra TEXT
        )
        """)

    # ---- RESPONSES ----
    # Rebuild responses to guarantee FK to runs(id) and provider_sources column
    if _table_exists(cur, "responses"):
        cur.execute("""
        CREATE TABLE responses_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            response_text TEXT,
            latency_ms INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_usd REAL,
            provider_sources TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)
        existing = _colnames(cur, "responses")
        wanted  = ["id","run_id","response_text","latency_ms","tokens_in","tokens_out","cost_usd","provider_sources"]
        select_list = _safe_select_list(existing, wanted)
        # id might not exist previously; if not, NULL => autoincrement on insert
        cur.execute(f"""
            INSERT INTO responses_new ({", ".join(wanted)})
            SELECT {select_list} FROM responses
        """)
        cur.execute("DROP TABLE responses")
        cur.execute("ALTER TABLE responses_new RENAME TO responses")
    else:
        cur.execute("""
        CREATE TABLE responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            response_text TEXT,
            latency_ms INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_usd REAL,
            provider_sources TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)

    # ---- METRICS ----
    if _table_exists(cur, "metrics"):
        cur.execute("""
        CREATE TABLE metrics_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            presence REAL,
            sentiment REAL,
            trust_authority REAL,
            trust_sunday REAL,
            details TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)
        existing = _colnames(cur, "metrics")
        wanted  = ["id","run_id","presence","sentiment","trust_authority","trust_sunday","details"]
        select_list = _safe_select_list(existing, wanted)
        cur.execute(f"""
            INSERT INTO metrics_new ({", ".join(wanted)})
            SELECT {select_list} FROM metrics
        """)
        cur.execute("DROP TABLE metrics")
        cur.execute("ALTER TABLE metrics_new RENAME TO metrics")
    else:
        cur.execute("""
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            presence REAL,
            sentiment REAL,
            trust_authority REAL,
            trust_sunday REAL,
            details TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)

    # Turn FK checks back on
    _ensure_fk_on(cur.connection)

def init_db():
    con = _connect()
    cur = con.cursor()

    # Create minimal shells if absent (so migration logic has tables to look at)
    if not _table_exists(cur, "runs"):
        cur.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            prompt_id TEXT,
            category TEXT,
            mode TEXT,
            question TEXT,
            prompt_text TEXT,
            market TEXT,
            lang TEXT,
            extra TEXT
        )
        """)

    if not _table_exists(cur, "responses"):
        cur.execute("""
        CREATE TABLE responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            response_text TEXT,
            latency_ms INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_usd REAL,
            provider_sources TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)

    if not _table_exists(cur, "metrics"):
        cur.execute("""
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            presence REAL,
            sentiment REAL,
            trust_authority REAL,
            trust_sunday REAL,
            details TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)

    # Run migration if older shapes are detected
    if _needs_migration(cur):
        _migrate(cur)

    # Ensure brands and brand_runs tables exist
    _ensure_brands_table()

    con.commit()

# ---------- Inserts ----------

def insert_run(provider, model, prompt_id, category, mode, question, prompt_text,
               market=None, lang=None, extra=None) -> int:
    con = _connect()
    cur = con.cursor()
    run_ts = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO runs (run_ts, provider, model, prompt_id, category, mode, question, prompt_text, market, lang, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_ts, provider, model, prompt_id, category, mode, question, prompt_text,
        market, lang, json.dumps(extra or {})
    ))
    run_id = cur.lastrowid
    con.commit()
    return run_id

def insert_response(run_id, response_text, latency_ms, tokens_in, tokens_out, cost_usd, provider_sources=None) -> int:
    con = _connect()
    cur = con.cursor()
    # Parent existence check against current PK
    cur.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,))
    if cur.fetchone() is None:
        raise RuntimeError(f"Run id {run_id} not found in 'runs' at {_DB_PATH}")

    cur.execute("""
        INSERT INTO responses (run_id, response_text, latency_ms, tokens_in, tokens_out, cost_usd, provider_sources)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, response_text, latency_ms, tokens_in, tokens_out, cost_usd,
        json.dumps(provider_sources or [])
    ))
    resp_id = cur.lastrowid
    con.commit()
    return resp_id

def insert_metrics(run_id, presence, sentiment, trust_authority, trust_sunday, details=None) -> int:
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,))
    if cur.fetchone() is None:
        raise RuntimeError(f"Run id {run_id} not found in 'runs' at {_DB_PATH}")

    cur.execute("""
        INSERT INTO metrics (run_id, presence, sentiment, trust_authority, trust_sunday, details)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        run_id, presence, sentiment, trust_authority, trust_sunday,
        json.dumps(details or {})
    ))
    mid = cur.lastrowid
    con.commit()
    return mid


# ---------- Recommendations ----------

def _ensure_recommendations_table():
    """Ensure the recommendations table exists."""
    con = _connect()
    cur = con.cursor()
    if not _table_exists(cur, "recommendations"):
        cur.execute("""
        CREATE TABLE recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            created_at TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            brand_name TEXT,
            content TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            tokens_used INTEGER
        )
        """)
        con.commit()


def insert_recommendation(
    job_id: Optional[str],
    analysis_type: str,
    content: str,
    brand_name: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tokens_used: Optional[int] = None
) -> int:
    """Insert a new recommendation/report."""
    _ensure_recommendations_table()
    con = _connect()
    cur = con.cursor()
    created_at = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO recommendations (job_id, created_at, analysis_type, brand_name, content, provider, model, tokens_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, created_at, analysis_type, brand_name, content, provider, model, tokens_used))
    rec_id = cur.lastrowid
    con.commit()
    return rec_id


def get_recommendations(job_id: str) -> List[Dict]:
    """Get recommendations for a specific job."""
    _ensure_recommendations_table()
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        SELECT id, job_id, created_at, analysis_type, brand_name, content, provider, model, tokens_used
        FROM recommendations WHERE job_id = ? ORDER BY created_at DESC
    """, (job_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def get_latest_recommendation(job_id: str, analysis_type: str = "visibility_report") -> Optional[Dict]:
    """Get the most recent recommendation of a specific type for a job."""
    _ensure_recommendations_table()
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        SELECT id, job_id, created_at, analysis_type, brand_name, content, provider, model, tokens_used
        FROM recommendations
        WHERE job_id = ? AND analysis_type = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (job_id, analysis_type))
    row = cur.fetchone()
    if row:
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    return None


# ---------- Brands History ----------

def _ensure_brands_table():
    """Ensure the brands table exists for tracking brand run history."""
    con = _connect()
    cur = con.cursor()
    if not _table_exists(cur, "brands"):
        cur.execute("""
        CREATE TABLE brands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_name TEXT NOT NULL,
            industry TEXT,
            market TEXT,
            company_id TEXT,
            created_at TEXT NOT NULL,
            last_run_at TEXT,
            total_runs INTEGER DEFAULT 0,
            total_queries INTEGER DEFAULT 0,
            avg_visibility REAL,
            extra TEXT
        )
        """)
        # Create index for faster lookups
        cur.execute("CREATE INDEX IF NOT EXISTS idx_brands_name ON brands(brand_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_brands_company ON brands(company_id)")
        con.commit()

    # Ensure brand_runs table exists for linking brands to runs
    if not _table_exists(cur, "brand_runs"):
        cur.execute("""
        CREATE TABLE brand_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_id INTEGER NOT NULL,
            job_id TEXT,
            run_at TEXT NOT NULL,
            providers TEXT,
            mode TEXT,
            total_queries INTEGER,
            visibility_pct REAL,
            avg_sentiment REAL,
            avg_trust REAL,
            competitor_summary TEXT,
            extra TEXT,
            FOREIGN KEY(brand_id) REFERENCES brands(id)
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_brand_runs_brand ON brand_runs(brand_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_brand_runs_job ON brand_runs(job_id)")
        con.commit()


def get_or_create_brand(
    brand_name: str,
    industry: Optional[str] = None,
    market: Optional[str] = None,
    company_id: Optional[str] = None,
) -> int:
    """Get existing brand or create new one. Returns brand_id."""
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    # Normalize brand name (trim whitespace)
    brand_name = brand_name.strip() if brand_name else ""

    # Try to find existing brand (case-insensitive, trimmed)
    cur.execute("""
        SELECT id FROM brands
        WHERE LOWER(TRIM(brand_name)) = LOWER(TRIM(?))
        AND (company_id = ? OR (company_id IS NULL AND ? IS NULL))
    """, (brand_name, company_id, company_id))
    row = cur.fetchone()

    if row:
        # Update industry/market if they've changed (keep brand up-to-date)
        if industry or market:
            cur.execute("""
                UPDATE brands SET
                    industry = COALESCE(?, industry),
                    market = COALESCE(?, market)
                WHERE id = ?
            """, (industry, market, row[0]))
            con.commit()
        return row[0]

    # Create new brand
    created_at = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO brands (brand_name, industry, market, company_id, created_at, total_runs, total_queries)
        VALUES (?, ?, ?, ?, ?, 0, 0)
    """, (brand_name, industry, market, company_id, created_at))
    brand_id = cur.lastrowid
    con.commit()
    return brand_id


def record_brand_run(
    brand_id: int,
    job_id: Optional[str],
    providers: List[str],
    mode: str,
    total_queries: int,
    visibility_pct: float,
    avg_sentiment: Optional[float] = None,
    avg_trust: Optional[float] = None,
    competitor_summary: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> int:
    """Record a run for a brand and update brand stats."""
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    run_at = datetime.now(timezone.utc).isoformat()

    # Insert brand run record
    cur.execute("""
        INSERT INTO brand_runs (brand_id, job_id, run_at, providers, mode, total_queries,
                                visibility_pct, avg_sentiment, avg_trust, competitor_summary, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        brand_id, job_id, run_at, json.dumps(providers), mode, total_queries,
        visibility_pct, avg_sentiment, avg_trust,
        json.dumps(competitor_summary or {}), json.dumps(extra or {})
    ))
    run_record_id = cur.lastrowid

    # Update brand stats
    cur.execute("""
        UPDATE brands SET
            last_run_at = ?,
            total_runs = total_runs + 1,
            total_queries = total_queries + ?,
            avg_visibility = (
                SELECT AVG(visibility_pct) FROM brand_runs WHERE brand_id = ?
            )
        WHERE id = ?
    """, (run_at, total_queries, brand_id, brand_id))

    con.commit()
    return run_record_id


def get_all_brands(company_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """Get all tracked brands, optionally filtered by company_id."""
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    if company_id:
        cur.execute("""
            SELECT id, brand_name, industry, market, company_id, created_at,
                   last_run_at, total_runs, total_queries, avg_visibility, extra
            FROM brands
            WHERE company_id = ?
            ORDER BY last_run_at DESC NULLS LAST, created_at DESC
            LIMIT ?
        """, (company_id, limit))
    else:
        cur.execute("""
            SELECT id, brand_name, industry, market, company_id, created_at,
                   last_run_at, total_runs, total_queries, avg_visibility, extra
            FROM brands
            ORDER BY last_run_at DESC NULLS LAST, created_at DESC
            LIMIT ?
        """, (limit,))

    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def get_brand_by_id(brand_id: int) -> Optional[Dict]:
    """Get a brand by its ID."""
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    cur.execute("""
        SELECT id, brand_name, industry, market, company_id, created_at,
               last_run_at, total_runs, total_queries, avg_visibility, extra
        FROM brands WHERE id = ?
    """, (brand_id,))

    row = cur.fetchone()
    if row:
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    return None


def get_brand_by_name(brand_name: str, company_id: Optional[str] = None) -> Optional[Dict]:
    """Get a brand by its name."""
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    if company_id:
        cur.execute("""
            SELECT id, brand_name, industry, market, company_id, created_at,
                   last_run_at, total_runs, total_queries, avg_visibility, extra
            FROM brands
            WHERE LOWER(brand_name) = LOWER(?) AND company_id = ?
        """, (brand_name, company_id))
    else:
        cur.execute("""
            SELECT id, brand_name, industry, market, company_id, created_at,
                   last_run_at, total_runs, total_queries, avg_visibility, extra
            FROM brands
            WHERE LOWER(brand_name) = LOWER(?)
            ORDER BY last_run_at DESC NULLS LAST
            LIMIT 1
        """, (brand_name,))

    row = cur.fetchone()
    if row:
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    return None


def get_brand_run_history(brand_id: int, limit: int = 20) -> List[Dict]:
    """Get run history for a specific brand."""
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    cur.execute("""
        SELECT id, brand_id, job_id, run_at, providers, mode, total_queries,
               visibility_pct, avg_sentiment, avg_trust, competitor_summary, extra
        FROM brand_runs
        WHERE brand_id = ?
        ORDER BY run_at DESC
        LIMIT ?
    """, (brand_id, limit))

    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    results = []
    for row in rows:
        result = dict(zip(columns, row))
        # Parse JSON fields
        if result.get("providers"):
            try:
                result["providers"] = json.loads(result["providers"])
            except (json.JSONDecodeError, TypeError, ValueError):
                result["providers"] = []
        if result.get("competitor_summary"):
            try:
                result["competitor_summary"] = json.loads(result["competitor_summary"])
            except (json.JSONDecodeError, TypeError, ValueError):
                result["competitor_summary"] = {}
        if result.get("extra"):
            try:
                result["extra"] = json.loads(result["extra"])
            except (json.JSONDecodeError, TypeError, ValueError):
                result["extra"] = {}
        results.append(result)

    return results


def delete_brand(brand_id: int) -> bool:
    """Delete a brand and all its run history."""
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    # Delete run history first
    cur.execute("DELETE FROM brand_runs WHERE brand_id = ?", (brand_id,))
    # Delete the brand
    cur.execute("DELETE FROM brands WHERE id = ?", (brand_id,))

    deleted = cur.rowcount > 0
    con.commit()
    return deleted


def get_all_brand_runs(
    company_id: Optional[str] = None,
    limit: int = 50,
    since_days: int = 30
) -> List[Dict]:
    """
    Get all brand runs with brand info for Previous Runs view.
    Returns properly grouped runs with job_id for detailed viewing.
    """
    _ensure_brands_table()
    con = _connect()
    cur = con.cursor()

    query = """
        SELECT
            br.id,
            br.brand_id,
            br.job_id,
            br.run_at,
            br.providers,
            br.mode,
            br.total_queries,
            br.visibility_pct,
            br.avg_sentiment,
            br.avg_trust,
            br.competitor_summary,
            br.extra,
            b.brand_name,
            b.industry,
            b.market,
            b.company_id
        FROM brand_runs br
        JOIN brands b ON br.brand_id = b.id
        WHERE br.run_at >= datetime('now', ?)
    """
    params = [f'-{since_days} days']

    if company_id:
        query += " AND b.company_id = ?"
        params.append(company_id)

    query += " ORDER BY br.run_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, params)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    results = []
    for row in rows:
        result = dict(zip(columns, row))
        # Parse JSON fields
        if result.get("providers"):
            try:
                result["providers"] = json.loads(result["providers"])
            except (json.JSONDecodeError, TypeError, ValueError):
                result["providers"] = []
        if result.get("competitor_summary"):
            try:
                result["competitor_summary"] = json.loads(result["competitor_summary"])
            except (json.JSONDecodeError, TypeError, ValueError):
                result["competitor_summary"] = {}
        if result.get("extra"):
            try:
                result["extra"] = json.loads(result["extra"])
            except (json.JSONDecodeError, TypeError, ValueError):
                result["extra"] = {}
        results.append(result)

    return results


def clear_all_run_data() -> Dict[str, int]:
    """
    Clear all run data from the database.
    Returns counts of deleted records.
    """
    con = _connect()
    cur = con.cursor()

    # Delete from child tables first (foreign key constraints)
    cur.execute("DELETE FROM metrics")
    metrics_deleted = cur.rowcount

    cur.execute("DELETE FROM responses")
    responses_deleted = cur.rowcount

    cur.execute("DELETE FROM runs")
    runs_deleted = cur.rowcount

    # Clear brand_runs table
    _ensure_brands_table()
    cur.execute("DELETE FROM brand_runs")
    brand_runs_deleted = cur.rowcount

    con.commit()

    return {
        "runs_deleted": runs_deleted,
        "responses_deleted": responses_deleted,
        "metrics_deleted": metrics_deleted,
        "brand_runs_deleted": brand_runs_deleted,
    }


# ---------- Leads Management ----------

def _ensure_leads_table():
    """Ensure the leads table exists for tracking lead submissions."""
    con = _connect()
    cur = con.cursor()
    if not _table_exists(cur, "leads"):
        cur.execute("""
        CREATE TABLE leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            email TEXT NOT NULL,
            website TEXT,
            industry TEXT,
            service TEXT NOT NULL,
            contact_name TEXT,
            status TEXT DEFAULT 'new',
            email_sent INTEGER DEFAULT 0,
            email_id TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(email)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_created ON leads(created_at)")
        con.commit()


def insert_lead(
    company: str,
    email: str,
    service: str,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    contact_name: Optional[str] = None,
    email_sent: bool = False,
    email_id: Optional[str] = None,
) -> int:
    """Insert a new lead into the database."""
    _ensure_leads_table()
    con = _connect()
    cur = con.cursor()
    created_at = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO leads (company, email, website, industry, service, contact_name,
                          email_sent, email_id, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)
    """, (company, email, website, industry, service, contact_name,
          1 if email_sent else 0, email_id, created_at))
    lead_id = cur.lastrowid
    con.commit()
    return lead_id


def get_all_leads(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    include_emails: bool = True
) -> List[Dict]:
    """
    Get all leads, optionally filtered by status.
    If include_emails is False, email addresses are masked for privacy.
    """
    _ensure_leads_table()
    con = _connect()
    cur = con.cursor()

    query = """
        SELECT id, company, email, website, industry, service, contact_name,
               status, email_sent, email_id, notes, created_at, updated_at
        FROM leads
    """
    params = []

    if status:
        query += " WHERE status = ?"
        params.append(status)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cur.execute(query, params)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    results = []
    for row in rows:
        result = dict(zip(columns, row))
        # Mask email if not including full emails
        if not include_emails and result.get("email"):
            email = result["email"]
            if "@" in email:
                local, domain = email.split("@", 1)
                masked_local = local[0] + "***" if len(local) > 1 else "***"
                result["email"] = f"{masked_local}@{domain}"
        results.append(result)

    return results


def get_lead_by_id(lead_id: int) -> Optional[Dict]:
    """Get a specific lead by ID."""
    _ensure_leads_table()
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        SELECT id, company, email, website, industry, service, contact_name,
               status, email_sent, email_id, notes, created_at, updated_at
        FROM leads WHERE id = ?
    """, (lead_id,))
    row = cur.fetchone()
    if row:
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    return None


def update_lead_status(lead_id: int, status: str, notes: Optional[str] = None) -> bool:
    """Update a lead's status and optionally add notes."""
    _ensure_leads_table()
    con = _connect()
    cur = con.cursor()
    updated_at = datetime.now(timezone.utc).isoformat()

    if notes:
        cur.execute("""
            UPDATE leads SET status = ?, notes = ?, updated_at = ?
            WHERE id = ?
        """, (status, notes, updated_at, lead_id))
    else:
        cur.execute("""
            UPDATE leads SET status = ?, updated_at = ?
            WHERE id = ?
        """, (status, updated_at, lead_id))

    updated = cur.rowcount > 0
    con.commit()
    return updated


def get_leads_stats() -> Dict:
    """Get lead statistics for admin dashboard."""
    _ensure_leads_table()
    con = _connect()
    cur = con.cursor()

    # Total leads
    cur.execute("SELECT COUNT(*) FROM leads")
    total = cur.fetchone()[0]

    # By status
    cur.execute("""
        SELECT status, COUNT(*) as count
        FROM leads GROUP BY status
    """)
    by_status = {row[0]: row[1] for row in cur.fetchall()}

    # By service
    cur.execute("""
        SELECT service, COUNT(*) as count
        FROM leads GROUP BY service ORDER BY count DESC
    """)
    by_service = {row[0]: row[1] for row in cur.fetchall()}

    # Recent (last 7 days)
    cur.execute("""
        SELECT COUNT(*) FROM leads
        WHERE created_at >= datetime('now', '-7 days')
    """)
    recent = cur.fetchone()[0]

    # Email success rate
    cur.execute("SELECT COUNT(*) FROM leads WHERE email_sent = 1")
    emails_sent = cur.fetchone()[0]

    return {
        "total": total,
        "by_status": by_status,
        "by_service": by_service,
        "recent_7_days": recent,
        "emails_sent": emails_sent,
        "email_success_rate": round(emails_sent / total * 100, 1) if total > 0 else 0,
    }


# ---------- Admin Users ----------

def _ensure_admin_users_table():
    """Ensure the admin_users table exists."""
    con = _connect()
    cur = con.cursor()
    if not _table_exists(cur, "admin_users"):
        cur.execute("""
        CREATE TABLE admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'demo',
            created_at TEXT NOT NULL,
            last_login TEXT
        )
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_username ON admin_users(username)")
        con.commit()


def create_admin_user(username: str, password_hash: str, role: str = "demo") -> int:
    """Create a new admin user. Role can be 'admin' or 'demo'."""
    _ensure_admin_users_table()
    con = _connect()
    cur = con.cursor()
    created_at = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO admin_users (username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?)
    """, (username, password_hash, role, created_at))
    user_id = cur.lastrowid
    con.commit()
    return user_id


def get_admin_user(username: str) -> Optional[Dict]:
    """Get admin user by username."""
    _ensure_admin_users_table()
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        SELECT id, username, password_hash, role, created_at, last_login
        FROM admin_users WHERE username = ?
    """, (username,))
    row = cur.fetchone()
    if row:
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    return None


def update_admin_last_login(username: str) -> bool:
    """Update admin user's last login timestamp."""
    _ensure_admin_users_table()
    con = _connect()
    cur = con.cursor()
    last_login = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        UPDATE admin_users SET last_login = ? WHERE username = ?
    """, (last_login, username))
    updated = cur.rowcount > 0
    con.commit()
    return updated
