# db.py
import os, json, sqlite3, threading
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

# Thread-local storage for connections
_DB_PATH = os.path.abspath(os.getenv("DB_PATH", "geo_tracker.db"))
_local = threading.local()

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
