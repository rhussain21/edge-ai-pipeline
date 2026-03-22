"""
Persistent cache for LLM task outputs and source health tracking.

Auto-detects backend via device_config:
  - Mac:    SQLite file  (discovery/cache.db)  — local, zero-config
  - Jetson: PostgreSQL   (industry_signals DB)  — same port 5432, visible in DBeaver

Cache keys are built from a hash of (task_type + input_text) so identical
classification requests return cached results without burning LLM tokens.
"""

import hashlib
import json
import os
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ── Backend detection ────────────────────────────────────────────────

def _detect_cache_backend() -> str:
    """Return 'postgres' on Jetson, 'sqlite' everywhere else."""
    try:
        from device_config import config
        if getattr(config, 'DB_BACKEND', '').lower() in ('postgres', 'postgresql'):
            return 'postgres'
    except ImportError:
        pass
    return 'sqlite'


# ── Mixin: backend-aware SQL execution ───────────────────────────────

class _CacheMixin:
    """Provides _exec / _fetchone / _fetchall that work on both backends.

    Subclasses call ``self._setup_backend(db_path)`` in their ``__init__``.
    All SQL uses ``?`` placeholders — converted to ``%s`` for PostgreSQL.
    """

    def _setup_backend(self, db_path: str = None):
        self._backend = _detect_cache_backend()
        if self._backend == 'postgres':
            import psycopg2
            self._pg_conn = psycopg2.connect(
                host=os.getenv('PG_HOST', 'localhost'),
                port=os.getenv('PG_PORT', '5432'),
                database=os.getenv('PG_DB', 'industry_signals'),
                user=os.getenv('PG_USER', 'redwan'),
                password=os.getenv('PG_PASSWORD', ''),
            )
            self._pg_conn.autocommit = True
        else:
            if db_path is None:
                db_path = os.path.join(os.path.dirname(__file__), "cache.db")
            self._sqlite_path = db_path

    # ── low-level helpers ────────────────────────────────────────────

    def _exec(self, query: str, params=None):
        """Execute a write query (CREATE, INSERT, UPDATE, DELETE)."""
        if self._backend == 'postgres':
            query = query.replace('?', '%s')
            cur = self._pg_conn.cursor()
            cur.execute(query, params)
        else:
            with sqlite3.connect(self._sqlite_path) as conn:
                conn.execute(query, params or ())

    def _fetchone(self, query: str, params=None):
        if self._backend == 'postgres':
            query = query.replace('?', '%s')
            cur = self._pg_conn.cursor()
            cur.execute(query, params)
            return cur.fetchone()
        else:
            with sqlite3.connect(self._sqlite_path) as conn:
                return conn.execute(query, params or ()).fetchone()

    def _fetchall(self, query: str, params=None):
        if self._backend == 'postgres':
            query = query.replace('?', '%s')
            cur = self._pg_conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
        else:
            with sqlite3.connect(self._sqlite_path) as conn:
                return conn.execute(query, params or ()).fetchall()

    def _create_table(self, ddl: str):
        """Execute CREATE TABLE, adapting AUTOINCREMENT → SERIAL for PG."""
        if self._backend == 'postgres':
            ddl = ddl.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
        self._exec(ddl)

    def _has_column(self, table: str, column: str) -> bool:
        """Check if a column exists in a table (migration helper)."""
        if self._backend == 'postgres':
            row = self._fetchone(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = ? AND column_name = ?",
                (table, column),
            )
            return row is not None
        else:
            try:
                self._fetchone(f"SELECT {column} FROM {table} LIMIT 1")
                return True
            except sqlite3.OperationalError:
                return False


# ── TaskCache ────────────────────────────────────────────────────────

class TaskCache(_CacheMixin):
    """
    Persistent cache for LLM task outputs.

    Mac  → SQLite file (discovery/cache.db)
    Jetson → PostgreSQL table ``llm_task_cache`` in the industry_signals DB

    Each entry is keyed by (task_type, input_hash) and stores the JSON output.
    """

    def __init__(self, db_path: str = None):
        self._setup_backend(db_path)
        self._init_db()

    def _init_db(self):
        self._create_table("""
            CREATE TABLE IF NOT EXISTS llm_task_cache (
                task_type   TEXT NOT NULL,
                input_hash  TEXT NOT NULL,
                input_text  TEXT,
                output_json TEXT NOT NULL,
                model       TEXT,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (task_type, input_hash)
            )
        """)

    @staticmethod
    def _hash(text: str) -> str:
        """Deterministic hash of input text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def has(self, task_type: str, input_text: str) -> bool:
        """Check if a cached result exists."""
        h = self._hash(input_text)
        row = self._fetchone(
            "SELECT 1 FROM llm_task_cache WHERE task_type = ? AND input_hash = ?",
            (task_type, h),
        )
        return row is not None

    def get(self, task_type: str, input_text: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached JSON output, or None if not found."""
        h = self._hash(input_text)
        row = self._fetchone(
            "SELECT output_json FROM llm_task_cache WHERE task_type = ? AND input_hash = ?",
            (task_type, h),
        )
        if row:
            return json.loads(row[0])
        return None

    def put(self, task_type: str, input_text: str, output: Dict[str, Any], model: str = None):
        """Store a result in the cache. Overwrites existing entries."""
        h = self._hash(input_text)
        now = datetime.utcnow().isoformat()
        output_json = json.dumps(output)
        params = (task_type, h, input_text[:500], output_json, model, now)

        if self._backend == 'postgres':
            self._exec("""
                INSERT INTO llm_task_cache
                    (task_type, input_hash, input_text, output_json, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (task_type, input_hash) DO UPDATE SET
                    input_text  = EXCLUDED.input_text,
                    output_json = EXCLUDED.output_json,
                    model       = EXCLUDED.model,
                    created_at  = EXCLUDED.created_at
            """, params)
        else:
            self._exec("""
                INSERT OR REPLACE INTO llm_task_cache
                   (task_type, input_hash, input_text, output_json, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
            """, params)

        logger.debug(f"Cached {task_type} result (hash={h[:8]}...)")

    def clear(self, task_type: str = None):
        """Clear cache entries. If task_type given, clear only that type."""
        if task_type:
            self._exec("DELETE FROM llm_task_cache WHERE task_type = ?", (task_type,))
        else:
            self._exec("DELETE FROM llm_task_cache")

    def stats(self) -> Dict[str, int]:
        """Return count of cached entries by task type."""
        rows = self._fetchall(
            "SELECT task_type, COUNT(*) FROM llm_task_cache GROUP BY task_type"
        )
        return {row[0]: row[1] for row in rows}


# ── SourceHealthTracker ──────────────────────────────────────────────

class SourceHealthTracker(_CacheMixin):
    """
    Tracks download attempt outcomes and discovery classification rates per source.

    Mac  → shares discovery/cache.db with TaskCache
    Jetson → tables in PostgreSQL industry_signals DB (port 5432)

    Tables:
        download_attempts      — one row per download try
        source_discovery_stats — one row per discovery run per source
    """

    def __init__(self, db_path: str = None):
        self._setup_backend(db_path)
        self._init_tables()

    def _init_tables(self):
        self._create_table("""
            CREATE TABLE IF NOT EXISTS download_attempts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url  TEXT NOT NULL,
                item_title  TEXT,
                adapter     TEXT,
                content_type TEXT,
                success     INTEGER NOT NULL,
                error_type  TEXT,
                error_msg   TEXT,
                attempted_at TEXT NOT NULL
            )
        """)
        self._create_table("""
            CREATE TABLE IF NOT EXISTS source_discovery_stats (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url   TEXT NOT NULL,
                adapter      TEXT,
                approved     INTEGER NOT NULL DEFAULT 0,
                rejected     INTEGER NOT NULL DEFAULT 0,
                run_id       TEXT,
                recorded_at  TEXT NOT NULL
            )
        """)
        self._exec("CREATE INDEX IF NOT EXISTS idx_dl_source ON download_attempts(source_url)")
        self._exec("CREATE INDEX IF NOT EXISTS idx_ds_source ON source_discovery_stats(source_url)")

        # Migration: add content_type column if missing (for existing DBs)
        if not self._has_column('download_attempts', 'content_type'):
            self._exec("ALTER TABLE download_attempts ADD COLUMN content_type TEXT")

    def log_download(self, source_url: str, item_title: str, adapter: str,
                     success: bool, content_type: str = None,
                     error_type: str = None, error_msg: str = None):
        """Log a single download attempt outcome."""
        self._exec(
            """INSERT INTO download_attempts
               (source_url, item_title, adapter, content_type, success, error_type, error_msg, attempted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_url, item_title, adapter, content_type, int(success),
             error_type, error_msg[:300] if error_msg else None,
             datetime.utcnow().isoformat()),
        )

    def log_discovery_batch(self, source_url: str, adapter: str,
                            approved: int, rejected: int, run_id: str = None):
        """Log approved/rejected counts for a source after a discovery run."""
        self._exec(
            """INSERT INTO source_discovery_stats
               (source_url, adapter, approved, rejected, run_id, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source_url, adapter, approved, rejected, run_id,
             datetime.utcnow().isoformat()),
        )

    def get_source_health_report(self) -> list:
        """
        Return per-source health summary across all recorded history.

        Each entry includes: source_url, adapter, total_downloads, success_rate,
        error_403_rate, total_discovery_runs, avg_approval_rate, flagged.
        """
        dl_rows = self._fetchall("""
            SELECT source_url, adapter,
                   COUNT(*) as total,
                   SUM(success) as successes,
                   SUM(CASE WHEN error_type = '403' THEN 1 ELSE 0 END) as forbidden
            FROM download_attempts
            GROUP BY source_url, adapter
        """)

        disc_rows = self._fetchall("""
            SELECT source_url, adapter,
                   COUNT(*) as runs,
                   SUM(approved) as total_approved,
                   SUM(rejected) as total_rejected
            FROM source_discovery_stats
            GROUP BY source_url, adapter
        """)

        dl_map = {(r[0], r[1]): r for r in dl_rows}
        disc_map = {(r[0], r[1]): r for r in disc_rows}

        all_sources = set(dl_map.keys()) | set(disc_map.keys())
        report = []
        for key in sorted(all_sources):
            dl = dl_map.get(key)
            ds = disc_map.get(key)

            total_dl = dl[2] if dl else 0
            successes = dl[3] if dl else 0
            forbidden = dl[4] if dl else 0
            success_rate = round(successes / total_dl, 2) if total_dl else None
            error_403_rate = round(forbidden / total_dl, 2) if total_dl else None

            runs = ds[2] if ds else 0
            total_approved = ds[3] if ds else 0
            total_rejected = ds[4] if ds else 0
            total_classified = total_approved + total_rejected
            approval_rate = round(total_approved / total_classified, 2) if total_classified else None

            flagged = (
                (error_403_rate is not None and error_403_rate >= 0.5) or
                (approval_rate is not None and approval_rate < 0.2 and total_classified >= 5)
            )

            report.append({
                "source_url": key[0],
                "adapter": key[1],
                "downloads": {"total": total_dl, "success_rate": success_rate, "error_403_rate": error_403_rate},
                "discovery": {"runs": runs, "avg_approval_rate": approval_rate},
                "flagged": flagged,
            })

        return report

    def get_flagged_sources(self, min_403_rate: float = 0.5, min_rejection_rate: float = 0.8) -> list:
        """
        Return sources that consistently fail or produce low-quality candidates.

        Args:
            min_403_rate: Flag if 403 errors exceed this fraction of download attempts.
            min_rejection_rate: Flag if rejections exceed this fraction of classified candidates.
        """
        report = self.get_source_health_report()
        return [
            r for r in report
            if (r["downloads"]["error_403_rate"] or 0) >= min_403_rate
            or (r["discovery"]["avg_approval_rate"] is not None
                and r["discovery"]["avg_approval_rate"] <= (1 - min_rejection_rate))
        ]
