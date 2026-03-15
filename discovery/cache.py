"""
Persistent cache for LLM task outputs.

Designed to be backed by SQLite initially, swappable to your relationalDB later.
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


class TaskCache:
    """
    Simple persistent cache for LLM task outputs.

    Storage: SQLite file (default: discovery/cache.db).
    Each entry is keyed by (task_type, input_hash) and stores the JSON output.

    To swap this with your relational DB later, subclass and override
    get / put / has methods, or replace with a thin wrapper around
    db.query() / db.execute() targeting an `llm_task_cache` table.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "cache.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM llm_task_cache WHERE task_type = ? AND input_hash = ?",
                (task_type, h),
            ).fetchone()
        return row is not None

    def get(self, task_type: str, input_text: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached JSON output, or None if not found."""
        h = self._hash(input_text)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT output_json FROM llm_task_cache WHERE task_type = ? AND input_hash = ?",
                (task_type, h),
            ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, task_type: str, input_text: str, output: Dict[str, Any], model: str = None):
        """Store a result in the cache. Overwrites existing entries."""
        h = self._hash(input_text)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_task_cache
                   (task_type, input_hash, input_text, output_json, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (task_type, h, input_text[:500], json.dumps(output), model, datetime.utcnow().isoformat()),
            )
        logger.debug(f"Cached {task_type} result (hash={h[:8]}...)")

    def clear(self, task_type: str = None):
        """Clear cache entries. If task_type given, clear only that type."""
        with sqlite3.connect(self.db_path) as conn:
            if task_type:
                conn.execute("DELETE FROM llm_task_cache WHERE task_type = ?", (task_type,))
            else:
                conn.execute("DELETE FROM llm_task_cache")

    def stats(self) -> Dict[str, int]:
        """Return count of cached entries by task type."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT task_type, COUNT(*) FROM llm_task_cache GROUP BY task_type"
            ).fetchall()
        return {row[0]: row[1] for row in rows}
