"""
Logging Configuration for AI Industry Signals.

Two systems:
  1. Python logging (console + file) — standard module-level logging
  2. SystemLogger  (PostgreSQL system_logs table) — structured event log
     for pipeline runs, errors, cleanup, discovery — queryable from
     DBeaver and Streamlit.

Usage:
    from logging_config import syslog

    syslog.info('pipeline', 'transcription', 'Transcribed Ep.252 in 38s',
                content_id=42, duration_sec=38.2)
    syslog.error('pipeline', 'transcription', 'CUDA OOM on Ep.250',
                 content_id=50, details={'gpu_mem': '7.8GB'})
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

# ── Standard Python logging ──────────────────────────────────────────

def setup_logging(level=logging.DEBUG, log_file=None):
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/system_{timestamp}.log"
    
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
            logging.FileHandler(log_file.replace('.log', '_errors.log'), mode='w')
        ]
    )
    
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) and 'errors' in handler.baseFilename:
            handler.setLevel(logging.ERROR)
    
    loggers_to_debug = [
        'agents',
        'agents.factory', 
        'agents.sql_agent',
        'agents.vector_agent',
        'agents.web_agent',
        'agents.router_agent',
        'db_vector',
        'db_relational',
        'tools',
        'tools.web_search',
        'etl',
        'discovery',
    ]
    
    for logger_name in loggers_to_debug:
        logging.getLogger(logger_name).setLevel(level)
    
    print(f"Logging configured:")
    print(f"   Level: {logging.getLevelName(level)}")
    print(f"   Console: YES")
    print(f"   File: {log_file}")
    print(f"   Errors: {log_file.replace('.log', '_errors.log')}")

def setup_quiet_logging():
    setup_logging(level=logging.INFO, log_file="logs/production.log")

def setup_debug_logging():
    setup_logging(level=logging.DEBUG, log_file="logs/debug.log")

def setup_silent_logging():
    setup_logging(level=logging.ERROR, log_file="logs/errors.log")


# ── SystemLogger (PostgreSQL-backed) ─────────────────────────────────

class SystemLogger:
    """
    Structured event logger that writes to the system_logs table in
    the industry_signals database.

    Also prints to console so you get real-time feedback.

    Levels: INFO, WARNING, ERROR, CRITICAL
    Sources: pipeline, discovery, cleanup, api, system
    """

    def __init__(self):
        self._db = None
        self._run_id = None

    def _get_db(self):
        """Lazy-load DB connection (avoids import cycles at module load)."""
        if self._db is None:
            try:
                from device_config import config
                from db_relational import relationalDB
                self._db = relationalDB(config.DB_PATH)
            except Exception as e:
                print(f"[syslog] WARNING: Could not connect to DB: {e}")
        return self._db

    def start_run(self, source: str = 'system') -> str:
        """Start a new run and return its run_id. Logs a RUN_START event."""
        self._run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
        self.info(source, 'run_start', f'Run started: {self._run_id}')
        return self._run_id

    def end_run(self, source: str = 'system', summary: str = ''):
        """Log a RUN_END event and clear run_id."""
        self.info(source, 'run_end', f'Run completed: {self._run_id}. {summary}')
        self._run_id = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    # ── Log methods ──

    def info(self, source: str, action: str, message: str, **kwargs):
        self._log('INFO', source, action, message, **kwargs)

    def warning(self, source: str, action: str, message: str, **kwargs):
        self._log('WARNING', source, action, message, **kwargs)

    def error(self, source: str, action: str, message: str, **kwargs):
        self._log('ERROR', source, action, message, **kwargs)

    def critical(self, source: str, action: str, message: str, **kwargs):
        self._log('CRITICAL', source, action, message, **kwargs)

    def _log(self, level: str, source: str, action: str, message: str,
             content_id: int = None, duration_sec: float = None,
             details: dict = None, **extra):
        """Write one row to system_logs and print to console."""
        now = datetime.utcnow().isoformat()
        details_json = json.dumps({**details, **extra}) if (details or extra) else None

        # Console output
        icon = {'INFO': '●', 'WARNING': '⚠', 'ERROR': '✗', 'CRITICAL': '‼'}.get(level, '?')
        ts = datetime.utcnow().strftime('%H:%M:%S')
        print(f"[{ts}] {icon} {level:<8} {source}/{action}: {message}")

        # DB insert
        db = self._get_db()
        if db is None:
            return
        try:
            db.execute("""
                INSERT INTO system_logs
                    (timestamp, level, source, action, message, details_json,
                     content_id, duration_sec, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [now, level, source, action, message, details_json,
                  content_id, duration_sec, self._run_id])
        except Exception as e:
            print(f"[syslog] Failed to write log: {e}")


# ── Singleton instance — import this everywhere ──
syslog = SystemLogger()


# ── CLI test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n1. Standard Python logging:")
    setup_debug_logging()
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.error("This is an error message")
    
    print("\n2. SystemLogger (DB-backed):")
    run_id = syslog.start_run('system')
    syslog.info('pipeline', 'transcription', 'Transcribed Ep.252', content_id=42, duration_sec=38.2)
    syslog.warning('discovery', 'search', 'Tavily rate limit approaching', details={'remaining': 50})
    syslog.error('pipeline', 'transcription', 'CUDA OOM on large file', content_id=99)
    syslog.end_run('system', summary='3 events logged')
