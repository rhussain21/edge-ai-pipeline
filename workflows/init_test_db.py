#!/usr/bin/env python3
"""
Initialize a minimal DuckDB test database with only content + signals tables.

This is intended for staging/test runs where you want a lightweight DB while
keeping the same schema shape for the two primary tables used by testing.

Usage:
    python workflows/init_test_db.py
    python workflows/init_test_db.py --db-path Database/industry_signals_test.db --force
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
from device_config import config


def init_test_db(db_path: str, force: bool = False) -> None:
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    if force and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing DB: {db_path}")

    con = duckdb.connect(db_path)
    try:
        # content table (same columns as db_relational._init_db_duckdb)
        con.execute('''
            CREATE TABLE IF NOT EXISTS content (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT,
                pub_date TEXT,
                file_path TEXT UNIQUE NOT NULL,
                audio_url TEXT,
                duration_seconds REAL,
                file_size_mb REAL,
                content_hash TEXT,
                transcript TEXT,
                language TEXT,
                transcription_date TEXT,
                transcription_model TEXT,
                extraction_hardware TEXT,
                extraction_status TEXT DEFAULT 'pending',
                vectorization_status TEXT DEFAULT 'pending',
                signal_processed BOOLEAN DEFAULT FALSE,
                screening_status TEXT DEFAULT 'pending',
                screening_reason TEXT,
                screened_at TEXT,
                marked_for_deletion BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                segments TEXT,
                metadata_json TEXT
            )
        ''')

        con.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON content(source_type)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_extraction_status ON content(extraction_status)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_content_date ON content(pub_date)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON content(content_hash)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_signal_processed ON content(signal_processed)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_screening_status ON content(screening_status)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_marked_deletion ON content(marked_for_deletion)')

        # signals table (same columns as db_relational._init_db_duckdb)
        con.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                signal_type TEXT NOT NULL,
                entity TEXT NOT NULL,
                description TEXT,
                industry TEXT,
                impact_level TEXT,
                confidence REAL,
                timeline TEXT,
                metadata_json TEXT,
                source_content_id INTEGER NOT NULL,
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                vectorized BOOLEAN DEFAULT FALSE,
                vectorized_at TEXT,
                context_window TEXT,
                enriched_text TEXT,
                enrichment_version TEXT
            )
        ''')

        con.execute('CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_signals_entity ON signals(entity)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_signals_industry ON signals(industry)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_signals_content ON signals(source_content_id)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_signal_vectorized ON signals(vectorized)')

        print(f"Initialized minimal test DB at: {db_path}")
        tables = con.execute("SHOW TABLES").fetchall()
        print(f"Tables: {[t[0] for t in tables]}")
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="Initialize minimal test DB (content + signals only)")
    parser.add_argument(
        '--db-path',
        default=config.DB_PATH,
        help=f"Target DB path (default: {config.DB_PATH})",
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete existing DB file before initialization',
    )
    args = parser.parse_args()

    init_test_db(args.db_path, force=args.force)


if __name__ == '__main__':
    main()
