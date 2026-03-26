import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class relationalDB:
    def __init__(self, db_path):
        if not db_path:
            raise ValueError("Database path cannot be empty")
        
        self.original_path = db_path
        self.backend = self._detect_backend()
        
        if self.backend == 'postgres':
            self._init_postgres()
        else:
            self._init_duckdb(db_path)
        
        self.init_db()
    
    def _detect_backend(self):
        """Auto-detect: Jetson -> PostgreSQL, Mac -> DuckDB."""
        backend = os.getenv('DB_BACKEND', '').lower()
        if backend in ('postgres', 'postgresql'):
            return 'postgres'
        if backend == 'duckdb':
            return 'duckdb'
        if os.path.exists('/etc/nv_tegra_release'):
            return 'postgres'
        return 'duckdb'
    
    def _init_postgres(self):
        """Initialize PostgreSQL connection using env vars."""
        import psycopg2
        self._psycopg2 = psycopg2
        config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5432'),
            'database': os.getenv('PG_DB', 'industry_signals'),
            'user': os.getenv('PG_USER', 'redwan'),
            'password': os.getenv('PG_PASSWORD', ''),
        }
        self.db_path = f"postgresql://{config['host']}:{config['port']}/{config['database']}"
        self.con = psycopg2.connect(**config)
        self.con.autocommit = True
        logger.info(f"Connected to PostgreSQL: {self.db_path}")
        print(f"Backend: PostgreSQL ({self.db_path})")
    
    def _init_duckdb(self, db_path):
        """Initialize DuckDB connection."""
        import duckdb
        self.db_path = self._ensure_valid_path(db_path)
        self.con = duckdb.connect(self.db_path)
        logger.info(f"Connected to DuckDB: {self.db_path}")
        print(f"Backend: DuckDB ({self.db_path})")
    
    def execute(self, query, params=None):
        """Execute SQL with automatic parameter style conversion.
        Returns a cursor-like object with fetchone()/fetchall()."""
        if self.backend == 'postgres':
            query = query.replace('?', '%s')
            cursor = self.con.cursor()
            cursor.execute(query, params)
            return cursor
        else:
            if params:
                return self.con.execute(query, params)
            return self.con.execute(query)
    
    def _ensure_valid_path(self, db_path):
        """Ensure directory exists, fallback to default if needed."""
        try:
            db_dir = os.path.dirname(db_path)
            
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"Created directory: {db_dir}")
            
            test_file = db_path.replace('.db', '_test.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                return db_path
            except (PermissionError, OSError):
                print(f"Cannot write to {db_path}, using default path")
                
        except Exception as e:
            print(f"Path validation failed: {e}")
        
        default_path = "Database/podcasts.db"
        default_dir = os.path.dirname(default_path)
        if default_dir and not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
        
        print(f"Using default path: {default_path}")
        return default_path

    def init_db(self):
        """Initialize database schema for the active backend."""
        if self.backend == 'postgres':
            self._init_db_postgres()
        else:
            self._init_db_duckdb()
        self._migrate_schema()
    
    def _init_db_postgres(self):
        """PostgreSQL schema with proper FK constraints and SERIAL IDs."""
        cursor = self.con.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS content (
                    id SERIAL PRIMARY KEY,
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
                    screened_at TIMESTAMP,
                    marked_for_deletion BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    segments TEXT,
                    metadata_json TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON content(source_type)')
            try:
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_extraction_status ON content(extraction_status)')
            except Exception as e:
                # Column doesn't exist yet - will be added by migration
                if "does not exist" in str(e) or "UndefinedColumn" in str(e):
                    print(f"PostgreSQL init: skipped idx_extraction_status (column missing pre-migration)")
                else:
                    raise
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_date ON content(pub_date)')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    signal_type TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    description TEXT,
                    industry TEXT,
                    impact_level TEXT,
                    confidence REAL,
                    timeline TEXT,
                    metadata_json TEXT,
                    source_content_id INTEGER NOT NULL REFERENCES content(id) ON UPDATE CASCADE,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vectorized BOOLEAN DEFAULT FALSE,
                    vectorized_at TIMESTAMP,
                    context_window TEXT,
                    enriched_text TEXT,
                    enrichment_version TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_entity ON signals(entity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_industry ON signals(industry)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_content ON signals(source_content_id)')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcript_segments (
                    id SERIAL PRIMARY KEY,
                    content_id INTEGER REFERENCES content(id) ON UPDATE CASCADE,
                    segment_index INTEGER,
                    start_time REAL,
                    end_time REAL,
                    speaker_id TEXT,
                    text TEXT,
                    confidence REAL,
                    ground_truth_text TEXT,
                    ground_truth_speaker TEXT,
                    is_corrected BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_content ON transcript_segments(content_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_speaker ON transcript_segments(speaker_id)')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS content_metadata (
                    id SERIAL PRIMARY KEY,
                    content_id INTEGER NOT NULL REFERENCES content(id) ON UPDATE CASCADE,
                    source_url TEXT,
                    publisher TEXT,
                    doc_type TEXT,
                    discovery_confidence REAL,
                    discovery_reason TEXT,
                    topic_tags TEXT,
                    authority TEXT,
                    adapter TEXT,
                    discovered_at TEXT,
                    download_timestamp TEXT,
                    processing_status TEXT DEFAULT 'downloaded',
                    original_format TEXT,
                    extraction_method TEXT,
                    podcast_name TEXT,
                    episode_description TEXT,
                    episode_author TEXT,
                    episode_summary TEXT,
                    episode_keywords TEXT,
                    episode_number TEXT,
                    episode_season TEXT,
                    episode_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_cm_content_id ON content_metadata(content_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cm_doc_type ON content_metadata(doc_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cm_adapter ON content_metadata(adapter)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cm_publisher ON content_metadata(publisher)')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    action TEXT,
                    message TEXT NOT NULL,
                    details_json TEXT,
                    content_id INTEGER,
                    duration_sec REAL,
                    run_id TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_source ON system_logs(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_run_id ON system_logs(run_id)')

            cursor.execute("SELECT setval(pg_get_serial_sequence('content','id'), COALESCE((SELECT MAX(id) FROM content), 0), true)")
            cursor.execute("SELECT setval(pg_get_serial_sequence('signals','id'), COALESCE((SELECT MAX(id) FROM signals), 0), true)")
            print("PostgreSQL schema initialized")
        finally:
            cursor.close()
    
    def _init_db_duckdb(self):
        """DuckDB schema without FK constraints (DuckDB FK limitations)."""
        self.con.execute('''
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
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON content(source_type)')
        try:
            self.con.execute('CREATE INDEX IF NOT EXISTS idx_extraction_status ON content(extraction_status)')
        except Exception as e:
            print(f"DuckDB init: skipped idx_extraction_status (column missing pre-migration): {e}")
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_content_date ON content(pub_date)')

        self.con.execute('''
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
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_signals_entity ON signals(entity)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_signals_industry ON signals(industry)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_signals_content ON signals(source_content_id)')

        self.con.execute('''
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY,
                content_id INTEGER,
                segment_index INTEGER,
                start_time REAL,
                end_time REAL,
                speaker_id TEXT,
                text TEXT,
                confidence REAL,
                ground_truth_text TEXT,
                ground_truth_speaker TEXT,
                is_corrected BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_segments_content ON transcript_segments(content_id)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_segments_speaker ON transcript_segments(speaker_id)')

        self.con.execute('''
            CREATE TABLE IF NOT EXISTS content_metadata (
                id INTEGER PRIMARY KEY,
                content_id INTEGER NOT NULL,
                source_url TEXT,
                publisher TEXT,
                doc_type TEXT,
                discovery_confidence REAL,
                discovery_reason TEXT,
                topic_tags TEXT,
                authority TEXT,
                adapter TEXT,
                discovered_at TEXT,
                download_timestamp TEXT,
                processing_status TEXT DEFAULT 'downloaded',
                original_format TEXT,
                extraction_method TEXT,
                podcast_name TEXT,
                episode_description TEXT,
                episode_author TEXT,
                episode_summary TEXT,
                episode_keywords TEXT,
                episode_number TEXT,
                episode_season TEXT,
                episode_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.con.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_cm_content_id ON content_metadata(content_id)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_cm_doc_type ON content_metadata(doc_type)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_cm_adapter ON content_metadata(adapter)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_cm_publisher ON content_metadata(publisher)')

        self.con.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                action TEXT,
                message TEXT NOT NULL,
                details_json TEXT,
                content_id INTEGER,
                duration_sec REAL,
                run_id TEXT
            )
        ''')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_logs_source ON system_logs(source)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_logs_run_id ON system_logs(run_id)')
    
    def _migrate_schema(self):
        """Add missing columns to existing database."""
        migrations = [
            ("content_hash", "ALTER TABLE content ADD COLUMN content_hash TEXT",
             "CREATE INDEX IF NOT EXISTS idx_content_hash ON content(content_hash)"),
            ("transcription_model", "ALTER TABLE content ADD COLUMN transcription_model TEXT", None),
            ("extraction_hardware", "ALTER TABLE content ADD COLUMN extraction_hardware TEXT", None),
            ("verification_status", "ALTER TABLE content ADD COLUMN verification_status TEXT DEFAULT 'unverified'", None),
            ("verified_by", "ALTER TABLE content ADD COLUMN verified_by TEXT", None),
            ("verified_at", "ALTER TABLE content ADD COLUMN verified_at TEXT", None),
            ("signal_processed", "ALTER TABLE content ADD COLUMN signal_processed BOOLEAN DEFAULT FALSE",
             "CREATE INDEX IF NOT EXISTS idx_signal_processed ON content(signal_processed)"),
            ("screening_status", "ALTER TABLE content ADD COLUMN screening_status TEXT DEFAULT 'pending'",
             "CREATE INDEX IF NOT EXISTS idx_screening_status ON content(screening_status)"),
            ("screening_reason", "ALTER TABLE content ADD COLUMN screening_reason TEXT", None),
            ("screened_at", "ALTER TABLE content ADD COLUMN screened_at TEXT", None),
            ("marked_for_deletion", "ALTER TABLE content ADD COLUMN marked_for_deletion BOOLEAN DEFAULT FALSE",
             "CREATE INDEX IF NOT EXISTS idx_marked_deletion ON content(marked_for_deletion)"),
            ("extraction_status", "ALTER TABLE content ADD COLUMN extraction_status TEXT DEFAULT 'pending'",
             "CREATE INDEX IF NOT EXISTS idx_extraction_status ON content(extraction_status)"),
        ]

        # Rename columns for existing databases (safe: only runs if old column exists)
        rename_migrations = [
            # (old_col, new_col, alter_sql) — postgres uses RENAME, duckdb uses RENAME too
            ("transcription_status", "extraction_status",
             "ALTER TABLE content RENAME COLUMN transcription_status TO extraction_status"),
            ("transcript_method", "transcription_model",
             "ALTER TABLE content RENAME COLUMN transcript_method TO transcription_model"),
            ("model_version", "extraction_hardware",
             "ALTER TABLE content RENAME COLUMN model_version TO extraction_hardware"),
        ]
        for old_col, new_col, alter_sql in rename_migrations:
            try:
                self.execute(f"SELECT {old_col} FROM content LIMIT 1").fetchone()
                # Old column exists — rename it
                try:
                    self.execute(alter_sql)
                    print(f"Migration: renamed '{old_col}' -> '{new_col}'")
                except Exception:
                    pass  # Already renamed or rename not supported
            except Exception:
                pass  # Old column doesn't exist — already migrated or new DB
        
        # Signal table migrations
        signal_migrations = [
            ("vectorized", "ALTER TABLE signals ADD COLUMN vectorized BOOLEAN DEFAULT FALSE",
             "CREATE INDEX IF NOT EXISTS idx_signal_vectorized ON signals(vectorized)"),
            ("vectorized_at", "ALTER TABLE signals ADD COLUMN vectorized_at TEXT", None),
            ("context_window", "ALTER TABLE signals ADD COLUMN context_window TEXT", None),
            ("enriched_text", "ALTER TABLE signals ADD COLUMN enriched_text TEXT", None),
            ("enrichment_version", "ALTER TABLE signals ADD COLUMN enrichment_version TEXT", None),
        ]

        for column, alter_sql, index_sql in migrations:
            try:
                self.execute(f"SELECT {column} FROM content LIMIT 1").fetchone()
            except Exception as e:
                if column in str(e):
                    self.execute(alter_sql)
                    if index_sql:
                        self.execute(index_sql)
                    print(f"Migration: added column '{column}'")
                else:
                    print(f"Migration check failed for '{column}': {e}")
        
        # Apply signal migrations
        for column, alter_sql, index_sql in signal_migrations:
            try:
                self.execute(f"SELECT {column} FROM signals LIMIT 1").fetchone()
            except Exception as e:
                if column in str(e):
                    self.execute(alter_sql)
                    if index_sql:
                        self.execute(index_sql)
                    print(f"Migration: added signal column '{column}'")
                else:
                    print(f"Migration check failed for signal '{column}': {e}")

        # DuckDB only: drop any FK constraints on signals table.
        # DuckDB FK enforcement causes UPDATE to fail on parent rows even when only
        # non-PK columns are changed — a known DuckDB limitation. PostgreSQL handles
        # FKs correctly and should keep its constraints.
        if self.backend != 'postgres':
            try:
                fk_constraints = self.con.execute("""
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_name = 'signals' AND constraint_type = 'FOREIGN KEY'
                """).fetchall()
                for (name,) in fk_constraints:
                    self.con.execute(f"ALTER TABLE signals DROP CONSTRAINT {name}")
                    print(f"Migration: dropped DuckDB FK constraint '{name}' on signals")
            except Exception as e:
                print(f"Migration: FK constraint check skipped ({e})")
    
    def test_connection(self):
        try:
            self.execute("SELECT 1").fetchone()
            return True
        except:
            return False
    
    def close(self):
        self.con.close()
    
    def add_content_metadata(self, content_data):
        extraction_status = content_data.get('extraction_status', 'pending')
        
        # Strip NUL characters from transcript to avoid PostgreSQL errors
        transcript = content_data.get('transcript')
        if transcript and isinstance(transcript, str):
            transcript = transcript.replace('\x00', '')
        
        values = (
            content_data.get('title'),
            content_data.get('content_type'),
            content_data.get('source_type'),
            content_data.get('source_name'),
            content_data.get('pub_date', ''),
            content_data.get('file_path'),
            content_data.get('audio_url', 'N/A'),
            content_data.get('duration_seconds'),
            content_data.get('file_size_mb'),
            content_data.get('content_hash'),
            transcript,
            content_data.get('language', ''),
            content_data.get('transcription_date', ''),
            content_data.get('transcription_model', 'N/A'),
            content_data.get('extraction_hardware', ''),
            extraction_status,
            content_data.get('vectorization_status', 'pending'),
            json.dumps(content_data.get('segments', [])),
            json.dumps(content_data.get('metadata', {}))
        )
        
        if self.backend == 'postgres':
            cursor = self.con.cursor()
            try:
                cursor.execute('''
                    INSERT INTO content (
                        title, content_type, source_type, source_name, pub_date, file_path,
                        audio_url, duration_seconds, file_size_mb, content_hash,
                        transcript, language, transcription_date, transcription_model, extraction_hardware,
                        extraction_status, vectorization_status, segments, metadata_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_path) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                ''', values)
                result = cursor.fetchone()
                return result[0] if result else None
            except Exception:
                self.con.rollback()
                raise
            finally:
                cursor.close()
        else:
            max_result = self.con.execute("SELECT COALESCE(MAX(id), 0) FROM content").fetchone()
            next_id = (max_result[0] if max_result else 0) + 1
            
            self.con.execute('''
                INSERT INTO content (
                    id, title, content_type, source_type, source_name, pub_date, file_path,
                    audio_url, duration_seconds, file_size_mb, content_hash,
                    transcript, language, transcription_date, transcription_model, extraction_hardware,
                    extraction_status, vectorization_status, segments, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (next_id,) + values)
            
            return next_id
    
    def add_content_metadata_record(self, content_id: int, metadata: dict):
        """Insert or update a row in content_metadata for a given content_id."""
        fields = {
            'source_url': metadata.get('url', metadata.get('source_url', '')),
            'publisher': metadata.get('publisher', ''),
            'doc_type': metadata.get('doc_type', ''),
            'discovery_confidence': metadata.get('confidence', metadata.get('discovery_confidence')),
            'discovery_reason': metadata.get('reason', metadata.get('discovery_reason', '')),
            'topic_tags': json.dumps(metadata.get('topic_tags', [])) if isinstance(metadata.get('topic_tags'), list) else metadata.get('topic_tags', ''),
            'authority': metadata.get('authority', ''),
            'adapter': metadata.get('adapter', ''),
            'discovered_at': metadata.get('discovered_at', ''),
            'download_timestamp': metadata.get('download_timestamp', ''),
            'processing_status': metadata.get('processing_status', 'downloaded'),
            'original_format': metadata.get('original_format', ''),
            'extraction_method': metadata.get('extraction_method', ''),
            'podcast_name': metadata.get('podcast_name', ''),
            'episode_description': metadata.get('description', metadata.get('episode_description', '')),
            'episode_author': metadata.get('author', metadata.get('episode_author', '')),
            'episode_summary': metadata.get('summary', metadata.get('episode_summary', '')),
            'episode_keywords': metadata.get('keywords', metadata.get('episode_keywords', '')),
            'episode_number': str(metadata.get('episode_number', '')) if metadata.get('episode_number') else '',
            'episode_season': str(metadata.get('season', metadata.get('episode_season', ''))) if metadata.get('season') or metadata.get('episode_season') else '',
            'episode_type': metadata.get('episode_type', ''),
        }

        columns = ['content_id'] + list(fields.keys())
        placeholders = ', '.join(['?' for _ in columns])
        values = [content_id] + list(fields.values())

        if self.backend == 'postgres':
            cursor = self.con.cursor()
            try:
                col_str = ', '.join(columns)
                ph_str = ', '.join(['%s' for _ in columns])
                update_str = ', '.join([f"{k} = EXCLUDED.{k}" for k in fields.keys()])
                cursor.execute(f'''
                    INSERT INTO content_metadata ({col_str})
                    VALUES ({ph_str})
                    ON CONFLICT (content_id) DO UPDATE SET {update_str}
                ''', values)
            except Exception:
                self.con.rollback()
                raise
            finally:
                cursor.close()
        else:
            # DuckDB: check if exists, then insert or update
            existing = self.con.execute(
                "SELECT id FROM content_metadata WHERE content_id = ?", [content_id]
            ).fetchone()
            if existing:
                set_clause = ', '.join([f"{k} = ?" for k in fields.keys()])
                self.con.execute(
                    f"UPDATE content_metadata SET {set_clause} WHERE content_id = ?",
                    list(fields.values()) + [content_id]
                )
            else:
                max_result = self.con.execute("SELECT COALESCE(MAX(id), 0) FROM content_metadata").fetchone()
                next_id = (max_result[0] if max_result else 0) + 1
                col_str = ', '.join(['id'] + columns)
                ph_str = ', '.join(['?' for _ in range(len(columns) + 1)])
                self.con.execute(
                    f"INSERT INTO content_metadata ({col_str}) VALUES ({ph_str})",
                    [next_id] + values
                )

    def get_content_metadata(self, content_id: int) -> dict:
        """Retrieve content_metadata record for a content_id."""
        rows = self.query(
            "SELECT * FROM content_metadata WHERE content_id = ?", [content_id]
        )
        return rows[0] if rows else {}

    def file_exists(self, file_path):
        try:
            result = self.execute("SELECT id FROM content WHERE file_path = ?", [file_path]).fetchone()
            return result is not None
        except Exception as e:
            print(f"Error checking file existence: {e}")
            return False
    
    def hash_exists(self, content_hash):
        try:
            result = self.execute("SELECT id, file_path FROM content WHERE content_hash = ?", [content_hash]).fetchone()
            return result
        except Exception as e:
            print(f"Error checking hash existence: {e}")
            return None
    
    def update_record(self, record_id, update_data):
        set_clauses = []
        values = []
        
        for field, value in update_data.items():
            if field not in ['id', 'created_at', 'updated_at']:
                # Strip NUL characters from string values (PostgreSQL rejects them)
                if isinstance(value, str):
                    value = value.replace('\x00', '')
                set_clauses.append(f"{field} = ?")
                values.append(value)
        
        if not set_clauses:
            return False
        
        set_clauses.append("updated_at = ?")
        values.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        values.append(record_id)
        
        query = f"UPDATE content SET {', '.join(set_clauses)} WHERE id = ?"
        
        try:
            self.execute(query, values)
            return True
        except Exception as e:
            print(f"Error updating record {record_id}: {e}")
            return False
    
    def _next_id(self, table: str) -> int | None:
        """
        Return the next safe integer ID for a manual INSERT.
        - DuckDB: MAX(id)+1 (no SERIAL support for manual inserts)
        - PostgreSQL: None (SERIAL / RETURNING id handles it automatically)
        """
        if self.backend == 'postgres':
            return None
        result = self.con.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()
        return (result[0] if result else 0) + 1

    def insert_signal(self, signal) -> int | None:
        """
        Insert a validated Pydantic signal model into the signals table.
        Backend-agnostic: callers never need to know about DuckDB vs PostgreSQL.
        Returns the new row ID.
        """
        timeline = getattr(signal, 'timeline', None)
        context_window = getattr(signal, 'context_window', None)
        enriched_text = getattr(signal, 'enriched_text', None)
        enrichment_version = getattr(signal, 'enrichment_version', None)
        values = (
            signal.signal_type, signal.entity, signal.description,
            signal.industry, getattr(signal, 'impact_level', None),
            signal.confidence, timeline, signal.metadata_json,
            signal.source_content_id,
            context_window, enriched_text, enrichment_version,
        )

        if self.backend == 'postgres':
            cursor = self.con.cursor()
            try:
                cursor.execute('''
                    INSERT INTO signals (
                        signal_type, entity, description, industry,
                        impact_level, confidence, timeline,
                        metadata_json, source_content_id,
                        context_window, enriched_text, enrichment_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', values)
                result = cursor.fetchone()
                return result[0] if result else None
            finally:
                cursor.close()
        else:
            next_id = self._next_id('signals')
            self.con.execute('''
                INSERT INTO signals (
                    id, signal_type, entity, description, industry,
                    impact_level, confidence, timeline,
                    metadata_json, source_content_id,
                    context_window, enriched_text, enrichment_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (next_id,) + values)
            return next_id

    def get_path_info(self):
        return {
            "original_requested": self.original_path,
            "actual_path": self.db_path,
            "using_default": self.original_path != self.db_path
        }

    def upsert_records(self, records: list) -> dict:
        """Bulk upsert records, skipping unchanged ones via content_hash."""
        inserted = 0
        updated = 0
        skipped = 0

        if not records:
            return {"inserted": 0, "updated": 0, "skipped": 0}

        ids = [r['id'] for r in records if r.get('id') is not None]
        existing = {}
        if ids:
            placeholders = ', '.join(['?' for _ in ids])
            rows = self.query(f"SELECT id, content_hash FROM content WHERE id IN ({placeholders})", ids)
            existing = {row['id']: row['content_hash'] for row in rows}

        for record in records:
            rid = record.get('id')
            if rid in existing and existing[rid] == record.get('content_hash'):
                skipped += 1
                continue

            try:
                self.execute("""
                    INSERT INTO content (
                        id, title, content_type, source_type, source_name,
                        pub_date, file_path, audio_url, duration_seconds,
                        file_size_mb, content_hash, transcript, language,
                        transcription_date, transcription_model, extraction_hardware,
                        extraction_status, vectorization_status,
                        screening_status, screening_reason, screened_at,
                        signal_processed, marked_for_deletion,
                        created_at, updated_at, segments, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        title = excluded.title,
                        content_type = excluded.content_type,
                        source_type = excluded.source_type,
                        source_name = excluded.source_name,
                        pub_date = excluded.pub_date,
                        file_path = excluded.file_path,
                        audio_url = excluded.audio_url,
                        duration_seconds = excluded.duration_seconds,
                        file_size_mb = excluded.file_size_mb,
                        content_hash = excluded.content_hash,
                        transcript = excluded.transcript,
                        language = excluded.language,
                        transcription_date = excluded.transcription_date,
                        transcription_model = excluded.transcription_model,
                        extraction_hardware = excluded.extraction_hardware,
                        extraction_status = excluded.extraction_status,
                        vectorization_status = excluded.vectorization_status,
                        screening_status = excluded.screening_status,
                        screening_reason = excluded.screening_reason,
                        screened_at = excluded.screened_at,
                        signal_processed = excluded.signal_processed,
                        marked_for_deletion = excluded.marked_for_deletion,
                        updated_at = excluded.updated_at,
                        segments = excluded.segments,
                        metadata_json = excluded.metadata_json
                """, (
                    rid,
                    record.get('title'), record.get('content_type'), record.get('source_type'),
                    record.get('source_name'), record.get('pub_date'), record.get('file_path'),
                    record.get('audio_url'), record.get('duration_seconds'), record.get('file_size_mb'),
                    record.get('content_hash'), record.get('transcript'), record.get('language'),
                    record.get('transcription_date'), record.get('transcription_model'),
                    record.get('extraction_hardware'),
                    record.get('extraction_status'), record.get('vectorization_status'),
                    record.get('screening_status'), record.get('screening_reason'),
                    record.get('screened_at'),
                    record.get('signal_processed', False), record.get('marked_for_deletion', False),
                    record.get('created_at'), record.get('updated_at'),
                    record.get('segments'), record.get('metadata_json')
                ))
                if rid in existing:
                    updated += 1
                else:
                    inserted += 1
            except Exception as e:
                logger.error(f"Upsert error for record {rid}: {e}")

        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def upsert_signals(self, records: list) -> dict:
        """Bulk upsert signals records by id."""
        inserted = 0
        updated = 0
        skipped = 0

        if not records:
            return {"inserted": 0, "updated": 0, "skipped": 0}

        ids = [r['id'] for r in records if r.get('id') is not None]
        existing = set()
        if ids:
            placeholders = ', '.join(['?' for _ in ids])
            rows = self.query(f"SELECT id FROM signals WHERE id IN ({placeholders})", ids)
            existing = {row['id'] for row in rows}

        for record in records:
            rid = record.get('id')
            try:
                self.execute("""
                    INSERT INTO signals (
                        id, signal_type, entity, description, industry,
                        impact_level, confidence, timeline,
                        metadata_json, source_content_id, extracted_at,
                        context_window, enriched_text, enrichment_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        signal_type = excluded.signal_type,
                        entity = excluded.entity,
                        description = excluded.description,
                        industry = excluded.industry,
                        impact_level = excluded.impact_level,
                        confidence = excluded.confidence,
                        timeline = excluded.timeline,
                        metadata_json = excluded.metadata_json,
                        source_content_id = excluded.source_content_id,
                        context_window = excluded.context_window,
                        enriched_text = excluded.enriched_text,
                        enrichment_version = excluded.enrichment_version
                """, (
                    rid,
                    record.get('signal_type'), record.get('entity'),
                    record.get('description'), record.get('industry'),
                    record.get('impact_level'), record.get('confidence'),
                    record.get('timeline'), record.get('metadata_json'),
                    record.get('source_content_id'), record.get('extracted_at'),
                    record.get('context_window'), record.get('enriched_text'),
                    record.get('enrichment_version')
                ))
                if rid in existing:
                    updated += 1
                else:
                    inserted += 1
            except Exception as e:
                logger.error(f"Upsert error for signal {rid}: {e}")

        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def upsert_logs(self, records: list) -> dict:
        """Bulk upsert system_logs records by id."""
        inserted = 0
        updated = 0
        skipped = 0

        if not records:
            return {"inserted": 0, "updated": 0, "skipped": 0}

        ids = [r['id'] for r in records if r.get('id') is not None]
        existing = set()
        if ids:
            placeholders = ', '.join(['?' for _ in ids])
            rows = self.query(f"SELECT id FROM system_logs WHERE id IN ({placeholders})", ids)
            existing = {row['id'] for row in rows}

        for record in records:
            rid = record.get('id')
            try:
                self.execute("""
                    INSERT INTO system_logs (
                        id, timestamp, level, source, action, message,
                        details_json, content_id, duration_sec, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        timestamp = excluded.timestamp,
                        level = excluded.level,
                        source = excluded.source,
                        action = excluded.action,
                        message = excluded.message,
                        details_json = excluded.details_json,
                        content_id = excluded.content_id,
                        duration_sec = excluded.duration_sec,
                        run_id = excluded.run_id
                """, (
                    rid,
                    record.get('timestamp'), record.get('level'),
                    record.get('source'), record.get('action'),
                    record.get('message'), record.get('details_json'),
                    record.get('content_id'), record.get('duration_sec'),
                    record.get('run_id')
                ))
                if rid in existing:
                    updated += 1
                else:
                    inserted += 1
            except Exception as e:
                logger.error(f"Upsert error for log {rid}: {e}")

        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def query(self, query_str: str, params=None):
        """Execute a query and return results as list of dicts."""
        try:
            cursor = self.execute(query_str, params)
            if cursor.description is None:
                return []
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            results = [dict(zip(columns, row)) for row in rows]
            # Convert datetime objects to strings for PostgreSQL consistency
            if self.backend == 'postgres':
                for row_dict in results:
                    for key, value in row_dict.items():
                        if isinstance(value, datetime):
                            row_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            return results
        except Exception as e:
            logger.error(f"Query error: {e}")
            return []


if __name__ == "__main__":

    from device_config import config

    REL_DB_PATH = os.getenv("REL_DB_PATH", config.DB_PATH)

    try:
        print("Creating database...")
        db = relationalDB(REL_DB_PATH)
        
        print(f"Backend: {db.backend}")
        print(f"Path: {db.db_path}")
        
        if db.test_connection():
            print("Database connection successful")
        else:
            print("Database connection failed")
        
        # Quick schema test
        count = db.query("SELECT COUNT(*) as cnt FROM content")
        print(f"Content records: {count[0]['cnt'] if count else 0}")
            
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'db' in locals():
            db.close()
            print("Database connection closed")
