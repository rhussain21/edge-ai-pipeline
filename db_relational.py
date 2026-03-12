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
                    transcript_method TEXT,
                    transcription_status TEXT DEFAULT 'pending',
                    vectorization_status TEXT DEFAULT 'pending',
                    signal_processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    segments TEXT,
                    metadata_json TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON content(source_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_status ON content(transcription_status)')
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
                    vectorized_at TIMESTAMP
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
                transcript_method TEXT,
                transcription_status TEXT DEFAULT 'pending',
                vectorization_status TEXT DEFAULT 'pending',
                signal_processed BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                segments TEXT,
                metadata_json TEXT
            )
        ''')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON content(source_type)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_content_status ON content(transcription_status)')
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
                vectorized_at TEXT
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
    
    def _migrate_schema(self):
        """Add missing columns to existing database."""
        migrations = [
            ("content_hash", "ALTER TABLE content ADD COLUMN content_hash TEXT",
             "CREATE INDEX IF NOT EXISTS idx_content_hash ON content(content_hash)"),
            ("speaker_count", "ALTER TABLE content ADD COLUMN speaker_count INTEGER", None),
            ("speakers_json", "ALTER TABLE content ADD COLUMN speakers_json TEXT", None),
            ("diarization_model", "ALTER TABLE content ADD COLUMN diarization_model TEXT", None),
            ("diarization_confidence", "ALTER TABLE content ADD COLUMN diarization_confidence REAL", None),
            ("transcription_model", "ALTER TABLE content ADD COLUMN transcription_model TEXT", None),
            ("model_version", "ALTER TABLE content ADD COLUMN model_version TEXT", None),
            ("inference_time_seconds", "ALTER TABLE content ADD COLUMN inference_time_seconds REAL", None),
            ("transcription_confidence", "ALTER TABLE content ADD COLUMN transcription_confidence REAL", None),
            ("word_error_rate", "ALTER TABLE content ADD COLUMN word_error_rate REAL", None),
            ("ground_truth_transcript", "ALTER TABLE content ADD COLUMN ground_truth_transcript TEXT", None),
            ("verification_status", "ALTER TABLE content ADD COLUMN verification_status TEXT DEFAULT 'unverified'", None),
            ("verified_by", "ALTER TABLE content ADD COLUMN verified_by TEXT", None),
            ("verified_at", "ALTER TABLE content ADD COLUMN verified_at TEXT", None),
            ("training_set_label", "ALTER TABLE content ADD COLUMN training_set_label TEXT", None),
            ("signal_processed", "ALTER TABLE content ADD COLUMN signal_processed BOOLEAN DEFAULT FALSE",
             "CREATE INDEX IF NOT EXISTS idx_signal_processed ON content(signal_processed)"),
        ]
        
        # Signal table migrations
        signal_migrations = [
            ("vectorized", "ALTER TABLE signals ADD COLUMN vectorized BOOLEAN DEFAULT FALSE",
             "CREATE INDEX IF NOT EXISTS idx_signal_vectorized ON signals(vectorized)"),
            ("vectorized_at", "ALTER TABLE signals ADD COLUMN vectorized_at TEXT", None),
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
    
    def test_connection(self):
        try:
            self.execute("SELECT 1").fetchone()
            return True
        except:
            return False
    
    def close(self):
        self.con.close()
    
    def add_content_metadata(self, content_data):
        if content_data.get('content_type') == 'text':
            transcription_status = 'NA'
        else:
            transcription_status = content_data.get('transcription_status', 'pending')
        
        values = (
            content_data.get('title'),
            content_data.get('content_type'),
            content_data.get('source_type'),
            content_data.get('source_name'),
            content_data.get('pub_date', ''),
            content_data.get('file_path'),
            content_data.get('audio_url', ''),
            content_data.get('duration_seconds'),
            content_data.get('file_size_mb'),
            content_data.get('content_hash'),
            content_data.get('transcript'),
            content_data.get('language', ''),
            content_data.get('transcription_date', ''),
            content_data.get('transcript_method', ''),
            transcription_status,
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
                        transcript, language, transcription_date, transcript_method, transcription_status,
                        vectorization_status, segments, metadata_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', values)
                result = cursor.fetchone()
                return result[0] if result else None
            finally:
                cursor.close()
        else:
            try:
                result = self.con.execute("SELECT nextval('content_id_seq')").fetchone()
                next_id = result[0] if result else None
            except:
                try:
                    self.con.execute("CREATE SEQUENCE content_id_seq START 1")
                    result = self.con.execute("SELECT nextval('content_id_seq')").fetchone()
                    next_id = result[0] if result else 1
                except:
                    max_result = self.con.execute("SELECT COALESCE(MAX(id), 0) FROM content").fetchone()
                    next_id = (max_result[0] if max_result else 0) + 1
            
            self.con.execute('''
                INSERT INTO content (
                    id, title, content_type, source_type, source_name, pub_date, file_path,
                    audio_url, duration_seconds, file_size_mb, content_hash,
                    transcript, language, transcription_date, transcript_method, transcription_status,
                    vectorization_status, segments, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (next_id,) + values)
            
            return next_id
    
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
                        transcription_date, transcript_method,
                        transcription_status, vectorization_status,
                        created_at, updated_at, segments, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        transcript_method = excluded.transcript_method,
                        transcription_status = excluded.transcription_status,
                        vectorization_status = excluded.vectorization_status,
                        updated_at = excluded.updated_at,
                        segments = excluded.segments,
                        metadata_json = excluded.metadata_json
                """, (
                    rid,
                    record.get('title'), record.get('content_type'), record.get('source_type'),
                    record.get('source_name'), record.get('pub_date'), record.get('file_path'),
                    record.get('audio_url'), record.get('duration_seconds'), record.get('file_size_mb'),
                    record.get('content_hash'), record.get('transcript'), record.get('language'),
                    record.get('transcription_date'), record.get('transcript_method'),
                    record.get('transcription_status'), record.get('vectorization_status'),
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
                        metadata_json, source_content_id, extracted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        signal_type = excluded.signal_type,
                        entity = excluded.entity,
                        description = excluded.description,
                        industry = excluded.industry,
                        impact_level = excluded.impact_level,
                        confidence = excluded.confidence,
                        timeline = excluded.timeline,
                        metadata_json = excluded.metadata_json,
                        source_content_id = excluded.source_content_id
                """, (
                    rid,
                    record.get('signal_type'), record.get('entity'),
                    record.get('description'), record.get('industry'),
                    record.get('impact_level'), record.get('confidence'),
                    record.get('timeline'), record.get('metadata_json'),
                    record.get('source_content_id'), record.get('extracted_at')
                ))
                if rid in existing:
                    updated += 1
                else:
                    inserted += 1
            except Exception as e:
                logger.error(f"Upsert error for signal {rid}: {e}")

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

    from dotenv import load_dotenv
    load_dotenv()

    REL_DB_PATH = os.getenv("REL_DB_PATH", "Database/industry_signals.db")

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
