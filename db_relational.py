import duckdb
import os
import json
from datetime import datetime


class relationalDB:
    def __init__(self, db_path):
        if not db_path:
            raise ValueError("Database path cannot be empty")
        
        self.original_path = db_path
        self.db_path = self._ensure_valid_path(db_path)
        self.con = duckdb.connect(self.db_path)
        self.init_db()
    
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
        """Initialize database schema"""
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
        self._migrate_schema()
    
    def _migrate_schema(self):
        """Add missing columns to existing database."""
        try:
            self.con.execute("SELECT content_hash FROM content LIMIT 1").fetchone()
            print("content_hash column already exists")
        except Exception as e:
            if "content_hash" in str(e) and "not found" in str(e):
                print("Adding content_hash column to existing database...")
                self.con.execute("ALTER TABLE content ADD COLUMN content_hash TEXT")
                print("Added content_hash column")
                
                # Add index for the new column
                self.con.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON content(content_hash)")
                print("Added content_hash index")
            else:
                print(f"Schema migration check failed: {e}")
    
    def test_connection(self):
        try:
            self.con.execute("SELECT 1").fetchone()
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
        ''', (
            next_id,
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
        ))
        
        return next_id
    
    def file_exists(self, file_path):
        try:
            result = self.con.execute("SELECT id FROM content WHERE file_path = ?", [file_path]).fetchone()
            return result is not None
        except Exception as e:
            print(f"Error checking file existence: {e}")
            return False
    
    def hash_exists(self, content_hash):
        try:
            result = self.con.execute("SELECT id, file_path FROM content WHERE content_hash = ?", [content_hash]).fetchone()
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
            self.con.execute(query, values)
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

    def query(self, query="SELECT * FROM content"):
        try:
            results = self.con.execute(query).fetchall()
            return results
        except Exception as e:
            print(f"Error executing query: {e}")
            return []


if __name__ == "__main__":

    

    from dotenv import load_dotenv
    load_dotenv()

    REL_DB_PATH = os.getenv("REL_DB_PATH", "Database/industry_signals.db")

    try:
        print("Creating database...")
        db = relationalDB(REL_DB_PATH)
        
        path_info = db.get_path_info()
        print(f"Original path requested: {path_info['original_requested']}")
        print(f"Actual path used: {path_info['actual_path']}")
        if path_info['using_default']:
            print("Note: Using default fallback path")
        
        if db.test_connection():
            print("Database connection successful")
        else:
            print("Database connection failed")
        
        if os.path.exists(db.db_path):
            print(f"Database file created at: {db.db_path}")
        else:
            print(f"Database file not found at: {db.db_path}")
            
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'db' in locals():
            db.close()
            print("Database connection closed")
