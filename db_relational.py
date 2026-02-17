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
        """Ensure directory exists, fallback to default if needed"""
        try:
            # Extract directory from path
            db_dir = os.path.dirname(db_path)
            
            # If directory doesn't exist, try to create it
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"Created directory: {db_dir}")
            
            # Test if we can write to this location
            test_file = db_path.replace('.db', '_test.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                return db_path  # Original path works
            except (PermissionError, OSError):
                print(f"Cannot write to {db_path}, using default path")
                
        except Exception as e:
            print(f"Path validation failed: {e}")
        
        # Fallback to default path
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
                content_type TEXT NOT NULL,        -- 'audio', 'text', 'video'
                source_type TEXT NOT NULL,          -- 'podcast', 'pdf', 'lecture', etc.
                source_name TEXT,                   -- podcast name, publication, etc.
                pub_date TEXT,
                file_path TEXT UNIQUE NOT NULL,
                
                -- Media-specific fields
                audio_url TEXT,                     -- For audio/video
                duration_seconds REAL,              -- For audio/video
                file_size_mb REAL,
                content_hash TEXT,                   -- For duplicate detection
                
                -- Text/transcription data
                transcript TEXT,
                language TEXT,
                transcription_date TEXT,
                transcript_method TEXT,             -- 'faster-whisper', 'ocr', etc.
                
                -- Processing status
                transcription_status TEXT DEFAULT 'pending',
                vectorization_status TEXT DEFAULT 'pending',
                
                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                
                -- JSON fields for flexible metadata
                segments TEXT,                      -- JSON array of segments
                metadata_json TEXT                  -- JSON object for additional metadata
            )
        ''')
        
        # Note: DuckDB doesn't support triggers like SQLite
        # We'll handle updated_at in application logic
        
        # Index for common queries
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON content(source_type)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_content_status ON content(transcription_status)')
        self.con.execute('CREATE INDEX IF NOT EXISTS idx_content_date ON content(pub_date)')
        # Note: content_hash index will be created in migration if needed
        
        # Check if we need to add content_hash column (for existing databases)
        self._migrate_schema()
    
    def _migrate_schema(self):
        """Add missing columns to existing database"""
        try:
            # Try to select from content_hash to see if it exists
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
        """Test if database is accessible"""
        try:
            self.con.execute("SELECT 1").fetchone()
            return True
        except:
            return False
    
    def close(self):
        """Close database connection"""
        self.con.close()
    
    def add_content_metadata(self, content_data):
        """Add content metadata to database with unique identifier"""
        # Set status based on content type
        if content_data.get('content_type') == 'text':
            transcription_status = 'NA'
        else:
            transcription_status = content_data.get('transcription_status', 'pending')
        
        # For DuckDB, use a sequence to get the next ID
        try:
            # Try to get the next ID from a sequence
            result = self.con.execute("SELECT nextval('content_id_seq')").fetchone()
            next_id = result[0] if result else None
        except:
            # If sequence doesn't exist, create it and get max ID + 1
            try:
                self.con.execute("CREATE SEQUENCE content_id_seq START 1")
                result = self.con.execute("SELECT nextval('content_id_seq')").fetchone()
                next_id = result[0] if result else 1
            except:
                # Fallback: get max existing ID + 1
                max_result = self.con.execute("SELECT COALESCE(MAX(id), 0) FROM content").fetchone()
                next_id = (max_result[0] if max_result else 0) + 1
        
        # Insert with explicit ID
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
        """Check if file path already exists in database"""
        try:
            result = self.con.execute("SELECT id FROM content WHERE file_path = ?", [file_path]).fetchone()
            return result is not None
        except Exception as e:
            print(f"Error checking file existence: {e}")
            return False
    
    def hash_exists(self, content_hash):
        """Check if content hash already exists in database"""
        try:
            result = self.con.execute("SELECT id, file_path FROM content WHERE content_hash = ?", [content_hash]).fetchone()
            return result
        except Exception as e:
            print(f"Error checking hash existence: {e}")
            return None
    
    def update_record(self, record_id, update_data):
        """Update individual record by ID - manually update updated_at"""
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        
        for field, value in update_data.items():
            if field not in ['id', 'created_at', 'updated_at']:  # Don't update these fields
                set_clauses.append(f"{field} = ?")
                values.append(value)
        
        if not set_clauses:
            return False  # Nothing to update
        
        # Add updated_at timestamp
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
        """Return information about which path is being used"""
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
    try:
        print("Creating database...")
        db = relationalDB("Database/industry_signals.db")
        
        # Show path information
        path_info = db.get_path_info()
        print(f"Original path requested: {path_info['original_requested']}")
        print(f"Actual path used: {path_info['actual_path']}")
        if path_info['using_default']:
            print("Note: Using default fallback path")
        
        # Test connection
        if db.test_connection():
            print("Database connection successful")
        else:
            print("Database connection failed")
        
        # Check if database file was created
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
