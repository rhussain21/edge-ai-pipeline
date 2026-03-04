"""PostgreSQL Database for AI Industry Signals."""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class RelationalDB:
    def __init__(self, db_config: Dict[str, str]):
        self.config = db_config
        self.conn = None
        self._test_connection()
    
    def _test_connection(self):
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            logger.info(f"Connected to PostgreSQL: {version}")
            cursor.close()
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def connect(self):
        if not self.conn or self.conn.closed:
            try:
                self.conn = psycopg2.connect(**self.config)
                self.conn.autocommit = False
                logger.debug("PostgreSQL connection established")
            except psycopg2.Error as e:
                logger.error(f"PostgreSQL connection error: {e}")
                raise
        return self.conn
    
    def init_db(self):
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_metadata (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    content_type TEXT NOT NULL,
                    file_size_mb REAL,
                    file_hash TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_timestamp TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    source_url TEXT,
                    metadata JSONB,
                    full_text TEXT,
                    segments JSONB,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_type ON content_metadata(content_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_processing_status ON content_metadata(processing_status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_date ON content_metadata(created_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_url ON content_metadata(source_url)
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vector_mappings (
                    id SERIAL PRIMARY KEY,
                    content_id INTEGER REFERENCES content_metadata(id) ON DELETE CASCADE,
                    segment_index INTEGER,
                    vector_id INTEGER,
                    chunk_text TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(content_id, segment_index)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id SERIAL PRIMARY KEY,
                    content_id INTEGER REFERENCES content_metadata(id) ON DELETE CASCADE,
                    log_level TEXT,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("PostgreSQL database schema initialized")
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            cursor.close()
    
    def add_content_metadata(self, file_path: str, content_type: str, file_size_mb: float,
                           source_url: Optional[str] = None, metadata: Optional[Dict] = None,
                           full_text: Optional[str] = None, file_hash: Optional[str] = None) -> Optional[int]:
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO content_metadata 
                (file_path, content_type, file_size_mb, source_url, metadata, full_text, file_hash, processing_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_path) DO UPDATE SET
                    content_type = EXCLUDED.content_type,
                    file_size_mb = EXCLUDED.file_size_mb,
                    source_url = EXCLUDED.source_url,
                    metadata = EXCLUDED.metadata,
                    full_text = EXCLUDED.full_text,
                    file_hash = EXCLUDED.file_hash,
                    processing_status = 'processed'
                RETURNING id
            """, (file_path, content_type, file_size_mb, source_url,
                  json.dumps(metadata) if metadata else None,
                  full_text, file_hash, 'processed'))
            
            result = cursor.fetchone()
            conn.commit()
            
            if result:
                content_id = result[0]
                logger.info(f"Added content metadata: {file_path} -> ID: {content_id}")
                return content_id
            else:
                cursor.execute("""
                    SELECT id FROM content_metadata WHERE file_path = %s
                """, (file_path,))
                result = cursor.fetchone()
                return result[0] if result else None
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error adding content metadata: {e}")
            return None
        finally:
            cursor.close()
    
    def update_content_segments(self, content_id: int, segments: List[Dict]):
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE content_metadata 
                SET segments = %s, processing_status = 'processed'
                WHERE id = %s
            """, (json.dumps(segments), content_id))
            
            conn.commit()
            logger.debug(f"Updated segments for content ID: {content_id}")
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error updating segments: {e}")
        finally:
            cursor.close()
    
    def add_vector_mapping(self, content_id: int, segment_index: int, vector_id: int, chunk_text: str):
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO vector_mappings (content_id, segment_index, vector_id, chunk_text)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (content_id, segment_index) DO UPDATE SET
                    vector_id = EXCLUDED.vector_id,
                    chunk_text = EXCLUDED.chunk_text
            """, (content_id, segment_index, vector_id, chunk_text))
            
            conn.commit()
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error adding vector mapping: {e}")
        finally:
            cursor.close()
    
    def get_content_by_id(self, content_id: int) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT * FROM content_metadata WHERE id = %s
            """, (content_id,))
            
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                if result_dict.get('metadata'):
                    result_dict['metadata'] = json.loads(result_dict['metadata'])
                if result_dict.get('segments'):
                    result_dict['segments'] = json.loads(result_dict['segments'])
                return result_dict
            return None
            
        except psycopg2.Error as e:
            logger.error(f"Error getting content by ID: {e}")
            return None
        finally:
            cursor.close()
    
    def get_content_by_file_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT * FROM content_metadata WHERE file_path = %s
            """, (file_path,))
            
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                if result_dict.get('metadata'):
                    result_dict['metadata'] = json.loads(result_dict['metadata'])
                if result_dict.get('segments'):
                    result_dict['segments'] = json.loads(result_dict['segments'])
                return result_dict
            return None
            
        except psycopg2.Error as e:
            logger.error(f"Error getting content by file path: {e}")
            return None
        finally:
            cursor.close()
    
    def count_documents(self) -> int:
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM content_metadata WHERE processing_status = 'processed'")
            count = cursor.fetchone()[0]
            return count
            
        except psycopg2.Error as e:
            logger.error(f"Error counting documents: {e}")
            return 0
        finally:
            cursor.close()
    
    def count_all(self) -> int:
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM content_metadata")
            count = cursor.fetchone()[0]
            return count
            
        except psycopg2.Error as e:
            logger.error(f"Error counting all documents: {e}")
            return 0
        finally:
            cursor.close()
    
    def get_recent_content(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT id, file_path, content_type, created_date, processing_status
                FROM content_metadata 
                WHERE created_date >= NOW() - INTERVAL '%s days'
                ORDER BY created_date DESC
                LIMIT %s
            """, (days, limit))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting recent content: {e}")
            return []
        finally:
            cursor.close()
    
    def get_by_content_type(self, content_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT id, file_path, content_type, created_date, processing_status
                FROM content_metadata 
                WHERE content_type = %s
                ORDER BY created_date DESC
                LIMIT %s
            """, (content_type, limit))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting content by type: {e}")
            return []
        finally:
            cursor.close()
    
    def get_pending_content(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT * FROM content_metadata 
                WHERE processing_status = 'pending'
                ORDER BY created_date ASC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting pending content: {e}")
            return []
        finally:
            cursor.close()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            stats = {}
            cursor.execute("""
                SELECT processing_status, COUNT(*) 
                FROM content_metadata 
                GROUP BY processing_status
            """)
            status_counts = dict(cursor.fetchall())
            stats['by_status'] = status_counts
            
            cursor.execute("""
                SELECT content_type, COUNT(*) 
                FROM content_metadata 
                WHERE processing_status = 'processed'
                GROUP BY content_type
            """)
            type_counts = dict(cursor.fetchall())
            stats['by_type'] = type_counts
            
            cursor.execute("""
                SELECT DATE(created_date) as date, COUNT(*) 
                FROM content_metadata 
                WHERE created_date >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_date)
                ORDER BY date DESC
            """)
            recent_activity = dict(cursor.fetchall())
            stats['recent_activity'] = recent_activity
            
            return stats
            
        except psycopg2.Error as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
        finally:
            cursor.close()
    
    def log_processing_event(self, content_id: int, level: str, message: str):
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO processing_logs (content_id, log_level, message)
                VALUES (%s, %s, %s)
            """, (content_id, level, message))
            
            conn.commit()
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error logging processing event: {e}")
        finally:
            cursor.close()
    
    def update_processing_status(self, content_id: int, status: str, error_message: Optional[str] = None):
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE content_metadata 
                SET processing_status = %s, processed_timestamp = NOW(), error_message = %s
                WHERE id = %s
            """, (status, error_message, content_id))
            
            conn.commit()
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error updating processing status: {e}")
        finally:
            cursor.close()
    
    def search_content(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT id, file_path, content_type, created_date,
                       ts_rank(to_tsvector('english', full_text), plainto_tsquery('english', %s)) as rank
                FROM content_metadata 
                WHERE full_text @@ plainto_tsquery('english', %s)
                    AND processing_status = 'processed'
                ORDER BY rank DESC
                LIMIT %s
            """, (query, query, limit))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            logger.error(f"Error searching content: {e}")
            return []
        finally:
            cursor.close()
    
    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.debug("PostgreSQL connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_postgres_db(host='localhost', port='5432', database='industry_signals', 
                   user='jetson_user', password='your_password') -> RelationalDB:
    config = {
        'host': host,
        'port': port,
        'database': database,
        'user': user,
        'password': password
    }
    return RelationalDB(config)


if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'industry_signals',
        'user': 'jetson_user',
        'password': 'your_password'
    }
    
    with RelationalDB(db_config) as db:
        db.init_db()
        content_id = db.add_content_metadata(
            file_path="/test/example.pdf",
            content_type="pdf",
            file_size_mb=2.5,
            source_url="https://example.com",
            metadata={"title": "Test Document", "author": "Test Author"},
            full_text="This is a test document for the industry signals database."
        )
        
        print(f"Added content with ID: {content_id}")
        print(f"Total documents: {db.count_documents()}")
        print(f"Recent content: {db.get_recent_content(days=1)}")
        
        stats = db.get_processing_stats()
        print(f"Processing stats: {stats}")
