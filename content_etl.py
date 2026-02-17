import os
import json
from datetime import datetime
from pathlib import Path
import re
import hashlib
import pdfplumber
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("PyMuPDF available")
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available")
from bs4 import BeautifulSoup
import docx
import whisper
from content_sources import ContentSources
import tiktoken
from db_relational import relationalDB
from db_vector import VectorDB

# Detect Jetson/CUDA environment
CUDA_AVAILABLE = False
DEVICE = "cpu"

try:
    import torch
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        DEVICE = "cuda"
        print(f"CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
except ImportError:
    print("PyTorch not available, using CPU")

# Try to import faster-whisper (4x faster than regular whisper)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("faster-whisper available (4x speed boost)")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("faster-whisper not available, using regular whisper")

# Try to import MLX Whisper for M2 acceleration (MacBook)
try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
    print("MLX Whisper available for M2 acceleration")
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    print("MLX Whisper not available")

# Jetson-specific Whisper optimization
JETSON_AVAILABLE = False
try:
    # Check if running on Jetson
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            if 'jetson' in model.lower():
                JETSON_AVAILABLE = True
                print(f"Detected Jetson device: {model}")
except:
    pass

class contentETL:
    def __init__(self, content_path, db=None, vdb=None):
        self.content_path = content_path
        self.db = db  # Store database reference
        self.vdb = vdb # Store Vector database reference
        self._whisper_model = None  # Lazy load transcription model
        self.sources = ContentSources(content_path)  # Content acquisition
        
        # Set device for this instance
        self.device = DEVICE
        self.cuda_available = CUDA_AVAILABLE
        self.jetson_available = JETSON_AVAILABLE
        
        print(f"ETL initialized with device: {self.device}")
        if self.cuda_available:
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
    def detect_file_type(self,file_path):
        """Detect file type from extension"""
        ext = Path(file_path).suffix.lower()
    
        type_mapping = {
        '.txt': 'text',
        '.md': 'text', 
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.html': 'html',
        '.htm': 'html',
        '.mp3': 'audio',
        '.wav': 'audio',
        '.m4a': 'audio',
        '.mp4': 'audio'
    }
        
        return type_mapping.get(ext, 'unknown')


    def extract_content(self,file_path, file_type=None):
        """Extract text content from various file types"""
        
        if file_type is None:
            file_type = self.detect_file_type(file_path)
        
        print(f"Extracting content from {file_path} (type: {file_type})")
             
        if file_type == 'text':
            content = self.extract_text_file(file_path)
        elif file_type == 'pdf':
            content = self.extract_pdf_text(file_path)
        elif file_type == 'docx':
            content = self.extract_docx_text(file_path)
        elif file_type == 'html':
            content = self.extract_html_text(file_path)
        elif file_type == 'audio':
            content = self.extract_audio_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"Content extraction completed, length: {len(content)}")
        # Return both content and file type
        return content, file_type    
    

    def extract_text_file(self,file_path):
        """Extract from .txt, .md files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    

    def extract_pdf_text(self, file_path):
        """Extract text from PDF files with multiple library fallbacks"""
        print(f"Attempting PDF extraction for {file_path}")
        
        # Try PyMuPDF first (better handling)
        if PYMUPDF_AVAILABLE:
            print("Using PyMuPDF...")
            try:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n"
                doc.close()
                if text.strip():
                    print(f"PyMuPDF extracted {len(text)} characters")
                    return text.strip()
                else:
                    print("PyMuPDF returned empty text")
            except Exception as e:
                print(f"PyMuPDF failed for {file_path}: {e}")
        
        # Fallback to pdfplumber
        print("Falling back to pdfplumber...")
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            print(f"pdfplumber failed for {file_path}: {e}")
        
        # If both fail, return error
        print(f"All PDF extraction methods failed for {file_path}")
        return f"PDF extraction failed: All methods failed"


    def extract_docx_text(self,file_path):
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")


    def extract_html_text(self,file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except ImportError:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    def extract_audio_text(self, file_path):
        """Extract transcript from audio files using MLX Whisper (M2) or Whisper (CPU)"""
        try:
            if MLX_WHISPER_AVAILABLE:
                return self._transcribe_with_mlx(file_path)
            else:
                return self._transcribe_with_whisper(file_path)
                
        except Exception as e:
            print(f"Audio transcription failed: {e}")
            return f"Transcription failed: {str(e)}"
    
    def _transcribe_with_mlx(self, file_path):
        """Transcribe using MLX for M2 acceleration"""
        try:
            print(f"Transcribing with MLX (M2): {file_path}")
            
            if MLX_WHISPER_AVAILABLE:
                result = mlx_whisper.transcribe(file_path)
                return result["text"].strip()
            else:
                print("mlx-whisper not installed, falling back to CPU")
                return self._transcribe_with_whisper(file_path)
                
        except Exception as e:
            print(f"MLX transcription failed: {e}")
            return self._transcribe_with_whisper(file_path)
    
    def _transcribe_with_whisper(self, file_path):
        """Transcribe using Whisper with optimal acceleration for platform"""
        try:
            # Lazy load Whisper model with device optimization
            if self._whisper_model is None:
                # Priority 1: faster-whisper (4x speed boost, works on CPU/CUDA)
                if FASTER_WHISPER_AVAILABLE:
                    print(f"Loading faster-whisper model on {self.device}...")
                    # Use float16 for GPU, int8 for CPU
                    compute_type = "float16" if self.cuda_available else "int8"
                    self._whisper_model = WhisperModel("base", device=self.device, compute_type=compute_type)
                    self._whisper_model_type = "faster"
                    print(f"faster-whisper loaded ({compute_type})")
                
                # Priority 2: MLX Whisper (MacBook M2 acceleration)
                elif MLX_WHISPER_AVAILABLE:
                    print("Loading MLX Whisper (M2 MacBook)...")
                    self._whisper_model = "mlx"  # Special flag for MLX
                    self._whisper_model_type = "mlx"
                    print("MLX Whisper ready")
                
                # Priority 3: Regular Whisper (fallback)
                else:
                    print(f"Loading regular Whisper on {self.device}...")
                    self._whisper_model = whisper.load_model("base", device=self.device)
                    self._whisper_model_type = "regular"
                    print("Regular Whisper loaded")
            
            print(f"Transcribing with {self._whisper_model_type} Whisper: {file_path}")
            
            # Choose transcription method based on model type
            if self._whisper_model_type == "faster":
                # faster-whisper returns segments generator
                segments, info = self._whisper_model.transcribe(file_path, beam_size=5)
                text = " ".join([segment.text for segment in segments])
                return text.strip()
            
            elif self._whisper_model_type == "mlx":
                # MLX Whisper
                result = mlx_whisper.transcribe(file_path, path_or_hf_repo="mlx-community/whisper-base")
                return result["text"].strip()
            
            else:
                # Regular Whisper
                result = self._whisper_model.transcribe(file_path, verbose=False)
                return result["text"].strip()
            
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            # Fallback to CPU if CUDA fails
            if self.cuda_available and "CUDA" in str(e):
                print("CUDA failed, falling back to CPU...")
                try:
                    if FASTER_WHISPER_AVAILABLE:
                        cpu_model = WhisperModel("base", device="cpu", compute_type="int8")
                        segments, info = cpu_model.transcribe(file_path)
                        text = " ".join([segment.text for segment in segments])
                        return text.strip()
                    else:
                        cpu_model = whisper.load_model("base", device="cpu")
                        result = cpu_model.transcribe(file_path, verbose=False)
                        return result["text"].strip()
                except Exception as fallback_error:
                    print(f"CPU fallback also failed: {fallback_error}")
            return f"Transcription failed: {str(e)}"

    def load_metadata(self, file_path):
        """Load metadata from JSON file if exists - HANDLES ALL CONTENT TYPES"""
        # Try different metadata file patterns
        metadata_patterns = [
            file_path.replace('.mp3', '_metadata.json'),  # Audio files
            file_path + '_metadata.json',                  # Generic pattern
            file_path.replace('.pdf', '_metadata.json'),  # PDF files
            file_path.replace('.docx', '_metadata.json'),  # DOCX files
            file_path.replace('.txt', '_metadata.json'),   # Text files
        ]
        
        for metadata_file in metadata_patterns:
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except UnicodeDecodeError:
                    try:
                        with open(metadata_file, 'r', encoding='latin-1') as f:
                            return json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not read metadata file {metadata_file}: {e}")
                        continue
        
        return {}

    def _generate_file_hash(self, file_path):
        """Generate MD5 hash of file content for duplicate detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error generating hash for {file_path}: {e}")
            return None
    
    def _create_basic_metadata(self, file_path, file_type):
        """Create basic metadata for files without existing metadata"""
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
        filename = Path(file_path).stem
        
        metadata = {
            'title': filename,
            'source': filename,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'file_size_mb': file_size_mb,
            'original_format': file_type,
            'extraction_method': 'automatic',
            'processing_status': 'downloaded',
            'download_timestamp': datetime.now().isoformat()
        }
        
        # Save metadata file
        metadata_file = file_path.replace('.mp3', '_metadata.json')
        if not metadata_file.endswith('_metadata.json'):
            metadata_file = file_path + '_metadata.json'
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

    def chunk_text(self, text, chunk_size=1000, overlap=200, use_tokens=True):
        """
        Chunk text into smaller segments for vectorization
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (tokens or characters)
            overlap: Overlap between chunks
            use_tokens: If True, use token-based chunking; if False, use character-based
            
        Returns:
            List of chunk dictionaries with text, start, end positions
        """
        if not text or not text.strip():
            return []
        
        if use_tokens:
            try:
                # Use tiktoken for token-based chunking (more accurate for embeddings)
                encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
                tokens = encoding.encode(text)
                
                chunks = []
                start_idx = 0
                
                while start_idx < len(tokens):
                    end_idx = min(start_idx + chunk_size, len(tokens))
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = encoding.decode(chunk_tokens)
                    
                    # Get character positions for reference
                    chunk_start_char = len(encoding.decode(tokens[:start_idx]))
                    chunk_end_char = len(encoding.decode(tokens[:end_idx]))
                    
                    chunks.append({
                        'text': chunk_text.strip(),
                        'start_token': start_idx,
                        'end_token': end_idx,
                        'start_char': chunk_start_char,
                        'end_char': chunk_end_char,
                        'token_count': len(chunk_tokens)
                    })
                    
                    # Move start index with overlap
                    start_idx = max(start_idx + 1, end_idx - overlap)
                
                return chunks
                
            except (ImportError, UnicodeError, Exception) as e:
                # Fallback to character-based if tiktoken fails
                print(f"Warning: tiktoken chunking failed ({e}), falling back to character-based chunking")
                use_tokens = False
        
        # Character-based chunking
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = max(end - 100, start)
                for i in range(end, sentence_end, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'char_count': len(chunk_text)
                })
            
            # Move start with overlap
            start = max(start + 1, end - overlap)
        
        return chunks

    def add_content_data(self, file_path, title=None, content=None):
        """Add any supported file type to database - SMART ETL FLOW"""
        # 1. Check if file already exists in database (path check)
        if self.db.file_exists(file_path):
            print(f"Skipping: {file_path} already processed")
            print(f"  File path already exists in database")
            return None
        
        # 2. Extract content if not provided
        if content is None:
            content, file_type = self.extract_content(file_path)
        else:
            file_type = self.detect_file_type(file_path)
        
        # Skip if extraction failed completely
        if content and ("extraction failed" in content.lower() or "transcription failed" in content.lower()):
            print(f"Skipping {file_path} due to extraction failure")
            return None
        
        # 3. Generate file hash for duplicate detection
        content_hash = self._generate_file_hash(file_path)
        if content_hash is None:
            print(f"Error: Could not generate hash for {file_path}")
            return None
        
        # 4. Check for duplicate content (hash check)
        existing_duplicate = self.db.hash_exists(content_hash)
        if existing_duplicate:
            existing_id, existing_path = existing_duplicate
            print(f"Skipping: {file_path} is duplicate of existing file: {existing_path}")
            print(f"  Same content hash found (ID: {existing_id})")
            return None
        
        # 5. Load metadata if available, create basic if missing
        print(f"Loading metadata for: {file_path}")
        metadata = self.load_metadata(file_path)
        if not metadata:
            print(f"Creating basic metadata for: {file_path}")
            metadata = self._create_basic_metadata(file_path, file_type)
        print(f"Metadata loaded/created successfully")
        
        # 6. Use filename or metadata for title
        if title is None:
            if file_type == 'audio':
                title = metadata.get('episode_title', Path(file_path).stem)
            else:
                title = metadata.get('title', Path(file_path).stem)
        
        # 7. Get file metadata
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
        
        # 8. Convert duration from metadata if present
        duration_seconds = metadata.get('duration')
        if duration_seconds:
            try:
                # Handle different duration formats
                if isinstance(duration_seconds, str):
                    if ':' in duration_seconds:
                        # Format: "00:24:31" (HH:MM:SS)
                        parts = duration_seconds.split(':')
                        if len(parts) == 3:
                            duration_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        elif len(parts) == 2:
                            duration_seconds = int(parts[0]) * 60 + int(parts[1])
                        else:
                            duration_seconds = float(duration_seconds)
                    else:
                        duration_seconds = float(duration_seconds)
                else:
                    duration_seconds = float(duration_seconds)
            except (ValueError, TypeError):
                print(f"Warning: Could not parse duration '{duration_seconds}', setting to None")
                duration_seconds = None
        
        # 9. Chunk the content for vectorization
        print(f"Chunking content ({len(content)} characters)...")
        segments = self.chunk_text(content, chunk_size=1000, overlap=200, use_tokens=True)
        print(f"Created {len(segments)} segments from content")
        
        # 10. Prepare data for DB - HANDLES ALL CONTENT TYPES
        print("Preparing data for database insertion...")
        data = {
            'title': title,
            'content_type': 'audio' if file_type == 'audio' else 'text',
            'source_type': file_type,
            'source_name': metadata.get('podcast_name', metadata.get('source', Path(file_path).name)),
            'file_path': file_path,
            'transcript': content,
            'pub_date': metadata.get('pub_date', metadata.get('date', '')),
            'duration_seconds': duration_seconds,
            'file_size_mb': file_size_mb,
            'content_hash': content_hash,
            'segments': segments,  # Add chunks for vectorization
            'metadata': {
                **metadata,  # Include all metadata (RSS, PDF, etc.)
                'original_format': file_type,
                'extraction_method': 'transcription' if file_type == 'audio' else 'automatic',
                'file_size_mb': file_size_mb
            }
        }
        print("Data preparation completed")
        
        # 11. Add to DB
        print("Adding to relational database...")
        if self.db is None:
            raise ValueError("No database connection provided")
        
        result = self.db.add_content_metadata(data)
        print(f"Successfully added to database with ID: {result}")
        
        # 12. Mark as processed in metadata file (audio only)
        if file_type == 'audio':
            self.sources.mark_episode_processed(file_path, 'processed')
        
        print(f"Successfully processed: {file_path} (ID: {result})")
        return result

    def get_pending_vectorization(self, limit=100):
        """Get content that needs vectorization"""
        try:
            query = """
                SELECT id, title, transcript, segments, metadata_json 
                FROM content 
                WHERE vectorization_status = 'pending' 
                ORDER BY created_at DESC 
                LIMIT ?
            """
            results = self.db.con.execute(query, [limit]).fetchall()
            
            pending_items = []
            for row in results:
                content_id, title, transcript, segments_json, metadata_json = row
                
                # Parse JSON fields
                segments = json.loads(segments_json) if segments_json else []
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                pending_items.append({
                    'id': content_id,
                    'title': title,
                    'transcript': transcript,
                    'segments': segments,
                    'metadata': metadata
                })
            
            return pending_items
            
        except Exception as e:
            print(f"Error getting pending vectorization: {e}")
            return []

    def vectorize_pending_batch(self, limit=100):
        """Vectorize pending content in batches"""
        pending_items = self.get_pending_vectorization(limit)
        
        if not pending_items:
            print("No pending items to vectorize")
            return 0
        
        print(f"Vectorizing {len(pending_items)} items...")
        
        for item in pending_items:
            try:
                # Prepare documents for vector DB
                documents = []
                
                # Use segments if available, otherwise use full transcript
                if item['segments']:
                    for i, segment in enumerate(item['segments']):
                        doc = {
                            'id': f"{item['id']}_seg_{i}",
                            'content': segment['text'],
                            'metadata': {
                                'content_id': item['id'],
                                'title': item['title'],
                                'segment_index': i,
                                'segment_start': segment.get('start_char', 0),
                                'segment_end': segment.get('end_char', 0),
                                **item['metadata']
                            }
                        }
                        documents.append(doc)
                else:
                    # Fallback to full transcript
                    doc = {
                        'id': str(item['id']),
                        'content': item['transcript'],
                        'metadata': {
                            'content_id': item['id'],
                            'title': item['title'],
                            **item['metadata']
                        }
                    }
                    documents.append(doc)
                
                # Add to vector database
                if self.vdb:
                    # Separate texts and metadata for VectorDB
                    texts = [doc['content'] for doc in documents]
                    metadatas = [doc['metadata'] for doc in documents]
                    self.vdb.upsert_documents(texts, metadatas)
                else:
                    print("Warning: No vector database connected")
                
                # Update status in relational DB
                self.db.update_record(item['id'], {'vectorization_status': 'completed'})
                
                print(f"Vectorized: {item['title']} ({len(documents)} segments)")
                
            except Exception as e:
                print(f"Error vectorizing {item['title']}: {e}")
                # Mark as failed
                self.db.update_record(item['id'], {'vectorization_status': 'failed'})
        
        return len(pending_items)

    def process_pending_audio(self):
        """Process only audio files that haven't been processed yet"""
        pending_episodes = self.sources.get_pending_episodes()
        
        if not pending_episodes:
            print("No pending audio episodes to process")
            return []
        
        print(f"Found {len(pending_episodes)} pending episodes")
        processed_ids = []
        
        for episode in pending_episodes:
            file_path = episode['file_path']
            print(f"Processing: {episode['episode_title']}")
            
            try:
                result = self.add_content_data(file_path)
                if result is not None:
                    processed_ids.append(result)
                    print(f"Successfully processed: {episode['episode_title']} (ID: {result})")
                else:
                    print(f"Skipped: {episode['episode_title']} (already processed or duplicate)")
            except Exception as e:
                print(f"Failed to process {episode['episode_title']}: {e}")
                # Mark as failed
                self.sources.mark_episode_processed(file_path, 'failed')
        
        return processed_ids

    def process_directory(self, directory_path=None):
        """Process all supported files in a directory"""
        if directory_path is None:
            # Process all standardized directories
            directories = [
                self.sources.audio_dir,
                self.sources.text_dir,
                self.sources.pdf_dir,
                self.sources.video_dir
            ]
        else:
            directories = [directory_path]
            
        for directory in directories:
            if not os.path.exists(directory):
                continue
                
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)

                # Skip directories and system files
                if not os.path.isfile(filepath):
                    continue
                    
                # Skip hidden files (starting with .)
                if filename.startswith('.'):
                    continue
                
                # Skip metadata files (these are data, not content)
                if filename.endswith('_metadata.json'):
                    continue

                try:
                    print(f"Processing file: {filename}")
                    result = self.add_content_data(filepath)
                    if result is not None:
                        print(f"Added to database with ID: {result}")
                    # If result is None, file was skipped (already processed or duplicate)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")





if __name__ == '__main__':
    # Auto-detect project directory (works on both MacBook and Jetson)
    rel_path = os.path.dirname(os.path.abspath(__file__))


    content_dir = os.path.join(rel_path, 'content_files')
    db_path = os.path.join(rel_path, 'Database/industry_signals.db')
    vdb_path = os.path.join(rel_path, 'Vectors/industry_signals_vectors')
    #Initialize databases
    
    
    # Initialize relational database
    print("Intializing Relational Database...")
    db = relationalDB(db_path)
    db.init_db()
    
    # Initialize vector database (optional)
    print("Initializing Vector Database...")
    vdb = VectorDB("Vectors/")
    vdb.load("industry_signals_vectors")

    # Initialize ETL with both databases
    print("Initializing ETL...")
    etl = contentETL(content_dir, db=db, vdb=vdb)

    etl.process_directory()

    etl.vectorize_pending_batch(50)

    etl.vdb.save("industry_signals_vectors")

    
    # Show download statistics
    # print("\n=== Download Statistics ===")
    # stats = etl.sources.get_download_stats()
    # for key, value in stats.items():
    #     print(f"{key}: {value}")
    

    