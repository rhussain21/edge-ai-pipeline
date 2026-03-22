import os
import json
from datetime import datetime
from pathlib import Path
import re
import hashlib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_config import config
VECTOR_PATH = os.getenv("VECTOR_PATH", os.getenv("JETSON_VECTOR_PATH", os.getenv("VECTOR_DB_PATH", "Vectors/")))
import pdfplumber
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    #print("PyMuPDF available")
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available")
from bs4 import BeautifulSoup
import docx
import whisper
from etl.sources import ContentSources
import tiktoken
from db_relational import relationalDB
from db_vector import VectorDB
from etl.signals import SignalPipeline
from logging_config import syslog


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

# Check for Jetson CUDA specifically
try:
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            if 'jetson' in model.lower() and not CUDA_AVAILABLE:
                print(f"Detected Jetson device: {model}")
                print("Jetson detected but CUDA not available in PyTorch - checking for CUDA libraries")
                # Try to force CUDA detection on Jetson
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                    if result.returncode == 0:
                        print("NVIDIA GPU detected via nvidia-smi")
                        # Only set CUDA_AVAILABLE if PyTorch actually supports CUDA
                        try:
                            if torch.cuda.is_available():
                                CUDA_AVAILABLE = True
                                DEVICE = "cuda"
                                print("Forcing CUDA device for Jetson")
                            else:
                                print("PyTorch CUDA not available, using CPU")
                        except Exception as torch_error:
                            print(f"PyTorch CUDA support not available: {torch_error}")
                            print("Using CPU instead")
                    else:
                        print("No NVIDIA GPU detected via nvidia-smi")
                except:
                    print("Could not detect NVIDIA GPU via nvidia-smi")
except:
    pass

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("faster-whisper available")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("faster-whisper not available, using regular whisper")

try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
    print("MLX Whisper available")
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    print("MLX Whisper not available")

JETSON_AVAILABLE = False
try:
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
        self.db = db
        self.vdb = vdb
        self._whisper_model = None
        self.sources = ContentSources(content_path)
        self.device = DEVICE
        self.cuda_available = CUDA_AVAILABLE
        self.jetson_available = JETSON_AVAILABLE
        
        # Initialize signal vector database (sibling of corpus_vectors under VECTOR_PATH)
        signal_vector_path = os.path.join(VECTOR_PATH.rstrip("/"), "signal_vectors")
        self.signal_vdb = VectorDB(signal_vector_path, use_builtin_embeddings=True)
        # CUDA is already handled in VectorDB constructor now
        
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
        """Transcribe using Whisper with optimal acceleration for platform.
        
        Benchmark results (Jetson Orin Nano):
          faster-whisper/tiny/cpu/int8/no-vad = 0.068 RTF (~15x real-time)
          openai/tiny/cuda                    = 0.088 RTF
          openai/base/cpu                     = 0.289 RTF (slowest)
        
        Strategy: faster-whisper tiny on CPU is fastest and frees GPU for embeddings.
        """
        try:
            if self._whisper_model is None:
                if FASTER_WHISPER_AVAILABLE:
                    print("Loading faster-whisper (tiny, cpu, int8)...")
                    self._whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                    self._whisper_model_type = "faster"
                    print("faster-whisper loaded (tiny/cpu/int8)")
                
                elif MLX_WHISPER_AVAILABLE:
                    print("Loading MLX Whisper (M2 MacBook)...")
                    self._whisper_model = "mlx"
                    self._whisper_model_type = "mlx"
                    print("MLX Whisper ready")
                
                else:
                    print(f"Loading regular Whisper on {self.device}...")
                    self._whisper_model = whisper.load_model("tiny", device=self.device)
                    self._whisper_model_type = "regular"
                    print("Regular Whisper loaded")
            
            print(f"Transcribing with {self._whisper_model_type} Whisper: {file_path}")
            
            if self._whisper_model_type == "faster":
                segments, info = self._whisper_model.transcribe(
                    file_path, beam_size=1, vad_filter=False
                )
                text = " ".join([segment.text for segment in segments])
                return text.strip()
            
            elif self._whisper_model_type == "mlx":
                result = mlx_whisper.transcribe(file_path, path_or_hf_repo="mlx-community/whisper-tiny")
                return result["text"].strip()
            
            else:
                result = self._whisper_model.transcribe(file_path, verbose=False)
                return result["text"].strip()
            
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            if self.cuda_available and "CUDA" in str(e):
                print("CUDA failed, falling back to CPU...")
                try:
                    if FASTER_WHISPER_AVAILABLE:
                        cpu_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                        segments, info = cpu_model.transcribe(
                            file_path, beam_size=1, vad_filter=False
                        )
                        text = " ".join([segment.text for segment in segments])
                        return text.strip()
                    else:
                        cpu_model = whisper.load_model("tiny", device="cpu")
                        result = cpu_model.transcribe(file_path, verbose=False)
                        return result["text"].strip()
                except Exception as fallback_error:
                    print(f"CPU fallback also failed: {fallback_error}")
            return f"Transcription failed: {str(e)}"

    def load_metadata(self, file_path):
        """Load metadata from JSON file if exists."""
        stem = os.path.splitext(file_path)[0]
        metadata_patterns = [
            stem + '_metadata.json',
            file_path.replace('.mp3', '_metadata.json'),
            file_path.replace('.html', '_metadata.json'),
            file_path.replace('.htm', '_metadata.json'),
            file_path.replace('.pdf', '_metadata.json'),
            file_path.replace('.docx', '_metadata.json'),
            file_path.replace('.txt', '_metadata.json'),
            file_path + '_metadata.json',
        ]
        seen = set()
        for metadata_file in metadata_patterns:
            if metadata_file in seen:
                continue
            seen.add(metadata_file)
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (UnicodeDecodeError, ValueError):
                    try:
                        with open(metadata_file, 'r', encoding='latin-1') as f:
                            return json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not read metadata file {metadata_file}: {e}")
                        continue
        
        return {}

    def _generate_file_hash(self, file_path):
        """Generate MD5 hash of file content for duplicate detection."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error generating hash for {file_path}: {e}")
            return None
    
    def _create_basic_metadata(self, file_path, file_type):
        """Create basic metadata for files without existing metadata."""
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
        
        metadata_file = file_path.replace('.mp3', '_metadata.json')
        if not metadata_file.endswith('_metadata.json'):
            metadata_file = file_path + '_metadata.json'
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

    def chunk_text(self, text, chunk_size=1000, overlap=200, use_tokens=True):
        """Chunk text into smaller segments for vectorization."""
        if not text or not text.strip():
            return []
        
        if use_tokens:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(text)
                
                chunks = []
                start_idx = 0
                
                while start_idx < len(tokens):
                    end_idx = min(start_idx + chunk_size, len(tokens))
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = encoding.decode(chunk_tokens)
                    
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
                    
                    start_idx = max(start_idx + 1, end_idx - overlap)
                
                return chunks
                
            except (ImportError, UnicodeError, Exception) as e:
                print(f"Warning: tiktoken chunking failed ({e}), falling back to character-based chunking")
                use_tokens = False
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            if end < len(text):
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
            
            start = max(start + 1, end - overlap)
        
        return chunks

    def add_content_data(self, file_path, title=None, content=None):
        """Add any supported file type to database."""
        if self.db.file_exists(file_path):
            print(f"Skipping: {file_path} already processed")
            print(f"  File path already exists in database")
            return None
        
        if content is None:
            content, file_type = self.extract_content(file_path)
        else:
            file_type = self.detect_file_type(file_path)
        
        if content and ("extraction failed" in content.lower() or "transcription failed" in content.lower()):
            print(f"Skipping {file_path} due to extraction failure")
            return None
        
        content_hash = self._generate_file_hash(file_path)
        if content_hash is None:
            print(f"Error: Could not generate hash for {file_path}")
            return None
        
        existing_duplicate = self.db.hash_exists(content_hash)
        if existing_duplicate:
            existing_id, existing_path = existing_duplicate
            print(f"Skipping: {file_path} is duplicate of existing file: {existing_path}")
            print(f"  Same content hash found (ID: {existing_id})")
            return None
        
        print(f"Loading metadata for: {file_path}")
        metadata = self.load_metadata(file_path)
        if not metadata:
            print(f"Creating basic metadata for: {file_path}")
            metadata = self._create_basic_metadata(file_path, file_type)
        print(f"Metadata loaded/created successfully")
        
        if title is None:
            if file_type == 'audio':
                title = metadata.get('episode_title', Path(file_path).stem)
            else:
                title = metadata.get('title', Path(file_path).stem)
        
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
        duration_seconds = metadata.get('duration')
        if duration_seconds:
            try:
                if isinstance(duration_seconds, str):
                    if ':' in duration_seconds:
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
        
        print(f"Chunking content ({len(content)} characters)...")
        segments = self.chunk_text(content, chunk_size=1000, overlap=200, use_tokens=True)
        print(f"Created {len(segments)} segments from content")
        
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
            'segments': segments,
            'metadata': {
                **metadata,
                'original_format': file_type,
                'extraction_method': 'transcription' if file_type == 'audio' else 'automatic',
                'file_size_mb': file_size_mb
            }
        }
        print("Data preparation completed")
        
        print("Adding to relational database...")
        if self.db is None:
            raise ValueError("No database connection provided")
        
        result = self.db.add_content_metadata(data)
        print(f"Successfully added to database with ID: {result}")
        
        # Write structured metadata to content_metadata table
        try:
            self.db.add_content_metadata_record(result, data['metadata'])
            print(f"Metadata record added for content_id={result}")
        except Exception as e:
            print(f"Warning: Failed to write content_metadata record: {e}")
        
        if file_type == 'audio':
            self.sources.mark_episode_processed(file_path, 'processed')
        
        syslog.info('pipeline', 'content_added', f'Processed: {title[:80]}',
                     content_id=result, duration_sec=duration_seconds,
                     details={'content_type': data['content_type'], 'file_size_mb': file_size_mb})
        print(f"Successfully processed: {file_path} (ID: {result})")
        return result

    def _create_segments(self, transcript, max_chars=1000):
        """Create segments from transcript for vectorization."""
        if not transcript:
            return []
        
        segments = []
        words = transcript.split()
        current_segment = []
        current_length = 0
        
        for word in words:
            current_segment.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= max_chars:
                segments.append(' '.join(current_segment))
                current_segment = []
                current_length = 0
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments

    def get_pending_vectorization(self, limit=100):
        """Get content that needs vectorization."""
        try:
            query = """
                SELECT id, title, transcript, segments, metadata_json 
                FROM content 
                WHERE vectorization_status = 'pending' 
                ORDER BY created_at DESC 
                LIMIT ?
            """
            results = self.db.execute(query, [limit]).fetchall()
            
            pending_items = []
            for row in results:
                content_id, title, transcript, segments_json, metadata_json = row
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
        """Vectorize pending content in batches."""
        pending_items = self.get_pending_vectorization(limit)
        
        if not pending_items:
            print("No pending items to vectorize")
            return 0
        
        print(f"Vectorizing {len(pending_items)} items...")
        
        for item in pending_items:
            try:
                documents = []
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
                
                if self.vdb:
                    texts = [doc['content'] for doc in documents]
                    metadatas = [doc['metadata'] for doc in documents]
                    self.vdb.upsert_documents(texts, metadatas)
                else:
                    print("Warning: No vector database connected")
                
                self.db.update_record(item['id'], {'vectorization_status': 'completed'})
                
                print(f"Vectorized: {item['title']} ({len(documents)} segments)")
                
            except Exception as e:
                print(f"Error vectorizing {item['title']}: {e}")
                self.db.update_record(item['id'], {'vectorization_status': 'failed'})
        
        return len(pending_items)

    def get_pending_content(self, content_type=None, limit=None):
        """Get content pending transcription from DB."""
        try:
            query = """
                SELECT id, title, file_path, content_type, source_name
                FROM content
                WHERE transcription_status = 'pending'
            """
            params = []
            
            if content_type:
                query += " AND content_type = ?"
                params.append(content_type)
            
            query += " ORDER BY created_at ASC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            results = self.db.query(query, params)
            return [dict(r) for r in results]
            
        except Exception as e:
            print(f"Error getting pending content: {e}")
            return []

    def process_pending_content(self, content_type=None):
        """Process content pending transcription (audio, pdf, html, text)."""
        pending_content = self.get_pending_content(content_type=content_type)
        
        if not pending_content:
            print("No pending content to process")
            return []
        
        print(f"Found {len(pending_content)} pending items")
        processed_ids = []
        
        for item in pending_content:
            content_id = item['id']
            file_path = item['file_path']
            title = item['title']
            
            print(f"Processing: {title} (ID: {content_id})")
            
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"  File not found: {file_path}")
                    self.db.update_record(content_id, {'transcription_status': 'failed'})
                    continue
                
                # Extract content (transcribe audio)
                content, file_type = self.extract_content(file_path)
                if not content or "transcription failed" in content.lower():
                    print(f"  Transcription failed for: {title}")
                    self.db.update_record(content_id, {'transcription_status': 'failed'})
                    continue
                
                # Generate hash (no duplicate check since we're updating existing record)
                content_hash = self._generate_file_hash(file_path)
                if content_hash is None:
                    print(f"  Could not generate hash for: {file_path}")
                    self.db.update_record(content_id, {'transcription_status': 'failed'})
                    continue
                
                # Update existing record with transcription results
                is_audio = item.get('content_type') == 'audio'
                update_data = {
                    'transcript': content,
                    'transcription_date': datetime.now().isoformat() if is_audio else None,
                    'transcript_method': 'whisper' if is_audio else None,
                    'transcription_status': 'completed' if is_audio else 'NA',
                    'content_hash': content_hash,
                    'language': 'en',
                    'segments': json.dumps(self._create_segments(content)) if content else []
                }
                
                # Get file size and duration if available
                try:
                    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                    update_data['file_size_mb'] = file_size_mb
                except:
                    pass
                
                self.db.update_record(content_id, update_data)
                processed_ids.append(content_id)
                print(f"  ✓ Transcribed: {title}")
                
            except Exception as e:
                print(f"  ✗ Failed to process {title}: {e}")
                self.db.update_record(content_id, {'transcription_status': 'failed'})
        
        return processed_ids

    def process_directory(self, directory_path=None):
        """Process all supported files in a directory."""
        if directory_path is None:
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

                if not os.path.isfile(filepath):
                    continue
                if filename.startswith('.'):
                    continue
                if filename.endswith('_metadata.json'):
                    continue

                try:
                    print(f"Processing file: {filename}")
                    result = self.add_content_data(filepath)
                    if result is not None:
                        print(f"Added to database with ID: {result}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    def vectorize_pending_signals(self, batch_size=100):
        """Vectorize signals that haven't been vectorized yet"""
        # Get unvectorized signals
        signals = self.db.query("""
            SELECT id, signal_type, entity, description, source_content_id
            FROM signals 
            WHERE vectorized = FALSE OR vectorized IS NULL
            LIMIT ?
        """, (batch_size,))
        
        if not signals:
            print("No unvectorized signals found")
            return 0
        
        print(f"Vectorizing {len(signals)} signals...")
        
        # Vectorize them
        count = self.signal_vdb.add_signals(signals)
        
        # Mark as vectorized
        for signal in signals:
            self.db.execute("""
                UPDATE signals 
                SET vectorized = TRUE, vectorized_at = ?
                WHERE id = ?
            """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal['id']))
        
        print(f"Successfully vectorized {count} signals")
        return count
    
    def run_signal_vectorization(self):
        """Main signal vectorization loop"""
        total_vectorized = 0
        
        while True:
            count = self.vectorize_pending_signals()
            if count == 0:
                print(f"All signals vectorized. Total: {total_vectorized}")
                break
            
            total_vectorized += count
            print(f"Progress: {total_vectorized} signals vectorized")
        
        # Save signal vectors
        self.signal_vdb.save("signal_vectors")
        print("Signal vectors saved")
        
        return total_vectorized


if __name__ == '__main__':
    print("Loading environment variables...")
    media_dir = 'media/' #os.getenv("MEDIA_DIR", "media")
    db_path = 'Database/industry_signals.db' #os.getenv("JETSON_DB_PATH", "Database/industry_signals.db")
    vector_base = 'Vectors/' #os.getenv("JETSON_VECTOR_PATH", os.getenv("VECTOR_DB_PATH", "Vectors/"))

    print(f"Media directory: {media_dir}")
    print(f"Database path: {db_path}")
    print(f"Vector base: {vector_base}")

    print("Intializing Relational Database...")
    db = relationalDB(db_path)
    db.init_db()
    
    print("Initializing ETL for Signal Vectorization...")
    etl = contentETL(media_dir, db=db)
    
    print("Running signal vectorization...")
    total_signals = etl.run_signal_vectorization()
    print(f"Signal vectorization complete: {total_signals} signals processed")