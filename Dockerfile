
# Jetson ETL container for the AI Industry Signals pipeline
#
# This image builds a reproducible runtime for running the content ingestion
# and signal extraction pipeline on NVIDIA Jetson devices (JetPack 6).
#
# Includes:
# - NVIDIA CUDA-enabled PyTorch base image
# - SentenceTransformers for GPU embedding generation
# - FAISS for vector search
# - Whisper / faster-whisper for audio transcription
# - PDF, HTML, and DOCX content extraction libraries
#
# The container is designed to run the `content_etl.py` pipeline which:
# 1. Extracts content from media sources
# 2. Generates embeddings for semantic search
# 3. Stores metadata in a relational database
# 4. Builds vector indexes for retrieval and signal analysis
#
# Base image: nvcr.io/nvidia/pytorch:25.05-py3-igpu

FROM nvcr.io/nvidia/pytorch:25.05-py3-igpu

WORKDIR /workspace/ai_industry_signals

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -i https://pypi.org/simple --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    faiss-cpu \
    "sentence-transformers==3.0.1" \
    "transformers==4.41.2" \
    scikit-learn \
    python-dotenv \
    numpy \
    pdfplumber \
    pymupdf \
    beautifulsoup4 \
    python-docx \
    openai-whisper \
    faster-whisper \
    tiktoken \
    psycopg2-binary \
    feedparser \
    requests \
    langextract \
    lxml \
    tqdm

COPY . .

CMD ["python3", "content_etl.py"]
