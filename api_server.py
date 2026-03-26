"""FastAPI Server for Jetson ETL System."""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import logging
from pydantic import BaseModel
from db_relational import relationalDB
from db_vector import VectorDB
from device_config import config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jetson ETL API",
    description="API for syncing ETL data from Jetson to MacBook",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use device config for paths
DB_PATH = config.DB_PATH
VECTOR_PATH = config.VECTOR_PATH
CORPUS_VECTOR_PATH = config.CORPUS_VECTOR_PATH
SIGNAL_VECTOR_PATH = config.SIGNAL_VECTOR_PATH


db = relationalDB(DB_PATH)

corpus_vdb = VectorDB(CORPUS_VECTOR_PATH)
try:
    corpus_vdb.load("corpus_vectors")
    print(f"Loaded {len(corpus_vdb.documents)} corpus vectors")
except Exception as e:
    print(f"Warning: Could not load corpus vectors: {e}")

signal_vdb = VectorDB(SIGNAL_VECTOR_PATH)
try:
    signal_vdb.load("signal_vectors")
    print(f"Loaded {len(signal_vdb.documents)} signal vectors")
except Exception as e:
    print(f"Warning: Could not load signal vectors: {e}")


class StatsResponse(BaseModel):
    total_content: int
    content_by_type: Dict[str, int]
    total_corpus_vectors: int
    total_signal_vectors: int
    last_updated: str



@app.get("/")
async def root():
    return {
        "name": "Jetson ETL API",
        "version": "1.0.0",
        "description": "API for content aggregation, processing, and search",
        "endpoints": {
            "health": "/health - System health check",
            "stats": "/api/stats - Content and vector database statistics", 
            "search": "/api/search - Semantic search across content library",
            "sync": "/api/content/sync - Incremental content sync",
            "signals": "/api/signals/sync - Incremental signals sync",
            "logs": "/api/logs/sync - Incremental system logs sync",
            "content": "/api/content/item/{id} - Get specific content item",
            "docs": "/docs - Interactive API documentation (Swagger UI)",
            "openapi": "/openapi.json - OpenAPI specification"
        },
        "usage": {
            "search_example": "/api/search?text=manufacturing%20automation",
            "sync_example": "/api/content/sync?since=2024-01-01T00:00:00Z",
            "content_example": "/api/content/item/123"
        },
        "system": {
            "database": "DuckDB relational database",
            "vectors": "FAISS vector databases (corpus + signal)",
            "processing": "Automated ETL pipeline for content ingestion"
        }
    }

@app.get("/health")
async def health():
    db_status = "Connected" if db.test_connection() else "Not Connected"
    return {"database status": f"{db_status}"}


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    try:


        # Get content by type
        content_by_type_query = """
            SELECT content_type, COUNT(*) as count
            FROM content
            GROUP BY content_type
        """
        content_results = db.query(content_by_type_query)
        
        # Get overall stats
        overall_stats_query = """
            SELECT 
                COUNT(*) as total_content,
                COUNT(DISTINCT source_type) as unique_sources,
                AVG(file_size_mb) as avg_file_size,
                MAX(updated_at) as last_updated
            FROM content
        """
        overall_stats = db.query(overall_stats_query)
        
        print("Content by type:", content_results)
        print("Overall stats:", overall_stats)
        

        content_by_type = {row['content_type']: row['count'] for row in content_results}
        stats = overall_stats[0] if overall_stats else {}
        
        return StatsResponse(
            total_content=stats.get('total_content', 0),
            content_by_type=content_by_type,
            total_corpus_vectors=len(corpus_vdb.documents) if corpus_vdb else 0,
            total_signal_vectors=len(signal_vdb.documents) if signal_vdb else 0,
            last_updated=stats.get('last_updated', 'N/A')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.get("/api/content/sync")
async def sync_content(
    since: str = Query(..., description="ISO timestamp of last sync"),
    limit: int = Query(1000, ge=1, le=5000)
):
    try:
        query = """
            SELECT * FROM content 
            WHERE updated_at > ?
            ORDER BY updated_at ASC
            LIMIT ?
        """
        data = db.query(query, (since,limit))
        
        return {
            "count": len(data),
            "since": since,
            "data": data,
            "has_more": len(data) == limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing content: {str(e)}")


@app.get("/api/signals/sync")
async def sync_signals(
    since: str = Query(..., description="ISO timestamp of last sync"),
    limit: int = Query(1000, ge=1, le=5000)
):
    try:
        query = """
            SELECT * FROM signals 
            WHERE extracted_at > ?
            ORDER BY extracted_at ASC
            LIMIT ?
        """
        data = db.query(query, (since, limit))
        return {
            "count": len(data),
            "since": since,
            "data": data,
            "has_more": len(data) == limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing signals: {str(e)}")


@app.get("/api/logs/sync")
async def sync_logs(
    since: str = Query(..., description="ISO timestamp of last sync"),
    limit: int = Query(1000, ge=1, le=5000)
):
    try:
        query = """
            SELECT * FROM system_logs 
            WHERE timestamp > ?
            ORDER BY timestamp ASC
            LIMIT ?
        """
        data = db.query(query, (since, limit))
        return {
            "count": len(data),
            "since": since,
            "data": data,
            "has_more": len(data) == limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing logs: {str(e)}")


@app.get("/api/content/item/{content_id}")
async def get_content_by_id(content_id: int):
    try:
        query = "SELECT * FROM content WHERE id = ?"
        data = db.query(query, (content_id,))

        if not data:
            raise HTTPException(status_code=404, detail="Content not found")

        row = data[0]
        return {
            'title': row['title'],
            'source_type': row['source_type'],
            'publication_date': row['pub_date']
        } 
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching content: {str(e)}")


@app.get("/api/search")
async def search_vectors(
    text: str = Query(..., description="Search query text"),
    limit: int = Query(10, ge=1, le=100),
    threshold: float = Query(0.3, ge=0.0, le=1.0)
):
    try:
        if not corpus_vdb or not corpus_vdb.model:
            raise HTTPException(status_code=503, detail="Corpus vector search not available")
        
        response = corpus_vdb.search(text, top_k=limit, cosine_threshold=threshold)
        return {
            "query": text,
            "count": len(response),
            "results": response
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in vector search: {str(e)}")


@app.get("/api/signals/search")
async def search_signal_vectors(
    text: str = Query(..., description="Search query text"),
    limit: int = Query(10, ge=1, le=100),
    threshold: float = Query(0.3, ge=0.0, le=1.0)
):
    try:
        if not signal_vdb or not signal_vdb.model:
            raise HTTPException(status_code=503, detail="Signal vector search not available")
        
        response = signal_vdb.search(text, top_k=limit, cosine_threshold=threshold)
        return {
            "query": text,
            "count": len(response),
            "results": response
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in signal vector search: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting Jetson ETL API Server")
    print("="*60)
    print(f"Database: {DB_PATH}")
    print(f"Corpus Vectors: {CORPUS_VECTOR_PATH}")
    print(f"Signal Vectors: {SIGNAL_VECTOR_PATH}")
    if corpus_vdb:
        print(f"Corpus docs: {len(corpus_vdb.documents)}")
    if signal_vdb:
        print(f"Signal docs: {len(signal_vdb.documents)}")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  
    )
