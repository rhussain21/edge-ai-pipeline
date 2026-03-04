"""Vector Search Tool - Semantic search in vector database."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorSearchTool:
    name = "vector_search"
    description = """Search vector database for relevant text chunks based on semantic similarity."""
    
    def __init__(self, vector_db, relational_db=None):
        self.vdb = vector_db
        self.db = relational_db
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[str]:
        try:
            results = self.vdb.search(query=query, top_k=top_k)
            
            if not results:
                logger.info(f"No results found for query: '{query}'")
                return []
            
            text_chunks = []
            for r in results:
                if isinstance(r, (list, tuple)):
                    text = r[0]
                elif isinstance(r, dict):
                    text = r.get("document") or r.get("text") or str(r)
                else:
                    text = str(r)
                
                if text and text.strip():
                    text_chunks.append(text.strip())
            
            logger.info(f"Retrieved {len(text_chunks)} text chunks for query: '{query}'")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.vdb.search(query=query, top_k=top_k)
            
            if not results:
                return []
            
            enriched_results = []
            for r in results:
                if isinstance(r, dict):
                    text = r.get("document") or r.get("text") or str(r)
                    metadata = r.get("metadata", {})
                else:
                    text = str(r)
                    metadata = {}
                
                if text and text.strip():
                    enriched_results.append({
                        "text": text.strip(),
                        "metadata": metadata
                    })
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Error searching with metadata: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            if hasattr(self.vdb, 'index') and hasattr(self.vdb.index, 'ntotal'):
                return {
                    "total_vectors": self.vdb.index.ntotal,
                    "embedding_dim": getattr(self.vdb, 'embedding_dim', 'unknown'),
                    "storage_dir": getattr(self.vdb, 'storage_dir', 'unknown')
                }
            else:
                return {"status": "stats not available"}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
