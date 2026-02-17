import os
import faiss
import pickle
import numpy as np
from typing import Optional, List, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class VectorDB:
    """
    Simple FAISS-based vector database for semantic search.
    
    Unlike relational databases that store rows and columns,
    vector databases store mathematical representations (embeddings)
    of text for similarity search.
    """
    
    def __init__(self, vector_dir="vectors", embedding_dim=384, model_name: str = "all-MiniLM-L6-v2", use_builtin_embeddings: bool = True):
        """
        Initialize vector database.
        
        Args:
            vector_dir: Directory to store vector files
            embedding_dim: Size of text embeddings (384 for MiniLM model)
            model_name: SentenceTransformer model name (if using builtin embeddings)
            use_builtin_embeddings: Whether to use built-in embedding model
        """
        self.vector_dir = vector_dir
        self.embedding_dim = embedding_dim
        self.use_builtin_embeddings = use_builtin_embeddings
        
        # Initialize embedding model if requested
        if self.use_builtin_embeddings:
            print(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        else:
            self.model = None
        
        # Core components of vector database:
        self.index = None           # FAISS index - like the "engine" for fast search
        self.documents = []         # Original text content - like "rows" in RDB
        self.metadata = []          # Extra info about each document - like "columns"
        
        # Ensure directory exists
        self._ensure_directory_exists()
        print(f"Vector database initialized at: {self.vector_dir}")
    
    def _ensure_directory_exists(self):
        """Create vector directory if it doesn't exist, with fallback."""
        try:
            # Try to create the requested directory
            if not os.path.exists(self.vector_dir):
                os.makedirs(self.vector_dir, exist_ok=True)
                print(f"Created vector directory: {self.vector_dir}")
            
            # Test if we can write to this location
            test_file = os.path.join(self.vector_dir, "test.tmp")
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
        except (PermissionError, OSError) as e:
            # Fallback to default directory
            fallback_dir = "vectors_fallback"
            if not os.path.exists(fallback_dir):
                os.makedirs(fallback_dir, exist_ok=True)
            print(f"Cannot write to {self.vector_dir}, using fallback: {fallback_dir}")
            self.vector_dir = fallback_dir
    
    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Convert texts to embeddings using built-in model."""
        if not self.use_builtin_embeddings:
            raise ValueError("Built-in embeddings not enabled. Use add_vectors() with pre-computed embeddings.")
        
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings
    
    def upsert_documents(self, texts: List[str], metadata: List[Dict], vectors: Optional[np.ndarray] = None):
        """Build new index or add to existing one (smart upsert)."""
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have same length")
        
        if self.index is None or self.index.ntotal == 0:
            print("No index found — building new FAISS index.")
            self.build_index(texts, metadata, vectors)
        else:
            print("Existing index found — adding new documents.")
            self.add_documents(texts, metadata, vectors)
    
    def build_index(self, texts: List[str], metadata: List[Dict], vectors: Optional[np.ndarray] = None):
    
        """Build FAISS index from scratch."""
        if vectors is None:
            if not self.use_builtin_embeddings:
                raise ValueError("Must provide vectors when not using built-in embeddings")
            vectors = self.create_embeddings(texts)
        
        print(f"Building FAISS index for {len(texts)} documents...")
        
        # Normalize for cosine similarity
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        
        # Create and populate index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(vectors)
        
        # Store documents and metadata
        self.documents = texts
        self.metadata = metadata
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def add_documents(self, texts: List[str], metadata: List[Dict], vectors: Optional[np.ndarray] = None):
        """Add new documents to existing index."""
        if self.index is None:
            self.build_index(texts, metadata, vectors)
            return
        
        if vectors is None:
            if not self.use_builtin_embeddings:
                raise ValueError("Must provide vectors when not using built-in embeddings")
            vectors = self.create_embeddings(texts)
        
        print(f"Adding {len(texts)} new documents to index...")
        
        # Normalize for cosine similarity
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors)
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        
        print(f"Index now contains {self.index.ntotal} vectors")
    
    def add_vectors(self, vectors: np.ndarray, documents: List[str], metadata: List[Dict]):
        """Legacy method for backward compatibility."""
        self.add_documents(documents, metadata, vectors)
    
    def search(self, query: Union[str, np.ndarray], top_k: int = 5, cosine_threshold: float = 0.3, adaptive: bool = True) -> List[Dict]:
        """
        Search for semantically similar documents using cosine similarity.
        
        Args:
            query: Search query text OR pre-computed embedding vector
            top_k: Number of results to return
            cosine_threshold: Minimum cosine similarity to include
            adaptive: If True, lower threshold slightly when no matches found
        
        Returns:
            List of dicts with document, metadata, similarity, and index.
        """
        if self.index is None or self.index.ntotal == 0:
            print("No vectors in index to search")
            return []
        
        # Handle query text vs pre-computed vector
        if isinstance(query, str):
            if not self.use_builtin_embeddings:
                raise ValueError("Cannot search with text when not using built-in embeddings. Provide embedding vector instead.")
            print(f"[DEBUG] Encoding query: '{query[:50]}...'")
            query_vector = self.model.encode([query], convert_to_numpy=True)
        else:
            query_vector = query.reshape(1, -1)
        
        # Normalize query vector
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search the index
        similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(sim),
                    'index': int(idx)
                })
        
        # Filter by cosine threshold
        filtered = [r for r in results if r["similarity"] >= cosine_threshold]
        
        # Adaptive fallback
        if not filtered and adaptive:
            relaxed = cosine_threshold * 0.75
            print(f"[DEBUG] No results above {cosine_threshold:.2f}, relaxing to {relaxed:.2f}")
            filtered = [r for r in results if r["similarity"] >= relaxed]
        
        if not filtered:
            print(f"[DEBUG] No results passed threshold {cosine_threshold}")
            return []
        
        return filtered
    
    def save(self, filename="vectors"):
        """
        Save index and data to disk.
        
        This is like database backup - saves everything to files
        """
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index (the "engine")
        index_path = os.path.join(self.vector_dir, f"{filename}.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata (the "data")
        data_path = os.path.join(self.vector_dir, f"{filename}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"Saved vector database to {index_path} and {data_path}")
    
    def load(self, filename="vectors"):
        """
        Load index and data from disk.
        
        This is like database restore - loads everything from files
        """
        index_path = os.path.join(self.vector_dir, f"{filename}.faiss")
        data_path = os.path.join(self.vector_dir, f"{filename}.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            print(f"Vector files not found: {filename}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        print(f"Loaded {self.index.ntotal} vectors from {filename}")
        return True
    
    def get_stats(self):
        """Get database statistics - like SELECT COUNT(*) in SQL"""
        if self.index is None:
            return {"total_vectors": 0, "dimension": self.embedding_dim}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.embedding_dim,
            "total_documents": len(self.documents),
            "vector_dir": self.vector_dir,
            "index_type": type(self.index).__name__,
            "has_builtin_embeddings": self.use_builtin_embeddings
        }
    
    def cluster_documents(self, n_clusters: int = 4) -> Dict[int, List[str]]:
        """
        Group all indexed documents into N semantic clusters.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dict of cluster_id -> list of document titles/summaries
        """
        if self.index is None or not self.documents:
            raise ValueError("Index not built or empty")
        
        print(f"[DEBUG] Clustering {len(self.documents)} embeddings into {n_clusters} themes...")
        
        # Reconstruct embeddings from FAISS index
        xb = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            xb[i] = self.index.reconstruct(i)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(xb)
        
        # Group documents by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(kmeans.labels_):
            # Try to get a meaningful title from metadata, fallback to document preview
            title = self.metadata[i].get("title") or \
                   self.metadata[i].get("source_name") or \
                   self.documents[i][:50] + "..." if len(self.documents[i]) > 50 else self.documents[i]
            clusters[label].append(title)
        
        return clusters


if __name__ == "__main__":
    # Example usage with enhanced features
    print("Testing Enhanced Vector Database...")
    
    # Create database with built-in embeddings
    vector_db = VectorDB("Vectors", use_builtin_embeddings=True)
    
    # Load existing industry signals vectors
    print("\n=== Loading Industry Signals Vectors ===")
    load_success = vector_db.load("industry_signals_vectors")
    
    if not load_success:
        print("No existing vectors found. Creating test data...")
        # Test data (multi-media content)
        test_texts = [
            "AI and machine learning are transforming healthcare diagnostics",
            "Climate change impacts global agriculture and food security", 
            "Economic inflation affects housing market affordability",
            "Renewable energy sources include solar and wind power",
            "Social media platforms influence political discourse"
        ]
        
        test_metadata = [
            {"title": "AI in Healthcare", "content_type": "article", "source": "tech_journal"},
            {"title": "Climate & Agriculture", "content_type": "report", "source": "environmental_org"},
            {"title": "Economic Housing", "content_type": "news", "source": "financial_news"},
            {"title": "Renewable Energy", "content_type": "research", "source": "energy_lab"},
            {"title": "Social Media Politics", "content_type": "analysis", "source": "policy_institute"}
        ]
        
        # Test upsert functionality
        print("\n=== Testing Upsert ===")
        vector_db.upsert_documents(test_texts, test_metadata)
    else:
        print("Successfully loaded industry signals vectors!")
    
    # Test search with industry-relevant questions
    print("\n=== Testing Industry Signals Search ===")
    
    test_queries = [
        "How does smart manufacturing improve industrial efficiency?",
        "What are the latest trends in AI and automation?",
        "How is renewable energy transforming the power sector?",
        "What are the economic impacts of climate change?",
        "How do digital technologies affect supply chain management?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vector_db.search(query, top_k=3, cosine_threshold=0.2, adaptive=True)
        
        print(f"Results: {len(results)} found")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Similarity: {r['similarity']:.3f}")
            print(f"   Title: {r['metadata'].get('title', 'Unknown')}")
            print(f"   Source: {r['metadata'].get('source_name', 'Unknown')}")
            print(f"   Content: {r['document'][:100]}...")
    
    # Test clustering on industry signals
    print("\n=== Testing Industry Signals Clustering ===")
    stats = vector_db.get_stats()
    doc_count = stats.get('total_documents', 0)
    
    if doc_count > 0:
        # Determine optimal number of clusters (max 8, min 2)
        n_clusters = min(8, max(2, doc_count // 5))
        print(f"Creating {n_clusters} clusters from {doc_count} documents...")
        
        clusters = vector_db.cluster_documents(n_clusters=n_clusters)
        
        for cluster_id, docs in clusters.items():
            print(f"\nCluster {cluster_id + 1} ({len(docs)} documents):")
            for i, doc in enumerate(docs[:3], 1):  # Show first 3 docs per cluster
                print(f"  {i}. {doc}")
            if len(docs) > 3:
                print(f"  ... and {len(docs) - 3} more documents")
    
    # Test statistics
    print(f"\n=== Database Statistics ===")
    stats = vector_db.get_stats()
    for key, value in stats.items():
        print(f"📈 {key}: {value}")
    
    # Test save functionality
    print(f"\n=== Saving Updated Vectors ===")
    save_success = vector_db.save("industry_signals_vectors_updated")
    if save_success:
        print("Successfully saved updated vectors!")
    else:
        print("Failed to save vectors")
