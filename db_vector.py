import os
import faiss
import pickle
import numpy as np
from typing import Optional, List, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from dotenv import load_dotenv


class VectorDB:
    """FAISS-based vector database for semantic search."""

    def __init__(
        self,
        vector_dir: str = "vectors",
        embedding_dim: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
        use_builtin_embeddings: bool = True,
    ):
        self.vector_dir = vector_dir
        self.embedding_dim = embedding_dim
        self.use_builtin_embeddings = use_builtin_embeddings

        self.index = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []

        self._ensure_directory_exists()

        if self.use_builtin_embeddings:
            self.model, self.device = self._load_embedding_model(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        else:
            self.model = None
            self.device = None

        print(f"Vector database initialized at: {self.vector_dir}")

    def _load_embedding_model(self, model_name: str):
        """Load embedding model on the best available device.

        Device priority:
        1. VECTOR_DEVICE env var, if set
        2. CUDA, if available
        3. Apple MPS, if available
        4. CPU fallback
        """
        import platform
        import torch

        forced_device = os.getenv("VECTOR_DEVICE", "").strip().lower()
        allowed_devices = {"cpu", "cuda", "mps"}

        if forced_device and forced_device not in allowed_devices:
            raise ValueError(
                f"Invalid VECTOR_DEVICE='{forced_device}'. Use one of: cpu, cuda, mps"
            )

        if forced_device:
            device = forced_device
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"

        # Safety checks so a forced device fails clearly instead of strangely.
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VECTOR_DEVICE=cuda was requested, but CUDA is not available.")
        if device == "mps":
            mps_ok = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if not mps_ok:
                raise RuntimeError("VECTOR_DEVICE=mps was requested, but MPS is not available.")

        print(f"Loading embedding model: {model_name} on {device}...")
        try:
            model = SentenceTransformer(model_name, device=device)
        except RuntimeError as e:
            if device != "cpu":
                print(f"Failed to load model on {device}: {e}")
                print("Falling back to CPU...")
                device = "cpu"
                model = SentenceTransformer(model_name, device="cpu")
            else:
                raise

        try:
            if device == "cuda":
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif device == "mps":
                print("Using Apple Metal GPU acceleration")
            else:
                print("Using CPU for embeddings")
        except Exception:
            pass

        return model, device

    def _ensure_directory_exists(self):
        """Create vector directory if it doesn't exist, with fallback."""
        try:
            if not os.path.exists(self.vector_dir):
                os.makedirs(self.vector_dir, exist_ok=True)
                print(f"Created vector directory: {self.vector_dir}")

            test_file = os.path.join(self.vector_dir, "test.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)

        except (PermissionError, OSError):
            fallback_dir = "vectors_fallback"
            if not os.path.exists(fallback_dir):
                os.makedirs(fallback_dir, exist_ok=True)
            print(f"Cannot write to {self.vector_dir}, using fallback: {fallback_dir}")
            self.vector_dir = fallback_dir

    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Convert texts to normalized embeddings using built-in model."""
        if not self.use_builtin_embeddings or self.model is None:
            raise ValueError(
                "Built-in embeddings not enabled. Use add_vectors() with pre-computed embeddings."
            )

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def upsert_documents(
        self,
        texts: List[str],
        metadata: List[Dict],
        vectors: Optional[np.ndarray] = None,
    ):
        """Build new index or add to existing one."""
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have same length")

        if self.index is None or self.index.ntotal == 0:
            print("No index found — building new FAISS index.")
            self.build_index(texts, metadata, vectors)
        else:
            print("Existing index found — adding new documents.")
            self.add_documents(texts, metadata, vectors)

    def build_index(
        self,
        texts: List[str],
        metadata: List[Dict],
        vectors: Optional[np.ndarray] = None,
    ):
        """Build FAISS index from scratch."""
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have same length")

        if not texts:
            print("No texts provided. Skipping index build.")
            return

        if vectors is None:
            if not self.use_builtin_embeddings:
                raise ValueError("Must provide vectors when not using built-in embeddings")
            vectors = self.create_embeddings(texts)
        else:
            vectors = np.asarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors)

        print(f"Building FAISS index for {len(texts)} documents...")

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(vectors)
        self.documents = list(texts)
        self.metadata = list(metadata)

        print(f"Index built with {self.index.ntotal} vectors")

    def add_documents(
        self,
        texts: List[str],
        metadata: List[Dict],
        vectors: Optional[np.ndarray] = None,
    ):
        """Add documents to an existing FAISS index."""
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have same length")

        if not texts:
            return 0

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        if vectors is None:
            if not self.use_builtin_embeddings:
                raise ValueError("Must provide vectors when not using built-in embeddings")
            embeddings_array = self.create_embeddings(texts)
        else:
            embeddings_array = np.asarray(vectors, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)

        self.index.add(embeddings_array)
        self.documents.extend(texts)
        self.metadata.extend(metadata)

        print(f"Added {len(texts)} documents. Total vectors: {self.index.ntotal}")
        return len(texts)

    def add_signals(self, signals: List[Dict]):
        """Add signal embeddings to vector store."""
        if not signals:
            return 0

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.metadata = []

        texts = []
        metadata = []

        for signal in signals:
            signal_text = f"{signal['signal_type']}: {signal['entity']} - {signal['description']}"
            texts.append(signal_text)
            metadata.append(signal)

        embeddings_array = self.create_embeddings(texts, show_progress=True)
        self.index.add(embeddings_array)

        self.documents.extend(texts)
        self.metadata.extend(metadata)

        print(f"Added {len(signals)} signals. Total vectors: {self.index.ntotal}")
        return len(signals)

    def add_vectors(self, vectors: np.ndarray, documents: List[str], metadata: List[Dict]):
        """Backward-compatible wrapper."""
        return self.add_documents(documents, metadata, vectors)

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        cosine_threshold: float = 0.3,
        adaptive: bool = True,
    ) -> List[Dict]:
        """Search for semantically similar documents."""
        if self.index is None or self.index.ntotal == 0:
            print("No vectors in index to search")
            return []

        if isinstance(query, str):
            if not self.use_builtin_embeddings or self.model is None:
                raise ValueError(
                    "Cannot search with text when not using built-in embeddings. Provide embedding vector instead."
                )
            print(f"[DEBUG] Encoding query: '{query[:50]}...'")
            query_vector = self.model.encode([query], convert_to_numpy=True)
        else:
            query_vector = query.reshape(1, -1)

        query_vector = np.asarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1:
                results.append(
                    {
                        "document": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "similarity": float(sim),
                        "index": int(idx),
                    }
                )

        filtered = [r for r in results if r["similarity"] >= cosine_threshold]

        if not filtered and adaptive:
            relaxed = cosine_threshold * 0.75
            print(f"[DEBUG] No results above {cosine_threshold:.2f}, relaxing to {relaxed:.2f}")
            filtered = [r for r in results if r["similarity"] >= relaxed]

        if not filtered:
            print(f"[DEBUG] No results passed threshold {cosine_threshold}")
            return []

        return filtered

    def save(self, filename: str = "vectors"):
        """Save index and metadata to disk."""
        if self.index is None:
            print("No index to save")
            return False

        index_path = os.path.join(self.vector_dir, f"{filename}.faiss")
        data_path = os.path.join(self.vector_dir, f"{filename}.pkl")

        faiss.write_index(self.index, index_path)

        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "embedding_dim": self.embedding_dim,
                },
                f,
            )

        print(f"Saved vector database to {index_path} and {data_path}")
        return True

    def load(self, filename: str = "vectors"):
        """Load index and metadata from disk."""
        index_path = os.path.join(self.vector_dir, f"{filename}.faiss")
        data_path = os.path.join(self.vector_dir, f"{filename}.pkl")

        if not os.path.exists(index_path) or not os.path.exists(data_path):
            print(f"Vector files not found: {filename}")
            return False

        self.index = faiss.read_index(index_path)

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.embedding_dim = data["embedding_dim"]

        print(f"Loaded {self.index.ntotal} vectors from {filename}")
        return True

    def get_stats(self):
        """Get database statistics."""
        if self.index is None:
            return {
                "total_vectors": 0,
                "dimension": self.embedding_dim,
                "device": self.device,
            }

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.embedding_dim,
            "total_documents": len(self.documents),
            "vector_dir": self.vector_dir,
            "index_type": type(self.index).__name__,
            "has_builtin_embeddings": self.use_builtin_embeddings,
            "device": self.device,
        }

    def cluster_documents(self, n_clusters: int = 4) -> Dict[int, List[str]]:
        """Group all indexed documents into N semantic clusters."""
        if self.index is None or not self.documents:
            raise ValueError("Index not built or empty")

        print(f"[DEBUG] Clustering {len(self.documents)} embeddings into {n_clusters} themes...")

        xb = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            xb[i] = self.index.reconstruct(i)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(xb)

        clusters = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(kmeans.labels_):
            title = (
                self.metadata[i].get("title")
                or self.metadata[i].get("source_name")
                or (
                    self.documents[i][:50] + "..."
                    if len(self.documents[i]) > 50
                    else self.documents[i]
                )
            )
            clusters[label].append(title)

        return clusters


if __name__ == "__main__":
    load_dotenv()

    VEC_DB_PATH = os.getenv("VEC_DB_PATH", "Vectors/corpus_vectors/")

    print("Testing Enhanced Vector Database...")

    vector_db = VectorDB(VEC_DB_PATH, use_builtin_embeddings=True)

    print("\n=== Loading Corpus Vectors ===")
    load_success = vector_db.load("corpus_vectors")

    if not load_success:
        print("No existing vectors found. Creating test data...")
        test_texts = [
            "AI and machine learning are transforming healthcare diagnostics",
            "Climate change impacts global agriculture and food security",
            "Economic inflation affects housing market affordability",
            "Renewable energy sources include solar and wind power",
            "Social media platforms influence political discourse",
        ]

        test_metadata = [
            {"title": "AI in Healthcare", "content_type": "article", "source": "tech_journal"},
            {"title": "Climate & Agriculture", "content_type": "report", "source": "environmental_org"},
            {"title": "Economic Housing", "content_type": "news", "source": "financial_news"},
            {"title": "Renewable Energy", "content_type": "research", "source": "energy_lab"},
            {"title": "Social Media Politics", "content_type": "analysis", "source": "policy_institute"},
        ]

        print("\n=== Testing Upsert ===")
        vector_db.upsert_documents(test_texts, test_metadata)
    else:
        print("Successfully loaded industry signals vectors!")

    print("\n=== Testing Industry Signals Search ===")

    test_queries = [
        "How does smart manufacturing improve industrial efficiency?",
        "What are the latest trends in AI and automation?",
        "How is renewable energy transforming the power sector?",
        "What are the economic impacts of climate change?",
        "How do digital technologies affect supply chain management?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vector_db.search(query, top_k=3, cosine_threshold=0.2, adaptive=True)

        print(f"Results: {len(results)} found")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Similarity: {r['similarity']:.3f}")
            print(f"   Title: {r['metadata'].get('title', 'Unknown')}")
            print(f"   Source: {r['metadata'].get('source_name', r['metadata'].get('source', 'Unknown'))}")
            print(f"   Content: {r['document'][:100]}...")

    print("\n=== Testing Industry Signals Clustering ===")
    stats = vector_db.get_stats()
    doc_count = stats.get("total_documents", 0)

    if doc_count > 0:
        n_clusters = min(8, max(2, doc_count // 5))
        print(f"Creating {n_clusters} clusters from {doc_count} documents...")

        clusters = vector_db.cluster_documents(n_clusters=n_clusters)

        for cluster_id, docs in clusters.items():
            print(f"\nCluster {cluster_id + 1} ({len(docs)} documents):")
            for i, doc in enumerate(docs[:3], 1):
                print(f"  {i}. {doc}")
            if len(docs) > 3:
                print(f"  ... and {len(docs) - 3} more documents")

    print("\n=== Database Statistics ===")
    stats = vector_db.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Saving Updated Vectors ===")
    save_success = vector_db.save("corpus_vectors")
    if save_success:
        print("Successfully saved updated vectors!")
    else:
        print("Failed to save vectors")