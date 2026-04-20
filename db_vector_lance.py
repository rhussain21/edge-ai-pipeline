"""
LanceDB vector database — Mac-only testing/development companion to FAISS.

Mirrors the public API of VectorDB (db_vector.py) so callers can swap
backends without code changes.  LanceDB stores vectors, documents, and
metadata in a single Lance-format directory — no separate .pkl sidecar.

Usage:
    from device_config import config
    from db_vector_lance import LanceVectorDB

    vdb = LanceVectorDB(config.LANCE_VECTOR_PATH)
    vdb.upsert_documents(texts, metadata_list)
    results = vdb.search("edge AI inference", top_k=5)

AnythingLLM integration:
    Point AnythingLLM's LanceDB path at the same LANCE_VECTOR_PATH directory.
    Both this class and AnythingLLM can read/write the same Lance tables.

NOTE: This module is intentionally NOT imported on Jetson.  The lancedb
package is only installed on Mac dev machines.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _check_mac_only():
    """Guard: refuse to initialize on non-Mac platforms."""
    import platform
    if platform.system() != "Darwin":
        raise RuntimeError(
            "LanceVectorDB is Mac-only. Use VectorDB (FAISS) on Jetson/Linux."
        )


class LanceVectorDB:
    """LanceDB-backed vector database for semantic search (Mac-only)."""

    # Default table names — match FAISS file naming convention
    CORPUS_TABLE = "corpus_vectors"
    SIGNAL_TABLE = "signal_vectors"

    def __init__(
        self,
        vector_dir: str = "Vectors/lance",
        embedding_dim: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
        use_builtin_embeddings: bool = True,
        trust_remote_code: bool = False,
    ):
        _check_mac_only()

        try:
            import lancedb  # noqa: F401
        except ImportError:
            raise ImportError(
                "lancedb is not installed. Install with: pip install lancedb"
            )

        self.vector_dir = vector_dir
        self.embedding_dim = embedding_dim
        self.use_builtin_embeddings = use_builtin_embeddings
        self._table_name = self.CORPUS_TABLE  # active table
        self._trust_remote_code = trust_remote_code

        os.makedirs(self.vector_dir, exist_ok=True)

        import lancedb as ldb
        self.db = ldb.connect(self.vector_dir)

        if self.use_builtin_embeddings:
            self.model, self.device = self._load_embedding_model(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        else:
            self.model = None
            self.device = None

        print(f"LanceDB initialized at: {self.vector_dir}")

    # ── Embedding model (shared with VectorDB) ───────────────────────

    def _load_embedding_model(self, model_name: str):
        """Load embedding model on the best available device."""
        import platform
        import torch
        from sentence_transformers import SentenceTransformer

        forced_device = os.getenv("VECTOR_DEVICE", "").strip().lower()
        if forced_device and forced_device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Invalid VECTOR_DEVICE='{forced_device}'")

        if forced_device:
            device = forced_device
        else:
            device = "cpu"
            if (
                platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"

        print(f"Loading embedding model: {model_name} on {device}...")
        try:
            model = SentenceTransformer(
                model_name, 
                device=device,
                trust_remote_code=self._trust_remote_code,
            )
        except RuntimeError:
            print(f"Failed on {device}, falling back to CPU...")
            device = "cpu"
            model = SentenceTransformer(
                model_name, 
                device="cpu",
                trust_remote_code=self._trust_remote_code,
            )

        return model, device

    # ── Embeddings ────────────────────────────────────────────────────

    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Convert texts to normalized embeddings."""
        if not self.use_builtin_embeddings or self.model is None:
            raise ValueError("Built-in embeddings not enabled.")

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(
            texts, show_progress_bar=show_progress, convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        return embeddings

    # ── Table helpers ─────────────────────────────────────────────────

    def _get_table(self, name: str = None):
        """Return an existing LanceDB table or None."""
        tbl_name = name or self._table_name
        if tbl_name in self.db.table_names():
            return self.db.open_table(tbl_name)
        return None

    def _create_or_open_table(self, name: str, records: list):
        """Create table from records, or append to existing."""
        if name in self.db.table_names():
            tbl = self.db.open_table(name)
            tbl.add(records)
            return tbl
        return self.db.create_table(name, records)

    @staticmethod
    def _build_records(
        texts: List[str],
        metadata: List[Dict],
        vectors: np.ndarray,
    ) -> list:
        """Build list-of-dicts for LanceDB insertion."""
        records = []
        now = datetime.now(timezone.utc).isoformat()
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            rec = {
                "vector": vectors[i].tolist(),
                "text": text,
                "title": meta.get("title", ""),
                "source_name": meta.get("source_name", meta.get("source", "")),
                "content_type": meta.get("content_type", ""),
                "content_id": meta.get("content_id", ""),
                "created_at": now,
                # Store full metadata as JSON string for lossless round-trip
                "metadata_json": _serialize_meta(meta),
            }
            records.append(rec)
        return records

    # ── Public API (mirrors VectorDB) ─────────────────────────────────

    def upsert_documents(
        self,
        texts: List[str],
        metadata: List[Dict],
        vectors: Optional[np.ndarray] = None,
        table_name: str = None,
    ):
        """Add documents to LanceDB (create table if needed)."""
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have same length")
        if not texts:
            return

        tbl_name = table_name or self._table_name

        if vectors is None:
            vectors = self.create_embeddings(texts)
        else:
            vectors = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms

        records = self._build_records(texts, metadata, vectors)
        tbl = self._create_or_open_table(tbl_name, records)
        print(f"Upserted {len(texts)} documents into '{tbl_name}'. Total rows: {tbl.count_rows()}")

    def build_index(
        self,
        texts: List[str],
        metadata: List[Dict],
        vectors: Optional[np.ndarray] = None,
        table_name: str = None,
    ):
        """Build table from scratch (drops existing)."""
        tbl_name = table_name or self._table_name

        # Drop existing table
        if tbl_name in self.db.table_names():
            self.db.drop_table(tbl_name)
            print(f"Dropped existing table '{tbl_name}'")

        self.upsert_documents(texts, metadata, vectors, table_name=tbl_name)

    def add_documents(
        self,
        texts: List[str],
        metadata: List[Dict],
        vectors: Optional[np.ndarray] = None,
        table_name: str = None,
    ):
        """Add documents to existing table."""
        return self.upsert_documents(texts, metadata, vectors, table_name=table_name)

    def add_signals(self, signals: List[Dict], table_name: str = None):
        """Add signal embeddings to LanceDB."""
        if not signals:
            return 0

        tbl_name = table_name or self.SIGNAL_TABLE
        texts = []
        metadata = []
        for signal in signals:
            signal_text = f"{signal['signal_type']}: {signal['entity']} - {signal['description']}"
            texts.append(signal_text)
            metadata.append(signal)

        self.upsert_documents(texts, metadata, table_name=tbl_name)
        return len(signals)

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        cosine_threshold: float = 0.3,
        adaptive: bool = True,
        table_name: str = None,
    ) -> List[Dict]:
        """Search for semantically similar documents."""
        import json

        tbl_name = table_name or self._table_name
        tbl = self._get_table(tbl_name)
        if tbl is None or tbl.count_rows() == 0:
            print("No vectors in LanceDB to search")
            return []

        if isinstance(query, str):
            if not self.use_builtin_embeddings or self.model is None:
                raise ValueError("Cannot search with text without built-in embeddings.")
            query_vector = self.create_embeddings([query], show_progress=False)[0]
        else:
            query_vector = np.asarray(query, dtype=np.float32).flatten()
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        # LanceDB search returns results sorted by distance (L2 by default).
        # We use cosine distance; LanceDB metric="cosine" returns distance in [0, 2].
        raw = (
            tbl.search(query_vector.tolist())
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

        results = []
        for row in raw:
            # LanceDB cosine _distance: 0 = identical, 2 = opposite.
            # Convert to similarity: sim = 1 - distance
            dist = row.get("_distance", 1.0)
            similarity = 1.0 - dist

            meta = _deserialize_meta(row.get("metadata_json", "{}"))
            results.append({
                "document": row.get("text", ""),
                "metadata": meta,
                "similarity": float(similarity),
            })

        filtered = [r for r in results if r["similarity"] >= cosine_threshold]

        if not filtered and adaptive:
            relaxed = cosine_threshold * 0.75
            print(f"[DEBUG] No results above {cosine_threshold:.2f}, relaxing to {relaxed:.2f}")
            filtered = [r for r in results if r["similarity"] >= relaxed]

        if not filtered:
            print(f"[DEBUG] No results passed threshold {cosine_threshold}")
            return []

        return filtered

    # ── Persistence ───────────────────────────────────────────────────
    # LanceDB is inherently persistent — data is written to disk on every
    # insert.  These methods exist for API compatibility with VectorDB.

    def save(self, filename: str = "corpus_vectors") -> bool:
        """No-op for LanceDB (already persistent). Returns True if table exists."""
        tbl = self._get_table(filename)
        if tbl is not None:
            print(f"LanceDB table '{filename}' already persisted ({tbl.count_rows()} rows)")
            return True
        print(f"LanceDB table '{filename}' does not exist — nothing to save")
        return False

    def load(self, filename: str = "corpus_vectors") -> bool:
        """'Load' a LanceDB table (just verifies it exists)."""
        self._table_name = filename
        tbl = self._get_table(filename)
        if tbl is not None:
            print(f"Loaded LanceDB table '{filename}' with {tbl.count_rows()} rows")
            return True
        print(f"LanceDB table '{filename}' not found")
        return False

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self, table_name: str = None) -> Dict:
        """Get database statistics."""
        tbl_name = table_name or self._table_name
        tbl = self._get_table(tbl_name)
        row_count = tbl.count_rows() if tbl else 0

        return {
            "backend": "lancedb",
            "total_vectors": row_count,
            "dimension": self.embedding_dim,
            "total_documents": row_count,
            "vector_dir": self.vector_dir,
            "table_name": tbl_name,
            "tables": self.db.table_names(),
            "has_builtin_embeddings": self.use_builtin_embeddings,
            "device": self.device,
        }

    def list_tables(self) -> list:
        """List all LanceDB tables."""
        return self.db.table_names()

    # ── AnythingLLM helpers ───────────────────────────────────────────

    def export_for_anythingllm(self, table_name: str = None) -> str:
        """
        Return the filesystem path that AnythingLLM should point to.

        AnythingLLM's LanceDB connector reads the same directory format
        that this class writes to.  Just point AnythingLLM's 'storage path'
        at the returned directory.
        """
        path = os.path.abspath(self.vector_dir)
        tbl_name = table_name or self._table_name
        tbl = self._get_table(tbl_name)
        if tbl:
            print(f"AnythingLLM can connect to: {path}")
            print(f"  Table: {tbl_name} ({tbl.count_rows()} rows)")
        else:
            print(f"No table '{tbl_name}' yet — create vectors first.")
        return path


# ── JSON helpers for metadata round-trip ──────────────────────────────

def _serialize_meta(meta: Dict) -> str:
    import json
    try:
        return json.dumps(meta, default=str)
    except (TypeError, ValueError):
        return "{}"


def _deserialize_meta(json_str: str) -> Dict:
    import json
    try:
        return json.loads(json_str) if json_str else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ── CLI test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from device_config import config

    if not config.is_mac:
        print("LanceVectorDB is Mac-only. Exiting.")
        raise SystemExit(1)

    lance_path = config.LANCE_VECTOR_PATH
    print(f"Testing LanceVectorDB at: {lance_path}")

    vdb = LanceVectorDB(lance_path, use_builtin_embeddings=True)

    print("\n=== Loading or Building Corpus ===")
    loaded = vdb.load("corpus_vectors")

    if not loaded:
        print("No existing Lance table. Creating test data...")
        test_texts = [
            "AI and machine learning are transforming healthcare diagnostics",
            "Climate change impacts global agriculture and food security",
            "Economic inflation affects housing market affordability",
            "Renewable energy sources include solar and wind power",
            "Edge AI inference on embedded devices reduces latency",
        ]
        test_metadata = [
            {"title": "AI in Healthcare", "content_type": "article", "source": "tech_journal"},
            {"title": "Climate & Agriculture", "content_type": "report", "source": "environmental_org"},
            {"title": "Economic Housing", "content_type": "news", "source": "financial_news"},
            {"title": "Renewable Energy", "content_type": "research", "source": "energy_lab"},
            {"title": "Edge AI Inference", "content_type": "research", "source": "ieee"},
        ]
        vdb.build_index(test_texts, test_metadata)

    print("\n=== Search Test ===")
    for q in [
        "How does smart manufacturing improve industrial efficiency?",
        "Edge computing for real-time AI inference",
    ]:
        print(f"\nQuery: '{q}'")
        results = vdb.search(q, top_k=3, cosine_threshold=0.2)
        print(f"Results: {len(results)}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['similarity']:.3f}] {r['metadata'].get('title', '?')}")

    print("\n=== Stats ===")
    for k, v in vdb.get_stats().items():
        print(f"  {k}: {v}")

    print("\n=== AnythingLLM Path ===")
    vdb.export_for_anythingllm()
