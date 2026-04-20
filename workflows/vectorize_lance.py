#!/usr/bin/env python3
"""
LanceDB Vectorization Workflow (Mac-only).

Reads approved content and enriched signals from DuckDB, creates embeddings,
and stores them in LanceDB for testing with AnythingLLM and other tools.

This runs INDEPENDENTLY of the FAISS vectorization pipeline. It reads the
same source data (DuckDB) but writes to a separate LanceDB directory.

Workflow:
    Jetson ──sync──► DuckDB (Mac) ──this script──► LanceDB (Mac)
                                   ──process.py──► FAISS   (production mirror)

Usage:
    # Full vectorization (corpus + signals)
    python workflows/vectorize_lance.py

    # Corpus only
    python workflows/vectorize_lance.py --corpus-only

    # Signals only
    python workflows/vectorize_lance.py --signals-only

    # Rebuild from scratch (drops existing tables)
    python workflows/vectorize_lance.py --rebuild

    # Limit batch size
    python workflows/vectorize_lance.py --batch 50

    # Show what AnythingLLM should point to
    python workflows/vectorize_lance.py --anythingllm-path

Notebook usage:
    from workflows.vectorize_lance import LanceVectorizer
    lv = LanceVectorizer()
    lv.vectorize_corpus(rebuild=True)
    lv.vectorize_signals()
    lv.summary()
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config

if not config.is_mac:
    print("ERROR: LanceDB vectorization is Mac-only.")
    sys.exit(1)


class LanceVectorizer:
    """Reads DuckDB, writes LanceDB. Importable from notebooks."""

    CORPUS_TABLE = "corpus_vectors"
    SIGNAL_TABLE = "signal_vectors"

    def __init__(self, db_path: str = None, lance_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        from db_relational import relationalDB
        from db_vector_lance import LanceVectorDB

        self.db = relationalDB(db_path or config.DB_PATH_ANALYTICS)
        self.model_name = model_name
        
        # Trust remote code for Nomic models (required for proper loading)
        trust_remote = model_name.startswith('nomic')
        
        self.lance = LanceVectorDB(
            vector_dir=lance_path or config.LANCE_VECTOR_PATH,
            model_name=model_name,
            use_builtin_embeddings=True,
            trust_remote_code=trust_remote,
        )
        print(f"Using embedding model: {model_name}")

    # ── Corpus vectorization ──────────────────────────────────────────

    def _get_approved_content(self, limit: int = 5000):
        """Get all approved content with transcripts/segments from DuckDB."""
        rows = self.db.query("""
            SELECT id, title, transcript, segments, metadata_json,
                   source_type, source_name, content_type, created_at
            FROM content
            WHERE screening_status = 'approved'
              AND extraction_status IN ('completed', 'NA')
              AND (do_not_vectorize = FALSE OR do_not_vectorize IS NULL)
              AND transcript IS NOT NULL
              AND transcript != ''
            ORDER BY created_at DESC
            LIMIT ?
        """, [limit])
        return rows

    def _get_lance_content_ids(self) -> set:
        """Get content_ids already in the Lance corpus table."""
        tbl = self.lance._get_table(self.CORPUS_TABLE)
        if tbl is None or tbl.count_rows() == 0:
            return set()
        try:
            df = tbl.to_pandas()
            return set(df['content_id'].dropna().unique())
        except Exception:
            return set()

    def vectorize_corpus(self, limit: int = 5000, batch_size: int = 64,
                         rebuild: bool = False):
        """
        Vectorize approved content into LanceDB corpus_vectors table.

        Args:
            limit: Max content items to process.
            batch_size: Embedding batch size.
            rebuild: If True, drop and rebuild the table from scratch.
        """
        print("\n── Corpus Vectorization ──")

        if rebuild:
            if self.CORPUS_TABLE in self.lance.db.table_names():
                self.lance.db.drop_table(self.CORPUS_TABLE)
                print(f"Dropped existing '{self.CORPUS_TABLE}' table")

        content = self._get_approved_content(limit)
        if not content:
            print("No approved content found in DuckDB.")
            return 0

        # Filter out already-vectorized content (incremental)
        existing_ids = set() if rebuild else self._get_lance_content_ids()
        new_content = [c for c in content if str(c['id']) not in existing_ids]

        if not new_content:
            print(f"All {len(content)} approved items already in LanceDB.")
            return 0

        print(f"Found {len(new_content)} new items to vectorize "
              f"({len(existing_ids)} already in Lance, {len(content)} total approved)")

        # Build texts + metadata from segments (or full transcript as fallback)
        all_texts = []
        all_metadata = []

        for item in new_content:
            segments = []
            if item.get('segments'):
                try:
                    segments = json.loads(item['segments']) if isinstance(item['segments'], str) else item['segments']
                except (json.JSONDecodeError, TypeError):
                    segments = []

            meta_base = {}
            if item.get('metadata_json'):
                try:
                    meta_base = json.loads(item['metadata_json']) if isinstance(item['metadata_json'], str) else item['metadata_json']
                except (json.JSONDecodeError, TypeError):
                    meta_base = {}

            if segments:
                for i, seg in enumerate(segments):
                    # Handle both dict segments and string segments
                    if isinstance(seg, dict):
                        text = seg.get('text', '').strip()
                        start_char = seg.get('start_char', 0)
                        end_char = seg.get('end_char', 0)
                    elif isinstance(seg, str):
                        text = seg.strip()
                        start_char = 0
                        end_char = len(seg)
                    else:
                        continue
                    
                    if not text or len(text) < 50:
                        continue
                    meta = {
                        'content_id': str(item['id']),
                        'title': item.get('title', ''),
                        'source_name': item.get('source_name', ''),
                        'source_type': item.get('source_type', ''),
                        'content_type': item.get('content_type', ''),
                        'segment_index': i,
                        'segment_start': start_char,
                        'segment_end': end_char,
                        **meta_base,
                    }
                    all_texts.append(text)
                    all_metadata.append(meta)
            else:
                # No segments — use full transcript
                text = (item.get('transcript') or '').strip()
                if not text or len(text) < 50:
                    continue
                meta = {
                    'content_id': str(item['id']),
                    'title': item.get('title', ''),
                    'source_name': item.get('source_name', ''),
                    'source_type': item.get('source_type', ''),
                    'content_type': item.get('content_type', ''),
                    **meta_base,
                }
                all_texts.append(text)
                all_metadata.append(meta)

        if not all_texts:
            print("No text segments to vectorize.")
            return 0

        print(f"Vectorizing {len(all_texts)} text chunks from {len(new_content)} items...")

        # Batch the upsert to avoid OOM on large corpora
        total = 0
        for start in range(0, len(all_texts), batch_size):
            end = min(start + batch_size, len(all_texts))
            batch_texts = all_texts[start:end]
            batch_meta = all_metadata[start:end]
            self.lance.upsert_documents(batch_texts, batch_meta,
                                        table_name=self.CORPUS_TABLE)
            total += len(batch_texts)
            print(f"  Batch {start // batch_size + 1}: "
                  f"{total}/{len(all_texts)} chunks vectorized")

        print(f"Corpus vectorization complete: {total} chunks from {len(new_content)} items")
        return total

    # ── Signal vectorization ──────────────────────────────────────────

    def _get_enriched_signals(self, limit: int = 2000):
        """Get enriched signals from DuckDB (falls back to description if no enriched_text)."""
        rows = self.db.query("""
            SELECT id, entity, signal_type, industry, enriched_text,
                   confidence, source_content_id, description,
                   COALESCE(NULLIF(enriched_text, ''), description) AS text_for_embedding
            FROM signals
            WHERE COALESCE(NULLIF(enriched_text, ''), description) IS NOT NULL
              AND COALESCE(NULLIF(enriched_text, ''), description) != ''
            ORDER BY id
            LIMIT ?
        """, [limit])
        return rows

    def _get_lance_signal_ids(self) -> set:
        """Get signal IDs already in the Lance signal table."""
        tbl = self.lance._get_table(self.SIGNAL_TABLE)
        if tbl is None or tbl.count_rows() == 0:
            return set()
        try:
            df = tbl.to_pandas()
            return set(df['content_id'].dropna().unique())
        except Exception:
            return set()

    def vectorize_signals(self, limit: int = 2000, batch_size: int = 64,
                          rebuild: bool = False):
        """
        Vectorize enriched signals into LanceDB signal_vectors table.

        Args:
            limit: Max signals to process.
            batch_size: Embedding batch size.
            rebuild: If True, drop and rebuild the table from scratch.
        """
        print("\n── Signal Vectorization ──")

        if rebuild:
            if self.SIGNAL_TABLE in self.lance.db.table_names():
                self.lance.db.drop_table(self.SIGNAL_TABLE)
                print(f"Dropped existing '{self.SIGNAL_TABLE}' table")

        signals = self._get_enriched_signals(limit)
        if not signals:
            print("No enriched signals found in DuckDB.")
            return 0

        existing_ids = set() if rebuild else self._get_lance_signal_ids()
        new_signals = [s for s in signals if str(s['id']) not in existing_ids]

        if not new_signals:
            print(f"All {len(signals)} signals already in LanceDB.")
            return 0

        print(f"Found {len(new_signals)} new signals to vectorize "
              f"({len(existing_ids)} already in Lance)")

        texts = []
        metadata = []
        for sig in new_signals:
            texts.append(sig['text_for_embedding'])
            metadata.append({
                'content_id': str(sig['id']),
                'title': f"{sig['signal_type']}: {sig['entity']}",
                'source_name': sig.get('industry', ''),
                'content_type': 'signal',
                'signal_type': sig['signal_type'],
                'entity': sig['entity'],
                'confidence': sig.get('confidence', 0),
                'parent_content_id': str(sig.get('source_content_id', '')),
                'description': sig.get('description', ''),
            })

        total = 0
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            self.lance.upsert_documents(texts[start:end], metadata[start:end],
                                        table_name=self.SIGNAL_TABLE)
            total += end - start
            print(f"  Batch {start // batch_size + 1}: "
                  f"{total}/{len(texts)} signals vectorized")

        print(f"Signal vectorization complete: {total} signals")
        return total

    # ── Summary / info ────────────────────────────────────────────────

    def summary(self):
        """Print status of both LanceDB tables."""
        print("\n── LanceDB Status ──")
        for tbl_name in [self.CORPUS_TABLE, self.SIGNAL_TABLE]:
            stats = self.lance.get_stats(table_name=tbl_name)
            print(f"  {tbl_name}: {stats['total_vectors']} rows")
        print(f"  Path: {os.path.abspath(self.lance.vector_dir)}")
        print(f"  Tables: {self.lance.list_tables()}")

    def anythingllm_path(self):
        """Print the path for AnythingLLM configuration."""
        return self.lance.export_for_anythingllm()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vectorize DuckDB content into LanceDB (Mac-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--corpus-only', action='store_true',
                        help='Only vectorize corpus (skip signals)')
    parser.add_argument('--signals-only', action='store_true',
                        help='Only vectorize signals (skip corpus)')
    parser.add_argument('--rebuild', action='store_true',
                        help='Drop and rebuild tables from scratch')
    parser.add_argument('--batch', type=int, default=64,
                        help='Embedding batch size (default: 64)')
    parser.add_argument('--limit', type=int, default=5000,
                        help='Max items to process (default: 5000)')
    parser.add_argument('--anythingllm-path', action='store_true',
                        help='Just print the AnythingLLM storage path and exit')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Embedding model name (default: all-MiniLM-L6-v2, try nomic-ai/nomic-embed-text-v1.5)')
    args = parser.parse_args()

    lv = LanceVectorizer(model_name=args.model)

    if args.anythingllm_path:
        lv.anythingllm_path()
        return

    corpus_count = 0
    signal_count = 0

    if not args.signals_only:
        corpus_count = lv.vectorize_corpus(
            limit=args.limit, batch_size=args.batch, rebuild=args.rebuild,
        )

    if not args.corpus_only:
        signal_count = lv.vectorize_signals(
            limit=args.limit, batch_size=args.batch, rebuild=args.rebuild,
        )

    lv.summary()

    print(f"\n{'=' * 50}")
    print(f"LANCE VECTORIZATION COMPLETE")
    print(f"  Corpus chunks: {corpus_count}")
    print(f"  Signals:       {signal_count}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
