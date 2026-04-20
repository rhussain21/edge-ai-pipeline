#!/usr/bin/env python3
"""
Pipeline test script - tests full ETL flow from downloaded content to vectorization.

Tests:
1. Content extraction (PDF, HTML, audio transcription)
2. Chunking and database insertion
3. Deduplication (file hash and URL)
4. Vectorization of content chunks
5. Signal extraction from content
"""

import os
import sys
from pathlib import Path
from pprint import pprint

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

# Device-aware config — loads .env.jetson or .env.mac automatically
from device_config import config

# Import from project modules
from etl.pipeline import contentETL
from etl.sources import ContentSources
from etl.signals import SignalPipeline
from etl.content_screener import ContentScreener
from db_relational import relationalDB
from db_vector import VectorDB
from llm_client import OllamaClient, GeminiClient

def test_content_extraction():
    """Test 1: Extract and process downloaded content"""
    print("\n" + "="*60)
    print("TEST 1: Content Extraction & Database Insertion")
    print("="*60)
    
    # Initialize components — paths from device_config
    db = relationalDB(config.DB_PATH)
    corpus_vdb = VectorDB(config.CORPUS_VECTOR_PATH, use_builtin_embeddings=True)
    
    etl = contentETL(config.MEDIA_DIR, db=db, vdb=corpus_vdb)
    sources = ContentSources(config.MEDIA_DIR, db=db)
    
    # Get pending content from DB
    pending = etl.get_pending_content()
    print(f"\nFound {len(pending)} pending items to process")
    
    if not pending:
        print("No pending content found. Run test_adapter.py first to download content.")
        return None, None, None
    
    # Show what we found
    print("\nPending items by type:")
    by_type = {}
    for item in pending:
        ctype = item.get('content_type', 'unknown')
        by_type[ctype] = by_type.get(ctype, 0) + 1
    for ctype, count in by_type.items():
        print(f"  {ctype}: {count}")
    
    # Process all pending items
    print(f"\nProcessing all {len(pending)} items...")
    
    try:
        # Process pending content (transcription happens here)
        processed_ids = etl.process_pending_content()
        
        if processed_ids:
            print(f"\n✓ Successfully processed {len(processed_ids)} items")
            print(f"  Processed IDs: {processed_ids[:5]}{'...' if len(processed_ids) > 5 else ''}")
        else:
            print("\n⊘ No items were processed (check logs above)")
                
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        processed_ids = []
    
    print(f"\n{len(processed_ids)} items successfully added to database")
    return db, corpus_vdb, processed_ids


def test_vectorization(db, corpus_vdb, processed_ids):
    """Test 2: Vectorize content chunks"""
    print("\n" + "="*60)
    print("TEST 2: Content Vectorization")
    print("="*60)
    
    if not processed_ids:
        print("No new content to vectorize. Checking for pending vectorization...")
    
    # Initialize ETL with vector DB
    etl = contentETL(config.MEDIA_DIR, db=db, vdb=corpus_vdb)
    
    # Get pending vectorization count
    pending = etl.get_pending_vectorization(limit=100)
    print(f"\nFound {len(pending)} items pending vectorization")
    
    if not pending:
        print("All content already vectorized!")
        return True
    
    # Vectorize in batches
    print("\nVectorizing content chunks...")
    vectorized_count = etl.vectorize_pending_batch(limit=100)
    
    print(f"\n✓ Vectorized {vectorized_count} content items")
    
    # Save corpus vectors
    print("\nSaving corpus vectors...")
    corpus_vdb.save("corpus_vectors")
    print("✓ Corpus vectors saved")
    
    return True


def _get_llm_client():
    """Instantiate the correct LLM client based on device config."""
    if config.LLM_PROVIDER == 'gemini':
        return GeminiClient(model=config.LLM_MODEL)
    else:
        return OllamaClient(model=config.LLM_MODEL, base_url=config.LLM_URL)


def test_content_screening(db):
    """Test 3: LLM quality gate — approve or reject content before signal extraction."""
    print("\n" + "="*60)
    print("STEP 1: Content Screening (LLM Quality Gate)")
    print("="*60)

    llm_client = _get_llm_client()
    screener = ContentScreener(db=db, llm_client=llm_client)

    # Count pending items before screening
    pending_count = db.query("""
        SELECT COUNT(*) as count FROM content
        WHERE (screening_status = 'pending' OR screening_status IS NULL)
          AND extraction_status IN ('completed', 'NA')
    """)[0]['count']

    if pending_count == 0:
        print("\nNo new content to screen. All vectorized items have been screened.")
    else:
        batch_size = min(pending_count, 20)
        print(f"\nFound {pending_count} unscreened items. Processing batch of {batch_size}...")
        print(f"Using {config.LLM_PROVIDER}/{config.LLM_MODEL}")
        print(f"Each item is evaluated for named companies, technologies, events, and metrics.")
        print(f"Items without substantive factual content are rejected.\n")

        results = screener.screen_pending(limit=20)

        print(f"\nBatch results ({results['approved'] + results['rejected'] + results['errors']} items screened):")
        print(f"  ✓ Approved:  {results['approved']}")
        print(f"  ✗ Rejected:  {results['rejected']}")
        if results['errors']:
            print(f"  ⚠ Errors:    {results['errors']}")

        # Show per-item decisions
        for item in results['details']:
            icon = '✓' if item['decision'] == 'approve' else '✗' if item['decision'] == 'reject' else '⚠'
            print(f"  {icon} [{item['id']}] {item['title']}")
            print(f"        {item['reason']}")

    # Show cumulative rejection totals from DB (all time, not just this batch)
    total_rejected = db.query("""
        SELECT COUNT(*) as count FROM content WHERE screening_status = 'rejected'
    """)[0]['count']
    rejected_details = screener.get_rejection_report(limit=100)
    if rejected_details:
        print(f"\nAll-time rejection report ({total_rejected} total items marked for future deletion):")
        for r in rejected_details:
            print(f"  - [{r['id']}] {r['title']}: {r['screening_reason']}")

    return True


def test_signal_extraction(db):
    """Test 4: Extract signals from APPROVED content only."""
    print("\n" + "="*60)
    print("STEP 2: Signal Extraction (approved content only)")
    print("="*60)

    # Count approved items that still need signal extraction
    total_approved = db.query("""
        SELECT COUNT(*) as count FROM content WHERE screening_status = 'approved'
    """)[0]['count']
    already_extracted = db.query("""
        SELECT COUNT(*) as count FROM content
        WHERE screening_status = 'approved' AND signal_processed = TRUE
    """)[0]['count']
    pending_extraction = db.query("""
        SELECT id, title, source_type FROM content
        WHERE screening_status = 'approved'
          AND (signal_processed = FALSE OR signal_processed IS NULL)
        LIMIT 5
    """)

    print(f"\nApproved content: {total_approved} total")
    print(f"  Already extracted: {already_extracted}")
    print(f"  Pending extraction: {total_approved - already_extracted}")

    if not pending_extraction:
        print("\nNo approved content pending signal extraction.")
        return True

    batch_size = len(pending_extraction)
    print(f"\nProcessing batch of {batch_size} (of {total_approved - already_extracted} remaining):")
    for item in pending_extraction:
        print(f"  [{item['source_type']}] {item['title']} (ID: {item['id']})")

    # SignalPipeline takes a model ID string and optional URL
    signal_pipeline = SignalPipeline(db, llm_client=config.LLM_MODEL, llm_url=config.LLM_URL)

    content_ids = [item['id'] for item in pending_extraction]
    print(f"\nExtracting signals via LangExtract ({config.LLM_MODEL})...")
    results = signal_pipeline.extract_from_batch(content_ids)

    # Report results
    success = sum(1 for r in results.values() if r['status'] == 'success')
    failed = sum(1 for r in results.values() if r['status'] == 'failed')
    total_stored = sum(r.get('signals_stored', 0) for r in results.values() if r['status'] == 'success')
    print(f"\nBatch results: {success} succeeded, {failed} failed, {total_stored} new signals")

    signal_count = db.query("SELECT COUNT(*) as count FROM signals")[0]['count']
    print(f"Total signals in database: {signal_count}")

    return True


def test_signal_vectorization(db):
    """Test 5: Vectorize extracted signals"""
    print("\n" + "="*60)
    print("TEST 5: Signal Vectorization")
    print("="*60)
    
    # Check for unvectorized signals
    unvectorized = db.query("""
        SELECT COUNT(*) as count 
        FROM signals 
        WHERE vectorized = FALSE OR vectorized IS NULL
    """)
    
    count = unvectorized[0]['count'] if unvectorized else 0
    print(f"\nFound {count} signals needing vectorization")
    
    if count == 0:
        print("All signals already vectorized!")
        return True
    
    # Initialize ETL for signal vectorization
    etl = contentETL(config.MEDIA_DIR, db=db)
    
    print("\nVectorizing signals...")
    total_vectorized = etl.run_signal_vectorization()
    
    print(f"\n✓ Vectorized {total_vectorized} signals")
    
    return True


def test_query_content(db, corpus_vdb):
    """Test 6: Query vectorized content"""
    print("\n" + "="*60)
    print("TEST 6: Vector Search Test")
    print("="*60)
    
    test_queries = [
        "robotics automation",
        "SCADA security",
        "edge AI manufacturing"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = corpus_vdb.search(query, top_k=3)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                print(f"  {i}. {metadata.get('title', 'Unknown')}")
                print(f"     Score: {result.get('similarity', 0):.3f}")
                print(f"     Snippet: {result.get('document', '')[:100]}...")
        except Exception as e:
            print(f"  Error: {e}")
    
    return True


def print_summary(db):
    """Print final summary statistics — authoritative source of truth."""
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    total = db.query("SELECT COUNT(*) as count FROM content")[0]['count']
    vectorized = db.query("SELECT COUNT(*) as count FROM content WHERE vectorization_status = 'completed'")[0]['count']
    approved = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'approved'")[0]['count']
    rejected = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'rejected'")[0]['count']
    unscreened = total - approved - rejected
    signals_done = db.query("SELECT COUNT(*) as count FROM content WHERE signal_processed = TRUE")[0]['count']
    total_signals = db.query("SELECT COUNT(*) as count FROM signals")[0]['count']

    print(f"\nContent: {total} items")
    print(f"  Vectorized:      {vectorized}")
    print(f"  Approved:        {approved}")
    print(f"  Rejected:        {rejected}")
    print(f"  Unscreened:      {unscreened}")
    print(f"  Signals done:    {signals_done} of {approved} approved")

    # Breakdown by source type
    content_stats = db.query("""
        SELECT 
            source_type,
            COUNT(*) as count,
            SUM(CASE WHEN screening_status = 'approved' THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN screening_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
            SUM(CASE WHEN signal_processed = TRUE THEN 1 ELSE 0 END) as signals_extracted
        FROM content
        GROUP BY source_type
    """)
    print("\nBy source type:")
    for stat in content_stats:
        print(f"  {stat['source_type']}: {stat['count']} items "
              f"(✓ {stat['approved']} approved, ✗ {stat['rejected']} rejected, "
              f"{stat['signals_extracted']} extracted)")

    # Signal breakdown
    signal_stats = db.query("""
        SELECT signal_type, COUNT(*) as count
        FROM signals GROUP BY signal_type ORDER BY count DESC
    """)
    if signal_stats:
        print(f"\nSignals: {total_signals} total")
        for stat in signal_stats:
            print(f"  {stat['signal_type']}: {stat['count']}")

    if approved > 0:
        print(f"\nAvg signals per approved item: {total_signals / approved:.1f}")
    else:
        print("\nNo approved content yet.")


if __name__ == "__main__":
    print("="*60)
    print("INDUSTRY SIGNALS - SCREENING & SIGNAL EXTRACTION TEST")
    print("="*60)
    print(f"LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
    
    try:
        # Initialize DBs (needed for all tests)
        db = relationalDB(config.DB_PATH)
        corpus_vdb = VectorDB(config.CORPUS_VECTOR_PATH, model_name="all-MiniLM-L6-v2")
        
        # Skip these working steps for faster testing:
        # # Test 1: Extract and insert content
        db, corpus_vdb, processed_ids = test_content_extraction()
        # 
        if db is None:
            print("\nNo content to process. Exiting.")
            sys.exit(0)
        # 
        # # Test 2: Vectorize content chunks (optional — skip on Jetson aggregator)
        skip_vectors = os.getenv('SKIP_VECTORIZATION', 'true').lower() in ('true', '1', 'yes')
        if skip_vectors:
            print("\n⊘ Vectorization SKIPPED (SKIP_VECTORIZATION=true)")
            print("  Set SKIP_VECTORIZATION=false to enable.")
        else:
            test_vectorization(db, corpus_vdb, processed_ids)
        
        # ── Overview: current state of the content table ──
        total = db.query("SELECT COUNT(*) as count FROM content")[0]['count']
        vectorized = db.query("SELECT COUNT(*) as count FROM content WHERE vectorization_status = 'completed'")[0]['count']
        approved = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'approved'")[0]['count']
        rejected = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'rejected'")[0]['count']
        unscreened = db.query("""
            SELECT COUNT(*) as count FROM content
            WHERE (screening_status = 'pending' OR screening_status IS NULL)
        """)[0]['count']
        signals_done = db.query("SELECT COUNT(*) as count FROM content WHERE signal_processed = TRUE")[0]['count']

        print(f"\n┌─ Content Overview ({total} items total) ───────────")
        print(f"│  Vectorized:    {vectorized}")
        print(f"│  Screened:      {approved + rejected} (approved: {approved}, rejected: {rejected})")
        print(f"│  Unscreened:    {unscreened}")
        print(f"│  Signals done:  {signals_done} of {approved} approved")
        print(f"└──────────────────────────────────────────")

        # Step 1: Screen unscreened content
        test_content_screening(db)

        # Step 2: Extract signals from approved content
        test_signal_extraction(db)
        
        # # Test 5: Vectorize signals
        # test_signal_vectorization(db)
        
        # # Test 6: Query test
        # test_query_content(db, corpus_vdb)
        
        # Summary
        print_summary(db)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
