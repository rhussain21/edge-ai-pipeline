#!/usr/bin/env python3
"""
Content Processing Workflow.

Processes pending content through the ETL pipeline:
  1. Content extraction (transcription for audio, text extraction for PDF/HTML)
  2. Content screening (LLM quality gate)
  3. Signal extraction (extract structured signals from approved content)

Usage:
    # Full pipeline (all steps)
    python workflows/process.py

    # Only transcribe/extract, nothing else
    python workflows/process.py --skip-screening --skip-signals

    # Limit signal extraction batch size
    python workflows/process.py --signal-batch 5

    # Process only audio content
    python workflows/process.py --content-type audio

Cron (Thursday 11 PM):
    0 23 * * 4  cd /home/redwan/ai_industry_signals && python workflows/process.py >> /var/log/process.log 2>&1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from logging_config import syslog
from db_relational import relationalDB
from etl.pipeline import contentETL
from etl.content_screener import ContentScreener
from etl.signals import SignalPipeline
from llm_client import OllamaClient, GeminiClient

import time


def get_llm_client():
    if config.LLM_PROVIDER == 'gemini':
        return GeminiClient(model=config.LLM_MODEL)
    else:
        return OllamaClient(model=config.LLM_MODEL, base_url=config.LLM_URL)


def run_processing(args):
    run_id = syslog.start_run('processing')

    print("=" * 60)
    print("PROCESSING WORKFLOW")
    print("=" * 60)
    print(f"Content type:      {args.content_type or 'all'}")
    print(f"Skip screening:    {args.skip_screening}")
    print(f"Skip signals:      {args.skip_signals}")
    print(f"Signal batch:      {args.signal_batch}")
    print()

    # ── Initialize ──
    db = relationalDB(config.DB_PATH)
    etl = contentETL(config.MEDIA_DIR, db=db)

    stats = {
        'extracted': 0,
        'screened_approved': 0,
        'screened_rejected': 0,
        'screened_errors': 0,
        'signals_extracted': 0,
    }

    # ── Step 1: Content Extraction ──
    print("\n── Step 1: Content Extraction ──")
    pending = etl.get_pending_content(content_type=args.content_type)
    print(f"Found {len(pending)} pending items")

    if pending:
        by_type = {}
        for item in pending:
            ctype = item.get('content_type', 'unknown')
            by_type[ctype] = by_type.get(ctype, 0) + 1
        for ctype, count in by_type.items():
            print(f"  {ctype}: {count}")

        processed_ids = etl.process_pending_content(content_type=args.content_type)
        stats['extracted'] = len(processed_ids)
        print(f"\nExtracted: {len(processed_ids)} items")

        syslog.info('processing', 'extraction_complete',
                    f'Extracted {len(processed_ids)} items',
                    details={'count': len(processed_ids), 'by_type': by_type})
    else:
        print("No pending content to extract.")

    # ── Step 2: Content Screening ──
    if args.skip_screening:
        print("\n── Step 2: Content Screening SKIPPED ──")
    else:
        print("\n── Step 2: Content Screening ──")
        pending_screen = db.query("""
            SELECT COUNT(*) as count FROM content
            WHERE (screening_status = 'pending' OR screening_status IS NULL)
              AND extraction_status IN ('completed', 'NA')
        """)[0]['count']

        print(f"Found {pending_screen} items pending screening")

        if pending_screen > 0:
            llm_client = get_llm_client()
            screener = ContentScreener(db=db, llm_client=llm_client)

            BATCH_SIZE = 20
            total_approved = 0
            total_rejected = 0
            total_errors = 0
            batch_num = 0

            while True:
                remaining = db.query("""
                    SELECT COUNT(*) as count FROM content
                    WHERE (screening_status = 'pending' OR screening_status IS NULL)
                      AND extraction_status IN ('completed', 'NA')
                """)[0]['count']

                if remaining == 0:
                    break

                batch_num += 1
                print(f"\n  Batch {batch_num} ({remaining} remaining)...")

                try:
                    results = screener.screen_pending(limit=BATCH_SIZE)
                    total_approved += results['approved']
                    total_rejected += results['rejected']
                    total_errors += results['errors']

                    print(f"  Batch {batch_num}: {results['approved']} approved, "
                          f"{results['rejected']} rejected, {results['errors']} errors")

                    # If an entire batch is errors, pause briefly then continue
                    if results['errors'] == BATCH_SIZE:
                        print("  Full batch errored — pausing 10s before retry...")
                        time.sleep(10)

                    # If no items were processed at all, break to avoid infinite loop
                    if results['approved'] + results['rejected'] + results['errors'] == 0:
                        print("  No items processed — breaking.")
                        break

                except Exception as e:
                    print(f"  Batch {batch_num} failed: {e}")
                    syslog.error('processing', 'screening_batch_failed', str(e))
                    time.sleep(10)
                    # Mark the first BATCH_SIZE pending items as 'error' to skip them
                    stuck = db.query("""
                        SELECT id FROM content
                        WHERE (screening_status = 'pending' OR screening_status IS NULL)
                          AND extraction_status IN ('completed', 'NA')
                        ORDER BY id LIMIT ?
                    """, [BATCH_SIZE])
                    for row in stuck:
                        db.update_record(row['id'], {'screening_status': 'error'})
                    total_errors += len(stuck)
                    continue

            stats['screened_approved'] = total_approved
            stats['screened_rejected'] = total_rejected
            stats['screened_errors'] = total_errors

            print(f"\n  Screening complete: {total_approved} approved, "
                  f"{total_rejected} rejected, {total_errors} errors")

            syslog.info('processing', 'screening_complete',
                        f"{total_approved} approved, {total_rejected} rejected, {total_errors} errors",
                        details={'approved': total_approved, 'rejected': total_rejected, 'errors': total_errors})
        else:
            print("No pending content to screen.")

    # ── Step 3: Signal Extraction ──
    if args.skip_signals:
        print("\n── Step 3: Signal Extraction SKIPPED ──")
    else:
        print("\n── Step 3: Signal Extraction ──")
        pending_signals = db.query("""
            SELECT id, title, source_type FROM content
            WHERE screening_status = 'approved'
              AND (signal_processed = FALSE OR signal_processed IS NULL)
            LIMIT ?
        """, [args.signal_batch])

        if pending_signals:
            print(f"Found {len(pending_signals)} items pending signal extraction")
            signal_pipeline = SignalPipeline(db, llm_client=config.LLM_MODEL, llm_url=config.LLM_URL)
            content_ids = [item['id'] for item in pending_signals]
            results = signal_pipeline.extract_from_batch(content_ids)

            success = sum(1 for r in results.values() if r['status'] == 'success')
            total_stored = sum(r.get('signals_stored', 0) for r in results.values() if r['status'] == 'success')
            stats['signals_extracted'] = total_stored
            print(f"Extracted: {success} succeeded, {total_stored} new signals")

            syslog.info('processing', 'signals_complete',
                        f'{success} items processed, {total_stored} signals extracted',
                        details={'success': success, 'signals': total_stored})
        else:
            print("No pending content for signal extraction.")

    # ── Summary ──
    total = db.query("SELECT COUNT(*) as count FROM content")[0]['count']
    approved = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'approved'")[0]['count']
    rejected = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'rejected'")[0]['count']
    still_pending = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'pending' OR screening_status IS NULL")[0]['count']
    signals_done = db.query("SELECT COUNT(*) as count FROM content WHERE signal_processed = TRUE")[0]['count']
    total_signals = db.query("SELECT COUNT(*) as count FROM signals")[0]['count']

    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"This run:")
    print(f"  Extracted:  {stats['extracted']}")
    print(f"  Approved:   {stats['screened_approved']}")
    print(f"  Rejected:   {stats['screened_rejected']}")
    print(f"  Errors:     {stats['screened_errors']}")
    print(f"  Signals:    {stats['signals_extracted']}")
    print(f"\nDatabase totals:")
    print(f"  Content:    {total}")
    print(f"  Approved:   {approved}")
    print(f"  Rejected:   {rejected}")
    print(f"  Still pending: {still_pending}")
    print(f"  Signals:    {total_signals} ({signals_done}/{approved} items done)")

    syslog.end_run('processing',
                   summary=f"Extracted {stats['extracted']}, "
                           f"{stats['screened_approved']} approved, {stats['screened_rejected']} rejected, "
                           f"{stats['signals_extracted']} signals")


def main():
    parser = argparse.ArgumentParser(
        description="Content processing workflow (extraction, screening, signals).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--content-type', type=str, default=None,
        choices=['audio', 'pdf', 'html', 'text'],
        help='Process only this content type (default: all)',
    )
    parser.add_argument(
        '--skip-screening', action='store_true',
        help='Skip LLM content screening',
    )
    parser.add_argument(
        '--skip-signals', action='store_true',
        help='Skip signal extraction',
    )
    parser.add_argument(
        '--signal-batch', type=int, default=5,
        help='Max items for signal extraction per run (default: 5)',
    )
    args = parser.parse_args()

    try:
        run_processing(args)
    except KeyboardInterrupt:
        print("\nProcessing interrupted.")
        syslog.warning('processing', 'interrupted', 'Processing interrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        syslog.error('processing', 'failed', str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
