#!/usr/bin/env python3
"""
Cleanup Signals from Rejected Content

Deletes signals from the signals table for all content that has been rejected
(screening_status = 'rejected'). This ensures the corpus doesn't contain signals
derived from low-quality content.

Usage:
    # Dry run — show what would be deleted (no DB changes)
    python workflows/cleanup_rejected_signals.py

    # Apply changes
    python workflows/cleanup_rejected_signals.py --apply

    # Only clean up signals from quality gate rejections
    python workflows/cleanup_rejected_signals.py --apply --gate-only

Cron (optional — run after retroactive quality gate):
    0 23 * * * 6  cd /home/redwan/ai_industry_signals && python workflows/cleanup_rejected_signals.py --apply >> /var/log/cleanup_rejected_signals.log 2>&1
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from db_relational import relationalDB
from logging_config import syslog


def run(args):
    run_id = syslog.start_run('cleanup_rejected_signals')
    db = relationalDB(config.DB_PATH)

    print(f"{'=' * 60}")
    print(f"CLEANUP SIGNALS FROM REJECTED CONTENT  ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'=' * 60}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"DB:   {config.DB_BACKEND} → {config.DB_PATH}")

    # Build WHERE clause
    conditions = ["screening_status = 'rejected'"]
    if args.gate_only:
        conditions.append("screening_reason LIKE '[quality_gate]%'")
        print("Scope: quality gate rejections only")
    else:
        print("Scope: ALL rejected content")

    where = f"WHERE {' AND '.join(conditions)}"

    # Get rejected content IDs
    rejected_content = db.query(f"""
        SELECT id, title, screening_reason, signal_processed
        FROM content
        {where}
        ORDER BY id
    """)

    total_content = len(rejected_content)
    print(f"Found {total_content} rejected content items\n")

    if not total_content:
        print("No rejected content found.")
        syslog.end_run('cleanup_rejected_signals', summary='0 rejected content')
        return

    # Count signals for each rejected content
    content_with_signals = []
    total_signals = 0

    for row in rejected_content:
        content_id = row['id']
        title = row.get('title', 'Untitled')
        reason = row.get('screening_reason', '')
        had_signals = row.get('signal_processed', False)

        signal_count = db.query("""
            SELECT COUNT(*) as count FROM signals
            WHERE source_content_id = ?
        """, [content_id])[0]['count']

        if signal_count > 0:
            content_with_signals.append({
                'content_id': content_id,
                'title': title,
                'reason': reason,
                'signal_count': signal_count,
                'had_signals': had_signals,
            })
            total_signals += signal_count

    print(f"{'=' * 60}")
    print(f"IMPACT: {len(content_with_signals)} content items have {total_signals} signals")
    print(f"{'=' * 60}")

    # Show breakdown by rejection reason
    by_reason = {}
    for item in content_with_signals:
        tag = item['reason'][:40] if item['reason'] else 'unknown'
        by_reason.setdefault(tag, {'content': 0, 'signals': 0})
        by_reason[tag]['content'] += 1
        by_reason[tag]['signals'] += item['signal_count']

    print("\nSignals by rejection reason:")
    for reason, stats in sorted(by_reason.items(), key=lambda x: -x[1]['signals']):
        print(f"  {reason}: {stats['content']} content, {stats['signals']} signals")

    # Show sample
    print(f"\nSample content with signals (first 20):")
    for item in content_with_signals[:20]:
        signals_tag = f" [{item['signal_count']} signals]"
        print(f"  [{item['content_id']:>4}] {item['title'][:55]}{signals_tag}")
    if len(content_with_signals) > 20:
        print(f"  ... and {len(content_with_signals) - 20} more")

    # Apply changes if requested
    if args.apply and content_with_signals:
        print(f"\nDeleting {total_signals} signals...")
        deleted = 0
        errors = 0

        for item in content_with_signals:
            content_id = item['content_id']
            try:
                # Delete signals for this content
                db.execute("""
                    DELETE FROM signals
                    WHERE source_content_id = ?
                """, [content_id])
                deleted += item['signal_count']
            except Exception as e:
                print(f"  ERROR deleting signals for content_id={content_id}: {e}")
                errors += 1

        print(f"Deleted {deleted} signals from {len(content_with_signals)} content items.")
        if errors:
            print(f"  {errors} errors occurred")

        syslog.info('cleanup_rejected_signals', 'applied',
                     f'Deleted {deleted} signals from {len(content_with_signals)} rejected content',
                     details={
                         'total_content': total_content,
                         'content_with_signals': len(content_with_signals),
                         'signals_deleted': deleted,
                         'errors': errors,
                         'by_reason': {k: v for k, v in by_reason.items()},
                     })
    elif not args.apply and content_with_signals:
        print(f"\nDry run — no changes made. Use --apply to delete {total_signals} signals.")

    syslog.end_run('cleanup_rejected_signals',
                    summary=f'{len(content_with_signals)} content, {total_signals} signals')


def main():
    parser = argparse.ArgumentParser(
        description="Delete signals from rejected content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--apply', action='store_true',
                        help='Apply changes to DB (default is dry run)')
    parser.add_argument('--gate-only', action='store_true',
                        help='Only clean up signals from quality gate rejections (screening_reason starts with [quality_gate])')
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
