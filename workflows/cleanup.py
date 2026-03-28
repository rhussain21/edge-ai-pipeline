#!/usr/bin/env python3
"""
Media Cleanup Workflow.

Marks and deletes files from media/ based on content table metadata:
  - Rejected content (screening_status = 'rejected')
  - Duplicate content (same content_hash, keep newest)
  - Old content (> 6 months, signal already extracted)
  - Processed audio files (transcript exists, signals extracted)

Usage:
    # Dry run (default) — shows what would be deleted
    python workflows/cleanup.py

    # Mark files for deletion in DB (no files deleted yet)
    python workflows/cleanup.py --mark

    # Actually delete marked files from disk
    python workflows/cleanup.py --delete

    # Mark + delete in one step
    python workflows/cleanup.py --mark --delete

    # Override age threshold (default 180 days)
    python workflows/cleanup.py --max-age-days 90

Cron (Saturday 11 PM):
    0 23 * * 6  cd /home/redwan/ai_industry_signals && python workflows/cleanup.py --mark --delete >> /var/log/cleanup.log 2>&1
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from db_relational import relationalDB
from logging_config import syslog


def get_db():
    return relationalDB(config.DB_PATH)


# ── Cleanup rules ────────────────────────────────────────────────────

def find_rejected(db) -> list:
    """Files with screening_status = 'rejected'."""
    rows = db.query("""
        SELECT id, title, file_path, content_type, file_size_mb
        FROM content
        WHERE screening_status = 'rejected'
          AND marked_for_deletion = FALSE
          AND file_path IS NOT NULL
    """)
    return [dict(r, reason='rejected') for r in rows]


def find_duplicates(db) -> list:
    """Files sharing the same content_hash — keep the newest, mark the rest."""
    dupes = db.query("""
        SELECT content_hash, COUNT(*) as cnt
        FROM content
        WHERE content_hash IS NOT NULL AND content_hash != ''
        GROUP BY content_hash
        HAVING COUNT(*) > 1
    """)
    to_delete = []
    for row in dupes:
        copies = db.query("""
            SELECT id, title, file_path, content_type, file_size_mb, created_at
            FROM content
            WHERE content_hash = ?
            ORDER BY created_at DESC
        """, [row['content_hash']])
        # Keep the first (newest), mark the rest
        for copy in copies[1:]:
            if copy['file_path']:
                to_delete.append(dict(copy, reason='duplicate'))
    return to_delete


def find_old_content(db, max_age_days: int = 180) -> list:
    """Content older than max_age_days where signals have been extracted."""
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
    rows = db.query("""
        SELECT id, title, file_path, content_type, file_size_mb, created_at
        FROM content
        WHERE created_at < ?
          AND signal_processed = TRUE
          AND marked_for_deletion = FALSE
          AND file_path IS NOT NULL
    """, [cutoff])
    return [dict(r, reason=f'old (>{max_age_days}d, signals done)') for r in rows]


def find_processed_audio(db) -> list:
    """Audio files where transcription is complete and signals extracted."""
    rows = db.query("""
        SELECT id, title, file_path, content_type, file_size_mb
        FROM content
        WHERE content_type = 'audio'
          AND extraction_status = 'completed'
          AND signal_processed = TRUE
          AND marked_for_deletion = FALSE
          AND file_path IS NOT NULL
    """)
    return [dict(r, reason='audio (transcribed + signals done)') for r in rows]


# ── Actions ──────────────────────────────────────────────────────────

def mark_for_deletion(db, candidates: list):
    """Set marked_for_deletion = TRUE in the content table."""
    if not candidates:
        return 0
    marked = 0
    for c in candidates:
        db.execute("UPDATE content SET marked_for_deletion = TRUE WHERE id = ?", [c['id']])
        marked += 1
    return marked


def delete_files(db):
    """Delete files on disk for all rows where marked_for_deletion = TRUE."""
    rows = db.query("""
        SELECT id, file_path, file_size_mb
        FROM content
        WHERE marked_for_deletion = TRUE
          AND file_path IS NOT NULL
    """)
    deleted = 0
    freed_mb = 0.0
    errors = 0
    for row in rows:
        fp = row['file_path']
        if os.path.exists(fp):
            try:
                size = row.get('file_size_mb') or 0
                os.remove(fp)
                deleted += 1
                freed_mb += size

                # Also delete companion metadata JSON if it exists
                meta_path = fp.rsplit('.', 1)[0] + '_metadata.json'
                if os.path.exists(meta_path):
                    os.remove(meta_path)

            except OSError as e:
                print(f"  ERROR deleting {fp}: {e}")
                errors += 1
        else:
            # File already gone — still counts as cleaned
            deleted += 1

    return deleted, freed_mb, errors


# ── Display ──────────────────────────────────────────────────────────

def print_candidates(candidates: list, label: str):
    if not candidates:
        print(f"\n  {label}: 0 files")
        return

    total_mb = sum(c.get('file_size_mb') or 0 for c in candidates)
    print(f"\n  {label}: {len(candidates)} files ({total_mb:.1f} MB)")
    for c in candidates:
        size = f"{c.get('file_size_mb') or 0:.1f}MB"
        title = (c.get('title') or 'Untitled')[:60]
        print(f"    [{c['id']}] {title}  ({size})")


def print_summary(all_candidates: list):
    total_files = len(all_candidates)
    total_mb = sum(c.get('file_size_mb') or 0 for c in all_candidates)
    by_reason = {}
    for c in all_candidates:
        reason = c.get('reason', 'unknown')
        by_reason.setdefault(reason, []).append(c)

    print(f"\n{'=' * 60}")
    print(f"CLEANUP SUMMARY: {total_files} files, {total_mb:.1f} MB")
    print(f"{'=' * 60}")
    for reason, items in sorted(by_reason.items()):
        mb = sum(c.get('file_size_mb') or 0 for c in items)
        print(f"  {reason}: {len(items)} files ({mb:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────

def run_cleanup(args):
    run_id = syslog.start_run('cleanup')

    print(f"{'=' * 60}")
    print(f"MEDIA CLEANUP  ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'=' * 60}")
    print(f"Mode: {'DRY RUN' if not args.mark else 'MARK' + (' + DELETE' if args.delete else '')}")
    print(f"Age threshold: {args.max_age_days} days")

    db = get_db()

    # Show current state
    already_marked = db.query(
        "SELECT COUNT(*) as count FROM content WHERE marked_for_deletion = TRUE"
    )[0]['count']
    if already_marked:
        print(f"Already marked for deletion: {already_marked} rows")

    # ── Identify candidates ──
    rejected = find_rejected(db)
    duplicates = find_duplicates(db)
    old = find_old_content(db, max_age_days=args.max_age_days)
    audio = find_processed_audio(db)

    # Deduplicate across categories
    seen_ids = set()
    all_candidates = []
    for candidate in rejected + duplicates + old + audio:
        if candidate['id'] not in seen_ids:
            seen_ids.add(candidate['id'])
            all_candidates.append(candidate)

    # ── Display ──
    print_candidates(rejected, "Rejected content")
    print_candidates(duplicates, "Duplicates (keeping newest)")
    print_candidates(old, f"Old content (>{args.max_age_days} days, signals done)")
    print_candidates(audio, "Processed audio (transcribed + signals done)")
    print_summary(all_candidates)

    total_mb = sum(c.get('file_size_mb') or 0 for c in all_candidates)

    if not all_candidates:
        print("\nNothing to clean up.")
        syslog.end_run('cleanup', summary='Nothing to clean up')
        return

    # ── Mark ──
    if args.mark:
        marked = mark_for_deletion(db, all_candidates)
        print(f"\nMarked {marked} rows for deletion in content table.")
        syslog.info('cleanup', 'mark', f'Marked {marked} files for deletion',
                    details={'total_mb': round(total_mb, 1),
                             'rejected': len(rejected), 'duplicates': len(duplicates),
                             'old': len(old), 'audio': len(audio)})
    else:
        print(f"\nDry run — {len(all_candidates)} files would be marked. Use --mark to proceed.")

    # ── Delete ──
    if args.delete:
        if not args.mark:
            print("WARNING: --delete requires --mark (or previously marked rows).")
        deleted, freed_mb, errors = delete_files(db)
        print(f"\nDeleted {deleted} files from disk ({freed_mb:.1f} MB freed)")
        syslog.info('cleanup', 'delete', f'Deleted {deleted} files ({freed_mb:.1f} MB freed)',
                    details={'deleted': deleted, 'freed_mb': round(freed_mb, 1), 'errors': errors})
        if errors:
            print(f"  {errors} files failed to delete")
            syslog.warning('cleanup', 'delete', f'{errors} files failed to delete')
    elif args.mark:
        print("Files marked but NOT deleted. Run with --mark --delete to remove from disk.")

    syslog.end_run('cleanup',
                   summary=f'{len(all_candidates)} candidates, {total_mb:.1f} MB')


def main():
    parser = argparse.ArgumentParser(
        description="Media file cleanup workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mark', action='store_true',
                        help='Mark candidates for deletion in DB')
    parser.add_argument('--delete', action='store_true',
                        help='Delete files on disk (only those marked)')
    parser.add_argument('--max-age-days', type=int, default=180,
                        help='Age threshold in days (default: 180)')
    args = parser.parse_args()

    try:
        run_cleanup(args)
    except KeyboardInterrupt:
        print("\nCleanup interrupted.")
        syslog.warning('cleanup', 'interrupted', 'Cleanup interrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f"\nCleanup failed: {e}")
        syslog.error('cleanup', 'failed', str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
