#!/usr/bin/env python3
"""
Retroactive Quality Gate — re-screen ALL existing content through rule-based checks.

Runs ContentQualityGate against every content row and updates:
  - screening_status → 'rejected' (if it fails)
  - screening_reason → '[quality_gate] <reason>'
  - screened_at      → current UTC timestamp
  - marked_for_deletion → TRUE
  - do_not_vectorize → TRUE

Content that passes the gate is left untouched (preserves existing LLM screening decisions).

Usage:
    # Dry run — show what would be rejected (no DB changes)
    python workflows/retroactive_quality_gate.py

    # Apply changes
    python workflows/retroactive_quality_gate.py --apply

    # Only check content that was previously approved (re-evaluate)
    python workflows/retroactive_quality_gate.py --apply --include-approved

    # Only check unscreened content
    python workflows/retroactive_quality_gate.py --apply --unscreened-only
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from db_relational import relationalDB
from etl.content_screener import ContentQualityGate
from logging_config import syslog


def run(args):
    run_id = syslog.start_run('retroactive_quality_gate')
    db = relationalDB(config.DB_PATH)

    print(f"{'=' * 60}")
    print(f"RETROACTIVE QUALITY GATE  ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'=' * 60}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"DB:   {config.DB_BACKEND} → {config.DB_PATH}")

    # Build WHERE clause based on flags
    conditions = []
    if args.unscreened_only:
        conditions.append("(screening_status IS NULL OR screening_status = 'pending')")
        print("Scope: unscreened content only")
    elif not args.include_approved:
        # Default: skip already-rejected content (no point re-rejecting)
        conditions.append("(screening_status IS NULL OR screening_status != 'rejected')")
        print("Scope: all non-rejected content")
    else:
        print("Scope: ALL content (including previously approved)")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    rows = db.query(f"""
        SELECT id, title, transcript, screening_status, signal_processed,
               LENGTH(transcript) as char_len
        FROM content
        {where}
        ORDER BY id
    """)

    total = len(rows)
    print(f"Found {total} content items to evaluate\n")

    if not total:
        print("Nothing to evaluate.")
        syslog.end_run('retroactive_quality_gate', summary='0 items')
        return

    # Run quality gate on each item
    gate_fail = []
    gate_pass = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for i, row in enumerate(rows, 1):
        content_id = row['id']
        title = row.get('title', 'Untitled')
        transcript = row.get('transcript') or ''
        current_status = row.get('screening_status', 'pending')
        had_signals = row.get('signal_processed', False)
        char_len = row.get('char_len', 0) or 0

        passed, reason = ContentQualityGate.check(transcript, title)

        if not passed:
            gate_fail.append({
                'id': content_id,
                'title': title,
                'reason': reason,
                'char_len': char_len,
                'prev_status': current_status,
                'had_signals': had_signals,
            })
        else:
            gate_pass.append(content_id)

    # Display results
    print(f"{'=' * 60}")
    print(f"RESULTS: {len(gate_fail)} FAIL / {len(gate_pass)} PASS  (of {total})")
    print(f"{'=' * 60}")

    # Group failures by reason
    by_reason = {}
    for f in gate_fail:
        # Extract the reason tag (before any parenthetical details)
        tag = f['reason'].split(' (')[0]
        by_reason.setdefault(tag, []).append(f)

    print("\nFailures by reason:")
    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        print(f"  {reason}: {len(items)}")

    # Show previously-approved items that would now be rejected
    was_approved = [f for f in gate_fail if f['prev_status'] == 'approved']
    if was_approved:
        print(f"\n⚠  {len(was_approved)} previously APPROVED items would be rejected:")
        for f in was_approved[:20]:
            signals_tag = " [has signals]" if f['had_signals'] else ""
            print(f"    [{f['id']:>4}] {f['char_len']:>6,} chars  {f['title'][:55]}{signals_tag}")
        if len(was_approved) > 20:
            print(f"    ... and {len(was_approved) - 20} more")

    # Show sample failures
    print(f"\nSample failures (first 20):")
    for f in gate_fail[:20]:
        print(f"  [{f['id']:>4}] {f['char_len']:>6,} chars  [{f['prev_status']:>8}] "
              f"{f['reason'][:25]:25}  {f['title'][:45]}")
    if len(gate_fail) > 20:
        print(f"  ... and {len(gate_fail) - 20} more")

    # Apply changes if requested
    if args.apply and gate_fail:
        print(f"\nApplying changes to {len(gate_fail)} items...")
        applied = 0
        for f in gate_fail:
            try:
                db.update_record(f['id'], {
                    'screening_status': 'rejected',
                    'screening_reason': f"[quality_gate] {f['reason']}",
                    'screened_at': now,
                    'marked_for_deletion': True,
                    'do_not_vectorize': True,
                })
                applied += 1
            except Exception as e:
                print(f"  ERROR updating id={f['id']}: {e}")

        print(f"Updated {applied} content records.")
        syslog.info('retroactive_quality_gate', 'applied',
                     f'Rejected {applied} items via quality gate',
                     details={
                         'total_evaluated': total,
                         'gate_fail': len(gate_fail),
                         'gate_pass': len(gate_pass),
                         'was_approved': len(was_approved),
                         'reasons': {k: len(v) for k, v in by_reason.items()},
                     })
    elif not args.apply and gate_fail:
        print(f"\nDry run — no changes made. Use --apply to update {len(gate_fail)} records.")

    syslog.end_run('retroactive_quality_gate',
                    summary=f'{len(gate_fail)} fail / {len(gate_pass)} pass of {total}')


def main():
    parser = argparse.ArgumentParser(
        description="Retroactive quality gate — re-screen existing content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--apply', action='store_true',
                        help='Apply changes to DB (default is dry run)')
    parser.add_argument('--include-approved', action='store_true',
                        help='Also re-evaluate previously approved content')
    parser.add_argument('--unscreened-only', action='store_true',
                        help='Only evaluate content that has not been screened yet')
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
