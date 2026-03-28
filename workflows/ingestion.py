#!/usr/bin/env python3
"""
Source Discovery & Content Ingestion Workflow.

Discovers new sources via RSS, web search, and GitHub adapters,
classifies them with LLM, and downloads approved content to media/.

Usage:
    # Default: RSS + web, topic=all, max 5 web queries
    python workflows/ingestion.py

    # Only RSS feeds (no API calls)
    python workflows/ingestion.py --source-types rss

    # Web + GitHub, filter to a topic, cap queries
    python workflows/ingestion.py --source-types web github --topic "edge AI" --max-searches 10

    # Skip LLM classification (approve everything)
    python workflows/ingestion.py --skip-classification

    # Dry run: discover but don't download
    python workflows/ingestion.py --dry-run

Cron (Wednesday 11 PM):
    0 23 * * 3  cd /home/redwan/ai_industry_signals && python workflows/ingestion.py >> /var/log/ingestion.log 2>&1
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from logging_config import syslog
from llm_client import OllamaClient, GeminiClient
from discovery.source_discovery_service import SourceDiscoveryService
from discovery.cache import SourceHealthTracker
from etl.sources import ContentSources
from db_relational import relationalDB


def get_llm_client():
    if config.LLM_PROVIDER == 'gemini':
        return GeminiClient(model=config.LLM_MODEL)
    else:
        return OllamaClient(model=config.LLM_MODEL, base_url=config.LLM_URL)


def run_ingestion(args):
    run_id = syslog.start_run('ingestion')

    print("=" * 60)
    print("INGESTION WORKFLOW")
    print("=" * 60)
    print(f"Source types:  {', '.join(args.source_types)}")
    print(f"Topic filter:  {args.topic or 'all'}")
    print(f"Max searches:  {args.max_searches}")
    print(f"Classification: {'skip' if args.skip_classification else 'enabled'}")
    print(f"Dry run:       {args.dry_run}")
    print(f"LLM:           {config.LLM_PROVIDER}/{config.LLM_MODEL}")
    print()

    # ── Step 1: Source Discovery ──
    print("── Step 1: Source Discovery ──")
    llm_client = get_llm_client()
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'discovery', 'configs')

    sd = SourceDiscoveryService(
        llm_generate_fn=llm_client.generate,
        config_dir=config_dir,
    )

    results = sd.run(
        adapters=args.source_types,
        topic_filter=args.topic,
        max_queries=args.max_searches,
        skip_classification=args.skip_classification,
    )

    print(f"\nDiscovery results:")
    print(f"  Queries executed: {results.queries_generated}")
    print(f"  Candidates found: {results.candidates_found}")
    print(f"  Deduped:          {results.candidates_deduped}")
    print(f"  Approved:         {results.candidates_approved}")
    print(f"  Rejected:         {results.candidates_rejected}")
    if results.errors:
        print(f"  Errors:           {len(results.errors)}")
        for err in results.errors[:5]:
            print(f"    - {err}")

    syslog.info('ingestion', 'discovery_complete',
                f'{results.candidates_approved} approved from {results.candidates_found} candidates',
                details={
                    'queries': results.queries_generated,
                    'found': results.candidates_found,
                    'approved': results.candidates_approved,
                    'rejected': results.candidates_rejected,
                    'source_types': args.source_types,
                    'topic': args.topic,
                })

    # ── Step 2: Download Approved Content ──
    approved = sd.get_approved_for_ingestion(results)
    print(f"\n── Step 2: Download ({len(approved)} items) ──")

    if not approved:
        print("No new content to download.")
        syslog.end_run('ingestion', summary='No new content found')
        return

    if args.dry_run:
        print("\nDRY RUN — would download:")
        for i, item in enumerate(approved[:20], 1):
            print(f"  {i}. [{item['source_type']}] {item['title'][:70]}")
            print(f"     {item['url']}")
        if len(approved) > 20:
            print(f"  ... and {len(approved) - 20} more")
        syslog.end_run('ingestion', summary=f'Dry run: {len(approved)} items would be downloaded')
        return

    db = relationalDB(config.DB_PATH)
    health_tracker = SourceHealthTracker()
    downloader = ContentSources(config.MEDIA_DIR, db=db, health_tracker=health_tracker)

    downloader.download_approved(approved)

    syslog.info('ingestion', 'download_complete',
                f'Downloaded {len(approved)} items',
                details={'count': len(approved)})

    # ── Step 3: Health Report ──
    print("\n── Source Health Report ──")
    flagged = health_tracker.get_flagged_sources()
    if flagged:
        print(f"Flagged sources ({len(flagged)}):")
        for s in flagged:
            print(f"  [{s['adapter']}] {s['source_url']}")
            print(f"    Downloads: {s['downloads']}")
            print(f"    Discovery: {s['discovery']}")
    else:
        print("No sources flagged.")

    syslog.end_run('ingestion',
                   summary=f'{results.candidates_approved} approved, {len(approved)} downloaded')

    print(f"\n{'=' * 60}")
    print("INGESTION COMPLETE")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Source discovery and content ingestion workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--source-types', nargs='+', default=['rss', 'web'],
        choices=['rss', 'web', 'github'],
        help='Adapter types to use (default: rss web)',
    )
    parser.add_argument(
        '--topic', type=str, default=None,
        help='Topic filter for query generation (default: all topics)',
    )
    parser.add_argument(
        '--max-searches', type=int, default=5,
        help='Max web/github search queries to execute (default: 5)',
    )
    parser.add_argument(
        '--skip-classification', action='store_true',
        help='Skip LLM classification — approve all candidates',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Discover only, do not download',
    )
    args = parser.parse_args()

    try:
        run_ingestion(args)
    except KeyboardInterrupt:
        print("\nIngestion interrupted.")
        syslog.warning('ingestion', 'interrupted', 'Ingestion interrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f"\nIngestion failed: {e}")
        syslog.error('ingestion', 'failed', str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
