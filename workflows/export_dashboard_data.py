#!/usr/bin/env python3
"""
Export Dashboard Data — generates summarized JSON snapshots for the admin dashboard.

Runs on Jetson after processing, writes JSON files that can be:
  1. Served from a local Streamlit app (reads JSON directly)
  2. Pushed to Netlify/Vercel as static JSON API (no Jetson needed)
  3. Uploaded to S3/R2/GitHub Pages as a static data layer

The dashboard reads these pre-computed snapshots instead of querying
the DB directly — fast loads, works offline, deployable anywhere.

Usage:
    python workflows/export_dashboard_data.py

    # Push to Netlify (requires NETLIFY_AUTH_TOKEN + NETLIFY_SITE_ID in .env)
    python workflows/export_dashboard_data.py --push-netlify

    # Custom output directory
    python workflows/export_dashboard_data.py --output-dir /tmp/dashboard_data

Cron (Thursday 11:30 PM, after processing):
    30 23 * * 4  cd /home/redwan/ai_industry_signals && python workflows/export_dashboard_data.py --push-netlify >> /var/log/export.log 2>&1
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from db_relational import relationalDB
from logging_config import syslog


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard_data')


def _safe_div(a, b, default=0):
    return round(a / b, 2) if b else default


def export_pipeline_stats(db) -> dict:
    """Content pipeline funnel metrics + acceptance rates."""
    total = db.query("SELECT COUNT(*) as count FROM content")[0]['count']
    pending = db.query("SELECT COUNT(*) as count FROM content WHERE extraction_status = 'pending'")[0]['count']
    extracted = db.query("SELECT COUNT(*) as count FROM content WHERE extraction_status IN ('completed', 'NA')")[0]['count']
    failed_extraction = db.query("SELECT COUNT(*) as count FROM content WHERE extraction_status = 'failed'")[0]['count']
    vectorized = db.query("SELECT COUNT(*) as count FROM content WHERE vectorization_status = 'completed'")[0]['count']
    approved = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'approved'")[0]['count']
    rejected = db.query("SELECT COUNT(*) as count FROM content WHERE screening_status = 'rejected'")[0]['count']
    signals_done = db.query("SELECT COUNT(*) as count FROM content WHERE signal_processed = TRUE")[0]['count']
    total_signals = db.query("SELECT COUNT(*) as count FROM signals")[0]['count']
    marked_deletion = db.query("SELECT COUNT(*) as count FROM content WHERE marked_for_deletion = TRUE")[0]['count']

    by_type = db.query("""
        SELECT source_type, content_type, COUNT(*) as count,
               SUM(CASE WHEN screening_status = 'approved' THEN 1 ELSE 0 END) as approved,
               SUM(CASE WHEN screening_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
               SUM(CASE WHEN signal_processed = TRUE THEN 1 ELSE 0 END) as signals_extracted
        FROM content GROUP BY source_type, content_type
    """)

    # Rejection reasons breakdown
    rejection_reasons = db.query("""
        SELECT screening_reason, COUNT(*) as count
        FROM content
        WHERE screening_status = 'rejected' AND screening_reason IS NOT NULL
        GROUP BY screening_reason
        ORDER BY count DESC
        LIMIT 20
    """)

    return {
        'exported_at': datetime.utcnow().isoformat(),
        'totals': {
            'content': total,
            'pending': pending,
            'extracted': extracted,
            'failed_extraction': failed_extraction,
            'vectorized': vectorized,
            'approved': approved,
            'rejected': rejected,
            'signals_done': signals_done,
            'total_signals': total_signals,
            'marked_deletion': marked_deletion,
            'acceptance_rate': _safe_div(approved, approved + rejected) * 100,
            'extraction_success_rate': _safe_div(extracted, extracted + failed_extraction) * 100,
            'avg_signals_per_doc': _safe_div(total_signals, signals_done),
        },
        'by_source_type': [dict(r) for r in by_type],
        'rejection_reasons': [dict(r) for r in rejection_reasons],
        'funnel': [
            {'stage': 'Downloaded', 'count': total},
            {'stage': 'Extracted', 'count': extracted},
            {'stage': 'Screened', 'count': approved + rejected},
            {'stage': 'Approved', 'count': approved},
            {'stage': 'Signals Extracted', 'count': signals_done},
        ],
    }


def export_signal_stats(db) -> dict:
    """Signal type distribution, top entities, confidence, co-occurrence."""
    signal_types = db.query("""
        SELECT signal_type, COUNT(*) as count,
               AVG(confidence) as avg_confidence,
               MIN(confidence) as min_confidence,
               MAX(confidence) as max_confidence
        FROM signals GROUP BY signal_type ORDER BY count DESC
    """)

    top_entities = db.query("""
        SELECT entity, signal_type, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM signals
        WHERE entity IS NOT NULL AND entity != ''
        GROUP BY entity, signal_type
        ORDER BY count DESC
        LIMIT 80
    """)

    # Confidence distribution buckets
    confidence_dist = db.query("""
        SELECT
            CASE
                WHEN confidence >= 0.9 THEN '0.90-1.00'
                WHEN confidence >= 0.7 THEN '0.70-0.89'
                WHEN confidence >= 0.5 THEN '0.50-0.69'
                WHEN confidence >= 0.3 THEN '0.30-0.49'
                ELSE '0.00-0.29'
            END as bucket,
            COUNT(*) as count
        FROM signals
        GROUP BY bucket
        ORDER BY bucket
    """)

    # Signal density: avg signals per document, per 1000 words
    density_rows = db.query("""
        SELECT c.id, c.title, c.source_type, c.source_name,
               LENGTH(c.transcript) as transcript_len,
               COUNT(s.id) as signal_count
        FROM content c
        JOIN signals s ON s.source_content_id = c.id
        WHERE c.signal_processed = TRUE
        GROUP BY c.id, c.title, c.source_type, c.source_name, c.transcript
    """)
    density_data = []
    for r in density_rows:
        row = dict(r)
        tlen = row.get('transcript_len') or 0
        word_count = tlen / 5  # rough word estimate
        row['word_count'] = int(word_count)
        row['signal_density'] = round(row['signal_count'] / (word_count / 1000), 2) if word_count > 0 else 0
        density_data.append(row)

    # By industry breakdown
    industry_dist = db.query("""
        SELECT industry, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM signals
        WHERE industry IS NOT NULL AND industry != ''
        GROUP BY industry
        ORDER BY count DESC
        LIMIT 30
    """)

    # Top sources by signal yield
    source_yield = db.query("""
        SELECT c.source_name, c.source_type,
               COUNT(DISTINCT c.id) as doc_count,
               COUNT(s.id) as signal_count,
               AVG(s.confidence) as avg_confidence
        FROM content c
        JOIN signals s ON s.source_content_id = c.id
        WHERE c.source_name IS NOT NULL
        GROUP BY c.source_name, c.source_type
        ORDER BY signal_count DESC
        LIMIT 25
    """)

    # Entity co-occurrence (which entities appear together in same content)
    cooccurrence = db.query("""
        SELECT s1.entity as entity_a, s2.entity as entity_b, COUNT(*) as count
        FROM signals s1
        JOIN signals s2 ON s1.source_content_id = s2.source_content_id
            AND s1.entity < s2.entity
        WHERE s1.entity IS NOT NULL AND s1.entity != ''
          AND s2.entity IS NOT NULL AND s2.entity != ''
        GROUP BY s1.entity, s2.entity
        ORDER BY count DESC
        LIMIT 100
    """)

    # Impact level distribution
    impact_dist = db.query("""
        SELECT impact_level, COUNT(*) as count
        FROM signals
        WHERE impact_level IS NOT NULL AND impact_level != ''
        GROUP BY impact_level
        ORDER BY count DESC
    """)

    return {
        'exported_at': datetime.utcnow().isoformat(),
        'signal_types': [dict(r) for r in signal_types],
        'top_entities': [dict(r) for r in top_entities],
        'confidence_distribution': [dict(r) for r in confidence_dist],
        'signal_density': density_data,
        'industry_distribution': [dict(r) for r in industry_dist],
        'source_yield': [dict(r) for r in source_yield],
        'impact_distribution': [dict(r) for r in impact_dist],
        'entity_cooccurrence': [dict(r) for r in cooccurrence],
    }


def export_corpus_quality(db) -> dict:
    """Corpus composition, content quality metrics, extraction metadata."""
    # Content type breakdown
    type_breakdown = db.query("""
        SELECT content_type, source_type, COUNT(*) as count,
               AVG(file_size_mb) as avg_size_mb,
               SUM(CASE WHEN extraction_status = 'completed' THEN 1 ELSE 0 END) as extracted,
               SUM(CASE WHEN screening_status = 'approved' THEN 1 ELSE 0 END) as approved
        FROM content
        GROUP BY content_type, source_type
        ORDER BY count DESC
    """)

    # Extraction hardware usage
    hardware_usage = db.query("""
        SELECT extraction_hardware, transcription_model, COUNT(*) as count
        FROM content
        WHERE extraction_hardware IS NOT NULL AND extraction_hardware != ''
        GROUP BY extraction_hardware, transcription_model
        ORDER BY count DESC
    """)

    # Source name distribution (top publishers)
    top_sources = db.query("""
        SELECT source_name, source_type, COUNT(*) as count,
               SUM(CASE WHEN screening_status = 'approved' THEN 1 ELSE 0 END) as approved,
               SUM(CASE WHEN screening_status = 'rejected' THEN 1 ELSE 0 END) as rejected
        FROM content
        WHERE source_name IS NOT NULL
        GROUP BY source_name, source_type
        ORDER BY count DESC
        LIMIT 30
    """)

    # File size distribution
    size_dist = db.query("""
        SELECT
            CASE
                WHEN file_size_mb < 1 THEN '< 1 MB'
                WHEN file_size_mb < 5 THEN '1-5 MB'
                WHEN file_size_mb < 20 THEN '5-20 MB'
                WHEN file_size_mb < 50 THEN '20-50 MB'
                ELSE '50+ MB'
            END as size_bucket,
            COUNT(*) as count
        FROM content
        WHERE file_size_mb IS NOT NULL
        GROUP BY size_bucket
    """)

    # Language distribution
    lang_dist = db.query("""
        SELECT language, COUNT(*) as count
        FROM content
        WHERE language IS NOT NULL AND language != ''
        GROUP BY language
        ORDER BY count DESC
    """)

    # Corpus age distribution
    cutoff_7d = (datetime.utcnow() - timedelta(days=7)).isoformat()
    cutoff_30d = (datetime.utcnow() - timedelta(days=30)).isoformat()
    cutoff_90d = (datetime.utcnow() - timedelta(days=90)).isoformat()
    age_dist = db.query("""
        SELECT
            CASE
                WHEN created_at > ? THEN 'Last 7 days'
                WHEN created_at > ? THEN 'Last 30 days'
                WHEN created_at > ? THEN 'Last 90 days'
                ELSE 'Older'
            END as age_bucket,
            COUNT(*) as count
        FROM content
        GROUP BY age_bucket
    """, [cutoff_7d, cutoff_30d, cutoff_90d])

    return {
        'exported_at': datetime.utcnow().isoformat(),
        'type_breakdown': [dict(r) for r in type_breakdown],
        'hardware_usage': [dict(r) for r in hardware_usage],
        'top_sources': [dict(r) for r in top_sources],
        'size_distribution': [dict(r) for r in size_dist],
        'language_distribution': [dict(r) for r in lang_dist],
        'age_distribution': [dict(r) for r in age_dist],
    }


def export_temporal_data(db) -> dict:
    """Time series data for content and signals."""
    # Content over time (by day)
    content_daily = db.query("""
        SELECT CAST(created_at AS DATE) as date, source_type, COUNT(*) as count
        FROM content
        WHERE created_at IS NOT NULL
        GROUP BY CAST(created_at AS DATE), source_type
        ORDER BY date
    """)

    # Signals over time
    signals_daily = db.query("""
        SELECT CAST(extracted_at AS DATE) as date, signal_type, COUNT(*) as count
        FROM signals
        WHERE extracted_at IS NOT NULL
        GROUP BY CAST(extracted_at AS DATE), signal_type
        ORDER BY date
    """)

    # Acceptance/rejection over time
    screening_daily = db.query("""
        SELECT CAST(screened_at AS DATE) as date,
               SUM(CASE WHEN screening_status = 'approved' THEN 1 ELSE 0 END) as approved,
               SUM(CASE WHEN screening_status = 'rejected' THEN 1 ELSE 0 END) as rejected
        FROM content
        WHERE screened_at IS NOT NULL
        GROUP BY CAST(screened_at AS DATE)
        ORDER BY date
    """)

    # Cumulative content growth
    cumulative = db.query("""
        SELECT CAST(created_at AS DATE) as date, COUNT(*) as count
        FROM content
        WHERE created_at IS NOT NULL
        GROUP BY CAST(created_at AS DATE)
        ORDER BY date
    """)

    return {
        'exported_at': datetime.utcnow().isoformat(),
        'content_timeline': [dict(r) for r in content_daily],
        'signal_timeline': [dict(r) for r in signals_daily],
        'screening_timeline': [dict(r) for r in screening_daily],
        'cumulative_content': [dict(r) for r in cumulative],
    }


def export_signal_explorer(db) -> dict:
    """Individual signal rows for the interactive Signal Explorer table."""
    signals = db.query("""
        SELECT s.id, s.signal_type, s.entity, s.description,
               s.industry, s.impact_level, s.confidence, s.timeline,
               s.extracted_at, s.source_content_id,
               c.title as source_title, c.source_type, c.source_name,
               c.pub_date
        FROM signals s
        JOIN content c ON c.id = s.source_content_id
        ORDER BY s.extracted_at DESC
        LIMIT 500
    """)

    return {
        'exported_at': datetime.utcnow().isoformat(),
        'signals': [dict(r) for r in signals],
    }


def export_system_logs(db) -> dict:
    """Recent system logs for monitoring."""
    recent_logs = db.query("""
        SELECT timestamp, level, source, action, message, details_json, run_id
        FROM system_logs
        ORDER BY timestamp DESC
        LIMIT 200
    """)

    # Run summary (last 20 runs)
    runs = db.query("""
        SELECT run_id, MIN(timestamp) as started, MAX(timestamp) as ended,
               COUNT(*) as events,
               SUM(CASE WHEN level = 'ERROR' THEN 1 ELSE 0 END) as errors,
               SUM(CASE WHEN level = 'WARNING' THEN 1 ELSE 0 END) as warnings
        FROM system_logs
        WHERE run_id IS NOT NULL
        GROUP BY run_id
        ORDER BY started DESC
        LIMIT 20
    """)

    # Error summary (last 7 days)
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    error_summary = db.query("""
        SELECT source, action, COUNT(*) as count
        FROM system_logs
        WHERE level IN ('ERROR', 'CRITICAL')
          AND timestamp > ?
        GROUP BY source, action
        ORDER BY count DESC
    """, [week_ago])

    return {
        'exported_at': datetime.utcnow().isoformat(),
        'recent_logs': [dict(r) for r in recent_logs],
        'runs': [dict(r) for r in runs],
        'error_summary_7d': [dict(r) for r in error_summary],
    }


def export_discovery_stats(db) -> dict:
    """Discovery run statistics from system_logs."""
    discovery_runs = db.query("""
        SELECT timestamp, message, details_json
        FROM system_logs
        WHERE source = 'discovery' AND action = 'run_complete'
        ORDER BY timestamp DESC
        LIMIT 20
    """)

    ingestion_runs = db.query("""
        SELECT timestamp, message, details_json
        FROM system_logs
        WHERE source = 'ingestion' AND action IN ('discovery_complete', 'download_complete')
        ORDER BY timestamp DESC
        LIMIT 20
    """)

    # Parse details_json for aggregation
    runs = []
    for row in discovery_runs:
        details = json.loads(row['details_json']) if row.get('details_json') else {}
        runs.append({
            'timestamp': row['timestamp'],
            'queries': details.get('queries', 0),
            'found': details.get('raw', details.get('found', 0)),
            'approved': details.get('approved', 0),
            'rejected': details.get('rejected', 0),
            'deduped': details.get('deduped', 0),
        })

    return {
        'exported_at': datetime.utcnow().isoformat(),
        'discovery_runs': runs,
        'ingestion_logs': [dict(r) for r in ingestion_runs],
    }


def push_to_netlify(output_dir: str):
    """Deploy dashboard_data/ as a Netlify site (static JSON API)."""
    token = os.getenv('NETLIFY_AUTH_TOKEN')
    site_id = os.getenv('NETLIFY_SITE_ID')

    if not token or not site_id:
        print("WARNING: NETLIFY_AUTH_TOKEN and NETLIFY_SITE_ID required for Netlify push.")
        print("  Set them in .env.jetson or export them.")
        return False

    try:
        import subprocess
        # Use netlify-cli deploy
        cmd = [
            'npx', 'netlify-cli', 'deploy',
            '--dir', output_dir,
            '--site', site_id,
            '--auth', token,
            '--prod',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"Netlify deploy succeeded.")
            return True
        else:
            print(f"Netlify deploy failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("netlify-cli not found. Install: npm install -g netlify-cli")
        print("Alternative: use GitHub Pages or direct API upload.")
        return False
    except Exception as e:
        print(f"Netlify push failed: {e}")
        return False


def run_export(args):
    run_id = syslog.start_run('export')

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"EXPORT DASHBOARD DATA")
    print(f"{'=' * 60}")
    print(f"Output: {output_dir}")

    db = relationalDB(config.DB_PATH_ANALYTICS)

    # Export all datasets
    datasets = {
        'pipeline_stats.json': export_pipeline_stats,
        'signal_stats.json': export_signal_stats,
        'corpus_quality.json': export_corpus_quality,
        'temporal_data.json': export_temporal_data,
        'signal_explorer.json': export_signal_explorer,
        'system_logs.json': export_system_logs,
        'discovery_stats.json': export_discovery_stats,
    }

    for filename, export_fn in datasets.items():
        filepath = os.path.join(output_dir, filename)
        try:
            data = export_fn(db)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  Exported: {filename}")
        except Exception as e:
            print(f"  Failed: {filename} — {e}")
            syslog.error('export', 'export_failed', f'Failed to export {filename}: {e}')

    # Write an index.html so Netlify serves it properly
    index_path = os.path.join(output_dir, 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html><head><title>AI Industry Signals API</title></head>
<body>
<h1>AI Industry Signals — Dashboard Data API</h1>
<ul>
  <li><a href="pipeline_stats.json">Pipeline Stats</a></li>
  <li><a href="signal_stats.json">Signal Stats</a></li>
  <li><a href="temporal_data.json">Temporal Data</a></li>
  <li><a href="system_logs.json">System Logs</a></li>
  <li><a href="discovery_stats.json">Discovery Stats</a></li>
</ul>
<p>Last updated: <span id="ts"></span></p>
<script>
fetch('pipeline_stats.json').then(r=>r.json()).then(d=>{
  document.getElementById('ts').textContent = d.exported_at;
});
</script>
</body></html>""")

    # Push to Netlify if requested
    if args.push_netlify:
        print("\nPushing to Netlify...")
        success = push_to_netlify(output_dir)
        if success:
            syslog.info('export', 'netlify_push', 'Dashboard data pushed to Netlify')

    syslog.end_run('export', summary=f'Exported {len(datasets)} datasets to {output_dir}')

    print(f"\nExport complete. Files in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export summarized dashboard data to JSON files.",
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for JSON files (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--push-netlify', action='store_true',
        help='Deploy dashboard_data/ to Netlify after export',
    )
    args = parser.parse_args()

    try:
        run_export(args)
    except Exception as e:
        print(f"\nExport failed: {e}")
        syslog.error('export', 'failed', str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
