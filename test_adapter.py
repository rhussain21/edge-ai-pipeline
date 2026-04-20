#!/usr/bin/env python3
"""
Adapter test script. Usage:
   
    python3 test_adapter.py rss
    python3 test_adapter.py web
    python3 test_adapter.py github
    python3 test_adapter.py academic
    python3 test_adapter.py institution
    python3 test_adapter.py stackoverflow
"""

import sys
import os
from unittest.case import skip
from pprint import pprint

# Set path FIRST before any local imports
sys.path.insert(0, os.path.dirname(__file__))

# Device-aware config — loads .env.jetson or .env.mac automatically
from device_config import config
from discovery.source_discovery_service import SourceDiscoveryService
from etl.sources import ContentSources

from tools.rss_reader import RSSAdapter
from tools.web_search import WebSearchAdapter
from tools.github_search import GitHubAdapter
from tools.academic_search import AcademicSearchAdapter
from tools.institution_search import InstitutionSearchAdapter
from tools.stackoverflow_search import StackOverflowAdapter
from discovery.query_planner import QueryPlanner
from discovery.cache import SourceHealthTracker
from llm_client import OllamaClient, GeminiClient
from db_relational import relationalDB

def get_adapter(adapter_type: str, planner: QueryPlanner):
    if adapter_type == "rss":
        return RSSAdapter(feed_configs=planner.get_rss_feeds())
    elif adapter_type == "web":
        return WebSearchAdapter()
    elif adapter_type == "github":
        return GitHubAdapter(min_stars=5)
    elif adapter_type == "academic":
        return AcademicSearchAdapter()
    elif adapter_type == "institution":
        return InstitutionSearchAdapter()
    elif adapter_type == "stackoverflow":
        return StackOverflowAdapter()
    else:
        raise ValueError(f"Unknown adapter: {adapter_type}. Choose: rss, web, github, academic, institution, stackoverflow")


def test_adapter(adapter_type: str = "web"):
    planner = QueryPlanner('discovery/configs')
    adapter = get_adapter(adapter_type, planner)

    print(f"=== Testing {adapter_type.upper()} adapter ===\n")

    if adapter_type == "rss":
        candidates = adapter.fetch_all_feeds(limit_per_feed=3)
    else:
        queries = planner.plan_queries(adapters=[adapter_type], topic_filter="PLC")[:3]
        print(f"Running {len(queries)} queries...\n")
        candidates = []
        for q in queries:
            results = adapter.search(q)
            print(f"  Query: '{q.query}' → {len(results)} results")
            candidates.extend(results)

    print(f"\nTotal candidates: {len(candidates)}\n")
    for i, c in enumerate(candidates[:10], 1):
        print(f"{i}. [{c.source_type}] {c.title}")
        print(f"   Publisher: {c.publisher}")
        print(f"   URL: {c.url}")
        print(f"   Snippet: {(c.snippet or '')[:100]}...")
        print()


if __name__ == "__main__":
    #adapter_type = sys.argv[1] if len(sys.argv) > 1 else "rss"
    #test_adapter(adapter_type)

    # LLM client: Mac uses Gemini API, Jetson uses local Ollama
    if config.LLM_PROVIDER == 'gemini':
        llm_client = GeminiClient(model=config.LLM_MODEL)
    else:
        llm_client = OllamaClient(model=config.LLM_MODEL, base_url=config.LLM_URL)
    config_dir = 'discovery/configs'
    sd = SourceDiscoveryService(llm_generate_fn=llm_client.generate, config_dir=config_dir)
    
    # Limit web queries to preserve Tavily quota (1000/month free tier)
    # max_web_queries only caps web adapter, not academic/stackoverflow/etc
    adapter_type = sys.argv[1] if len(sys.argv) > 1 else "rss"
    results = sd.run(
        adapters=[adapter_type], 
        topic_filter="LLM manufacturing", 
        max_web_queries=5,  # Only limits web adapter (Tavily)
        skip_classification=True
    )


    approved = sd.get_approved_for_ingestion(results)

    pprint(approved)
    
    health_tracker = SourceHealthTracker()
    db = relationalDB(config.DB_PATH)
    content_downloader = ContentSources(config.MEDIA_DIR, db=db, health_tracker=health_tracker)
    
    # Unified download — routes podcasts to RSS matching, web content to direct URL download
    content_downloader.download_approved(approved)

    print("\n=== SOURCE HEALTH REPORT ===")
    flagged = health_tracker.get_flagged_sources()
    if flagged:
        print(f"Flagged sources ({len(flagged)}):")
        for s in flagged:
            print(f"  [{s['adapter']}] {s['source_url']}")
            print(f"    Downloads: {s['downloads']}")
            print(f"    Discovery: {s['discovery']}")
    else:
        print("No sources flagged yet (more runs needed to accumulate history)")

    # print(f"\n=== APPROVED ({len(results.approved)}) ===")
    # for i, cc in enumerate(results.approved[:10], 1):
    #     print(f"{i}. [{cc.candidate.source_type}] {cc.candidate.title}")
    #     print(f"   Publisher: {cc.candidate.publisher}")
    #     print(f"   doc_type: {cc.classification.doc_type} | confidence: {cc.classification.confidence:.2f}")
    #     print(f"   Reason: {cc.classification.reason}")
    #     print()
    
    # print(f"\n=== REJECTED ({len(results.rejected)}) ===")
    # for i, cc in enumerate(results.rejected[:5], 1):
    #     print(f"{i}. [{cc.candidate.source_type}] {cc.candidate.title}")
    #     print(f"   Publisher: {cc.candidate.publisher}")
    #     print(f"   URL: {cc.candidate.url}")
    #     print(f"   Reason: {cc.classification.reason}")
    #     print()
