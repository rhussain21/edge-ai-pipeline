"""
Source Discovery — entry point and convenience wrapper.

Thin interface over the discovery/ module. Import and use this from your
main pipeline, scripts, or Streamlit dashboards.

Architecture:
    discovery/configs/*.json   → QueryPlanner    → SearchQuery objects
    SearchQuery objects        → Adapters (RSS / Web / GitHub) → CandidateSource objects
    CandidateSource objects    → Deduper          → unique CandidateSource objects
    unique CandidateSource     → CandidateClassifier (LLM) → ClassifiedCandidate objects
    ClassifiedCandidate        → DiscoveryResult  → approved list for ingestion

Usage:
    from source_discovery import SourceDiscovery

    sd = SourceDiscovery(llm_generate_fn=ollama.generate, db=my_relational_db)
    result = sd.discover(topic="PLC programming", adapters=["web", "rss"])
    for item in result.approved:
        print(item.candidate.title)
"""

import json
import logging
import os
from typing import List, Set, Optional, Callable, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from discovery.source_discovery_service import SourceDiscoveryService
from discovery.models import DiscoveryResult, ClassifiedCandidate

logger = logging.getLogger(__name__)


class SourceDiscovery:
    """
    High-level API for source discovery.

    Wraps SourceDiscoveryService and wires in your DB + LLM.

    Args:
        llm_generate_fn: Your OllamaClient.generate method.
        db: Your relationalDB instance (for fetching existing URLs to dedup against).
        config_dir: Path to discovery/configs/ (auto-detected if None).
    """

    def __init__(self, llm_generate_fn: Callable = None, db=None, config_dir: str = None):
        self.db = db
        self.llm_fn = llm_generate_fn

        # Fetch existing URLs from DB for dedup
        existing_urls = self._load_existing_urls()

        self.service = SourceDiscoveryService(
            llm_generate_fn=llm_generate_fn,
            existing_urls=existing_urls,
            config_dir=config_dir,
        )

    def _load_existing_urls(self) -> Set[str]:
        """Pull known URLs from DB so we don't re-discover existing content."""
        if not self.db:
            return set()
        try:
            rows = self.db.query("SELECT url FROM content WHERE url IS NOT NULL")
            urls = {row['url'] for row in rows if row.get('url')}
            logger.info(f"Loaded {len(urls)} existing URLs for dedup")
            return urls
        except Exception as e:
            logger.warning(f"Could not load existing URLs: {e}")
            return set()

    def discover(
        self,
        topic: str = None,
        adapters: List[str] = None,
        skip_classification: bool = False,
    ) -> DiscoveryResult:
        """
        Run source discovery.

        Args:
            topic: Optional topic filter (e.g., "PLC programming").
                   If None, discovers across all configured topics.
            adapters: Which adapters to use. Default: ["web", "github", "rss"]
            skip_classification: If True, skip LLM step (useful for testing).

        Returns:
            DiscoveryResult with .approved and .rejected lists.
        """
        result = self.service.run(
            adapters=adapters,
            topic_filter=topic,
            skip_classification=skip_classification,
        )
        self._log_summary(result)
        return result

    def discover_rss(self) -> DiscoveryResult:
        """Convenience: discover from RSS feeds only (no web search, no GitHub)."""
        return self.service.run_rss_only()

    def get_approved_records(self, result: DiscoveryResult) -> List[Dict[str, Any]]:
        """Convert approved candidates to dicts for DB insertion."""
        return self.service.get_approved_for_ingestion(result)

    def save_report(self, result: DiscoveryResult, path: str = "discovery_report.json"):
        """Save discovery run summary and approved items to JSON."""
        report = {
            "summary": result.summary(),
            "approved": [cc.to_dict() for cc in result.approved],
            "rejected_count": len(result.rejected),
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Discovery report saved to {path}")

    @staticmethod
    def _log_summary(result: DiscoveryResult):
        s = result.summary()
        logger.info("=" * 50)
        logger.info(f"Discovery Run: {s['run_id']}")
        logger.info(f"  Queries:    {s['queries_generated']}")
        logger.info(f"  Found:      {s['candidates_found']}")
        logger.info(f"  Deduped:    {s['candidates_deduped']}")
        logger.info(f"  Approved:   {s['candidates_approved']}")
        logger.info(f"  Rejected:   {s['candidates_rejected']}")
        if s['errors']:
            logger.warning(f"  Errors:     {len(s['errors'])}")
        logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Example: run RSS-only discovery without LLM classification
    sd = SourceDiscovery()
    result = sd.discover(adapters=["rss"], skip_classification=True)

    print(f"\nFound {len(result.approved)} sources:")
    for cc in result.approved[:10]:
        c = cc.candidate
        print(f"  [{c.source_type}] {c.title[:70]}")
        print(f"    URL: {c.url}")
        print()

    sd.save_report(result)
