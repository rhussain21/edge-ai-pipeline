"""
Source Discovery Service — top-level orchestrator.

Pipeline:
    1. Load JSON configs via QueryPlanner
    2. Generate search queries
    3. Dispatch queries to adapters (RSS, Web, GitHub)
    4. Collect and normalize candidates
    5. Deduplicate against each other and existing DB records
    6. Batch classify with LLM
    7. Return approved candidates for ingestion

Usage:
    from discovery import SourceDiscoveryService

    service = SourceDiscoveryService(
        llm_generate_fn=ollama_client.generate,
        existing_urls=set_of_known_urls,
    )
    result = service.run(topic_filter="PLC programming")
    for item in result.approved:
        print(item.candidate.title, item.classification.doc_type)
"""

import logging
import uuid
from datetime import datetime
from typing import List, Set, Optional, Callable, Dict, Any

from discovery.models import (
    SearchQuery, CandidateSource, ClassifiedCandidate, DiscoveryResult
)
from discovery.query_planner import QueryPlanner
from discovery.candidate_classifier import CandidateClassifier
from discovery.deduper import Deduper
from discovery.cache import TaskCache
from tools.rss_reader import RSSAdapter
from tools.web_search import WebSearchAdapter
from tools.github_search import GitHubAdapter

logger = logging.getLogger(__name__)


class SourceDiscoveryService:
    """
    Orchestrates the full source discovery pipeline.

    Wiring guide — connect your existing components:
        - llm_generate_fn: your OllamaClient.generate method
        - existing_urls: query your DB for known URLs before each run
        - db: pass your relationalDB for persisting approved candidates

    Example:
        from llm_client import OllamaClient
        from db_relational import relationalDB

        ollama = OllamaClient(model="llama3:latest")
        db = relationalDB("Database/industry_signals.db")

        # Get existing URLs to avoid re-discovering known content
        existing = {row['url'] for row in db.query("SELECT url FROM content WHERE url IS NOT NULL")}

        service = SourceDiscoveryService(
            llm_generate_fn=ollama.generate,
            existing_urls=existing,
        )
        result = service.run(adapters=["web", "github", "rss"])
    """

    def __init__(
        self,
        llm_generate_fn: Callable = None,
        existing_urls: Set[str] = None,
        config_dir: str = None,
        cache_db_path: str = None,
        classifier_batch_size: int = 10,
        classifier_temperature: float = 0.1,
        github_min_stars: int = 5,
    ):
        """
        Args:
            llm_generate_fn: Your LLM generate function.
                             Signature: generate(prompt, system_prompt, temperature) -> str
                             Pass OllamaClient().generate
            existing_urls: Set of URLs already in your content DB (for dedup).
            config_dir: Path to discovery/configs/ JSON files.
            cache_db_path: Path to SQLite cache DB for LLM outputs.
            classifier_batch_size: Max candidates per LLM batch call.
            classifier_temperature: LLM temperature for classification.
            github_min_stars: Minimum stars for GitHub repo search.
        """
        # Query planner (loads JSON configs)
        self.planner = QueryPlanner(config_dir=config_dir)

        # Adapters
        self.adapters = {
            "rss": RSSAdapter(feed_configs=self.planner.get_rss_feeds()),
            "web": WebSearchAdapter(),
            "github": GitHubAdapter(min_stars=github_min_stars),
        }

        # Deduper
        self.deduper = Deduper(existing_urls=existing_urls)

        # Classifier (optional — skip if no LLM provided)
        self.classifier = None
        if llm_generate_fn:
            cache = TaskCache(db_path=cache_db_path) if cache_db_path else TaskCache()
            self.classifier = CandidateClassifier(
                llm_generate_fn=llm_generate_fn,
                cache=cache,
                batch_size=classifier_batch_size,
                temperature=classifier_temperature,
            )

    def run(
        self,
        adapters: List[str] = None,
        topic_filter: str = None,
        skip_classification: bool = False,
        rss_only: bool = False,
    ) -> DiscoveryResult:
        """
        Execute a full discovery run.

        Args:
            adapters: Which adapters to use. Default: ["web", "github", "rss"]
            topic_filter: Optional topic string to narrow query generation.
            skip_classification: If True, skip LLM classification (useful for testing).
            rss_only: Shortcut to only run RSS feeds (no query planning needed).

        Returns:
            DiscoveryResult with approved and rejected candidates.
        """
        if adapters is None:
            adapters = ["web", "github", "rss"]

        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow().isoformat()

        result = DiscoveryResult(
            run_id=run_id,
            started_at=started_at,
        )

        logger.info(f"Discovery run {run_id} started — adapters: {adapters}, topic: {topic_filter}")

        # ── Step 1: Generate queries ──
        if rss_only:
            queries = []
        else:
            queries = self.planner.plan_queries(adapters=adapters, topic_filter=topic_filter)
        result.queries_generated = len(queries)
        logger.info(f"Generated {len(queries)} search queries")

        # ── Step 2: Execute searches ──
        all_candidates: List[CandidateSource] = []

        # RSS: always fetch all feeds (keyword filtering happens inside adapter)
        if "rss" in adapters:
            rss_adapter = self.adapters.get("rss")
            if rss_adapter:
                try:
                    # For RSS, also run any RSS-specific queries from the planner
                    rss_queries = [q for q in queries if q.adapter == "rss"]
                    if rss_queries:
                        for q in rss_queries:
                            all_candidates.extend(rss_adapter.search(q))
                    else:
                        # No specific queries — just fetch all configured feeds
                        all_candidates.extend(rss_adapter.fetch_all_feeds())
                except Exception as e:
                    error_msg = f"RSS adapter error: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

        # Web + GitHub: dispatch queries to their respective adapters
        for query in queries:
            if query.adapter == "rss":
                continue  # Already handled above

            adapter = self.adapters.get(query.adapter)
            if not adapter:
                logger.warning(f"No adapter registered for: {query.adapter}")
                continue

            try:
                candidates = adapter.search(query)
                all_candidates.extend(candidates)
            except Exception as e:
                error_msg = f"Adapter {query.adapter} error for '{query.query}': {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        result.candidates_found = len(all_candidates)
        logger.info(f"Collected {len(all_candidates)} raw candidates")

        # ── Step 3: Deduplicate ──
        unique_candidates = self.deduper.deduplicate(all_candidates)
        result.candidates_deduped = len(all_candidates) - len(unique_candidates)
        logger.info(f"After dedup: {len(unique_candidates)} unique candidates")

        # ── Step 4: Classify ──
        if skip_classification or not self.classifier:
            if not self.classifier:
                logger.warning("No LLM configured — skipping classification. All candidates approved.")
            # Without classification, approve everything
            for c in unique_candidates:
                result.approved.append(ClassifiedCandidate(
                    candidate=c,
                    classification=__import__('discovery.models', fromlist=['Classification']).Classification(
                        relevant=True, confidence=0.0, reason="No LLM — auto-approved"
                    ),
                ))
            result.candidates_approved = len(unique_candidates)
        else:
            classified = self.classifier.classify_batch(unique_candidates)
            for cc in classified:
                if cc.approved:
                    result.approved.append(cc)
                else:
                    result.rejected.append(cc)
            result.candidates_approved = len(result.approved)
            result.candidates_rejected = len(result.rejected)

        result.completed_at = datetime.utcnow().isoformat()

        logger.info(
            f"Discovery run {run_id} complete: "
            f"{result.candidates_approved} approved, "
            f"{result.candidates_rejected} rejected, "
            f"{result.candidates_deduped} deduped"
        )

        return result

    def run_rss_only(self, limit_per_feed: int = 20) -> DiscoveryResult:
        """Convenience: run only RSS feed discovery."""
        return self.run(adapters=["rss"], rss_only=True)

    def get_approved_for_ingestion(self, result: DiscoveryResult) -> List[Dict[str, Any]]:
        """
        Convert approved candidates to dicts ready for DB insertion.

        Returns records compatible with your content table schema.
        Map these to your contentETL or relationalDB.upsert_records().

        TODO: Wire this to your DB layer. Example:
            records = service.get_approved_for_ingestion(result)
            for record in records:
                db.execute("INSERT INTO content_queue ...", record)
        """
        records = []
        for cc in result.approved:
            c = cc.candidate
            cl = cc.classification
            records.append({
                "title": c.title,
                "url": c.url,
                "source_type": c.source_type,
                "publisher": c.publisher,
                "snippet": c.snippet,
                "doc_type": cl.doc_type,
                "topic_tags": cl.topic_tags,
                "authority": cl.authority_guess,
                "confidence": cl.confidence,
                "reason": cl.reason,
                "adapter": c.adapter,
                "query_used": c.query_used,
                "discovered_at": c.discovered_at,
            })
        return records
