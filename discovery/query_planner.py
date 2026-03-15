"""
Query Planner — deterministic query generation from JSON config files.

Loads domain knowledge (vendors, standards, topics, search patterns, etc.)
and combinatorially generates targeted search queries for each adapter.

No LLM needed here. Queries are explicit and reproducible.
Optional: add lightweight LLM query expansion later via expand_queries().
"""

import json
import os
import logging
from typing import List, Dict, Any
from itertools import product

from discovery.models import SearchQuery

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


class QueryPlanner:
    """
    Generates search queries from structured JSON configs.

    Config files (in discovery/configs/):
        - vendors.json       — automation vendors and their product lines
        - standards.json     — IEC, ISO, NIST standards
        - topics.json        — core domain topics (PLC, SCADA, robotics, etc.)
        - industry_orgs.json — standards bodies, trade orgs
        - doc_types.json     — preferred document types
        - search_patterns.json — query templates with {vendor}, {topic} slots
        - rss_feeds.json     — configured RSS feed URLs

    The planner combines these configs to produce SearchQuery objects
    targeting specific adapters.
    """

    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or CONFIG_DIR
        self.configs: Dict[str, Any] = {}
        self._load_configs()

    def _load_configs(self):
        """Load all JSON config files from the config directory."""
        if not os.path.isdir(self.config_dir):
            logger.warning(f"Config directory not found: {self.config_dir}")
            return

        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                key = filename.replace(".json", "")
                filepath = os.path.join(self.config_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        self.configs[key] = json.load(f)
                    logger.debug(f"Loaded config: {key} ({filepath})")
                except Exception as e:
                    logger.error(f"Failed to load config {filepath}: {e}")

        logger.info(f"Loaded {len(self.configs)} config files: {list(self.configs.keys())}")

    def get_config(self, name: str) -> Any:
        """Get a specific config by name (without .json extension)."""
        return self.configs.get(name, {})

    def plan_queries(self, adapters: List[str] = None, topic_filter: str = None) -> List[SearchQuery]:
        """
        Generate all search queries for the given adapters.

        Args:
            adapters: List of adapter names to generate queries for.
                      Default: ["web", "github"]
            topic_filter: Optional topic string to narrow query generation.

        Returns:
            List of SearchQuery objects ready to dispatch to adapters.
        """
        if adapters is None:
            adapters = ["web", "github"]

        queries = []

        if "web" in adapters:
            queries.extend(self._plan_web_queries(topic_filter))

        if "github" in adapters:
            queries.extend(self._plan_github_queries(topic_filter))

        if "rss" in adapters:
            # RSS doesn't use generated queries — it uses configured feeds directly.
            # But we can still generate keyword-filter queries for RSS.
            queries.extend(self._plan_rss_queries(topic_filter))

        logger.info(f"QueryPlanner generated {len(queries)} queries for adapters: {adapters}")
        return queries

    def _plan_web_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate web search queries from patterns × vendors × topics."""
        patterns = self.get_config("search_patterns").get("web_patterns", [])
        vendors = self.get_config("vendors").get("vendors", [])
        topics = self.get_config("topics").get("topics", [])
        doc_types = self.get_config("doc_types").get("doc_types", [])
        standards = self.get_config("standards").get("standards", [])

        # Apply topic filter
        if topic_filter:
            topics = [t for t in topics if topic_filter.lower() in t.get("name", "").lower()
                      or topic_filter.lower() in " ".join(t.get("keywords", [])).lower()]

        queries = []

        # Pattern-based queries: fill {vendor}, {topic}, {doc_type} slots
        for pattern in patterns:
            template = pattern.get("template", "")
            tags = pattern.get("tags", [])

            # Vendor × Topic combinations
            for vendor in vendors:
                vendor_name = vendor.get("name", "")
                for topic in topics:
                    topic_name = topic.get("name", "")
                    query_str = (
                        template
                        .replace("{vendor}", vendor_name)
                        .replace("{topic}", topic_name)
                    )
                    # Only add if we actually filled something
                    if query_str != template:
                        queries.append(SearchQuery(
                            query=query_str.strip(),
                            adapter="web",
                            source_config=f"pattern/{pattern.get('name', 'unknown')}",
                            tags=tags + topic.get("keywords", [])[:3],
                            limit=10,
                        ))

            # Doc type queries: "{topic} {doc_type}"
            for topic in topics:
                for dt in doc_types:
                    dt_name = dt.get("name", "")
                    query_str = f"{topic.get('name', '')} {dt_name}"
                    queries.append(SearchQuery(
                        query=query_str.strip(),
                        adapter="web",
                        source_config=f"topic_doctype/{topic.get('name', '')}",
                        tags=[dt_name] + topic.get("keywords", [])[:2],
                        limit=5,
                    ))

        # Standards queries
        for std in standards:
            std_name = std.get("name", "")
            std_id = std.get("id", "")
            query_str = f"{std_id} {std_name} specification document"
            queries.append(SearchQuery(
                query=query_str.strip(),
                adapter="web",
                source_config=f"standards/{std_id}",
                tags=["standards", std.get("category", "")],
                limit=5,
            ))

        return queries

    def _plan_github_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate GitHub search queries from topics and vendors."""
        topics = self.get_config("topics").get("topics", [])
        vendors = self.get_config("vendors").get("vendors", [])

        if topic_filter:
            topics = [t for t in topics if topic_filter.lower() in t.get("name", "").lower()
                      or topic_filter.lower() in " ".join(t.get("keywords", [])).lower()]

        queries = []

        # Topic-based GitHub searches
        for topic in topics:
            for kw in topic.get("github_keywords", topic.get("keywords", []))[:3]:
                queries.append(SearchQuery(
                    query=kw,
                    adapter="github",
                    source_config=f"topics/{topic.get('name', '')}",
                    tags=topic.get("keywords", [])[:3],
                    limit=10,
                ))

        # Vendor GitHub repos
        for vendor in vendors:
            github_org = vendor.get("github_org")
            if github_org:
                queries.append(SearchQuery(
                    query=f"org:{github_org}",
                    adapter="github",
                    source_config=f"vendors/{vendor.get('name', '')}",
                    tags=["vendor", vendor.get("name", "")],
                    limit=10,
                ))

        return queries

    def _plan_rss_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate RSS keyword-filter queries from topics."""
        topics = self.get_config("topics").get("topics", [])

        if topic_filter:
            topics = [t for t in topics if topic_filter.lower() in t.get("name", "").lower()
                      or topic_filter.lower() in " ".join(t.get("keywords", [])).lower()]

        queries = []
        for topic in topics:
            queries.append(SearchQuery(
                query=topic.get("name", ""),
                adapter="rss",
                source_config=f"topics/{topic.get('name', '')}",
                tags=topic.get("keywords", [])[:3],
                limit=20,
            ))

        return queries

    def get_rss_feeds(self) -> List[dict]:
        """Return the configured RSS feed list for the RSSAdapter."""
        return self.get_config("rss_feeds").get("feeds", [])
