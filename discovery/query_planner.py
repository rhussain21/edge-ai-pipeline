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

    def _filter_topics(self, topic_filter: str = None) -> List[Dict[str, Any]]:
        """Filter topics by topic_filter. If no match, synthesize a topic from the filter string.
        
        This ensures every adapter gets at least some queries even for ad-hoc
        filter terms like 'Manufacturing AI' that aren't in topics.json.
        """
        topics = self.get_config("topics").get("topics", [])
        if not topic_filter:
            return topics

        filtered = [t for t in topics if topic_filter.lower() in t.get("name", "").lower()
                    or topic_filter.lower() in " ".join(t.get("keywords", [])).lower()]

        if filtered:
            return filtered

        # No topic matched — synthesize one from the filter string
        logger.info(f"topic_filter '{topic_filter}' matched no configured topic — using as direct query")
        return [{
            "name": topic_filter,
            "keywords": [topic_filter],
            "github_keywords": [topic_filter.lower().replace(' ', '-')],
        }]

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

        if "academic" in adapters:
            queries.extend(self._plan_academic_queries(topic_filter))

        if "institution" in adapters:
            queries.extend(self._plan_institution_queries(topic_filter))

        if "stackoverflow" in adapters:
            queries.extend(self._plan_stackoverflow_queries(topic_filter))

        # Site-scoped web searches (trade pubs, consulting, communities)
        if "web" in adapters:
            queries.extend(self._plan_site_scoped_queries(topic_filter))

        logger.info(f"QueryPlanner generated {len(queries)} queries for adapters: {adapters}")
        return queries

    def _plan_web_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate web search queries from patterns × vendors × topics."""
        patterns = self.get_config("search_patterns").get("web_patterns", [])
        vendors = self.get_config("vendors").get("vendors", [])
        topics = self._filter_topics(topic_filter)
        doc_types = self.get_config("doc_types").get("doc_types", [])
        standards = self.get_config("standards").get("standards", [])

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
        topics = self._filter_topics(topic_filter)
        vendors = self.get_config("vendors").get("vendors", [])

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
        topics = self._filter_topics(topic_filter)

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

    def _plan_academic_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate academic search queries from topics, standards, and academic_sources config.
        
        Strategy: short keyword queries work best for arXiv's API.
        Generates 3 types of queries per matched topic:
          1. Topic name (e.g., "PLC Programming")
          2. Individual high-value keywords (e.g., "ladder logic", "structured text")
          3. Related standards by ID only (e.g., "IEC 61131")
        """
        topics = self._filter_topics(topic_filter)
        standards = self.get_config("standards").get("standards", [])

        queries = []

        for topic in topics:
            topic_name = topic.get("name", "")
            keywords = topic.get("keywords", [])

            # Query 1: Topic name directly
            if topic_name:
                queries.append(SearchQuery(
                    query=topic_name,
                    adapter="academic",
                    source_config=f"academic/topic/{topic_name}",
                    tags=keywords[:3] + ["academic"],
                    limit=10,
                ))

            # Query 2-3: Individual keywords (short, specific terms)
            for kw in keywords[:3]:
                queries.append(SearchQuery(
                    query=kw,
                    adapter="academic",
                    source_config=f"academic/keyword/{kw}",
                    tags=[kw, "academic"],
                    limit=5,
                ))

        # Standards queries — only standards matching the topic filter's category
        matched_categories = set()
        for topic in topics:
            topic_name = topic.get("name", "").lower()
            # Map topic names to standard categories
            if "plc" in topic_name:
                matched_categories.add("plc")
            if "safety" in topic_name:
                matched_categories.update(["safety"])
            if "network" in topic_name:
                matched_categories.add("networking")
            if "cyber" in topic_name or "security" in topic_name:
                matched_categories.add("cybersecurity")
            if "robot" in topic_name:
                matched_categories.add("robotics")
            if "scada" in topic_name or "hmi" in topic_name:
                matched_categories.add("scada")
            if "process" in topic_name or "batch" in topic_name:
                matched_categories.add("process")
            if "manufacturing" in topic_name or "mes" in topic_name:
                matched_categories.add("mes")

        for std in standards:
            cat = std.get("category", "")
            # If no topic filter, include all standards; otherwise filter by category
            if topic_filter and cat not in matched_categories:
                continue
            std_id = std.get("id", "")
            queries.append(SearchQuery(
                query=std_id,
                adapter="academic",
                source_config=f"academic/standards/{std_id}",
                tags=["standards", "academic", cat],
                limit=5,
            ))

        return queries

    def _plan_institution_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate institution search queries from topics and institutions config.
        
        Strategy:
          - Template queries use topic name (e.g. "NIST PLC Programming manufacturing")
          - Program/directorate queries are paired with topics for relevance
          - Skip templates with unfilled placeholders
        """
        topics = self._filter_topics(topic_filter)
        inst_cfg = self.get_config("institutions").get("institutions", {})

        queries = []
        seen = set()

        for inst_name, inst_data in inst_cfg.items():
            # Template-based queries
            for template in inst_data.get("search_templates", []):
                for topic in topics:
                    query_str = template.replace("{topic}", topic.get("name", ""))
                    # Skip if unfilled placeholders remain
                    if "{" in query_str or "}" in query_str:
                        continue
                    if query_str == template:
                        continue
                    q = query_str.strip()
                    if q in seen:
                        continue
                    seen.add(q)
                    queries.append(SearchQuery(
                        query=q,
                        adapter="institution",
                        source_config=f"institution/{inst_name}/{topic.get('name', '')}",
                        tags=topic.get("keywords", [])[:3] + ["institution"],
                        limit=5,
                    ))

            # Topic-specific queries for each program/directorate
            # Pairs program with topic for relevance (instead of generic "manufacturing automation")
            for program in inst_data.get("programs", []) + inst_data.get("directorates", []):
                for topic in topics:
                    q = f"{program} {topic.get('name', '')}".strip()
                    if q in seen:
                        continue
                    seen.add(q)
                    queries.append(SearchQuery(
                        query=q,
                        adapter="institution",
                        source_config=f"institution/{inst_name}/{program}",
                        tags=["institution", inst_name],
                        limit=5,
                    ))

        return queries

    def _plan_stackoverflow_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate Stack Overflow search queries from topics.
        
        Uses shorter, tag-friendly queries for better SO API results.
        Prefers 1-2 word queries over long keyword concatenations.
        """
        topics = self._filter_topics(topic_filter)

        queries = []
        for topic in topics:
            # Use topic name as primary query (shorter, cleaner)
            topic_name = topic.get("name", "")
            keywords = topic.get("keywords", [])
            
            if topic_name:
                # Primary query: just the topic name
                queries.append(SearchQuery(
                    query=topic_name,
                    adapter="stackoverflow",
                    source_config=f"stackoverflow/{topic_name}",
                    tags=keywords[:3],  # Tags for filtering
                    limit=10,
                ))
            
            # Secondary queries: individual high-value keywords
            for kw in keywords[:2]:  # Only first 2 keywords to avoid spam
                if len(kw.split()) <= 2:  # Prefer short keywords
                    queries.append(SearchQuery(
                        query=kw,
                        adapter="stackoverflow",
                        source_config=f"stackoverflow/{topic_name}/{kw}",
                        tags=[kw] + keywords[:2],
                        limit=5,
                    ))

        return queries

    def _plan_site_scoped_queries(self, topic_filter: str = None) -> List[SearchQuery]:
        """Generate site-scoped web searches from web_search_sites.json.

        These queries use Tavily's include_domains to target specific
        trade publications, consulting firms, and communities.
        Time-filtered to last 7 days to save API tokens on weekly runs.
        """
        site_cfg = self.get_config("web_search_sites").get("site_searches", [])
        topics = self._filter_topics(topic_filter)

        queries = []
        for site in site_cfg:
            domain = site.get("domain", "")
            label = site.get("label", domain)
            if not domain:
                continue

            for topic in topics[:3]:  # Limit combinations to save tokens
                query = SearchQuery(
                    query=f"{topic.get('name', '')} manufacturing automation",
                    adapter="web",
                    source_config=f"site_search/{label}",
                    tags=[site.get("category", ""), topic.get("name", "")],
                    limit=5,
                )
                # Attach extra attributes for WebSearchAdapter to pick up
                query.days = 7
                query.include_domains = [domain]
                queries.append(query)

        return queries

    def get_rss_feeds(self) -> List[dict]:
        """Return the configured RSS feed list for the RSSAdapter."""
        return self.get_config("rss_feeds").get("feeds", [])
