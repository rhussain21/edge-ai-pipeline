"""
RSS discovery adapter.

Checks configured RSS feeds for new items and normalizes them into CandidateSource.
Integrates with your existing ContentSources class for podcast feeds,
and also supports generic RSS/Atom feeds for articles and blog posts.
"""

import logging
from datetime import datetime
from typing import List, Optional

import feedparser

from discovery.models import CandidateSource, SearchQuery

logger = logging.getLogger(__name__)


class RSSAdapter:
    """
    Discovers content from RSS/Atom feeds.

    Two modes:
      1. Configured feeds — iterate a list of feed URLs from rss_feeds.json
      2. Query-based — not typical for RSS, but can filter feed items by keyword
    """

    adapter_name = "rss"

    def __init__(self, feed_configs: List[dict] = None):
        """
        Args:
            feed_configs: List of feed config dicts from rss_feeds.json, each with:
                - url: RSS feed URL
                - name: Human-readable feed name
                - category: e.g. "podcast", "blog", "news"
                - publisher: e.g. "Siemens", "Automation World"
        """
        self.feed_configs = feed_configs or []

    def search(self, query: SearchQuery) -> List[CandidateSource]:
        """
        Fetch items from all configured feeds and optionally filter by query keywords.

        If query.query is non-empty, only return items whose title or summary
        contains one of the query terms (case-insensitive).
        """
        candidates = []
        keywords = query.query.lower().split() if query.query.strip() else []

        for feed_cfg in self.feed_configs:
            try:
                items = self._fetch_feed(feed_cfg, limit=query.limit)
                for item in items:
                    # Keyword filter if query is specified
                    if keywords:
                        text = f"{item.title} {item.snippet or ''}".lower()
                        if not any(kw in text for kw in keywords):
                            continue
                    item.query_used = query.query
                    candidates.append(item)
            except Exception as e:
                logger.error(f"RSS adapter error for {feed_cfg.get('url')}: {e}")

        logger.info(f"RSS adapter returned {len(candidates)} candidates")
        return candidates[:query.limit]

    def fetch_all_feeds(self, limit_per_feed: int = 20) -> List[CandidateSource]:
        """
        Fetch latest items from all configured feeds without keyword filtering.
        Useful for scheduled discovery runs.
        """
        candidates = []
        for feed_cfg in self.feed_configs:
            try:
                items = self._fetch_feed(feed_cfg, limit=limit_per_feed)
                candidates.extend(items)
            except Exception as e:
                logger.error(f"RSS fetch error for {feed_cfg.get('url')}: {e}")
        return candidates

    def _fetch_feed(self, feed_cfg: dict, limit: int = 20) -> List[CandidateSource]:
        """Parse a single RSS feed and return normalized candidates."""
        url = feed_cfg.get("url", "")
        name = feed_cfg.get("name", url)
        category = feed_cfg.get("category", "rss_article")
        publisher = feed_cfg.get("publisher")

        logger.debug(f"Fetching RSS feed: {name} ({url})")
        feed = feedparser.parse(url)

        if feed.bozo:
            if not feed.entries:
                logger.warning(f"Feed parse error for {url}: {feed.bozo_exception}")
                return []
            else:
                logger.debug(f"Feed has minor parse issues but extracted {len(feed.entries)} entries: {url}")

        candidates = []
        for entry in feed.entries[:limit]:
            link = entry.get("link", "")
            title = entry.get("title", "Untitled")

            # Build snippet from summary or description
            snippet = entry.get("summary") or entry.get("description") or ""
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            # Publication date
            pub_date = entry.get("published") or entry.get("updated") or ""

            candidates.append(CandidateSource(
                title=title,
                url=link,
                snippet=snippet,
                source_type=category,
                publisher=publisher or feed.feed.get("title", ""),
                discovered_at=datetime.utcnow().isoformat(),
                adapter=self.adapter_name,
                raw_metadata={
                    "feed_url": url,
                    "feed_name": name,
                    "pub_date": pub_date,
                    "author": entry.get("author", ""),
                    "tags": [t.get("term", "") for t in entry.get("tags", [])],
                },
            ))

        logger.debug(f"Parsed {len(candidates)} items from {name}")
        return candidates
