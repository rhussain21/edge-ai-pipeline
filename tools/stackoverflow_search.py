"""
Stack Overflow Search Adapter — Stack Exchange API v2.3.

Searches Stack Overflow for Q&A threads relevant to industrial automation,
PLC programming, SCADA, OPC UA, and related topics.

Unique content type: Q&A with votes, accepted answers, code snippets, and tags.

API docs: https://api.stackexchange.com/docs
Rate limits:
  - Without key: 300 requests/day per IP
  - With key (SO_API_KEY): 10,000 requests/day

Usage:
    from tools.stackoverflow_search import StackOverflowAdapter

    adapter = StackOverflowAdapter()
    results = adapter.search(query)  # SearchQuery object
"""

import logging
import os
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

from discovery.models import CandidateSource, SearchQuery

logger = logging.getLogger(__name__)

# Tags relevant to industrial automation on Stack Overflow
AUTOMATION_TAGS = [
    "plc", "scada", "opc-ua", "modbus", "opcua", "industrial",
    "ethercat", "profinet", "mqtt", "ladder-logic", "siemens",
    "allen-bradley", "beckhoff", "twincat", "codesys", "iec-61131",
    "robotics", "ros", "pid-controller", "embedded", "can-bus",
]


class StackOverflowProvider:
    """
    Stack Exchange API v2.3 provider for Stack Overflow.

    Searches questions by relevance with optional tag filtering.
    Responses are gzip-compressed by default.
    """

    BASE_URL = "https://api.stackexchange.com/2.3"

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Stack Exchange API key. Falls back to SO_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("SO_API_KEY")

    def search(self, query: str, limit: int = 10, tags: List[str] = None,
               min_score: int = 1, days_back: int = None) -> List[Dict[str, Any]]:
        """
        Search Stack Overflow questions.

        Args:
            query: Search string.
            limit: Max results.
            tags: Optional list of SO tags to filter by.
            min_score: Minimum question score (filters low-quality).
            days_back: If set, only return questions from the last N days.

        Returns:
            List of standardized result dicts.
        """
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": "stackoverflow",
            "pagesize": min(limit, 30),
            "filter": "withbody",  # Include question body text
        }
        if days_back:
            from datetime import timedelta
            fromdate = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
            params["fromdate"] = fromdate
        if self.api_key:
            params["key"] = self.api_key
        if tags:
            params["tagged"] = ";".join(tags[:5])  # API supports max 5 tags

        try:
            resp = requests.get(
                f"{self.BASE_URL}/search/advanced",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Stack Overflow API error: {e}")
            return []

        # Check quota
        quota_remaining = data.get("quota_remaining", "?")
        logger.debug(f"SO API quota remaining: {quota_remaining}")

        results = []
        for item in data.get("items", []):
            score = item.get("score", 0)
            if score < min_score:
                continue

            # Extract answer count and accepted status
            answer_count = item.get("answer_count", 0)
            is_answered = item.get("is_answered", False)

            # Build snippet from body (HTML) — strip tags for plain text
            body_html = item.get("body", "")
            snippet = self._strip_html(body_html)[:500]

            tags_list = item.get("tags", [])
            owner = item.get("owner", {})

            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": snippet,
                "score": score,
                "answer_count": answer_count,
                "is_answered": is_answered,
                "view_count": item.get("view_count", 0),
                "tags": tags_list,
                "author": owner.get("display_name", ""),
                "author_reputation": owner.get("reputation", 0),
                "published_date": datetime.utcfromtimestamp(
                    item.get("creation_date", 0)
                ).isoformat() if item.get("creation_date") else "",
                "last_activity": datetime.utcfromtimestamp(
                    item.get("last_activity_date", 0)
                ).isoformat() if item.get("last_activity_date") else "",
                "provider": "stackoverflow",
            })

        logger.info(f"Stack Overflow returned {len(results)} results for: '{query}'")
        return results

    def search_by_tags(self, tags: List[str], limit: int = 10,
                       sort: str = "votes") -> List[Dict[str, Any]]:
        """
        Browse top questions by tags (useful for discovering popular topics).

        Args:
            tags: List of SO tags.
            sort: One of: activity, votes, creation, hot, week, month.
        """
        params = {
            "order": "desc",
            "sort": sort,
            "tagged": ";".join(tags[:5]),
            "site": "stackoverflow",
            "pagesize": min(limit, 30),
            "filter": "withbody",
        }
        if self.api_key:
            params["key"] = self.api_key

        try:
            resp = requests.get(
                f"{self.BASE_URL}/questions",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Stack Overflow tag browse error: {e}")
            return []

        results = []
        for item in data.get("items", []):
            body_html = item.get("body", "")
            snippet = self._strip_html(body_html)[:500]
            owner = item.get("owner", {})

            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": snippet,
                "score": item.get("score", 0),
                "answer_count": item.get("answer_count", 0),
                "is_answered": item.get("is_answered", False),
                "view_count": item.get("view_count", 0),
                "tags": item.get("tags", []),
                "author": owner.get("display_name", ""),
                "published_date": datetime.utcfromtimestamp(
                    item.get("creation_date", 0)
                ).isoformat() if item.get("creation_date") else "",
                "provider": "stackoverflow",
            })

        return results

    @staticmethod
    def _strip_html(html: str) -> str:
        """Quick HTML tag stripping for snippets."""
        import re
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def is_available(self) -> bool:
        return True


# ── Discovery Adapter ────────────────────────────────────────────────────

class StackOverflowAdapter:
    """
    Stack Overflow discovery adapter for the discovery pipeline.

    Normalizes SO Q&A results into CandidateSource objects.
    Filters by automation-relevant tags when possible.
    """

    adapter_name = "stackoverflow"

    def __init__(self, api_key: str = None, relevant_tags: List[str] = None):
        self.provider = StackOverflowProvider(api_key=api_key)
        self.relevant_tags = relevant_tags or AUTOMATION_TAGS

    # Map common keywords to actual SO tags (SO tags are hyphenated, lowercase)
    KEYWORD_TO_SO_TAG = {
        "plc": "plc", "programmable logic controller": "plc",
        "ladder logic": "ladder-logic", "structured text": "structured-text",
        "iec 61131-3": "iec-61131", "iec 61131": "iec-61131",
        "scada": "scada", "hmi": "hmi",
        "opc ua": "opc-ua", "opcua": "opc-ua", "opc-ua": "opc-ua",
        "modbus": "modbus", "mqtt": "mqtt",
        "profinet": "profinet", "ethercat": "ethercat", "ethernet/ip": "ethernet-ip",
        "siemens": "siemens", "allen-bradley": "allen-bradley",
        "beckhoff": "beckhoff", "twincat": "twincat", "codesys": "codesys",
        "robotics": "robotics", "ros": "ros",
        "pid controller": "pid-controller", "pid control": "pid",
        "can bus": "can-bus", "can-bus": "can-bus",
        "industrial": "industrial", "embedded": "embedded",
        "servo drive": "servo", "vfd": "vfd",
        "cnc": "cnc", "stepper motor": "stepper",
        "digital twin": "digital-twin", "industry 4.0": "industry-4.0",
        "predictive maintenance": "predictive-maintenance",
        "anomaly detection": "anomaly-detection",
        "ot security": "scada", "iec 62443": "iec-62443",
    }

    def search(self, query: SearchQuery) -> List[CandidateSource]:
        """Search Stack Overflow and return normalized candidates."""
        # Map query tags/keywords to actual SO tags
        so_tags = set()
        for t in (query.tags or []):
            mapped = self.KEYWORD_TO_SO_TAG.get(t.lower())
            if mapped:
                so_tags.add(mapped)
            elif t.lower() in [rt.lower() for rt in self.relevant_tags]:
                so_tags.add(t.lower())
        so_tags = list(so_tags)[:5]

        results = self.provider.search(
            query=query.query,
            limit=query.limit,
            tags=so_tags if so_tags else None,
            min_score=0,  # PLC/automation is niche; many relevant Qs have 0 score
            days_back=getattr(query, 'days_back', None),
        )

        candidates = []
        for r in results:
            candidates.append(CandidateSource(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                source_type="qa_thread",
                publisher="Stack Overflow",
                discovered_at=datetime.utcnow().isoformat(),
                adapter=self.adapter_name,
                query_used=query.query,
                raw_metadata={
                    "provider": "stackoverflow",
                    "score": r.get("score", 0),
                    "answer_count": r.get("answer_count", 0),
                    "is_answered": r.get("is_answered", False),
                    "view_count": r.get("view_count", 0),
                    "tags": r.get("tags", []),
                    "author": r.get("author", ""),
                    "author_reputation": r.get("author_reputation", 0),
                    "published_date": r.get("published_date", ""),
                    "last_activity": r.get("last_activity", ""),
                },
            ))

        logger.info(f"SO adapter returned {len(candidates)} candidates for: '{query.query}'")
        return candidates

    def is_available(self) -> bool:
        return self.provider.is_available()
