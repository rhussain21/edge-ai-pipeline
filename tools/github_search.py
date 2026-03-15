"""
GitHub discovery adapter.

Searches GitHub for repositories, gists, and code related to industrial automation.
Uses the GitHub REST API (no auth required for basic search, but rate-limited).
Set GITHUB_TOKEN in .env for higher rate limits.
"""

import logging
import os
import requests
from datetime import datetime
from typing import List

from discovery.models import CandidateSource, SearchQuery

logger = logging.getLogger(__name__)

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"


class GitHubAdapter:
    """
    Discovers GitHub repositories relevant to industrial automation topics.

    Rate limits:
      - Unauthenticated: 10 requests/minute
      - Authenticated (GITHUB_TOKEN): 30 requests/minute
    """

    adapter_name = "github"

    def __init__(self, token: str = None, min_stars: int = 5):
        """
        Args:
            token: GitHub personal access token. Falls back to GITHUB_TOKEN env var.
            min_stars: Minimum stars filter to reduce noise.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.min_stars = min_stars
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
            logger.info("GitHubAdapter initialized with auth token")
        else:
            logger.info("GitHubAdapter initialized without auth (rate-limited)")

    def search(self, query: SearchQuery) -> List[CandidateSource]:
        """
        Search GitHub repositories and normalize results.

        Appends 'stars:>N' to the query to filter low-quality repos.
        """
        search_query = f"{query.query} stars:>={self.min_stars}"

        params = {
            "q": search_query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(query.limit, 30),  # GitHub max is 100
        }

        try:
            resp = requests.get(
                GITHUB_SEARCH_URL,
                params=params,
                headers=self.headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"GitHub search error: {e}")
            return []

        candidates = []
        for repo in data.get("items", []):
            candidates.append(CandidateSource(
                title=repo.get("full_name", ""),
                url=repo.get("html_url", ""),
                snippet=repo.get("description") or "",
                source_type="github_repo",
                publisher=repo.get("owner", {}).get("login", ""),
                discovered_at=datetime.utcnow().isoformat(),
                adapter=self.adapter_name,
                query_used=query.query,
                raw_metadata={
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "language": repo.get("language", ""),
                    "license": (repo.get("license") or {}).get("spdx_id", ""),
                    "topics": repo.get("topics", []),
                    "updated_at": repo.get("updated_at", ""),
                    "created_at": repo.get("created_at", ""),
                    "open_issues": repo.get("open_issues_count", 0),
                },
            ))

        logger.info(f"GitHub search returned {len(candidates)} repos for: '{query.query}'")
        return candidates

    def is_available(self) -> bool:
        """GitHub API is always available, but rate-limited without token."""
        return True
