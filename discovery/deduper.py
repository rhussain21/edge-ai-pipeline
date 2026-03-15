"""
Deduplication for discovery candidates.

Handles:
  1. URL normalization (strip tracking params, trailing slashes, etc.)
  2. Exact URL dedup
  3. Title similarity dedup (fuzzy matching)
  4. Placeholder for future content-hash dedup against existing DB records
"""

import logging
import re
from typing import List, Set, Optional
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from discovery.models import CandidateSource

logger = logging.getLogger(__name__)

# URL query params to strip (tracking, analytics, etc.)
STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "ref", "fbclid", "gclid", "mc_cid", "mc_eid", "source", "sxsrf",
}


class Deduper:
    """
    Deduplicates CandidateSource lists.

    Usage:
        deduper = Deduper(existing_urls=set_from_db)
        unique = deduper.deduplicate(candidates)
    """

    def __init__(
        self,
        existing_urls: Set[str] = None,
        title_similarity_threshold: float = 0.85,
    ):
        """
        Args:
            existing_urls: Set of URLs already in your DB. Candidates matching
                           these will be removed. Pass db.query("SELECT url FROM content")
                           results here.
            title_similarity_threshold: Jaccard similarity threshold for title dedup.
                                        0.85 = very similar titles are considered dupes.
        """
        self.existing_urls = set()
        if existing_urls:
            self.existing_urls = {self.normalize_url(u) for u in existing_urls}
        self.title_threshold = title_similarity_threshold

    def deduplicate(self, candidates: List[CandidateSource]) -> List[CandidateSource]:
        """
        Full dedup pipeline:
          1. Normalize URLs
          2. Remove exact URL duplicates within the batch
          3. Remove URLs already in existing DB
          4. Remove near-duplicate titles
          5. Return unique candidates

        Returns:
            Deduplicated list of CandidateSource.
        """
        original_count = len(candidates)

        # Step 1+2: URL dedup within batch
        seen_urls: Set[str] = set()
        url_unique = []
        for c in candidates:
            norm = self.normalize_url(c.url)
            if norm not in seen_urls:
                seen_urls.add(norm)
                url_unique.append(c)

        after_url = len(url_unique)

        # Step 3: Remove existing DB URLs
        if self.existing_urls:
            db_filtered = [
                c for c in url_unique
                if self.normalize_url(c.url) not in self.existing_urls
            ]
        else:
            db_filtered = url_unique

        after_db = len(db_filtered)

        # Step 4: Title similarity dedup
        title_unique = self._title_dedup(db_filtered)
        after_title = len(title_unique)

        removed = original_count - after_title
        logger.info(
            f"Dedup: {original_count} → {after_title} candidates "
            f"(url_dedup=-{original_count - after_url}, "
            f"db_dedup=-{after_url - after_db}, "
            f"title_dedup=-{after_db - after_title})"
        )

        return title_unique

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize a URL for dedup comparison:
          - Lowercase scheme and host
          - Strip trailing slashes
          - Remove tracking query params
          - Sort remaining query params
          - Strip fragments
        """
        if not url:
            return ""

        try:
            parsed = urlparse(url.strip())

            # Lowercase scheme and netloc
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()

            # Strip www. prefix
            if netloc.startswith("www."):
                netloc = netloc[4:]

            # Strip trailing slash from path
            path = parsed.path.rstrip("/")

            # Filter and sort query params
            params = parse_qs(parsed.query, keep_blank_values=False)
            filtered = {k: v for k, v in params.items() if k.lower() not in STRIP_PARAMS}
            sorted_query = urlencode(sorted(filtered.items()), doseq=True) if filtered else ""

            return urlunparse((scheme, netloc, path, "", sorted_query, ""))

        except Exception:
            return url.strip().lower().rstrip("/")

    def _title_dedup(self, candidates: List[CandidateSource]) -> List[CandidateSource]:
        """Remove candidates with near-duplicate titles using Jaccard similarity."""
        if not candidates:
            return []

        unique = [candidates[0]]

        for candidate in candidates[1:]:
            is_dupe = False
            for existing in unique:
                sim = self._title_similarity(candidate.title, existing.title)
                if sim >= self.title_threshold:
                    is_dupe = True
                    break
            if not is_dupe:
                unique.append(candidate)

        return unique

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """Jaccard similarity between two title strings (word-level)."""
        if not a or not b:
            return 0.0

        words_a = set(re.findall(r'\w+', a.lower()))
        words_b = set(re.findall(r'\w+', b.lower()))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)

    # TODO: Add content-hash dedup
    # def content_hash_dedup(self, candidates, db):
    #     """Compare candidate content hashes against existing DB hashes.
    #     Requires downloading content first, so this runs after initial approval."""
    #     pass
