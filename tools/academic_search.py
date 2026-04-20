"""
Academic Search Adapter — arXiv + OpenAlex + IEEE Xplore.

Searches academic repositories for research papers, conference proceedings,
and preprints related to industrial automation topics.

Designed for expansion: add new academic providers by implementing a provider
class with a `search(query, limit)` method returning standardized dicts.

Usage:
    from tools.academic_search import AcademicSearchAdapter

    adapter = AcademicSearchAdapter()
    results = adapter.search(query)  # SearchQuery object
"""

import logging
import os
import re
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from discovery.models import CandidateSource, SearchQuery

logger = logging.getLogger(__name__)


# ── Provider: arXiv ──────────────────────────────────────────────────────

class ArxivProvider:
    """
    arXiv API provider. Free, no API key required.

    Docs: https://info.arxiv.org/help/api/basics.html
    Terms: https://info.arxiv.org/help/api/tou.html

    Rate limit: 1 request per 3 seconds, single connection at a time.

    Query strategy — fallback ladder (tries unfiltered first, then category-scoped):
      1. Phrase search in all fields, no category filter
      2. Token AND search in all fields, no category filter
      3. Phrase + category filter
      4. Token AND + category filter
    First attempt that returns >0 entries wins.

    Categories for industrial automation (broader than robotics-only):
        cs.CR  — Security (ICS attacks, PLC security, SCADA)
        cs.NI  — Networking (OPC UA, industrial protocols)
        cs.SE  — Software Engineering (PLC code, tooling)
        cs.PL  — Programming Languages (IEC 61131-3)
        eess.SY — Systems and Control (alias: cs.SY)
        cs.RO  — Robotics
        cs.AI  — Artificial Intelligence
        cs.CE  — Computational Engineering
    """

    BASE_URL = "http://export.arxiv.org/api/query"
    DEFAULT_CATEGORIES = [
        "cs.CR", "cs.NI", "cs.SE", "cs.PL",
        "eess.SY", "cs.SY", "cs.RO", "cs.AI", "cs.CE",
    ]
    DELAY_SECONDS = 3.1  # arXiv requires >= 3s between requests

    def __init__(
        self,
        categories: List[str] = None,
        delay_seconds: float = 3.1,
        num_retries: int = 2,
        timeout: float = 20.0,
    ):
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.delay_seconds = max(delay_seconds, 3.0)
        self.num_retries = max(num_retries, 0)
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "industry-signals-pipeline/0.1"
        self._last_request_ts = 0.0
        # Circuit breaker
        self._consecutive_429s = 0
        self._max_429s = 3
        self._disabled = False

    # ── Query building ────────────────────────────────────────────────

    @staticmethod
    def _tokenize(query: str) -> List[str]:
        """Tokenize query for arXiv: lowercase, strip quotes, split on punctuation."""
        q = query.strip()
        q = re.sub(r"[\"'`]", " ", q)
        q = re.sub(r"[-_/,:;()\[\]{}]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return [t for t in q.split(" ") if t][:12]

    def _cat_clause(self) -> str:
        return "(" + " OR ".join(f"cat:{c}" for c in self.categories) + ")"

    def _build_attempts(self, user_query: str) -> List[Tuple[str, str]]:
        """
        Build a ladder of (name, search_query) attempts.

        Returns list of (attempt_name, arXiv search_query string) tuples.
        Tries phrase first (most precise), then token AND (more flexible),
        each without then with category filters.
        """
        uq = user_query.strip()
        tokens = self._tokenize(uq)

        # Phrase: all:"user query"
        phrase = f'all:"{uq}"' if uq else ""
        # Token AND: all:"tok1" AND all:"tok2" AND ...
        token_and = " AND ".join(f'all:"{t}"' for t in tokens) if tokens else ""

        cat = self._cat_clause()

        attempts = []
        if phrase:
            attempts.append(("phrase_unfiltered", phrase))
        if token_and and token_and != phrase:
            attempts.append(("tokens_unfiltered", token_and))
        if phrase:
            attempts.append(("phrase_filtered", f"{phrase} AND {cat}"))
        if token_and and token_and != phrase:
            attempts.append(("tokens_filtered", f"{token_and} AND {cat}"))

        return attempts

    # ── Rate limiting ─────────────────────────────────────────────────

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.delay_seconds:
            wait = self.delay_seconds - elapsed
            logger.debug(f"arXiv rate limit: waiting {wait:.1f}s")
            time.sleep(wait)

    # ── HTTP request ──────────────────────────────────────────────────

    def _fetch(self, params: Dict[str, Any]) -> Tuple[int, str]:
        """Make rate-limited request. Returns (status_code, response_text)."""
        self._rate_limit()
        resp = self._session.get(self.BASE_URL, params=params, timeout=self.timeout)
        self._last_request_ts = time.time()
        logger.debug(f"arXiv URL: {resp.url}")
        return resp.status_code, resp.text

    # ── Entry parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_entries(feed) -> List[Dict[str, Any]]:
        results = []
        for entry in getattr(feed, "entries", []) or []:
            pdf_link = ""
            for link in entry.get("links", []):
                if link.get("type") == "application/pdf" or link.get("title") == "pdf":
                    pdf_link = link.get("href", "")
                    break

            authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))

            results.append({
                "title": (entry.get("title", "") or "").replace("\n", " ").strip(),
                "url": entry.get("link", ""),
                "pdf_url": pdf_link,
                "snippet": (entry.get("summary", "") or "").replace("\n", " ").strip()[:500],
                "authors": authors,
                "published_date": entry.get("published", ""),
                "updated_date": entry.get("updated", ""),
                "categories": [t.get("term", "") for t in entry.get("tags", [])],
                "provider": "arxiv",
            })
        return results

    # ── Main search ───────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10, days_back: int = None) -> List[Dict[str, Any]]:
        """
        Search arXiv with fallback ladder. First attempt returning >0 entries wins.

        Args:
            query: Search query string.
            limit: Max results.
            days_back: If set, only return papers submitted in the last N days.

        Returns standardized dicts with: title, url, pdf_url, snippet, authors,
        published_date, updated_date, categories, provider.
        """
        if self._disabled:
            return []

        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed — required for arXiv adapter")
            return []

        attempts = self._build_attempts(query)
        if not attempts:
            return []

        # Add date range filter if days_back specified
        # arXiv format: submittedDate:[YYYYMMDDTTTT+TO+YYYYMMDDTTTT]
        date_filter = ""
        if days_back:
            from datetime import timedelta
            end = datetime.utcnow()
            start = end - timedelta(days=days_back)
            date_filter = f" AND submittedDate:[{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359]"

        base_params = {
            "start": 0,
            "max_results": max(1, min(limit, 100)),
            "sortBy": "submittedDate" if days_back else "relevance",
            "sortOrder": "descending",
        }

        last_error = None

        for attempt_name, search_query in attempts:
            params = {**base_params, "search_query": search_query + date_filter}

            for retry in range(self.num_retries + 1):
                try:
                    status, text = self._fetch(params)

                    if status == 429:
                        self._consecutive_429s += 1
                        logger.warning(
                            f"arXiv 429 ({self._consecutive_429s}/{self._max_429s}) "
                            f"attempt={attempt_name} retry={retry}"
                        )
                        if self._consecutive_429s >= self._max_429s:
                            self._disabled = True
                            logger.error("arXiv disabled after repeated 429 errors")
                            return []
                        continue  # retry same attempt

                    if status != 200:
                        last_error = f"HTTP {status} on {attempt_name}"
                        logger.warning(f"arXiv {last_error} (retry {retry+1}/{self.num_retries+1})")
                        continue

                    # Reset 429 counter on any successful HTTP
                    self._consecutive_429s = 0

                    feed = feedparser.parse(text)
                    if getattr(feed, "bozo", 0):
                        logger.debug(f"arXiv bozo ({attempt_name}): {getattr(feed, 'bozo_exception', None)}")

                    results = self._parse_entries(feed)

                    if results:
                        logger.info(
                            f"arXiv returned {len(results)} results "
                            f"(attempt={attempt_name}) for: '{query}'"
                        )
                        return results

                    # 0 entries — break retry loop, try next attempt
                    logger.debug(f"arXiv 0 entries for attempt={attempt_name} query={search_query}")
                    break

                except requests.RequestException as e:
                    last_error = f"{attempt_name}: {e}"
                    logger.warning(f"arXiv error (retry {retry+1}/{self.num_retries+1}): {last_error}")

        logger.warning(f"arXiv exhausted all attempts for: '{query}' (last_error={last_error})")
        return []

    def is_available(self) -> bool:
        if self._disabled:
            return False
        return True


# ── Provider: IEEE Xplore ────────────────────────────────────────────────

class IEEEProvider:
    """
    IEEE Xplore API provider.

    Requires: IEEE_API_KEY environment variable.
    Free tier: 200 requests/day, 10 calls/second, 10 results per request.
    Docs: https://developer.ieee.org/docs/read/Metadata_API_overview

    Valid search params (per IEEE docs):
      querytext, apikey, max_records, start_record,
      start_year, end_year, sort_order, sort_field,
      content_type, open_access, start_date, end_date,
      publication_number, publisher, article_number

    Relevant content types: Conferences, Journals, Standards, Books.
    """

    BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("IEEE_API_KEY")
        # Rate limiting: 10 calls/second, 200 calls/day
        self._last_call_time = 0
        self._daily_calls = 0
        self._daily_reset = self._get_midnight_utc()
        self._min_interval = 0.11  # 1/9.09 ≈ 110ms to stay under 10 calls/second
        # Circuit breaker: disable after consecutive errors
        self._consecutive_errors = 0
        self._max_errors = 5
        self._disabled = False

    # ── Debug helpers ─────────────────────────────────────────────────

    def _mask_key(self) -> str:
        """Return masked API key for safe logging (first4...last4)."""
        if not self.api_key:
            return "<not set>"
        k = self.api_key.strip()
        if len(k) <= 8:
            return k[:2] + "***"
        return k[:4] + "..." + k[-4:]

    @staticmethod
    def _log_curl(url: str):
        """Log a curl-equivalent command for manual debugging."""
        logger.info(f'IEEE curl equivalent:\n  curl "{url}"')

    def _log_response(self, resp, query: str):
        """Log response details for debugging."""
        logger.info(
            f"IEEE response: status={resp.status_code} "
            f"url={resp.url} query='{query[:50]}'"
        )
        if resp.status_code != 200:
            body_snippet = resp.text[:300] if resp.text else "<empty>"
            logger.warning(
                f"IEEE non-200 body: {body_snippet}"
            )

    # ── Minimal test ──────────────────────────────────────────────────

    def test_connection(self):
        """
        Minimal hardcoded test to isolate auth vs query issues.

        Usage:
            from tools.academic_search import IEEEProvider
            p = IEEEProvider()
            p.test_connection()
        """
        print(f"IEEE API key: {self._mask_key()}")
        print(f"IEEE BASE_URL: {self.BASE_URL}")

        params = {
            "querytext": "robotics",
            "max_records": 1,
            "apikey": self.api_key,
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            print(f"Final URL: {resp.url}")
            print(f'curl "{resp.url}"')
            print(f"Status: {resp.status_code}")
            print(f"Body (first 500 chars): {resp.text[:500]}")
        except Exception as e:
            print(f"Request failed: {e}")

    # ── Search ────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search IEEE Xplore and return standardized result dicts."""
        if not self.api_key:
            logger.warning("IEEE_API_KEY not set — skipping IEEE search")
            return []

        logger.debug(f"IEEE search: key={self._mask_key()} query='{query[:50]}'")

        # Check daily limit
        now = time.time()
        if now > self._daily_reset:
            self._daily_calls = 0
            self._daily_reset = self._get_midnight_utc()

        if self._daily_calls >= 200:
            logger.warning("IEEE daily limit reached (200 calls) — skipping search")
            return []

        # Rate limit: wait if too soon since last call
        time_since_last = now - self._last_call_time
        if time_since_last < self._min_interval:
            sleep_time = self._min_interval - time_since_last
            time.sleep(sleep_time)

        params = {
            "querytext": query,
            "max_records": min(limit, 25),
            "start_year": 2020,
            "apikey": self.api_key,
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)

            # ── Debug logging ──
            self._log_curl(resp.url)
            self._log_response(resp, query)

            # ── Error classification ──
            if resp.status_code == 403:
                self._consecutive_errors += 1
                logger.warning(
                    f"IEEE 403 Forbidden ({self._consecutive_errors}/{self._max_errors}) "
                    f"key={self._mask_key()} — auth/provisioning issue"
                )
                if self._consecutive_errors >= self._max_errors:
                    self._disabled = True
                    logger.error(
                        f"IEEE provider disabled after {self._max_errors} consecutive "
                        f"403 errors — check API key provisioning at developer.ieee.org"
                    )
                return []

            if resp.status_code == 429:
                self._consecutive_errors += 1
                logger.warning(
                    f"IEEE 429 Rate Limited ({self._consecutive_errors}/{self._max_errors}) "
                    f"— back off"
                )
                if self._consecutive_errors >= self._max_errors:
                    self._disabled = True
                    logger.error(f"IEEE provider disabled after {self._max_errors} consecutive rate limits")
                return []

            if resp.status_code >= 500:
                self._consecutive_errors += 1
                logger.warning(
                    f"IEEE {resp.status_code} Server Error "
                    f"({self._consecutive_errors}/{self._max_errors})"
                )
                if self._consecutive_errors >= self._max_errors:
                    self._disabled = True
                    logger.error(f"IEEE provider disabled after {self._max_errors} consecutive server errors")
                return []

            resp.raise_for_status()
            data = resp.json()

            # Success — reset error counter
            self._consecutive_errors = 0
            self._daily_calls += 1
            self._last_call_time = time.time()
            logger.debug(f"IEEE API call #{self._daily_calls}/200 today")
        except requests.RequestException as e:
            logger.error(f"IEEE Xplore API error: {e}")
            return []

        results = []
        for article in data.get("articles", []):
            authors_list = article.get("authors", {}).get("authors", [])
            authors = ", ".join([a.get("full_name", "") for a in authors_list])

            results.append({
                "title": article.get("title", ""),
                "url": article.get("html_url", article.get("pdf_url", "")),
                "pdf_url": article.get("pdf_url", ""),
                "snippet": article.get("abstract", "")[:500],
                "authors": authors,
                "published_date": article.get("publication_date", ""),
                "doi": article.get("doi", ""),
                "content_type": article.get("content_type", ""),
                "publication_title": article.get("publication_title", ""),
                "provider": "ieee",
            })

        logger.info(f"IEEE Xplore returned {len(results)} results for: '{query}'")
        return results

    def _get_midnight_utc(self) -> float:
        """Return Unix timestamp for next midnight UTC (daily reset)."""
        now = datetime.utcnow()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return tomorrow.timestamp()

    def is_available(self) -> bool:
        """Check if provider is available and not disabled by circuit breaker."""
        if self._disabled:
            return False
        return bool(self.api_key)


# ── Provider: OpenAlex ───────────────────────────────────────────────────

class OpenAlexProvider:
    """
    OpenAlex API provider for scholarly metadata.

    Free API with optional key (OPENALEX_API_KEY). Key gives $1/day budget;
    unauthenticated is "best effort." Recommended to always use a key.

    Docs: https://docs.openalex.org
    Rate: 100 req/s hard limit; daily budget-based throttling with 429s.

    Key design decisions:
      - Cursor pagination (no 10k cap vs basic paging).
      - Abstracts reconstructed from abstract_inverted_index.
      - PDF URL priority: content_url (cached) > best_oa_location > locations > oa_url.
      - select= used to minimize response size and API cost.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        api_key: str = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        per_page: int = 50,
        min_delay: float = 0.05,
    ):
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max(max_retries, 0)
        self.per_page = max(1, min(per_page, 100))
        self.min_delay = min_delay
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "industry-signals-pipeline/1.0 (OpenAlexProvider)"
        self._last_request_ts = 0.0
        # Circuit breaker
        self._consecutive_errors = 0
        self._max_errors = 5
        self._disabled = False

    # ── Abstract reconstruction ────────────────────────────────────────

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
        """Reconstruct plaintext from OpenAlex abstract_inverted_index."""
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        max_pos = -1
        for positions in inverted_index.values():
            if positions:
                m = max(positions)
                if m > max_pos:
                    max_pos = m
        if max_pos < 0:
            return ""
        words = [""] * (max_pos + 1)
        for token, positions in inverted_index.items():
            for p in positions:
                if 0 <= p <= max_pos and not words[p]:
                    words[p] = token
        return " ".join(w for w in words if w)

    # ── PDF URL extraction ─────────────────────────────────────────────

    @staticmethod
    def _best_pdf_url(work: Dict[str, Any]) -> str:
        """
        Extract best PDF URL from work object.

        Priority: content_url (OpenAlex cached) > best_oa_location.pdf_url
        > primary_location.pdf_url > locations[*].pdf_url > oa_url if .pdf.
        """
        # OpenAlex cached content (strongest signal)
        has_content = work.get("has_content") or {}
        if has_content.get("pdf") and work.get("content_url"):
            return work["content_url"]

        # best_oa_location
        best_oa = work.get("best_oa_location") or {}
        if best_oa.get("pdf_url"):
            return best_oa["pdf_url"]

        # primary_location
        primary = work.get("primary_location") or {}
        if primary.get("pdf_url"):
            return primary["pdf_url"]

        # Scan all locations
        for loc in (work.get("locations") or []):
            if isinstance(loc, dict) and loc.get("pdf_url"):
                return loc["pdf_url"]

        # OA URL fallback — only if it looks like a PDF
        oa = work.get("open_access") or {}
        oa_url = oa.get("oa_url") or ""
        if isinstance(oa_url, str) and oa_url.lower().endswith(".pdf"):
            return oa_url

        return ""

    @staticmethod
    def _best_url(work: Dict[str, Any]) -> str:
        """Best landing page URL: DOI > primary_location > OpenAlex ID."""
        doi = work.get("doi")
        if doi:
            return doi
        primary = work.get("primary_location") or {}
        if primary.get("landing_page_url"):
            return primary["landing_page_url"]
        return work.get("id", "")

    # ── HTTP with backoff ──────────────────────────────────────────────

    def _request_json(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make rate-limited request with exponential backoff. Returns JSON or None."""
        import random as _random

        url = f"{self.BASE_URL}{path}"
        if self.api_key:
            params = dict(params)
            params["api_key"] = self.api_key

        for attempt in range(self.max_retries + 1):
            # Politeness delay
            elapsed = time.time() - self._last_request_ts
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)

            try:
                resp = self._session.get(url, params=params, timeout=self.timeout, allow_redirects=True)
                self._last_request_ts = time.time()

                logger.debug(
                    "OpenAlex status=%s remaining=%s reset=%s url=%s",
                    resp.status_code,
                    resp.headers.get("X-RateLimit-Remaining", "?"),
                    resp.headers.get("X-RateLimit-Reset", "?"),
                    resp.url[:120],
                )

                if resp.status_code == 200:
                    self._consecutive_errors = 0
                    return resp.json()

                # Retryable: 429, 403 (budget), 5xx
                if resp.status_code in (429, 403) or resp.status_code >= 500:
                    self._consecutive_errors += 1
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        sleep_s = min(float(retry_after), 30.0)
                    else:
                        sleep_s = min(1.0 * (2 ** attempt), 30.0) + _random.uniform(0, 0.25)
                    logger.warning(
                        "OpenAlex %s (attempt %d/%d). Sleeping %.1fs",
                        resp.status_code, attempt + 1, self.max_retries + 1, sleep_s,
                    )
                    if self._consecutive_errors >= self._max_errors:
                        self._disabled = True
                        logger.error("OpenAlex disabled after %d consecutive errors", self._max_errors)
                        return None
                    time.sleep(sleep_s)
                    continue

                # Non-retryable client error (400, 404, etc.)
                logger.error("OpenAlex error %s: %s", resp.status_code, resp.text[:300])
                return None

            except (requests.Timeout, requests.ConnectionError) as e:
                self._consecutive_errors += 1
                sleep_s = min(1.0 * (2 ** attempt), 30.0) + _random.uniform(0, 0.25)
                logger.warning("OpenAlex transport error (attempt %d): %r. Sleeping %.1fs", attempt + 1, e, sleep_s)
                if attempt >= self.max_retries:
                    return None
                time.sleep(sleep_s)

        return None

    # ── Main search ────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 20, days_back: int = None) -> List[Dict[str, Any]]:
        """
        Search OpenAlex /works with cursor pagination.

        Returns standardized dicts with: title, url, pdf_url, snippet, authors,
        published_date, doi, provider, publication_title, cited_by_count, concepts.
        """
        if self._disabled:
            return []

        q = (query or "").strip()
        if not q:
            return []

        # Build filter
        filters = []
        if days_back:
            from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            filters.append(f"from_publication_date:{from_date}")

        select_fields = ",".join([
            "id", "doi", "title", "display_name", "publication_date",
            "publication_year", "cited_by_count", "authorships",
            "primary_location", "best_oa_location", "locations",
            "open_access", "abstract_inverted_index", "topics", "concepts",
        ])

        # Cursor pagination — collects up to `limit` results
        results = []
        cursor = "*"
        remaining = max(0, int(limit))

        while remaining > 0:
            params = {
                "search": q,
                "per_page": min(self.per_page, remaining),
                "cursor": cursor,
                "sort": "relevance_score:desc",
                "select": select_fields,
            }
            if filters:
                params["filter"] = ",".join(filters)

            data = self._request_json("/works", params)
            if not data:
                break

            meta = data.get("meta") or {}
            works = data.get("results") or []

            if not works:
                break

            for w in works:
                abstract = self._reconstruct_abstract(w.get("abstract_inverted_index"))
                snippet = abstract.strip()[:500] if abstract else ""

                # Authors
                authors = []
                for a in (w.get("authorships") or []):
                    author_obj = (a.get("author") or {}) if isinstance(a, dict) else {}
                    name = author_obj.get("display_name")
                    if name:
                        authors.append(name)

                # Publication title (journal/conference name)
                source = ((w.get("primary_location") or {}).get("source") or {})
                pub_title = source.get("display_name", "") if isinstance(source, dict) else ""

                # Topics / concepts
                topic_names = []
                for t in (w.get("topics") or []):
                    if isinstance(t, dict) and t.get("display_name"):
                        topic_names.append(t["display_name"])
                if not topic_names:
                    for c in (w.get("concepts") or []):
                        if isinstance(c, dict) and c.get("display_name"):
                            topic_names.append(c["display_name"])

                results.append({
                    "title": (w.get("title") or w.get("display_name") or "").strip(),
                    "url": self._best_url(w),
                    "pdf_url": self._best_pdf_url(w),
                    "snippet": snippet,
                    "authors": ", ".join(authors),
                    "published_date": w.get("publication_date") or "",
                    "doi": w.get("doi") or "",
                    "provider": "openalex",
                    "publication_title": pub_title,
                    "cited_by_count": int(w.get("cited_by_count") or 0),
                    "concepts": topic_names,
                })

                remaining -= 1
                if remaining <= 0:
                    break

            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor

        logger.info(f"OpenAlex returned {len(results)} results for: '{q}'")
        return results

    def is_available(self) -> bool:
        if self._disabled:
            return False
        # OpenAlex works without a key (best-effort), but key is recommended
        return True


# ── Unified Academic Adapter ─────────────────────────────────────────────

class AcademicSearchAdapter:
    """
    Unified academic search adapter for the discovery pipeline.

    Searches arXiv, OpenAlex, and IEEE Xplore (when API key is available),
    normalizes results into CandidateSource objects.

    Expandable: add new providers to self.providers dict.
    """

    adapter_name = "academic"

    def __init__(self, providers: Dict[str, Any] = None):
        """
        Args:
            providers: Optional dict of {name: provider_instance}.
                       Defaults to arXiv + OpenAlex + IEEE (if key available).
        """
        if providers is not None:
            self.providers = providers
        else:
            self.providers = {
                "arxiv": ArxivProvider(),
                "openalex": OpenAlexProvider(),
                "ieee": IEEEProvider(),
            }

    def search(self, query: SearchQuery) -> List[CandidateSource]:
        """
        Search all available academic providers and return normalized candidates.

        Args:
            query: SearchQuery with .query, .limit, .tags, etc.

        Returns:
            List of CandidateSource objects.
        """
        candidates = []

        # Determine which providers to use (default: all available)
        target_providers = getattr(query, 'providers', None)

        for name, provider in self.providers.items():
            if target_providers and name not in target_providers:
                continue
            if not provider.is_available():
                logger.debug(f"Academic provider '{name}' not available, skipping")
                continue

            try:
                # Pass days_back to providers that support it
                search_kwargs = {"query": query.query, "limit": query.limit}
                days_back = getattr(query, 'days_back', None)
                if days_back and 'days_back' in (getattr(provider.search, '__code__', None) and provider.search.__code__.co_varnames or []):
                    search_kwargs["days_back"] = days_back
                results = provider.search(**search_kwargs)
                for r in results:
                    candidates.append(CandidateSource(
                        title=r.get("title", "Untitled"),
                        url=r.get("url", ""),
                        snippet=r.get("snippet", ""),
                        source_type="research_paper",
                        publisher=r.get("publication_title", r.get("provider", name)),
                        discovered_at=datetime.utcnow().isoformat(),
                        adapter=self.adapter_name,
                        query_used=query.query,
                        raw_metadata={
                            "provider": r.get("provider", name),
                            "authors": r.get("authors", ""),
                            "published_date": r.get("published_date", ""),
                            "pdf_url": r.get("pdf_url", ""),
                            "doi": r.get("doi", ""),
                            "categories": r.get("categories", []),
                            "content_type": r.get("content_type", ""),
                            "cited_by_count": r.get("cited_by_count", 0),
                            "concepts": r.get("concepts", []),
                        },
                    ))
            except Exception as e:
                logger.error(f"Academic provider '{name}' error: {e}")

        logger.info(f"Academic adapter returned {len(candidates)} candidates for: '{query.query}'")
        return candidates

    def is_available(self) -> bool:
        """At least one provider must be available."""
        return any(p.is_available() for p in self.providers.values())
