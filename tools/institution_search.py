"""
Institution Search Adapter — NIST + NSF.

Searches government research institutions for standards documents,
technical reports, and funded research projects.

Designed for expansion: add new institution providers by implementing a
provider class with `search(query, limit)` returning standardized dicts.

Usage:
    from tools.institution_search import InstitutionSearchAdapter

    adapter = InstitutionSearchAdapter()
    results = adapter.search(query)  # SearchQuery object
"""

import logging
import os
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

from discovery.models import CandidateSource, SearchQuery

logger = logging.getLogger(__name__)


# ── Provider: NIST ───────────────────────────────────────────────────────

class NISTProvider:
    """
    NIST search provider. Free, no API key required.

    Searches NIST publications (technical notes, special publications, etc.)
    via the NIST public search API.

    Docs: https://csrc.nist.gov/publications
    Also covers: NIST Cybersecurity Framework, Manufacturing Extension Partnership.
    """

    CSRC_URL = "https://csrc.nist.gov/publications/search"
    GENERAL_URL = "https://www.nist.gov/search"

    # Terms that indicate a query is relevant for CSRC (cybersecurity repo)
    # Ordered by specificity: higher-priority terms produce better CSRC results
    CSRC_KEYWORDS_PRIORITY = [
        # Tier 1: OT/ICS-specific terms (best CSRC results)
        'scada', 'iec 62443', '800-82', 'ot security', 'ics',
        # Tier 2: Focused cybersecurity terms
        'zero trust', 'network segmentation', 'intrusion', 'incident response',
        'access control', 'vulnerability', 'ransomware', 'malware',
        'firewall', 'encryption', 'authentication', 'nist framework',
        # Tier 3: Broad terms (weakest CSRC results, but still relevant for routing)
        'industrial control', 'cybersecurity', 'cyber', 'security', 'threat',
    ]
    # Flat set for quick membership check
    CSRC_KEYWORDS = set(CSRC_KEYWORDS_PRIORITY)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search NIST publications via CSRC and/or general search.
        
        Routes intelligently: only queries CSRC for cybersecurity-related terms.
        Always queries NIST general search.
        """
        results = []
        q_lower = query.lower()

        # Only query CSRC if query contains cybersecurity-related terms
        if any(kw in q_lower for kw in self.CSRC_KEYWORDS):
            # CSRC search works best with single focused keywords, not phrases.
            # Extract the best matching CSRC keyword from the query.
            csrc_query = self._extract_csrc_keyword(query)
            if csrc_query:
                results.extend(self._search_csrc(csrc_query, limit))

        # NIST general publications via site search (always)
        results.extend(self._search_general(query, limit))

        return results[:limit]

    def _extract_csrc_keyword(self, query: str) -> str:
        """Extract the best single keyword for CSRC search.
        
        CSRC returns much better results with focused terms like 'SCADA'
        vs phrases like 'SCADA and HMI' or 'Industrial Cybersecurity manufacturing'.
        
        Strategy: use priority-ordered list — first match wins.
        'scada' (tier 1) beats 'security' (tier 3).
        """
        q_lower = query.lower()
        for kw in self.CSRC_KEYWORDS_PRIORITY:
            if kw in q_lower:
                # Preserve original case from query
                idx = q_lower.find(kw)
                if idx >= 0:
                    return query[idx:idx + len(kw)]
                return kw
        return query.strip()

    def _search_csrc(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search NIST CSRC publications (cybersecurity, standards, frameworks)."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not available for NIST CSRC")
            return []

        params = {"keywords": query, "status": "Final"}
        try:
            resp = requests.get(self.CSRC_URL, params=params, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; IndustrySignalsBot/1.0)',
            })
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            results = []
            for row in soup.select('table.table tbody tr')[:limit]:
                cells = row.select('td')
                if len(cells) < 3:
                    continue

                # CSRC table: cell[0]=series, cell[1]=number, cell[2]=title+link,
                #             cell[3]=status, cell[4]=release date
                series = cells[0].get_text(strip=True)   # e.g. "SP", "IR", "CSWP"
                number = cells[1].get_text(strip=True)    # e.g. "1308", "800-82"
                doc_id = f"{series} {number}" if series else number

                # Title is in cell[2] inside <a> tag
                title_link = cells[2].select_one('a[href]')
                if not title_link:
                    continue
                pub_title = title_link.get_text(strip=True)
                href = title_link.get('href', '')
                if href and not href.startswith('http'):
                    href = f"https://csrc.nist.gov{href}"

                # Release date from cell[4]
                pub_date = cells[4].get_text(strip=True) if len(cells) > 4 else ""

                results.append({
                    "title": f"{doc_id}: {pub_title}" if doc_id else pub_title,
                    "url": href,
                    "snippet": pub_title[:500],
                    "authors": "NIST",
                    "published_date": pub_date,
                    "provider": "nist_csrc",
                    "doc_type": "standard",
                })

            logger.info(f"NIST CSRC returned {len(results)} results for: '{query}'")
            return results

        except requests.RequestException as e:
            logger.error(f"NIST CSRC search error: {e}")
            return []

    def _search_general(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search NIST general publications via site search.
        
        NOTE: NIST general search (nist.gov/search) is JS-rendered and cannot
        be scraped server-side. Returns empty until a headless browser or
        NIST API alternative is found. CSRC search is the primary provider.
        """
        logger.debug(f"NIST general search skipped (JS-rendered page): '{query}'")
        return []

    def is_available(self) -> bool:
        return True


# ── Provider: NSF ────────────────────────────────────────────────────────

class NSFProvider:
    """
    NSF Awards API provider. Free, no API key required.

    Searches funded research awards related to manufacturing and automation.
    Docs: https://www.research.gov/common/webapi/awardapisearch-v1.htm

    Useful for discovering cutting-edge funded research projects.
    """

    BASE_URL = "https://api.nsf.gov/services/v1/awards.json"

    def search(self, query: str, limit: int = 10, days_back: int = None) -> List[Dict[str, Any]]:
        """Search NSF awards database.
        
        Uses quoted phrases for multi-word queries to avoid irrelevant
        results from individual word matching.

        Args:
            query: Search keywords.
            limit: Max results.
            days_back: If set, only return awards started in the last N days.
        """
        # Quote multi-word queries for phrase matching
        kw = query.strip()
        if ' ' in kw and not kw.startswith('"'):
            kw = f'"{kw}"'

        params = {
            "keyword": kw,
            "printFields": "id,title,abstractText,piFirstName,piLastName,"
                           "startDate,expDate,awardeeName,fundProgramName",
            "offset": 1,
            "rpp": min(limit, 25),
        }
        if days_back:
            from datetime import timedelta
            start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%m/%d/%Y")
            params["dateStart"] = start

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"NSF API error: {e}")
            return []

        results = []
        for award in data.get("response", {}).get("award", []):
            award_id = award.get("id", "")
            pi_name = f"{award.get('piFirstName', '')} {award.get('piLastName', '')}".strip()
            abstract = award.get("abstractText", "")

            results.append({
                "title": award.get("title", ""),
                "url": f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={award_id}",
                "snippet": abstract[:500] if abstract else "",
                "authors": pi_name,
                "published_date": award.get("startDate", ""),
                "end_date": award.get("expDate", ""),
                "institution": award.get("awardeeName", ""),
                "program": award.get("fundProgramName", ""),
                "provider": "nsf",
                "doc_type": "funded_research",
            })

        logger.info(f"NSF returned {len(results)} awards for: '{query}'")
        return results

    def is_available(self) -> bool:
        return True


# ── Unified Institution Adapter ──────────────────────────────────────────

class InstitutionSearchAdapter:
    """
    Unified institution search adapter for the discovery pipeline.

    Searches NIST and NSF (both free, no API keys needed).
    Normalizes results into CandidateSource objects.

    Expandable: add new providers to self.providers dict.
    Example future additions: DOE, Fraunhofer, EU Horizon.
    """

    adapter_name = "institution"

    def __init__(self, providers: Dict[str, Any] = None):
        if providers is not None:
            self.providers = providers
        else:
            self.providers = {
                "nist": NISTProvider(),
                "nsf": NSFProvider(),
            }

    def search(self, query: SearchQuery) -> List[CandidateSource]:
        """Search all available institution providers."""
        candidates = []

        target_providers = getattr(query, 'providers', None)

        for name, provider in self.providers.items():
            if target_providers and name not in target_providers:
                continue
            if not provider.is_available():
                continue

            try:
                # Pass days_back to providers that support it (NSF)
                search_kwargs = {"query": query.query, "limit": query.limit}
                days_back = getattr(query, 'days_back', None)
                if days_back and hasattr(provider.search, '__code__') and 'days_back' in provider.search.__code__.co_varnames:
                    search_kwargs["days_back"] = days_back
                results = provider.search(**search_kwargs)
                for r in results:
                    candidates.append(CandidateSource(
                        title=r.get("title", "Untitled"),
                        url=r.get("url", ""),
                        snippet=r.get("snippet", ""),
                        source_type=r.get("doc_type", "technical_report"),
                        publisher=r.get("institution", r.get("authors", name.upper())),
                        discovered_at=datetime.utcnow().isoformat(),
                        adapter=self.adapter_name,
                        query_used=query.query,
                        raw_metadata={
                            "provider": r.get("provider", name),
                            "authors": r.get("authors", ""),
                            "published_date": r.get("published_date", ""),
                            "doc_type": r.get("doc_type", ""),
                            "program": r.get("program", ""),
                            "institution": r.get("institution", ""),
                        },
                    ))
            except Exception as e:
                logger.error(f"Institution provider '{name}' error: {e}")

        logger.info(f"Institution adapter returned {len(candidates)} candidates for: '{query.query}'")
        return candidates

    def is_available(self) -> bool:
        return any(p.is_available() for p in self.providers.values())
