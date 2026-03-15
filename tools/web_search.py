"""Internet Search Tool - Search the web using Tavily, DuckDuckGo, Google, or Brave."""

from typing import List, Dict, Any, Optional
import requests, os
import json
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

class InternetSearchTool:
    name = "internet_search"
    description = """Search the internet for current information, news, and external data."""
    
    def __init__(self, provider: str = "tavily", api_key: Optional[str] = None, 
                 search_engine_id: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        
        if self.provider not in ["tavily", "duckduckgo", "google", "brave"]:
            raise ValueError(f"Provider must be one of: tavily, duckduckgo, google, brave")
        
        if self.provider == "tavily" and not api_key:
            raise ValueError("Tavily requires api_key")
        elif self.provider == "google" and not (api_key and search_engine_id):
            raise ValueError("Google requires api_key and search_engine_id")
        elif self.provider == "brave" and not api_key:
            raise ValueError("Brave requires api_key")
    
    def execute(self, action: str, query: str, max_results: int = 5, **kwargs) -> Dict[str, Any]:
        if action != "search":
            return {"error": f"Unknown action '{action}'. Only 'search' is supported"}
        
        try:
            if self.provider == "tavily":
                return self._search_tavily(query, max_results)
            elif self.provider == "duckduckgo":
                return self._search_duckduckgo(query, max_results)
            elif self.provider == "google":
                return self._search_google(query, max_results)
            elif self.provider == "brave":
                return self._search_brave(query, max_results)
        except Exception as e:
            logger.error(f"Internet search error ({self.provider}): {e}")
            return {"error": f"Search failed: {str(e)}", "provider": self.provider}
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        return self.execute("search", query=query, max_results=max_results)
    
    def _search_tavily(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": "basic",
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [],
                "exclude_domains": []
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            
            if data.get("answer"):
                results.append({
                    "title": "AI-Generated Summary",
                    "url": "",
                    "snippet": data.get("answer", ""),
                    "relevance_score": 1.0
                })
            
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "relevance_score": item.get("score", 0.8)
                })
            
            return {
                "query": query,
                "provider": "tavily",
                "count": len(results),
                "results": results[:max_results]
            }
            
        except Exception as e:
            return {"error": f"Tavily search failed: {str(e)}", "provider": "tavily"}
    
    def _search_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            encoded_query = quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("AbstractSource", "DuckDuckGo"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", ""),
                    "relevance_score": 1.0
                })
            
            if data.get("RelatedTopics"):
                for topic in data.get("RelatedTopics", [])[:max_results-len(results)]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " ").title(),
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", ""),
                            "relevance_score": 0.8
                        })
            
            return {
                "query": query,
                "provider": "duckduckgo",
                "count": len(results),
                "results": results[:max_results]
            }
            
        except Exception as e:
            return {"error": f"DuckDuckGo search failed: {str(e)}", "provider": "duckduckgo"}
    
    def _search_google(self, query: str, max_results: int) -> Dict[str, Any]:
        pass
    
    def _search_brave(self, query: str, max_results: int) -> Dict[str, Any]:
        pass
    
    def _format_results(self, raw_results: List[Dict], query: str) -> Dict[str, Any]:
        pass

# ---------------------------------------------------------------------------
# Discovery-oriented web search adapter (used by discovery/service.py)
# ---------------------------------------------------------------------------

from typing import Protocol as _Protocol

try:
    from discovery.models import CandidateSource, SearchQuery as _SearchQuery
    _DISCOVERY_MODELS = True
except ImportError:
    _DISCOVERY_MODELS = False


class SearchProvider(_Protocol):
    """Protocol for pluggable search backends."""
    def search(self, query: str, limit: int = 10) -> List[dict]: ...


class PlaceholderSearchProvider:
    """Placeholder — returns empty results. Swap in Tavily/Brave/etc."""
    def search(self, query: str, limit: int = 10) -> List[dict]:
        logger.warning(f"PlaceholderSearchProvider: '{query}' — no real provider configured")
        return []


class TavilySearchProvider:
    """Tavily search provider. Requires: pip install tavily-python + TAVILY_API_KEY."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def search(self, query: str, limit: int = 10) -> List[dict]:
        if not self.api_key:
            logger.error("TAVILY_API_KEY not set")
            return []
        try:
            client = self._get_client()
            response = client.search(query, max_results=limit, search_depth="basic")
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "published_date": item.get("published_date", ""),
                }
                for item in response.get("results", [])
            ]
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []


class WebSearchAdapter:
    """
    Web search adapter for the discovery pipeline.
    Normalizes results into CandidateSource objects.
    """

    adapter_name = "web"

    def __init__(self, provider=None):
        if provider is None:
            if os.getenv("TAVILY_API_KEY"):
                self.provider = TavilySearchProvider()
            else:
                self.provider = PlaceholderSearchProvider()
        else:
            self.provider = provider

    def search(self, query) -> list:
        """Execute web search and normalize results to CandidateSource."""
        if not _DISCOVERY_MODELS:
            logger.error("discovery.models not available — cannot normalize results")
            return []

        raw_results = self.provider.search(query.query, limit=query.limit)
        candidates = []
        for result in raw_results:
            url = result.get("url", "")
            if not url:
                continue
            candidates.append(CandidateSource(
                title=result.get("title", "Untitled"),
                url=url,
                snippet=result.get("snippet", ""),
                source_type="web_page",
                publisher=self._extract_domain(url),
                adapter=self.adapter_name,
                query_used=query.query,
                raw_metadata={
                    "search_score": result.get("score", 0.0),
                    "published_date": result.get("published_date", ""),
                },
            ))
        return candidates

    def is_available(self) -> bool:
        return not isinstance(self.provider, PlaceholderSearchProvider)

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ""


if __name__ == "__main__":
    print("=== Testing Tavily ===")
    tavily_tool = InternetSearchTool(
        provider="tavily",
        api_key="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"
    )
    results = tavily_tool.search("what is an AI agent", max_results=3)
    print(f"Results: {results}")

    print("=== Testing DuckDuckGo ===")
    duckduckgo_tool = InternetSearchTool(provider="duckduckgo")
    results = duckduckgo_tool.search("AI agent", max_results=3)
    print(f"Results: {results}")
