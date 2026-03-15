"""
Tools package — Shared capabilities used by agents and discovery.

Agent tools:
    - InternetSearchTool  — web search (Tavily/DDG/Brave)
    - DBQueryTool         — structured SQL queries
    - VectorSearchTool    — FAISS semantic search

Discovery tools:
    - RSSAdapter          — RSS/Atom feed reader
    - GitHubAdapter       — GitHub repo search
    - WebSearchAdapter    — web search for discovery pipeline
"""

from tools.base import BaseTool
from tools.vector_search import VectorSearchTool
from tools.db_query import DBQueryTool
from tools.web_search import InternetSearchTool
from tools.rss_reader import RSSAdapter
from tools.github_search import GitHubAdapter

__all__ = [
    'BaseTool',
    'VectorSearchTool', 'DBQueryTool', 'InternetSearchTool',
    'RSSAdapter', 'GitHubAdapter',
]
