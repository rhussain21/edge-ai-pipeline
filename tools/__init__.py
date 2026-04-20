"""
Tools package — Shared capabilities used by agents and discovery.

Agent tools:
    - InternetSearchTool  — web search (Tavily/DDG/Brave)
    - DBQueryTool         — structured SQL queries
    - VectorSearchTool    — FAISS semantic search
    - WebScraper          — HTML-to-text extraction (BeautifulSoup)

Discovery tools:
    - RSSAdapter              — RSS/Atom feed reader
    - GitHubAdapter           — GitHub repo search
    - WebSearchAdapter        — web search for discovery pipeline
    - AcademicSearchAdapter   — arXiv + IEEE Xplore
    - InstitutionSearchAdapter — NIST + NSF
    - StackOverflowAdapter    — Stack Exchange API
    - PDFExtractor            — PDF download + text extraction
"""

from tools.base import BaseTool
from tools.vector_search import VectorSearchTool
from tools.db_query import DBQueryTool
from tools.web_search import InternetSearchTool
from tools.rss_reader import RSSAdapter
from tools.github_search import GitHubAdapter
from tools.academic_search import AcademicSearchAdapter
from tools.institution_search import InstitutionSearchAdapter
from tools.stackoverflow_search import StackOverflowAdapter
from tools.web_scraper import WebScraper
from tools.pdf_extractor import PDFExtractor

__all__ = [
    'BaseTool',
    'VectorSearchTool', 'DBQueryTool', 'InternetSearchTool', 'WebScraper',
    'RSSAdapter', 'GitHubAdapter',
    'AcademicSearchAdapter', 'InstitutionSearchAdapter', 'StackOverflowAdapter',
    'PDFExtractor',
]
