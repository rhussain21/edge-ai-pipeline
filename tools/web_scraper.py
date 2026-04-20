"""
Web Scraper — stateless HTML-to-text utility.

Extracts clean, relevant text from HTML pages using BeautifulSoup.
Designed to be called by etl/sources.py during ingestion, or by agents directly.

Usage:
    from tools.web_scraper import WebScraper

    scraper = WebScraper()
    result = scraper.scrape("https://example.com/article")
    print(result['text'])       # Clean extracted text
    print(result['metadata'])   # Title, author, pub_date, etc.
"""

import logging
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

# Tags that typically contain the main article content
ARTICLE_TAGS = ['article', 'main', '[role="main"]']

# Tags to strip entirely (ads, nav, footers, scripts, etc.)
STRIP_TAGS = [
    'script', 'style', 'nav', 'footer', 'header', 'aside',
    'iframe', 'noscript', 'form', 'button', 'svg', 'figure',
    'figcaption', 'img', 'video', 'audio', 'source', 'picture',
]

# CSS classes/ids commonly associated with non-content elements
NOISE_PATTERNS = re.compile(
    r'(sidebar|widget|advert|promo|popup|modal|cookie|consent|newsletter|'
    r'social|share|comment|related|recommend|footer|nav|menu|breadcrumb|'
    r'signup|subscribe|banner|sponsor)',
    re.IGNORECASE
)

# Minimum text length to consider a page successfully scraped
MIN_TEXT_LENGTH = 100

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; IndustrySignalsBot/1.0)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


class WebScraper:
    """
    Stateless HTML scraper. Takes a URL or raw HTML string, returns clean text.

    Features:
        - Extracts article body preferentially (article/main tags)
        - Strips navigation, ads, scripts, footers
        - Extracts metadata (title, author, pub_date, description)
        - Normalizes whitespace
        - Falls back to full-page extraction if no article container found
    """

    def __init__(self, timeout: int = 30, max_content_length: int = 10 * 1024 * 1024):
        """
        Args:
            timeout: HTTP request timeout in seconds.
            max_content_length: Max response size in bytes (default 10MB).
        """
        self.timeout = timeout
        self.max_content_length = max_content_length

    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Fetch a URL and extract clean text + metadata.

        Returns:
            {
                'text': str,         # Clean extracted text
                'title': str,        # Page title
                'metadata': dict,    # author, pub_date, description, domain, url, word_count
                'success': bool,
                'error': str | None,
            }
        """
        try:
            response = requests.get(url, timeout=self.timeout, headers=DEFAULT_HEADERS)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'html' not in content_type and 'text' not in content_type:
                return self._error_result(f"Not an HTML page: {content_type}", url)

            if len(response.content) > self.max_content_length:
                return self._error_result("Content too large", url)

            return self.extract(response.text, url=url)

        except requests.RequestException as e:
            logger.error(f"Scrape failed for {url}: {e}")
            return self._error_result(str(e), url)

    def extract(self, html: str, url: str = '') -> Dict[str, Any]:
        """
        Extract clean text from raw HTML string.

        Args:
            html: Raw HTML content.
            url: Optional source URL (for metadata).

        Returns:
            Same dict structure as scrape().
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Strip unwanted tags entirely
        for tag_name in STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Strip noisy elements by class/id
        self._strip_noise(soup)

        # Extract metadata before further stripping
        metadata = self._extract_metadata(soup, url)
        title = metadata.get('title', '')

        # Try to find article body
        text = self._extract_article_text(soup)

        if not text or len(text) < MIN_TEXT_LENGTH:
            # Fallback: extract from <body> or entire soup
            body = soup.find('body') or soup
            text = self._get_clean_text(body)

        if not text or len(text) < MIN_TEXT_LENGTH:
            return self._error_result("Insufficient text extracted", url)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        metadata['word_count'] = len(text.split())

        return {
            'text': text,
            'title': title,
            'metadata': metadata,
            'success': True,
            'error': None,
        }

    def _extract_article_text(self, soup: BeautifulSoup) -> Optional[str]:
        """Try to extract text from article/main content containers."""
        # Try semantic article containers
        for selector in ARTICLE_TAGS:
            if selector.startswith('['):
                container = soup.select_one(selector)
            else:
                container = soup.find(selector)
            if container:
                text = self._get_clean_text(container)
                if text and len(text) >= MIN_TEXT_LENGTH:
                    return text

        # Try common content class patterns
        content_patterns = [
            {'class_': re.compile(r'(article|post|entry|content|story)[-_]?(body|text|content)?', re.I)},
            {'id': re.compile(r'(article|post|entry|content|story)[-_]?(body|text|content)?', re.I)},
        ]
        for pattern in content_patterns:
            container = soup.find('div', **pattern)
            if container:
                text = self._get_clean_text(container)
                if text and len(text) >= MIN_TEXT_LENGTH:
                    return text

        return None

    def _get_clean_text(self, element) -> str:
        """Extract text from a BeautifulSoup element, preserving paragraph structure."""
        # Get text with newlines between block elements
        lines = []
        for child in element.descendants:
            if isinstance(child, str):
                stripped = child.strip()
                if stripped:
                    lines.append(stripped)
            elif child.name in ('p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                'li', 'tr', 'blockquote', 'pre', 'section'):
                lines.append('\n')

        text = ' '.join(lines)
        return text.strip()

    def _strip_noise(self, soup: BeautifulSoup):
        """Remove elements whose class or id matches noise patterns."""
        # Collect first, then decompose — avoids mutating tree during iteration
        to_remove = []
        for element in soup.find_all(True):
            if element.decomposed if hasattr(element, 'decomposed') else (element.parent is None):
                continue
            classes = ' '.join(element.get('class', []) or [])
            element_id = element.get('id', '') or ''
            if NOISE_PATTERNS.search(classes) or NOISE_PATTERNS.search(element_id):
                to_remove.append(element)
        for element in to_remove:
            try:
                element.decompose()
            except Exception:
                pass

    def _extract_metadata(self, soup: BeautifulSoup, url: str = '') -> Dict[str, Any]:
        """Extract page metadata from meta tags, Open Graph, etc."""
        meta = {
            'url': url,
            'domain': urlparse(url).netloc if url else '',
        }

        # Title: try og:title, then <title>
        og_title = soup.find('meta', property='og:title')
        if og_title:
            meta['title'] = og_title.get('content', '')
        elif soup.title:
            meta['title'] = soup.title.string or ''
        else:
            h1 = soup.find('h1')
            meta['title'] = h1.get_text(strip=True) if h1 else ''

        # Author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        meta['author'] = author_meta.get('content', '') if author_meta else ''

        # Published date
        for attr in ['article:published_time', 'datePublished', 'og:article:published_time']:
            date_meta = soup.find('meta', property=attr) or soup.find('meta', attrs={'name': attr})
            if date_meta:
                meta['pub_date'] = date_meta.get('content', '')
                break
        else:
            # Try JSON-LD
            ld_script = soup.find('script', type='application/ld+json')
            if ld_script and ld_script.string:
                try:
                    import json
                    ld = json.loads(ld_script.string)
                    if isinstance(ld, dict):
                        meta['pub_date'] = ld.get('datePublished', '')
                        if not meta.get('author'):
                            author_ld = ld.get('author', {})
                            if isinstance(author_ld, dict):
                                meta['author'] = author_ld.get('name', '')
                except (ValueError, KeyError):
                    pass
            if 'pub_date' not in meta:
                meta['pub_date'] = ''

        # Description
        desc_meta = (
            soup.find('meta', property='og:description')
            or soup.find('meta', attrs={'name': 'description'})
        )
        meta['description'] = desc_meta.get('content', '') if desc_meta else ''

        return meta

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace/newlines into clean paragraphs."""
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Clean up space around newlines
        text = re.sub(r' *\n *', '\n', text)
        return text.strip()

    @staticmethod
    def _error_result(error_msg: str, url: str = '') -> Dict[str, Any]:
        """Return a standardized error result."""
        return {
            'text': '',
            'title': '',
            'metadata': {'url': url, 'domain': urlparse(url).netloc if url else ''},
            'success': False,
            'error': error_msg,
        }
