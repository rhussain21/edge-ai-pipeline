"""
PDFExtractor — download and extract text from PDF files.

General-purpose PDF handling for the ingestion pipeline. Supports:
  - Downloading PDFs from any URL (with retries, streaming, content-type checks)
  - Extracting text via PyMuPDF (primary) with pdfplumber fallback
  - NUL-byte stripping (prevents PostgreSQL insert failures)
  - arXiv URL helpers for routing decisions

Usage (extract text from local file):
    from tools.pdf_extractor import PDFExtractor

    text, page_count = PDFExtractor.extract_text("path/to/paper.pdf")

Usage (download + extract):
    extractor = PDFExtractor(pdf_dir="media/pdf")
    result = extractor.download_and_extract(
        pdf_url="https://example.com/paper.pdf",
        title="Some Paper Title",
    )
    if result:
        print(result["text"][:200])
        print(result["pdf_path"])
"""

import logging
import os
import re
import time
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Download PDFs and extract full text."""

    USER_AGENT = "Mozilla/5.0 (compatible; IndustrySignalsBot/1.0)"
    MAX_RETRIES = 2
    RETRY_DELAY = 3.0  # seconds between retries
    TIMEOUT = 60  # PDF downloads can be large

    def __init__(self, pdf_dir: str = "media/pdf"):
        self.pdf_dir = pdf_dir
        os.makedirs(self.pdf_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def download_and_extract(
        self,
        pdf_url: str,
        title: str = "Untitled",
        source_url: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Download a PDF and extract full text.

        Args:
            pdf_url: Direct URL to the PDF file.
            title: Paper/document title for filename generation.
            source_url: Original page URL (e.g. arXiv /abs/ page). Stored in result.

        Returns:
            Dict with keys: text, pdf_path, source_url, pdf_url, page_count, char_count
            or None on failure.
        """
        if not pdf_url:
            logger.error(f"No PDF URL provided for: {title}")
            return None

        # Download PDF
        pdf_path = self._download_pdf(pdf_url, title)
        if not pdf_path:
            return None

        # Extract text
        text, page_count = self.extract_text(pdf_path)
        if not text or len(text.strip()) < 200:
            logger.warning(
                f"Insufficient text extracted from PDF "
                f"({len(text) if text else 0} chars): {title}"
            )
            return None

        logger.info(
            f"PDF extracted: {page_count} pages, "
            f"{len(text)} chars — {title[:60]}"
        )

        return {
            "text": text,
            "pdf_path": pdf_path,
            "source_url": source_url or pdf_url,
            "pdf_url": pdf_url,
            "page_count": page_count,
            "char_count": len(text),
        }

    @staticmethod
    def extract_text(pdf_path: str) -> Tuple[str, int]:
        """Extract text from a local PDF file.

        Tries PyMuPDF first (faster, better layout preservation),
        falls back to pdfplumber. Strips NUL bytes.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            (text, page_count) tuple. Empty string and 0 on failure.
        """
        text, pages = PDFExtractor._extract_with_pymupdf(pdf_path)
        if text and len(text.strip()) > 200:
            return text, pages

        logger.info(f"PyMuPDF extraction insufficient, trying pdfplumber: {pdf_path}")
        text, pages = PDFExtractor._extract_with_pdfplumber(pdf_path)
        return text, pages

    # ── arXiv URL helpers ─────────────────────────────────────────────

    @staticmethod
    def is_arxiv_url(url: str) -> bool:
        """Check if a URL points to arXiv."""
        if not url:
            return False
        host = urlparse(url).hostname or ""
        return "arxiv.org" in host

    @staticmethod
    def construct_arxiv_pdf_url(abs_url: str) -> str:
        """Construct PDF URL from an arXiv /abs/ URL.

        http://arxiv.org/abs/2401.12345v1 → http://arxiv.org/pdf/2401.12345v1
        """
        if not abs_url:
            return ""
        return re.sub(r'/abs/', '/pdf/', abs_url)

    # ── PDF download ───────────────────────────────────────────────────

    def _download_pdf(self, pdf_url: str, title: str) -> Optional[str]:
        """Download PDF to disk with retries. Returns filepath or None."""
        # Build filename from title
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'\s+', '_', clean_title).strip('_')
        if len(clean_title) > 180:
            clean_title = clean_title[:180]
        filename = f"{clean_title}.pdf"
        filepath = os.path.join(self.pdf_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            logger.info(f"PDF already on disk: {filename}")
            return filepath

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.debug(f"PDF download attempt {attempt}/{self.MAX_RETRIES}: {pdf_url}")
                resp = requests.get(
                    pdf_url,
                    timeout=self.TIMEOUT,
                    headers={"User-Agent": self.USER_AGENT},
                    stream=True,
                )
                resp.raise_for_status()

                # Verify we actually got a PDF
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" not in content_type and "octet-stream" not in content_type:
                    logger.warning(
                        f"Server returned non-PDF content-type: {content_type} "
                        f"for {pdf_url}"
                    )
                    # Still try — some servers serve PDFs with wrong content-type

                # Stream to disk
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)

                file_size = os.path.getsize(filepath)
                if file_size < 1000:
                    logger.warning(f"PDF suspiciously small ({file_size} bytes), retrying")
                    os.remove(filepath)
                    continue

                logger.info(f"PDF downloaded: {filename} ({file_size / 1024:.0f} KB)")
                return filepath

            except requests.RequestException as e:
                logger.warning(f"PDF download error (attempt {attempt}): {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)

        logger.error(f"PDF download failed after {self.MAX_RETRIES} attempts: {pdf_url}")
        return None

    # ── Text extraction backends ──────────────────────────────────────

    @staticmethod
    def _extract_with_pymupdf(pdf_path: str) -> Tuple[str, int]:
        """Extract text using PyMuPDF (fitz). Returns (text, page_count)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.debug("PyMuPDF (fitz) not installed")
            return "", 0

        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    pages.append(page_text)
            doc.close()

            text = "\n\n".join(pages)
            # Light cleanup: collapse excessive whitespace, remove form feeds
            text = re.sub(r'\f', '\n', text)
            text = re.sub(r'\n{4,}', '\n\n\n', text)
            # Strip NUL bytes that some PDFs contain (breaks PostgreSQL)
            text = text.replace('\x00', '')

            return text, len(pages)

        except Exception as e:
            logger.warning(f"PyMuPDF extraction error: {e}")
            return "", 0

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: str) -> Tuple[str, int]:
        """Extract text using pdfplumber (fallback). Returns (text, page_count)."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed — cannot extract PDF text")
            return "", 0

        try:
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text)

            text = "\n\n".join(pages)
            # Strip NUL bytes
            text = text.replace('\x00', '')
            return text, len(pages)

        except Exception as e:
            logger.warning(f"pdfplumber extraction error: {e}")
            return "", 0
