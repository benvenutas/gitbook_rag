"""
crawler.py
───────────────────────────────────────────────
Discovers all documentation pages (URLs, metadata) from llms.txt and sitemap.xml.
"""

from __future__ import annotations
import logging
from typing import List, Optional
from urllib.parse import urljoin
from data_ingestion.models import Page
from data_ingestion.utils import (
    extract_pages_from_llms,
    extract_metadata_from_sitemap,
    merge_pages_with_sitemap,
)

logger = logging.getLogger(__name__)


class DocCrawler:
    """Find all documentation pages and metadata."""

    def __init__(self, base_url: str, sitemap_path: str, llms_path: str):
        self.base_url = base_url
        self.sitemap_path = sitemap_path
        self.llms_path = llms_path

    def crawl(self, include_sections: Optional[List[str]] = None) -> List[Page]:
        """Return a list of discovered pages (metadata only)."""
        sitemap_url = urljoin(self.base_url, self.sitemap_path)
        llms_url = urljoin(self.base_url, self.llms_path)

        logger.info(f"🌐 Crawling sources: {llms_url} + {sitemap_url}")

        llms_pages = extract_pages_from_llms(llms_url, self.base_url)
        sitemap_meta = extract_metadata_from_sitemap(sitemap_url)
        pages = merge_pages_with_sitemap(llms_pages, sitemap_meta)

        if include_sections:
            pages = [p for p in pages if p.section in include_sections]

        logger.info(f"✅ Discovered {len(pages)} pages")
        return pages
