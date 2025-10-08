"""
scraper.py
Downloads and cleans markdown content for discovered pages.
"""

import logging
from typing import List, Optional, Callable
import requests
from data_ingestion.models import Page
from data_ingestion.utils import make_session

logger = logging.getLogger(__name__)


class DocScraper:
    """Downloads and cleans markdown for documentation pages."""

    def __init__(self, timeout: int = 15, clean_fn: Optional[Callable[[str], str]] = None):
        self.session = make_session()
        self.timeout = timeout
        self.clean_fn = clean_fn

    def scrape(self, pages: List[Page], limit: Optional[int] = None) -> List[Page]:
        """Fetch markdown for each page."""
        for i, page in enumerate(pages):
            if limit and i >= limit:
                break
            page.load(self.session, timeout=self.timeout, clean_fn=self.clean_fn)
        logger.info(f"✅ Scraped {sum(p.loaded for p in pages)} / {len(pages)} pages")
        return pages
