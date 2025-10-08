"""
models.py
Data models representing documentation pages and their metadata.
"""

from __future__ import annotations
from typing import Optional, Callable
from datetime import datetime, timezone
from pydantic import BaseModel, Field, HttpUrl
import requests
import logging

logger = logging.getLogger(__name__)


class Page(BaseModel):
    """Represents a documentation page and its metadata."""

    url: HttpUrl
    md_url: HttpUrl
    title: Optional[str] = None
    md: Optional[str] = None
    lastmod: Optional[datetime] = None
    priority: Optional[float] = None
    section: Optional[str] = None
    source: str = Field(default="llms.txt")
    loaded: bool = Field(default=False)
    scraped_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

    def load(
        self,
        session: requests.Session,
        timeout: int = 15,
        clean_fn: Optional[Callable[[str], str]] = None,
    ) -> bool:
        """Fetch the Markdown content for this page."""
        try:
            r = session.get(str(self.md_url), timeout=timeout)
            r.raise_for_status()
            content = r.text
            if clean_fn:
                content = clean_fn(content)
            self.md = content
            self.loaded = True
            self.scraped_at = datetime.now(timezone.utc)
            logger.debug(f"Loaded markdown for {self.url}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {self.url}: {e}")
            self.loaded = False
            return False

    def to_metadata(self) -> dict:
        """Return metadata as plain primitives suitable for vector stores."""
        return {
            "url": str(self.url),
            "md_url": str(self.md_url),
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "priority": self.priority,
            "lastmod": self.lastmod.isoformat() if self.lastmod else None,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
            "loaded": bool(self.loaded),
        }


    def __str__(self):
        ts = self.lastmod.strftime("%Y-%m-%d") if self.lastmod else "n/a"
        return f"<Page title={self.title!r} section={self.section} lastmod={ts}>"
