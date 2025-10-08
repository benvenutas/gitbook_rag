"""
utils.py
Utility functions for network requests and sitemap/llms parsing.
"""

import logging
import re
import requests
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET
from typing import List, Dict, Optional
from dateutil import parser as date_parser
from data_ingestion.models import Page
import data_ingestion.config as config
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


def safe_request(url: str, timeout: int = config.REQUEST_TIMEOUT) -> Optional[str]:
    try:
        resp = make_session().get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"⚠️ Failed request: {url} ({e})")
        return None


def extract_metadata_from_sitemap(sitemap_url: str) -> Dict[str, dict]:
    text = safe_request(sitemap_url)
    if not text:
        return {}
    root = ET.fromstring(text)
    ns = {'ns': root.tag[root.tag.find("{")+1:root.tag.find("}")]} if "{" in root.tag else {}
    meta = {}
    for url_tag in root.findall("ns:url" if ns else "url", ns):
        loc_elem = url_tag.find("ns:loc" if ns else "loc", ns)
        if not loc_elem.text:
            continue
        loc = loc_elem.text.strip()
        lastmod, priority = None, None
        lastmod_elem = url_tag.find("ns:lastmod" if ns else "lastmod", ns)
        if lastmod_elem is not None and lastmod_elem.text:
            try:
                lastmod = date_parser.parse(lastmod_elem.text.strip())
            except Exception:
                pass
        priority_elem = url_tag.find("ns:priority" if ns else "priority", ns)
        if priority_elem is not None and priority_elem.text:
            try:
                priority = float(priority_elem.text.strip())
            except ValueError:
                pass
        meta[loc] = {"lastmod": lastmod, "priority": priority}
    logger.info(f"Extracted metadata for {len(meta)} pages")
    return meta


def extract_pages_from_llms(llms_url: str, base_url: str) -> List[Page]:
    text = safe_request(llms_url)
    if not text:
        return []
    pattern = re.compile(r"- \[([^\]]+)\]\((/[^\)]+\.md)\)")
    matches = pattern.findall(text)
    pages = []
    for title, md_path in matches:
        web_url = urljoin(base_url, md_path.replace(".md", ""))
        md_url = urljoin(base_url, md_path)
        section = urlparse(web_url).path.strip("/").split("/")[0]
        pages.append(Page(url=web_url, md_url=md_url, title=title, section=section))
    logger.info(f"Extracted {len(pages)} pages from llms.txt")
    return pages


def merge_pages_with_sitemap(pages: List[Page], meta: Dict[str, dict]) -> List[Page]:
    for p in pages:
        if str(p.url) in meta:
            p.lastmod = meta[str(p.url)].get("lastmod")
            p.priority = meta[str(p.url)].get("priority")
    return pages

def convert_html_tables_to_markdown(md: str) -> str:
    """
    Detect and convert HTML <table> blocks into readable Markdown-formatted tables.
    Preserves Markdown syntax and structure.
    """
    soup = BeautifulSoup(md, "html.parser")

    for table in soup.find_all("table"):
        headers = [h.get_text(strip=True) for h in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if cells:
                rows.append(cells)

        # --- Build Markdown table string
        table_lines = ["**Table:**"]
        if headers:
            header_line = " | ".join(headers)
            separator = " | ".join(["---"] * len(headers))
            table_lines.append(header_line)
            table_lines.append(separator)
        for r in rows:
            table_lines.append(" | ".join(r))

        table_md = "\n".join(table_lines)
        table.replace_with("\n" + table_md + "\n")

    return str(soup)

def clean_markdown_content(md: str) -> str:
    # Remove full lines that contain only GitBook tags
    md = re.sub(r"^\s*\{%\s*.*?\s*%\}\s*$", "", md, flags=re.MULTILINE)

    md = convert_html_tables_to_markdown(md)
    
    return md.strip()
