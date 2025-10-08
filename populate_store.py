"""
populate_store.py
───────────────────────────────────────────────
Bootstraps the Chroma vector store by:
1. Crawling the Oxylabs docs via llms.txt + sitemap.
2. Scraping and cleaning Markdown content.
3. Chunking into semantic sections.
4. Embedding and persisting into Chroma DB.

This file can be run standalone OR imported into FastAPI startup.
"""

import os
import logging
from dotenv import load_dotenv

from data_ingestion.crawler import DocCrawler
from data_ingestion.scrapper import DocScraper
from data_ingestion.chunker import MarkdownChunker
from data_ingestion.utils import clean_markdown_content
from vector_store.chroma_store import ChromaStore
from data_ingestion.config import BASE_URL, SITEMAP_PATH, LLMS_PATH, EMBED_MODEL, CHROMA_DIR


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_chroma(persist_dir: str):
    """Initialize Chroma vector store if missing."""
    # Skip if already built
    if os.path.exists(persist_dir) and any(os.scandir(persist_dir)):
        logger.info(f"✅ Found existing Chroma index at {persist_dir}, skipping bootstrap.")
        return

    logger.info("🚀 No Chroma index found — building embeddings...")

    # Crawl docs
    crawler = DocCrawler(BASE_URL, SITEMAP_PATH, LLMS_PATH)
    pages = crawler.crawl(include_sections=["proxies", "advanced-proxy-solutions"])
    logger.info(f"🌐 Crawled {len(pages)} pages.")

    # Scrape content
    scraper = DocScraper(clean_fn=clean_markdown_content)
    scraped_pages = scraper.scrape(pages)
    loaded = [p for p in scraped_pages if p.loaded]
    logger.info(f"🧾 Scraped {len(loaded)} successfully loaded pages.")

    # Chunk
    chunker = MarkdownChunker()
    chunks = chunker.chunk_pages(loaded)
    logger.info(f"✂️ Created {len(chunks)} chunks total.")

    # Build Chroma index
    chroma_store = ChromaStore(persist_dir=persist_dir, embedding_model=EMBED_MODEL)
    chroma_store.build(chunks)
    logger.info(f"✅ Chroma vector store initialized and saved to {persist_dir}.")


if __name__ == "__main__":
    bootstrap_chroma(CHROMA_DIR)
