"""
config.py
Central configuration for the crawler.
Default values are safe for local testing, but can be
overridden via environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Base crawling target
BASE_URL = os.getenv("GITBOOK_BASE_URL", "https://developers.oxylabs.io")

# Paths (relative to base)
SITEMAP_PATH = os.getenv("GITBOOK_SITEMAP_PATH", "sitemap-pages.xml")
LLMS_PATH = os.getenv("GITBOOK_LLMS_PATH", "llms.txt")

# Embedding model + Chroma DB directory
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Networking    
USER_AGENT = os.getenv(
    "CRAWLER_USER_AGENT",
    "Candidate-BG-RAG-Crawler/1.0 (take-home task)"
)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
