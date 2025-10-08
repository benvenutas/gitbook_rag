"""
chunker.py
Splits cleaned Markdown documentation into semantically meaningful chunks
for vector embedding and retrieval.
"""

from __future__ import annotations
from typing import List, Optional
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain.schema import Document
import re
import logging
from data_ingestion.models import Page

logger = logging.getLogger(__name__)

class MarkdownChunker:
    """
    Combined header + recursive chunker for Oxylabs documentation.
    Split by Markdown headers.
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[tuple[str, str]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ]

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split Markdown text into retrievable chunks and attach metadata."""
        if not text:
            logger.warning("⚠️ Empty document provided to chunker.")
            return []

        # normalized = normalize_markdown_for_embedding(text)
        header_docs = self.header_splitter.split_text(text)

        docs: List[Document] = []

        for doc in header_docs:
            sub_chunks = self.recursive_splitter.split_documents([doc])
            for i, chunk in enumerate(sub_chunks):
                chunk.metadata.update(metadata or {})
                chunk.metadata["chunk_index"] = i
                docs.append(chunk)

        logger.info(f"✅ Created {len(docs)} chunks.")
        return docs

    def chunk_pages(self, pages: List[Page], limit: Optional[int] = None) -> List[Document]:
        """Chunk multiple Page objects (with .md loaded) into LangChain Documents."""
        all_docs: List[Document] = []

        for i, page in enumerate(pages):
            if limit and i >= limit:
                break

            if not getattr(page, "md", None):
                logger.warning(f"⚠️ Page has no markdown: {page.url}")
                continue

            metadata = page.to_metadata() if hasattr(page, "to_metadata") else {}
            docs = self.chunk(page.md, metadata=metadata)
            all_docs.extend(docs)
            logger.info(f"📄 {page.title}: {len(docs)} chunks")

        logger.info(f"✅ Total {len(all_docs)} document chunks.")
        return all_docs

# TODO: Consider extending chunking logic to better handle GitBook-specific content blocks.
#       Examples include:
#         - `{% hint %}` informational callouts
#         - `{% tabs %}` and `{% tab %}` code/language sections
#         - Embedded HTML tables (`<table>...</table>`)
#       These could be segmented using sub-level headers (e.g., `######`) or
#       regex-based semantic boundaries to improve retrieval granularity
#       while preserving Markdown hierarchy.
