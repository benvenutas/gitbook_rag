"""
chroma_store.py
Embeds chunked Markdown documents and indexes them
into a persistent Chroma vector store for retrieval.
"""

from __future__ import annotations
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import logging
import os

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Manages creation, persistence, and querying of a Chroma vector store.
    """

    def __init__(
        self,
        persist_dir: str = "./vector_store",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Args:
            persist_dir: directory to persist the Chroma database
            embedding_model: OpenAI embedding model name
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        logger.info(f"🧠 Using embedding model: {embedding_model}")

    def build(self, docs: List[Document], batch_size: int = 100) -> Chroma:
        """Build Chroma store with automatic batching to avoid OpenAI token limit."""
        if not docs:
            logger.warning("⚠️ No documents to embed.")
            return None

        store = None
        total = len(docs)
        for i in range(0, total, batch_size):
            batch = docs[i:i + batch_size]
            logger.info(f"🔹 Embedding batch {i // batch_size + 1}/{(total // batch_size) + 1}")

            if store is None:
                store = Chroma.from_documents(
                    batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                    collection_name="docs_collection"
                )
            else:
                store.add_documents(batch)

        logger.info(f"✅ Built Chroma store with {total} documents.")
        return store


    def load(self) -> Optional[Chroma]:
        """Load an existing Chroma vector store."""
        if not os.path.exists(self.persist_dir):
            logger.warning(f"⚠️ No Chroma store found at {self.persist_dir}")
            return None

        logger.info(f"📂 Loading existing Chroma store from {self.persist_dir}")
        return Chroma(
            collection_name="docs_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def as_retriever(self, store: Chroma, k: int = 3):
        """Return the store as a LangChain retriever."""
        return store.as_retriever(search_kwargs={"k": k})

    def query(self, store: Chroma, query_text: str, k: int = 3):
        """Run a similarity search over the Chroma store."""
        logger.info(f"🔎 Searching for top-{k} results for query: {query_text!r}")
        results = store.similarity_search(query_text, k=k)
        for i, r in enumerate(results):
            print(f"\nResult {i+1} — {r.metadata.get('title', 'Untitled')}")
            print(f"URL: {r.metadata.get('url')}")
            print(f"Content: {r.page_content[:300]}...\n")
        return results

    def count(self, store: Chroma) -> int:
        """Return the number of vectors in the store."""
        try:
            return store._collection.count()
        except Exception:
            return 0
