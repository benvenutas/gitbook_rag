"""
app.py
Minimal FastAPI app exposing a RAG question-answering endpoint.
"""

from fastapi import FastAPI, HTTPException
from fastapi import Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional
from rag_chain import create_retrieval_chain
from populate_store import bootstrap_chroma
from langchain.schema import Document
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)

logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    """Structured metadata for document chunks."""
    url: Optional[str] = Field(None, description="Source page URL.")
    section: Optional[str] = Field(None, description="Documentation section or category.")
    chunk_index: Optional[int] = Field(None, description="Chunk position in the document.")
    lastmod: Optional[datetime] = Field(None, description="Last modification timestamp of the page.")
    source: Optional[str] = Field(None, description="Original content source identifier.")
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Other metadata fields.")

    @classmethod
    def from_raw(cls, metadata: Dict[str, Any]) -> "DocumentMetadata":
        """Convert raw metadata dict from LangChain into structured fields."""
        known_fields = {"url", "section", "chunk_index", "lastmod", "source"}
        base = {k: metadata.get(k) for k in known_fields if k in metadata}

        # Handle lastmod normalization
        if isinstance(base.get("lastmod"), str):
            try:
                base["lastmod"] = datetime.fromisoformat(base["lastmod"])
            except ValueError:
                base["lastmod"] = None

        extra = {k: v for k, v in metadata.items() if k not in known_fields}
        return cls(**base, extra=extra)


class DocumentModel(BaseModel):
    """Schema for retrievable document chunks returned by the RAG pipeline."""
    page_content: str
    metadata: DocumentMetadata

    @classmethod
    def from_langchain(cls, doc: Document):
        """Convert LangChain Document (from Chroma) to API-safe schema."""
        return cls(
            page_content=doc.page_content.strip(),
            metadata=DocumentMetadata.from_raw(doc.metadata or {}),
        )

class QueryRequest(BaseModel):
    """Incoming request body with validation."""
    question: str = Field(..., description="Natural language question to query the documentation.")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Ensure question is non-empty, human-readable, and reasonable length."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) < 5:
            raise ValueError("Question must be at least 5 characters long.")
        if len(v) > 500:
            raise ValueError("Question too long (max 500 characters).")
        return v

    class Config:
        extra = "forbid"

        json_schema_extra = {
            "example": {
                "question": "How do I get IP addresses for Oxylabs integration?"
            }
        }

class QueryResponse(BaseModel):
    """Response containing the generated answer and source documents."""
    answer: str
    sources: List[DocumentModel]

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Use the ping command to find the IP address for integration...",
                "sources": [
                    {
                        "page_content": "### Using ISP, Datacenter, or Dedicated Datacenter Proxies...",
                        "metadata": {
                            "url": "https://developers.oxylabs.io/proxies/integration-guides/get-ip-address-for-integrations",
                            "section": "proxies",
                            "chunk_index": 0,
                            "lastmod": "2025-09-16T09:31:04.577000+00:00",
                            "source": "llms.txt",
                        },
                    }
                ],
            }
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event — runs before and after the app lifecycle."""
    logger.info("🔍 Checking Chroma vector store...")
    bootstrap_chroma(os.getenv("CHROMA_DIR"))
    logger.info("✅ Chroma vector store is ready.")

    # Initialize QA chain and store in app state
    app.state.qa_chain = create_retrieval_chain(persist_dir=os.getenv("CHROMA_DIR"), llm_model=os.getenv("LLM_MODEL"), search_arguments={"k": 3})

    yield  # 🚀 Application runs here

    # Optional cleanup logic
    logger.info("🧹 Shutting down API gracefully.")

app = FastAPI(
    title="Oxylabs RAG API",
    description="Retrieve answers from the Oxylabs documentation knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.post("/query", response_model=QueryResponse, tags=["retrieval"])
async def query_rag(request: Request, payload: QueryRequest):
    """Answer user questions using RAG pipeline."""
    try:
        qa_chain = request.app.state.qa_chain
        result = qa_chain.invoke({"query": payload.question})
        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])

        sources = [DocumentModel.from_langchain(doc) for doc in source_docs]

        logger.info(
            f"✅ Generated answer ({len(answer.split())} words) "
            f"from {len(sources)} source chunks."
        )

        logger.debug(f"🔍 Query: {payload.question}")
        logger.debug(f"💬 Answer preview: {answer[:300]}...")  # truncate long outputs
        logger.debug(f"📚 Sources: {[s.metadata.url for s in sources if s.metadata.url]}")
        
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.exception(f"Error while processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "RAG API is running."}
