# 🧠 Oxylabs RAG API  
*A Retrieval-Augmented Generation pipeline for Oxylabs Developer Documentation.*

## 📘 Overview

This project demonstrates a complete **RAG (Retrieval-Augmented Generation)** system that ingests Oxylabs developer documentation, stores it in a **vector database (Chroma)**, and serves a **FastAPI** endpoint to answer natural language questions over the docs.

It is designed as a **take-home task** but follows production-grade engineering practices:
- Modular ingestion (crawler, scraper, chunker)
- Persistent embeddings via **Chroma**
- Query pipeline using **LangChain** + **OpenAI models**
- Containerized API for easy deployment

## 🏗️ System Architecture

    ┌─────────────────────────────────────┐
    │          data_ingestion             │
    │  ┌────────┬────────┬────────────┐   │
    │  │crawler │scraper │chunker     │   │
    └──┴────────┴────────┴────────────┘───┘
                        │ 
                  Markdown Pages
                        │ 
                        ▼
            🧩  Vectorization (ChromaStore)
                        │ 
                    Embeddings
                        │ 
                        ▼
         ┌─────────────────────────────┐
         │     RAG FastAPI Service     │
         │  Retrieval → Generation     │
         └──────────────┬──────────────┘
                        │
                        ▼
              JSON Answer + Sources

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/benvenutas/gitbook_rag.git
cd gitbook_rag
```

### 2. Create a .env file
The API and ingestion use environment variables for configuration.
```bash
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
CHROMA_DIR=./chroma_db
GITBOOK_BASE_URL=https://developers.oxylabs.io
GITBOOK_SITEMAP_PATH=sitemap-pages.xml
GITBOOK_LLMS_PATH=llms.txt
LOG_LEVEL=INFO
```

## 🐳 Running with Docker Compose
Build and start the API:
```bash
docker compose up -d --build
```

### Access the docs:
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/

## 🧾 API Endpoints
### `POST /query`
Retrieve an answer from Oxylabs documentation.

**Request:**
```json
{
  "question": "How do I get IP addresses for Oxylabs integration?"
}
```

**Response:**
```json
{
  "answer": "Use the ping command to find the IP address for integration...",
  "sources": [
    {
      "page_content": "### Using ISP, Datacenter, or Dedicated Datacenter Proxies...",
      "metadata": {
        "url": "https://developers.oxylabs.io/proxies/integration-guides/get-ip-address-for-integrations",
        "section": "proxies",
        "chunk_index": 0,
        "lastmod": "2025-09-16T09:31:04.577000+00:00",
        "source": "llms.txt"
      }
    }
  ]
}
```

### `GET /`
Health check.

**Returns:**
```json
{"status": "ok", "message": "RAG API is running."}
```

## 🧩 Key Components
| Module                         | Description                                             |
| ------------------------------ | ------------------------------------------------------- |
| `data_ingestion/crawler.py`    | Reads sitemap and text lists to find documentation URLs |
| `data_ingestion/scraper.py`    | Fetches Markdown files and cleans HTML                  |
| `data_ingestion/chunker.py`    | Splits Markdown into embedding-ready text chunks        |
| `vector_store/chroma_store.py` | Manages Chroma vector database and embeddings           |
| `rag_chain.py`                 | Defines LangChain retrieval + LLM pipeline              |
| `app.py`                       | FastAPI app exposing `/query` endpoint                  |
| `populate_store.py`            | Bootstraps Chroma with crawled and embedded docs        |


## Example Workflow
1. On first startup, the API:
    - Crawls and scrapes the Oxylabs documentation
    - Chunks and embeds pages into Chroma
    - Persists embeddings locally under ./chroma_db
2. Subsequent startups reuse the existing vector store without re-crawling.
3. Queries to /query are retrieved and answered using OpenAI’s LLM via LangChain.

## Logging
Logs are structured for clarity:
```yaml
2025-10-08 16:20:42 [INFO] app: 🧠 Received query: 'How do I get IP addresses for Oxylabs integration?'
2025-10-08 16:20:44 [INFO] app: ✅ Generated answer (82 words) from 3 source chunks.
2025-10-08 16:20:44 [DEBUG] app: 💬 Answer preview: "Use the ping command..."
2025-10-08 16:20:44 [DEBUG] app: 📚 Sources: ['https://developers.oxylabs.io/...']
```

## Local Development
**Run without Docker**
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Tech Stack
| Component        | Library                                          |
| ---------------- | ------------------------------------------------ |
| Framework        | FastAPI                                          |
| Vector DB        | Chroma                                           |
| LLM + Embeddings | OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) |
| Framework        | LangChain                                        |
| HTML Parsing     | BeautifulSoup4, lxml,                            |
| Containerization | Docker Compose                                   |

## Author
**Candidate B.G.**
Take-home task submission for Oxylabs.