from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from vector_store.chroma_store import ChromaStore
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Optional

def create_retrieval_chain(
    persist_dir: str = "./chroma_db",
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    search_arguments: Optional[dict] = None,
) -> RetrievalQA:
    """
    Build a RAG QA chain using a persisted Chroma store and ChatOpenAI model.
    Returns a RetrievalQA instance.
    """
    # Load or build Chroma store
    chroma = ChromaStore(persist_dir=persist_dir, embedding_model=embedding_model)
    store = chroma.load()
    if store is None:
        raise RuntimeError("No vector store found — build it first.")

    # Convert to retriever interface
    retriever = VectorStoreRetriever(vectorstore=store, search_kwargs=search_arguments)

    # Create LLM
    llm = ChatOpenAI(model=llm_model, temperature=0.0)

    # Build the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    return qa
