#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI RAG (LangChain + Chroma + OpenAI) — Refactored with Pydantic
- Loads local Chroma vector store (persisted during ingest)
- Uses OpenAI chat model to answer questions with citations from your documents
- Swagger UI available at /docs (default FastAPI)

ENV (.env) keys (examples):
    # OpenAI
    OPENAI_API_KEY=sk-...
    LLM_MODEL=gpt-4o-mini
    EMBEDDING_MODEL=text-embedding-3-large
    EMBEDDING_DIM=3072            # optional reduced dim if used during ingest

    # Vector store
    CHROMA_PERSIST_DIR=.chroma/ntt_reports_openai_v1
    COLLECTION_NAME=ntt_reports_openai_v1

    # Retrieval
    RETRIEVAL_K=12                # how many to fetch from vector DB
    CONTEXT_K=6                   # how many to pass to the LLM
    CONTEXT_CHAR_LIMIT=1200       # per chunk char cap in prompt

Run:
    pip install -U fastapi uvicorn python-dotenv langchain-openai langchain-community chromadb langchain-core
    uvicorn app:app --reload --port 8080
"""
from fastapi import FastAPI

from config.settings import get_settings
from api.routes import create_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Load settings
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="RAG API (LangChain + Chroma + OpenAI)",
        description="A modular RAG system with Pydantic validation",
        version="2.0.0"
    )
    
    # Include routes
    router = create_router(settings)
    app.include_router(router)
    
    return app


# Create the app instance
app = create_app()

# Run: uvicorn app:app --reload --port 8080
