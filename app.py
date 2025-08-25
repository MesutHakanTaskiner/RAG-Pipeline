#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI RAG (LangChain + Chroma + OpenAI) — .env-config
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
    uvicorn rag_api_chroma:app --reload --port 8080
"""
from __future__ import annotations
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Load .env --------------------------------------------------------------
ENV_FILE = os.getenv("ENV_FILE", ".env")
if Path(ENV_FILE).exists():
    load_dotenv(ENV_FILE)
elif Path(".env").exists():
    load_dotenv(".env")

# --- Settings ---------------------------------------------------------------
@dataclass
class Settings:
    chroma_dir: Path
    collection: str
    embedding_model: str
    embedding_dim: Optional[int]
    llm_model: str
    retrieval_k: int
    context_k: int
    context_char_limit: int

    @classmethod
    def from_env(cls) -> "Settings":
        chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma/ntt_reports_openai_v1")).resolve()
        collection = os.getenv("COLLECTION_NAME", "ntt_reports_openai_v1")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        dim_val = os.getenv("EMBEDDING_DIM")
        embedding_dim = int(dim_val) if dim_val and dim_val.isdigit() else None
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        retrieval_k = int(os.getenv("RETRIEVAL_K", "12"))
        context_k = int(os.getenv("CONTEXT_K", "6"))
        context_char_limit = int(os.getenv("CONTEXT_CHAR_LIMIT", "1200"))
        return cls(
            chroma_dir=chroma_dir,
            collection=collection,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            llm_model=llm_model,
            retrieval_k=retrieval_k,
            context_k=context_k,
            context_char_limit=context_char_limit,
        )

CFG = Settings.from_env()

# --- LangChain imports ------------------------------------------------------
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Build vector store & LLM ----------------------------------------------

def make_embeddings() -> OpenAIEmbeddings:
    if CFG.embedding_dim:
        return OpenAIEmbeddings(model=CFG.embedding_model, dimensions=CFG.embedding_dim)
    return OpenAIEmbeddings(model=CFG.embedding_model)


def get_vectorstore() -> Chroma:
    embeddings = make_embeddings()
    vs = Chroma(
        collection_name=CFG.collection,
        embedding_function=embeddings,
        persist_directory=str(CFG.chroma_dir),
    )
    return vs


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CFG.llm_model, temperature=0.2)


# --- FastAPI schema ---------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    top_k: Optional[int] = None

class Source(BaseModel):
    doc_id: Optional[str]
    year: Optional[int]
    score: float
    page_start: Optional[int]
    page_end: Optional[int]
    section_path: Optional[str]
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: int

# --- Helpers ----------------------------------------------------------------

def _trim(txt: str, limit: int) -> str:
    t = re.sub(r"\s+", " ", txt).strip()
    return (t[:limit] + "…") if len(t) > limit else t


def _format_context(docs: List[Tuple[Document, float]], limit: int) -> str:
    blocks = []
    for doc, _score in docs:
        tag = f"[{doc.metadata.get('doc_id')} {doc.metadata.get('year')} p.{doc.metadata.get('page_start')}-{doc.metadata.get('page_end')}]"
        blocks.append(tag + "\n" + _trim(doc.page_content, limit))
    return "\n\n".join(blocks)


def _year_filter(y_from: int | None, y_to: int | None):
    """
    Build a Chroma 'where' filter. Chroma expects exactly ONE operator per field
    -> combine bounds via $and.
    """
    if y_from is None and y_to is None:
        return None

    # only lower bound
    if y_from is not None and y_to is None:
        return {"year": {"$gte": y_from}}

    # only upper bound
    if y_from is None and y_to is not None:
        return {"year": {"$lte": y_to}}

    # both bounds -> use $and
    if y_from is not None and y_to is not None:
        # if equal, can use $eq
        if y_from == y_to:
            return {"year": {"$eq": y_from}}
        return {
            "$and": [
                {"year": {"$gte": y_from}},
                {"year": {"$lte": y_to}},
            ]
        }


# --- App --------------------------------------------------------------------
app = FastAPI(title="RAG API (LangChain + Chroma + OpenAI)")

# Lazy init so the app can start even if Chroma dir is missing; we check on /health
_VS: Optional[Chroma] = None
_LLM: Optional[ChatOpenAI] = None


def _ensure_clients():
    global _VS, _LLM
    if _VS is None:
        _VS = get_vectorstore()
    if _LLM is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; please create a .env or export it")
        _LLM = get_llm()


@app.get("/health")
def health():
    try:
        _ensure_clients()
        # Try to count vectors
        try:
            count = _VS._collection.count()  # type: ignore[attr-defined]
        except Exception:
            count = None
        return {
            "status": "ok",
            "collection": CFG.collection,
            "persist_dir": str(CFG.chroma_dir),
            "vectors": count,
            "retrieval_k": CFG.retrieval_k,
            "context_k": CFG.context_k,
            "embedding_model": CFG.embedding_model,
            "llm_model": CFG.llm_model,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    _ensure_clients()
    t0 = time.time()

    k = req.top_k or CFG.retrieval_k
    flt = _year_filter(req.year_from, req.year_to)

    # 1) retrieve with scores from Chroma
    results = _VS.similarity_search_with_score(
        req.question,
        k=k,
        filter=flt,
    )

    # 2) pick context
    top_ctx = results[: CFG.context_k]

    # 3) build prompt
    system_msg = (
        "You are a careful analyst. Answer in Turkish, using only the provided context. "
        "Cite sources as [doc_id year p.start-p.end] at the end. If not in context, say you don't know."
    )
    context_block = _format_context(top_ctx, CFG.context_char_limit)
    user_msg = f"Context:\n{context_block}\n\nQuestion: {req.question}\n\nAnswer in Turkish."

    # 4) generate
    comp = _LLM.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    answer = comp.content.strip() if hasattr(comp, "content") else str(comp)

    latency_ms = int((time.time() - t0) * 1000)

    # Build sources output
    sources: List[Source] = []
    for doc, score in top_ctx:
        sources.append(Source(
            doc_id=doc.metadata.get("doc_id"),
            year=doc.metadata.get("year"),
            score=float(score),
            page_start=doc.metadata.get("page_start"),
            page_end=doc.metadata.get("page_end"),
            section_path=doc.metadata.get("section_path"),
            text=doc.page_content,
        ))

    return AskResponse(answer=answer, sources=sources, latency_ms=latency_ms)

# Run: uvicorn rag_api_chroma:app --reload --port 8080
