#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain-only INGEST script (marker-free): reads config from .env, embeds JSONL chunks, stores in local Chroma.
- Scans data/processed/<YEAR>/chunks_*.jsonl (override via .env)
- Uses OpenAIEmbeddings (text-embedding-3-large by default)
- Persists to an on-disk Chroma directory (no server)
- No QA — ingestion only

ENV (.env) keys (examples):
    # OpenAI
    OPENAI_API_KEY=sk-...
    EMBEDDING_MODEL=text-embedding-3-large
    # optional reduced dims for v3 embeddings (e.g., 1024, 512, 256)
    EMBEDDING_DIM=3072

    # Paths & names
    PROCESSED_ROOT=data/processed
    CHROMA_PERSIST_DIR=.chroma/ntt_reports_openai_v1
    COLLECTION_NAME=ntt_reports_openai_v1

    # Optional: custom .env path
    ENV_FILE=.env

Usage:
    pip install -U langchain-openai langchain-community chromadb langchain-core python-dotenv
    python ingest_chroma_env.py
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------- .env loading ----------------
try:
    from dotenv import load_dotenv
except Exception as e:
    raise SystemExit("Please install python-dotenv: pip install python-dotenv")

# Load .env (ENV_FILE overrides default path)
_ENV_PATH = os.getenv("ENV_FILE", ".env")
if Path(_ENV_PATH).exists():
    load_dotenv(_ENV_PATH)
else:
    # Fallback: try loading a default .env in CWD if present
    if Path(".env").exists():
        load_dotenv(".env")

# ---------------- Settings --------------------
@dataclass
class Settings:
    processed_root: Path
    persist_dir: Path
    collection_name: str
    embedding_model: str
    embedding_dim: int | None

    @classmethod
    def from_env(cls) -> "Settings":
        processed_root = Path(os.getenv("PROCESSED_ROOT", "data/processed")).resolve()
        persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma/ntt_reports_openai_v1")).resolve()
        collection_name = os.getenv("COLLECTION_NAME", "ntt_reports_openai_v1")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        dim_val = os.getenv("EMBEDDING_DIM")
        embedding_dim = int(dim_val) if dim_val and dim_val.isdigit() else None
        return cls(
            processed_root=processed_root,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

CFG = Settings.from_env()

# ---------------- Imports ---------------------
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------------- JSONL → Documents -----------

def load_jsonl_as_documents(jsonl_path: Path) -> tuple[list[Document], list[str]]:
    docs: list[Document] = []
    ids: list[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            meta = {
                "doc_id": rec.get("doc_id"),
                "year": rec.get("year"),
                "section_path": rec.get("section_path"),
                "page_start": rec.get("page_start"),
                "page_end": rec.get("page_end"),
                "source_path": rec.get("source_path"),
                "sha256": rec.get("sha256"),
            }
            docs.append(Document(page_content=text, metadata=meta))
            ids.append(rec.get("id") or f"{meta['doc_id']}::{len(ids)+1:04d}")
    return docs, ids

# ---------------- VectorStore factory ---------

def get_chroma(cfg: Settings) -> Chroma:
    # Pass dimensions if provided (OpenAI v3 embeddings support reduced dims)
    if cfg.embedding_dim:
        embeddings = OpenAIEmbeddings(model=cfg.embedding_model, dimensions=cfg.embedding_dim)
    else:
        embeddings = OpenAIEmbeddings(model=cfg.embedding_model)
    vs = Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
        persist_directory=str(cfg.persist_dir),
    )
    return vs

# ---------------- Ingest helpers --------------

def safe_upsert(vs: Chroma, documents: list[Document], ids: list[str]) -> int:
    if not documents:
        return 0
    try:
        vs.delete(ids=ids)
    except Exception:
        pass
    vs.add_documents(documents=documents, ids=ids)
    vs.persist()
    return len(documents)


def ingest_file(jsonl_path: Path, cfg: Settings = CFG) -> int:
    docs, ids = load_jsonl_as_documents(jsonl_path)
    if not docs:
        print(f"[SKIP] empty: {jsonl_path}")
        return 0
    vs = get_chroma(cfg)
    n = safe_upsert(vs, docs, ids)
    print(f"[OK] {jsonl_path} → {n} chunks → {cfg.collection_name}")
    return n


def ingest_all(root: Path | None = None, cfg: Settings = CFG) -> dict[str, int]:
    root = root or cfg.processed_root
    counts: dict[str, int] = {}
    files = sorted(root.rglob("chunks_*.jsonl"))
    if not files:
        print(f"[WARN] no JSONL files under {root}")
        return counts
    vs = get_chroma(cfg)
    total = 0
    for jf in files:
        docs, ids = load_jsonl_as_documents(jf)
        n = safe_upsert(vs, docs, ids)
        counts[str(jf)] = n
        total += n
        print(f"[OK] {jf} → {n} chunks")
    print(f"[DONE] upserted {total} chunks into '{cfg.collection_name}' at {cfg.persist_dir}")
    return counts

# ---------------- Entry point ---------------

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (load via .env or export)")
    ingest_all(CFG.processed_root, CFG)


if __name__ == "__main__":
    main()
