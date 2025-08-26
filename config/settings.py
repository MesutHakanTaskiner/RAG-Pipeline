#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration settings for the RAG application.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class Settings(BaseModel):
    """Application settings with Pydantic validation."""
    
    # Vector store settings
    chroma_dir: Path = Field(description="Chroma persistence directory")
    collection: str = Field(description="Chroma collection name")
    
    # Embedding settings
    embedding_model: str = Field(description="OpenAI embedding model")
    embedding_dim: Optional[int] = Field(None, description="Embedding dimensions")
    
    # LLM settings
    llm_model: str = Field(description="OpenAI chat model")
    
    # Retrieval settings
    retrieval_k: int = Field(description="Number of documents to retrieve")
    context_k: int = Field(description="Number of documents to use as context")
    context_char_limit: int = Field(description="Character limit per context chunk")
    
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Settings":
        """Load settings from environment variables."""
        # Load environment file
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
        elif Path(".env").exists():
            load_dotenv(".env")
        
        # Parse environment variables
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


def get_settings() -> Settings:
    """Get application settings."""
    env_file = os.getenv("ENV_FILE", ".env")
    return Settings.from_env(env_file)
