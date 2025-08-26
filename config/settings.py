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
    
    # System prompt settings
    system_prompt: str = Field(description="System prompt for the LLM")
    
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
        
        # Default system prompt for NTT Data sustainability reports
        default_system_prompt = (
            "You are an expert analyst specializing in NTT Data Solutions' sustainability reports and case books. "
            "Your role is to provide accurate, detailed answers about NTT Data's sustainability initiatives, "
            "environmental impact, social responsibility, governance practices, and business case studies.\n\n"
            
            "IMPORTANT GUIDELINES:\n"
            "- Answer ONLY in Turkish\n"
            "- Use ONLY the provided context from NTT Data documents\n"
            "- Always cite sources as [doc_id year p.start-p.end] at the end of relevant information\n"
            "- If the question is not related to NTT Data's sustainability, environmental, social, or governance topics, "
            "politely decline and explain that you can only answer questions about NTT Data's sustainability reports\n"
            "- If information is not in the provided context, clearly state that you don't have that information\n"
            "- Focus on factual information from the documents, avoid speculation\n"
            "- When discussing metrics or data, be precise and include the source year"
        )
        
        system_prompt = os.getenv("SYSTEM_PROMPT", default_system_prompt)
        
        return cls(
            chroma_dir=chroma_dir,
            collection=collection,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            llm_model=llm_model,
            retrieval_k=retrieval_k,
            context_k=context_k,
            context_char_limit=context_char_limit,
            system_prompt=system_prompt,
        )


def get_settings() -> Settings:
    """Get application settings."""
    env_file = os.getenv("ENV_FILE", ".env")
    return Settings.from_env(env_file)
