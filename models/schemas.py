#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class AskRequest(BaseModel):
    """Request model for asking questions to the RAG system."""
    
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    year_from: Optional[int] = Field(None, ge=2009, le=2026, description="Filter documents from this year")
    year_to: Optional[int] = Field(None, ge=2009, le=2026, description="Filter documents to this year")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of documents to retrieve")
    
    @validator('year_to')
    def validate_year_range(cls, v, values):
        """Ensure year_to is not less than year_from."""
        if v is not None and 'year_from' in values and values['year_from'] is not None:
            if v < values['year_from']:
                raise ValueError('year_to must be greater than or equal to year_from')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the main findings in the sustainability report?",
                "year_from": 2022,
                "year_to": 2024,
                "top_k": 10
            }
        }


class Source(BaseModel):
    """Source document information."""
    
    doc_id: Optional[str] = Field(None, description="Document identifier")
    year: Optional[int] = Field(None, description="Document year")
    score: float = Field(..., ge=0.0, description="Similarity score")
    page_start: Optional[int] = Field(None, ge=1, description="Starting page number")
    page_end: Optional[int] = Field(None, ge=1, description="Ending page number")
    section_path: Optional[str] = Field(None, description="Section path in document")
    text: str = Field(..., description="Source text content")
    
    @validator('page_end')
    def validate_page_range(cls, v, values):
        """Ensure page_end is not less than page_start."""
        if v is not None and 'page_start' in values and values['page_start'] is not None:
            if v < values['page_start']:
                raise ValueError('page_end must be greater than or equal to page_start')
        return v


class AskResponse(BaseModel):
    """Response model for RAG system answers."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents used")
    latency_ms: int = Field(..., ge=0, description="Response latency in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "Sürdürülebilirlik raporuna göre...",
                "sources": [
                    {
                        "doc_id": "sr_2023_cb",
                        "year": 2023,
                        "score": 0.85,
                        "page_start": 15,
                        "page_end": 16,
                        "section_path": "sustainability/environment",
                        "text": "Sample text from the document..."
                    }
                ],
                "latency_ms": 1250
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    collection: Optional[str] = Field(None, description="Vector collection name")
    persist_dir: Optional[str] = Field(None, description="Persistence directory")
    vectors: Optional[int] = Field(None, description="Number of vectors in collection")
    retrieval_k: Optional[int] = Field(None, description="Default retrieval count")
    context_k: Optional[int] = Field(None, description="Default context count")
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    llm_model: Optional[str] = Field(None, description="LLM model name")
    error: Optional[str] = Field(None, description="Error message if status is error")
