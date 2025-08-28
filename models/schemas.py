#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class AskRequest(BaseModel):
    """Request model for asking questions to the RAG system."""
    
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    year_from: Optional[int] = Field(None, ge=2009, le=2026, description="Filter documents from this year")
    year_to: Optional[int] = Field(None, le=2026, description="Filter documents to this year")
    top_k: Optional[int] = Field(None, ge=6, le=50, description="Number of documents to retrieve")
    use_mmr: Optional[bool] = Field(False, description="Apply Maximal Marginal Relevance for diversity")
    mmr_lambda: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="MMR lambda parameter (relevance vs diversity)")
    
    # New agentic features
    use_reasoning: Optional[bool] = Field(True, description="Enable multi-step reasoning and query decomposition")
    show_reasoning_trace: Optional[bool] = Field(False, description="Include reasoning trace in response")
    reasoning_depth: Optional[int] = Field(3, ge=1, le=5, description="Maximum reasoning depth for complex questions")
    
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
                "question": "2022'den 2024'e kadar çevresel performans nasıl değişti ve ana etkenler nelerdi?",
                "year_from": 2022,
                "year_to": 2024,
                "top_k": 10,
                "use_mmr": True,
                "mmr_lambda": 0.7,
                "use_reasoning": True,
                "show_reasoning_trace": True,
                "reasoning_depth": 3
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


class ReasoningStepResponse(BaseModel):
    """Response model for individual reasoning steps."""
    step_number: int = Field(..., description="Step number in reasoning process")
    description: str = Field(..., description="Step description")
    question: str = Field(..., description="Question being analyzed")
    analysis: str = Field(..., description="Step analysis")
    conclusion: str = Field(..., description="Step conclusion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this step")


class SubQuestionResponse(BaseModel):
    """Response model for sub-questions and their answers."""
    question: str = Field(..., description="Sub-question text")
    question_type: str = Field(..., description="Type of sub-question")
    priority: int = Field(..., description="Priority level")
    answer: str = Field(..., description="Answer to sub-question")
    sources: List[Source] = Field(..., description="Sources used for this sub-question")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this answer")
    reasoning_steps: List[ReasoningStepResponse] = Field(..., description="Reasoning steps for this sub-question")


class DecompositionResponse(BaseModel):
    """Response model for query decomposition information."""
    original_question: str = Field(..., description="Original complex question")
    question_type: str = Field(..., description="Identified question type")
    requires_decomposition: bool = Field(..., description="Whether decomposition was needed")
    reasoning: str = Field(..., description="Decomposition reasoning")
    sub_questions: List[SubQuestionResponse] = Field(..., description="Generated sub-questions and answers")


class AskResponse(BaseModel):
    """Response model for RAG system answers."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents used")
    latency_ms: int = Field(..., ge=0, description="Response latency in milliseconds")
    
    # New agentic features
    question_type: Optional[str] = Field(None, description="Identified question type")
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence score")
    decomposition: Optional[DecompositionResponse] = Field(None, description="Query decomposition details")
    reasoning_trace: Optional[str] = Field(None, description="Human-readable reasoning trace")
    validation_report: Optional[Dict[str, Any]] = Field(None, description="Answer validation report")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "2022'den 2024'e kadar çevresel performansta önemli iyileşmeler gözlenmiştir...",
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
                "latency_ms": 2150,
                "question_type": "comparison",
                "overall_confidence": 0.87,
                "decomposition": {
                    "original_question": "2022'den 2024'e kadar çevresel performans nasıl değişti?",
                    "question_type": "comparison",
                    "requires_decomposition": True,
                    "reasoning": "Karşılaştırmalı soru, yıllara göre ayrıştırma gerekiyor",
                    "sub_questions": []
                },
                "reasoning_trace": "🔍 SORU ANALİZİ:\nSoru Tipi: comparison\n...",
                "validation_report": {
                    "overall_score": 0.85,
                    "consistency_score": 0.87,
                    "completeness_score": 1.0,
                    "confidence_score": 0.87
                }
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
