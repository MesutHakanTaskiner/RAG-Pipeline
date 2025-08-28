#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Pydantic models and schemas.
"""
import pytest
from pydantic import ValidationError
from models.schemas import (
    AskRequest, AskResponse, Source, HealthResponse,
    ReasoningStepResponse, SubQuestionResponse, DecompositionResponse
)


def test_ask_request_valid():
    """Test valid AskRequest creation."""
    request = AskRequest(
        question="Test question?",
        year_from=2020,
        year_to=2023,
        use_mmr=True,
        mmr_lambda=0.7
    )
    
    assert request.question == "Test question?"
    assert request.year_from == 2020
    assert request.year_to == 2023
    assert request.use_mmr is True
    assert request.mmr_lambda == 0.7


def test_ask_request_validation_errors():
    """Test AskRequest validation errors."""
    # Test empty question
    with pytest.raises(ValidationError):
        AskRequest(question="")
    
    # Test question too long
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 1001)
    
    # Test invalid year range
    with pytest.raises(ValidationError):
        AskRequest(question="Test?", year_from=2025, year_to=2020)
    
    # Test invalid mmr_lambda
    with pytest.raises(ValidationError):
        AskRequest(question="Test?", mmr_lambda=1.5)


def test_source_model():
    """Test Source model creation and validation."""
    source = Source(
        doc_id="test_doc",
        year=2023,
        score=0.85,
        page_start=10,
        page_end=15,
        text="Sample text content"
    )
    
    assert source.doc_id == "test_doc"
    assert source.year == 2023
    assert source.score == 0.85
    assert source.page_start == 10
    assert source.page_end == 15
    assert source.text == "Sample text content"


def test_source_page_validation():
    """Test Source page range validation."""
    # Valid page range
    source = Source(
        score=0.8,
        text="Test",
        page_start=5,
        page_end=10
    )
    assert source.page_start == 5
    assert source.page_end == 10
    
    # Invalid page range
    with pytest.raises(ValidationError):
        Source(
            score=0.8,
            text="Test",
            page_start=10,
            page_end=5
        )


def test_ask_response_model():
    """Test AskResponse model creation."""
    sources = [
        Source(
            doc_id="doc1",
            year=2023,
            score=0.9,
            text="Test content"
        )
    ]
    
    response = AskResponse(
        answer="Test answer",
        sources=sources,
        latency_ms=1500,
        question_type="factual",
        overall_confidence=0.85
    )
    
    assert response.answer == "Test answer"
    assert len(response.sources) == 1
    assert response.latency_ms == 1500
    assert response.question_type == "factual"
    assert response.overall_confidence == 0.85


def test_health_response_model():
    """Test HealthResponse model creation."""
    # Successful health response
    health = HealthResponse(
        status="ok",
        collection="test_collection",
        vectors=1000,
        retrieval_k=12
    )
    
    assert health.status == "ok"
    assert health.collection == "test_collection"
    assert health.vectors == 1000
    assert health.retrieval_k == 12
    
    # Error health response
    error_health = HealthResponse(
        status="error",
        error="Connection failed"
    )
    
    assert error_health.status == "error"
    assert error_health.error == "Connection failed"


def test_reasoning_step_response():
    """Test ReasoningStepResponse model."""
    step = ReasoningStepResponse(
        step_number=1,
        description="Analysis step",
        question="What is the trend?",
        analysis="Detailed analysis",
        conclusion="Positive trend observed",
        confidence=0.8
    )
    
    assert step.step_number == 1
    assert step.description == "Analysis step"
    assert step.confidence == 0.8


def test_sub_question_response():
    """Test SubQuestionResponse model."""
    sources = [Source(score=0.8, text="Test content")]
    reasoning_steps = [
        ReasoningStepResponse(
            step_number=1,
            description="Test step",
            question="Sub question?",
            analysis="Analysis",
            conclusion="Conclusion",
            confidence=0.7
        )
    ]
    
    sub_response = SubQuestionResponse(
        question="Sub question?",
        question_type="factual",
        priority=1,
        answer="Sub answer",
        sources=sources,
        confidence=0.8,
        reasoning_steps=reasoning_steps
    )
    
    assert sub_response.question == "Sub question?"
    assert sub_response.question_type == "factual"
    assert sub_response.priority == 1
    assert len(sub_response.sources) == 1
    assert len(sub_response.reasoning_steps) == 1


def test_decomposition_response():
    """Test DecompositionResponse model."""
    sub_questions = [
        SubQuestionResponse(
            question="Sub question?",
            question_type="factual",
            priority=1,
            answer="Answer",
            sources=[Source(score=0.8, text="Test")],
            confidence=0.8,
            reasoning_steps=[]
        )
    ]
    
    decomposition = DecompositionResponse(
        original_question="Original question?",
        question_type="complex",
        requires_decomposition=True,
        reasoning="Complex question needs breakdown",
        sub_questions=sub_questions
    )
    
    assert decomposition.original_question == "Original question?"
    assert decomposition.question_type == "complex"
    assert decomposition.requires_decomposition is True
    assert len(decomposition.sub_questions) == 1
