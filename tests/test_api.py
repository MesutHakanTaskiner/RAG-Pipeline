#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for API routes and endpoints.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from api.routes import RAGRoutes, create_router
from config.settings import Settings
from models.schemas import AskRequest, AskResponse, HealthResponse
from pathlib import Path


@pytest.fixture
def mock_settings():
    """Create mock settings for API testing."""
    return Settings(
        chroma_dir=Path("test_chroma"),
        collection="test_collection",
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        llm_model="gpt-3.5-turbo",
        retrieval_k=5,
        context_k=3,
        context_char_limit=500,
        system_prompt="Test system prompt"
    )


@pytest.fixture
def mock_rag_routes(mock_settings):
    """Create mock RAG routes for testing."""
    with patch('api.routes.VectorStoreService'), \
         patch('api.routes.LLMService'), \
         patch('api.routes.MMRService'), \
         patch('api.routes.QueryDecomposer'), \
         patch('api.routes.ReasoningAgent'):
        return RAGRoutes(mock_settings)


class TestRAGRoutes:
    """Test cases for RAG API routes."""
    
    def test_rag_routes_initialization(self, mock_settings):
        """Test RAG routes initialization."""
        with patch('api.routes.VectorStoreService') as mock_vector, \
             patch('api.routes.LLMService') as mock_llm, \
             patch('api.routes.MMRService') as mock_mmr, \
             patch('api.routes.QueryDecomposer') as mock_decomposer, \
             patch('api.routes.ReasoningAgent') as mock_reasoning:
            
            routes = RAGRoutes(mock_settings)
            
            # Verify services are initialized
            mock_vector.assert_called_once_with(mock_settings)
            mock_llm.assert_called_once_with(mock_settings)
            mock_mmr.assert_called_once_with(mock_settings)
            mock_decomposer.assert_called_once_with(mock_settings)
            mock_reasoning.assert_called_once_with(mock_settings)
            
            # Verify router is created
            assert routes.router is not None
    
    @patch('api.routes.VectorStoreService')
    @patch('api.routes.LLMService')
    def test_health_check_success(self, mock_llm_service, mock_vector_service, mock_settings):
        """Test successful health check."""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_service.return_value = mock_vector_instance
        mock_vector_instance.get_vectorstore.return_value = Mock()
        mock_vector_instance.get_collection_count.return_value = 1000
        
        mock_llm_instance = Mock()
        mock_llm_service.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = Mock()
        
        with patch('api.routes.MMRService'), \
             patch('api.routes.QueryDecomposer'), \
             patch('api.routes.ReasoningAgent'):
            
            routes = RAGRoutes(mock_settings)
            response = routes.health_check()
            
            assert isinstance(response, HealthResponse)
            assert response.status == "ok"
            assert response.collection == "test_collection"
            assert response.vectors == 1000
            assert response.retrieval_k == 5
            assert response.context_k == 3
    
    @patch('api.routes.VectorStoreService')
    @patch('api.routes.LLMService')
    def test_health_check_error(self, mock_llm_service, mock_vector_service, mock_settings):
        """Test health check with error."""
        # Setup mocks to raise exception
        mock_vector_instance = Mock()
        mock_vector_service.return_value = mock_vector_instance
        mock_vector_instance.get_vectorstore.side_effect = Exception("Connection failed")
        
        with patch('api.routes.MMRService'), \
             patch('api.routes.QueryDecomposer'), \
             patch('api.routes.ReasoningAgent'):
            
            routes = RAGRoutes(mock_settings)
            response = routes.health_check()
            
            assert isinstance(response, HealthResponse)
            assert response.status == "error"
            assert "Connection failed" in response.error
    
    def test_ask_question_simple_rag(self, mock_rag_routes):
        """Test ask question with simple RAG (no reasoning)."""
        # Setup request
        request = AskRequest(
            question="What is AI?",
            use_reasoning=False,
            top_k=6
        )
        
        # Mock the _process_simple_rag method
        expected_response = AskResponse(
            answer="AI is artificial intelligence",
            sources=[],
            latency_ms=100
        )
        
        with patch.object(mock_rag_routes, '_process_simple_rag', return_value=expected_response):
            response = mock_rag_routes.ask_question(request)
            
            assert response == expected_response
    
    def test_ask_question_with_reasoning(self, mock_rag_routes):
        """Test ask question with reasoning enabled."""
        # Setup request
        request = AskRequest(
            question="What is AI?",
            use_reasoning=True,
            top_k=6
        )
        
        # Mock the _process_with_reasoning method
        expected_response = AskResponse(
            answer="AI is artificial intelligence with reasoning",
            sources=[],
            latency_ms=200,
            question_type="factual",
            overall_confidence=0.85
        )
        
        with patch.object(mock_rag_routes, '_process_with_reasoning', return_value=expected_response):
            response = mock_rag_routes.ask_question(request)
            
            assert response == expected_response
    
    def test_ask_question_error_handling(self, mock_rag_routes):
        """Test ask question error handling."""
        # Setup request
        request = AskRequest(
            question="What is AI?",
            use_reasoning=False
        )
        
        # Mock method to raise exception
        with patch.object(mock_rag_routes, '_process_simple_rag', side_effect=Exception("Processing error")):
            with pytest.raises(HTTPException) as exc_info:
                mock_rag_routes.ask_question(request)
            
            assert exc_info.value.status_code == 500
            assert "Error processing question" in str(exc_info.value.detail)
    
    @patch('api.routes.format_context_block')
    def test_process_simple_rag(self, mock_format_context, mock_rag_routes):
        """Test simple RAG processing."""
        from langchain_core.documents import Document

        # Setup mocks
        mock_doc = Document(page_content="AI content", metadata={"doc_id": "1", "year": 2023})
        mock_rag_routes.vector_service.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
        mock_format_context.return_value = "Formatted context"
        mock_rag_routes.llm_service.create_rag_prompts.return_value = ("system", "user")
        mock_rag_routes.llm_service.generate_answer.return_value = "Generated answer"

        # Create request
        request = AskRequest(question="What is AI?", top_k=6)

        # Test processing - mock time at the start_time parameter level
        response = mock_rag_routes._process_simple_rag(request, 0)

        assert response.answer == "Generated answer"
        # Since we're passing start_time=0 and the method calculates latency internally,
        # we should test that latency is calculated (non-negative)
        assert response.latency_ms >= 0
        assert len(response.sources) == 1
        assert response.sources[0].doc_id == "1"
    
    def test_build_sources(self, mock_rag_routes):
        """Test source building from context sources."""
        from langchain_core.documents import Document
        
        # Setup test data
        doc = Document(
            page_content="Test content",
            metadata={
                "doc_id": "test_doc",
                "year": 2023,
                "page_start": 1,
                "page_end": 2,
                "section_path": "test/section"
            }
        )
        context_sources = [(doc, 0.85)]
        
        # Test source building
        sources = mock_rag_routes._build_sources(context_sources)
        
        assert len(sources) == 1
        source = sources[0]
        assert source.doc_id == "test_doc"
        assert source.year == 2023
        assert source.score == 0.85
        assert source.page_start == 1
        assert source.page_end == 2
        assert source.section_path == "test/section"
        assert source.text == "Test content"
    
    def test_deduplicate_sources(self, mock_rag_routes):
        """Test source deduplication."""
        from langchain_core.documents import Document
        
        # Setup duplicate sources
        doc1 = Document(page_content="Content 1", metadata={"doc_id": "doc1", "page_start": 1})
        doc2 = Document(page_content="Content 2", metadata={"doc_id": "doc1", "page_start": 1})  # Duplicate
        doc3 = Document(page_content="Content 3", metadata={"doc_id": "doc2", "page_start": 1})
        
        all_sources = [(doc1, 0.9), (doc2, 0.8), (doc3, 0.7)]
        
        # Test deduplication
        unique_sources = mock_rag_routes._deduplicate_sources(all_sources)
        
        assert len(unique_sources) == 2  # Should remove one duplicate
        assert unique_sources[0][0].page_content == "Content 1"
        assert unique_sources[1][0].page_content == "Content 3"
    
    def test_format_context_for_reasoning(self, mock_rag_routes):
        """Test context formatting for reasoning."""
        from langchain_core.documents import Document
        
        # Setup test data
        doc = Document(
            page_content="This is test content for reasoning analysis.",
            metadata={"doc_id": "test_doc", "year": 2023, "page_start": 5}
        )
        context_sources = [(doc, 0.85)]
        
        # Test formatting
        formatted = mock_rag_routes._format_context_for_reasoning(context_sources)
        
        assert "[test_doc 2023 s.5]" in formatted
        assert "(Skor: 0.850)" in formatted
        assert "This is test content" in formatted


class TestCreateRouter:
    """Test router creation function."""
    
    def test_create_router(self, mock_settings):
        """Test router creation."""
        with patch('api.routes.RAGRoutes') as mock_rag_routes:
            mock_instance = Mock()
            mock_instance.router = Mock()
            mock_rag_routes.return_value = mock_instance
            
            router = create_router(mock_settings)
            
            mock_rag_routes.assert_called_once_with(mock_settings)
            assert router == mock_instance.router


class TestAPIIntegration:
    """Integration tests for API components."""
    
    @patch('api.routes.VectorStoreService')
    @patch('api.routes.LLMService')
    @patch('api.routes.MMRService')
    @patch('api.routes.QueryDecomposer')
    @patch('api.routes.ReasoningAgent')
    def test_full_api_flow(self, mock_reasoning, mock_decomposer, mock_mmr, 
                          mock_llm, mock_vector, mock_settings):
        """Test full API request flow."""
        from langchain_core.documents import Document
        
        # Setup comprehensive mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        mock_vector_instance.get_vectorstore.return_value = Mock()
        mock_vector_instance.get_collection_count.return_value = 1000
        mock_vector_instance.similarity_search_with_score.return_value = [
            (Document(page_content="Test content", metadata={"doc_id": "1", "year": 2023}), 0.9)
        ]
        
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = Mock()
        mock_llm_instance.create_rag_prompts.return_value = ("system", "user")
        mock_llm_instance.generate_answer.return_value = "Test answer"
        
        # Create routes and test
        routes = RAGRoutes(mock_settings)
        
        # Test health check
        health = routes.health_check()
        assert health.status == "ok"
        
        # Test simple question
        request = AskRequest(question="Test question?", use_reasoning=False)
        
        with patch('api.routes.format_context_block', return_value="context"):
            response = routes._process_simple_rag(request, 0)
        
        assert response.answer == "Test answer"
        # Test that latency is calculated (non-negative)
        assert response.latency_ms >= 0
        assert len(response.sources) == 1
