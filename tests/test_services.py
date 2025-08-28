#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for service layer components.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

from services.vector_store import VectorStoreService, create_year_filter
from services.llm import LLMService
from services.mmr import MMRService
from config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
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


class TestVectorStoreService:
    """Test cases for VectorStoreService."""
    
    def test_create_year_filter_no_bounds(self):
        """Test year filter creation with no bounds."""
        result = create_year_filter(None, None)
        assert result is None
    
    def test_create_year_filter_lower_bound_only(self):
        """Test year filter creation with only lower bound."""
        result = create_year_filter(2020, None)
        expected = {"year": {"$gte": 2020}}
        assert result == expected
    
    def test_create_year_filter_upper_bound_only(self):
        """Test year filter creation with only upper bound."""
        result = create_year_filter(None, 2023)
        expected = {"year": {"$lte": 2023}}
        assert result == expected
    
    def test_create_year_filter_equal_bounds(self):
        """Test year filter creation with equal bounds."""
        result = create_year_filter(2022, 2022)
        expected = {"year": {"$eq": 2022}}
        assert result == expected
    
    def test_create_year_filter_range(self):
        """Test year filter creation with range."""
        result = create_year_filter(2020, 2023)
        expected = {
            "$and": [
                {"year": {"$gte": 2020}},
                {"year": {"$lte": 2023}}
            ]
        }
        assert result == expected
    
    @patch('services.vector_store.OpenAIEmbeddings')
    @patch('services.vector_store.Chroma')
    def test_vector_store_initialization(self, mock_chroma, mock_embeddings, mock_settings):
        """Test vector store service initialization."""
        service = VectorStoreService(mock_settings)
        
        # Test lazy loading - embeddings not created yet
        assert service._embeddings is None
        assert service._vectorstore is None
        
        # Test embeddings creation
        embeddings = service._get_embeddings()
        mock_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            dimensions=1536
        )
        assert service._embeddings is not None
    
    @patch('services.vector_store.OpenAIEmbeddings')
    @patch('services.vector_store.Chroma')
    def test_vector_store_creation(self, mock_chroma, mock_embeddings, mock_settings):
        """Test vector store creation."""
        service = VectorStoreService(mock_settings)
        
        vectorstore = service.get_vectorstore()
        
        mock_chroma.assert_called_once()
        assert service._vectorstore is not None
    
    @patch('services.vector_store.OpenAIEmbeddings')
    @patch('services.vector_store.Chroma')
    def test_similarity_search_with_score(self, mock_chroma, mock_embeddings, mock_settings):
        """Test similarity search with scores."""
        # Setup mocks
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="test", metadata={"year": 2023}), 0.9)
        ]
        
        service = VectorStoreService(mock_settings)
        
        results = service.similarity_search_with_score("test query", 5, {"year": 2023})
        
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(
            query="test query",
            k=5,
            filter={"year": 2023}
        )
        assert len(results) == 1
        assert results[0][1] == 0.9


class TestLLMService:
    """Test cases for LLMService."""
    
    @patch('services.llm.ChatOpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_llm_initialization(self, mock_chat_openai, mock_settings):
        """Test LLM service initialization."""
        service = LLMService(mock_settings)
        
        # Test lazy loading
        assert service._llm is None
        
        # Test LLM creation
        llm = service.get_llm()
        mock_chat_openai.assert_called_once_with(
            model="gpt-3.5-turbo",
            temperature=0.2
        )
        assert service._llm is not None
    
    @patch.dict('os.environ', {}, clear=True)
    def test_llm_missing_api_key(self, mock_settings):
        """Test LLM service with missing API key."""
        service = LLMService(mock_settings)
        
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            service.get_llm()
    
    @patch('services.llm.ChatOpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_answer(self, mock_chat_openai, mock_settings):
        """Test answer generation."""
        # Setup mock
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        service = LLMService(mock_settings)
        
        result = service.generate_answer("System prompt", "User prompt")
        
        mock_llm.invoke.assert_called_once_with([
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"}
        ])
        assert result == "Test answer"
    
    def test_create_rag_prompts(self, mock_settings):
        """Test RAG prompt creation."""
        service = LLMService(mock_settings)
        
        system_prompt, user_prompt = service.create_rag_prompts(
            "What is AI?", 
            "Context about AI"
        )
        
        assert system_prompt == mock_settings.system_prompt
        assert "Context about AI" in user_prompt
        assert "What is AI?" in user_prompt
        assert "Answer in Turkish:" in user_prompt


class TestMMRService:
    """Test cases for MMRService."""
    
    def test_mmr_initialization(self, mock_settings):
        """Test MMR service initialization."""
        service = MMRService(mock_settings)
        
        assert service.settings == mock_settings
        assert service.lambda_param == 0.7
        assert service._embeddings is None
    
    def test_mmr_custom_lambda(self, mock_settings):
        """Test MMR service with custom lambda."""
        service = MMRService(mock_settings, lambda_param=0.5)
        assert service.lambda_param == 0.5
    
    def test_cosine_similarity(self, mock_settings):
        """Test cosine similarity calculation."""
        import numpy as np
        
        service = MMRService(mock_settings)
        
        # Test identical vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        similarity = service._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        similarity = service._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test zero vectors
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 0, 0])
        similarity = service._cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_apply_mmr_empty_input(self, mock_settings):
        """Test MMR with empty input."""
        service = MMRService(mock_settings)
        
        result = service.apply_mmr("query", [], 5)
        assert result == []
    
    def test_apply_mmr_k_larger_than_input(self, mock_settings):
        """Test MMR when k is larger than input size."""
        service = MMRService(mock_settings)
        
        docs = [
            (Document(page_content="doc1", metadata={"doc_id": "1"}), 0.9)
        ]
        
        result = service.apply_mmr("query", docs, 5)
        assert len(result) == 1
        assert result == docs
    
    @patch('services.mmr.OpenAIEmbeddings')
    def test_apply_mmr_simple_empty_input(self, mock_embeddings, mock_settings):
        """Test simple MMR with empty input."""
        service = MMRService(mock_settings)
        
        result = service.apply_mmr_simple("query", [], 5)
        assert result == []
    
    @patch('services.mmr.OpenAIEmbeddings')
    def test_apply_mmr_simple_single_document(self, mock_embeddings, mock_settings):
        """Test simple MMR with single document."""
        service = MMRService(mock_settings)
        
        docs = [
            (Document(page_content="single doc", metadata={"doc_id": "1"}), 0.9)
        ]
        
        result = service.apply_mmr_simple("query", docs, 5)
        assert len(result) == 1
        assert result[0][0].page_content == "single doc"


class TestServiceIntegration:
    """Integration tests for services working together."""
    
    @patch('services.vector_store.OpenAIEmbeddings')
    @patch('services.vector_store.Chroma')
    @patch('services.llm.ChatOpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_services_integration(self, mock_chat_openai, mock_chroma, mock_embeddings, mock_settings):
        """Test services working together."""
        # Setup mocks
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="test content", metadata={"year": 2023}), 0.9)
        ]
        
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Generated answer"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Test integration
        vector_service = VectorStoreService(mock_settings)
        llm_service = LLMService(mock_settings)
        
        # Simulate a simple RAG pipeline
        docs = vector_service.similarity_search_with_score("test query", 3)
        assert len(docs) == 1
        
        answer = llm_service.generate_answer("system", "user")
        assert answer == "Generated answer"
    
    def test_year_filter_edge_cases(self):
        """Test year filter with edge cases."""
        # Test with very old year
        result = create_year_filter(1900, None)
        assert result == {"year": {"$gte": 1900}}
        
        # Test with future year
        result = create_year_filter(None, 2050)
        assert result == {"year": {"$lte": 2050}}
        
        # Test with same year (edge case)
        result = create_year_filter(2023, 2023)
        assert result == {"year": {"$eq": 2023}}
