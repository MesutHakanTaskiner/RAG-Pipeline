#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for configuration settings.
"""
import pytest
import os
from pathlib import Path
from config.settings import Settings, get_settings


def test_settings_from_env():
    """Test settings creation from environment variables."""
    # Set some test environment variables
    os.environ["CHROMA_PERSIST_DIR"] = "test_chroma"
    os.environ["COLLECTION_NAME"] = "test_collection"
    os.environ["LLM_MODEL"] = "gpt-3.5-turbo"
    
    settings = Settings.from_env()
    
    assert settings.chroma_dir == Path("test_chroma").resolve()
    assert settings.collection == "test_collection"
    assert settings.llm_model == "gpt-3.5-turbo"
    
    # Clean up
    del os.environ["CHROMA_PERSIST_DIR"]
    del os.environ["COLLECTION_NAME"]
    del os.environ["LLM_MODEL"]


def test_settings_defaults():
    """Test default settings values."""
    # Clear any existing environment variables
    env_vars_to_clear = [
        "CHROMA_PERSIST_DIR", "COLLECTION_NAME", "EMBEDDING_MODEL",
        "LLM_MODEL", "RETRIEVAL_K", "CONTEXT_K", "CONTEXT_CHAR_LIMIT"
    ]
    
    original_values = {}
    for var in env_vars_to_clear:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]
    
    try:
        settings = Settings.from_env()
        
        assert settings.chroma_dir == Path(".chroma/ntt_reports_openai_v1").resolve()
        assert settings.collection == "ntt_reports_openai_v1"
        assert settings.embedding_model == "text-embedding-3-large"
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.retrieval_k == 12
        assert settings.context_k == 3
        assert settings.context_char_limit == 1200
        
    finally:
        # Restore original environment variables
        for var, value in original_values.items():
            os.environ[var] = value


def test_get_settings():
    """Test the get_settings function."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert hasattr(settings, 'chroma_dir')
    assert hasattr(settings, 'collection')
    assert hasattr(settings, 'llm_model')


def test_settings_validation():
    """Test settings validation."""
    settings = Settings.from_env()
    
    # Test that required fields are present
    assert settings.chroma_dir is not None
    assert settings.collection is not None
    assert settings.embedding_model is not None
    assert settings.llm_model is not None
    assert settings.retrieval_k > 0
    assert settings.context_k > 0
    assert settings.context_char_limit > 0
