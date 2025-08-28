#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for utility functions.
"""
import pytest
from langchain_core.documents import Document
from utils.text_processing import (
    trim_text, format_context_block, extract_metadata_safely, validate_text_length
)


def test_trim_text():
    """Test text trimming functionality."""
    # Test normal text
    text = "This is a normal text"
    result = trim_text(text, 50)
    assert result == "This is a normal text"
    
    # Test text that needs trimming
    long_text = "This is a very long text that should be trimmed"
    result = trim_text(long_text, 20)
    assert len(result) <= 21  # 20 chars + ellipsis
    assert result.endswith("…")
    
    # Test text with extra whitespace
    messy_text = "  This   has    extra   spaces  "
    result = trim_text(messy_text, 50)
    assert result == "This has extra spaces"
    
    # Test empty text
    result = trim_text("", 10)
    assert result == ""


def test_format_context_block():
    """Test context block formatting."""
    # Create test documents
    doc1 = Document(
        page_content="This is the first document content.",
        metadata={
            "doc_id": "doc1",
            "year": 2023,
            "page_start": 1,
            "page_end": 2
        }
    )
    
    doc2 = Document(
        page_content="This is the second document content with more text.",
        metadata={
            "doc_id": "doc2",
            "year": 2024,
            "page_start": 5,
            "page_end": 6
        }
    )
    
    documents_with_scores = [(doc1, 0.9), (doc2, 0.8)]
    
    result = format_context_block(documents_with_scores, 100)
    
    # Check that both documents are included
    assert "[doc1 2023 p.1-2]" in result
    assert "[doc2 2024 p.5-6]" in result
    assert "first document content" in result
    assert "second document content" in result
    
    # Test with character limit
    result_limited = format_context_block(documents_with_scores, 20)
    assert len(result_limited.split('\n\n')[0].split('\n')[1]) <= 21  # 20 chars + ellipsis


def test_format_context_block_missing_metadata():
    """Test context block formatting with missing metadata."""
    doc = Document(
        page_content="Document without complete metadata.",
        metadata={"doc_id": "partial_doc"}  # Missing year, page info
    )
    
    documents_with_scores = [(doc, 0.7)]
    result = format_context_block(documents_with_scores, 100)
    
    assert "[partial_doc unknown p.unknown-unknown]" in result
    assert "Document without complete metadata." in result


def test_extract_metadata_safely():
    """Test safe metadata extraction."""
    doc = Document(
        page_content="Test content",
        metadata={
            "doc_id": "test_doc",
            "year": 2023,
            "score": 0.85
        }
    )
    
    # Test existing key
    assert extract_metadata_safely(doc, "doc_id") == "test_doc"
    assert extract_metadata_safely(doc, "year") == 2023
    
    # Test missing key with default
    assert extract_metadata_safely(doc, "missing_key", "default") == "default"
    assert extract_metadata_safely(doc, "missing_key") is None
    
    # Test missing key without default
    assert extract_metadata_safely(doc, "nonexistent") is None


def test_validate_text_length():
    """Test text length validation."""
    # Valid text
    assert validate_text_length("Valid text", 1, 100) is True
    assert validate_text_length("Short", 1, 10) is True
    
    # Text too short
    assert validate_text_length("", 1, 100) is False
    assert validate_text_length("   ", 5, 100) is False  # Whitespace only
    
    # Text too long
    assert validate_text_length("Too long text", 1, 5) is False
    
    # Edge cases
    assert validate_text_length("Exact", 5, 5) is True
    assert validate_text_length("X", 1, 1) is True
    
    # Test with custom bounds
    assert validate_text_length("Medium length text", 10, 20) is True
    assert validate_text_length("Short", 10, 20) is False
    assert validate_text_length("This is way too long for the limit", 10, 20) is False


def test_validate_text_length_whitespace_handling():
    """Test that text length validation handles whitespace correctly."""
    # Leading/trailing whitespace should be stripped
    assert validate_text_length("  valid  ", 1, 10) is True
    assert validate_text_length("  toolongtext  ", 1, 5) is False
    
    # Only whitespace should be considered empty
    assert validate_text_length("   \n\t  ", 1, 10) is False


def test_format_context_block_empty_input():
    """Test context block formatting with empty input."""
    result = format_context_block([], 100)
    assert result == ""


def test_format_context_block_single_document():
    """Test context block formatting with single document."""
    doc = Document(
        page_content="Single document content.",
        metadata={
            "doc_id": "single",
            "year": 2023,
            "page_start": 10,
            "page_end": 10
        }
    )
    
    documents_with_scores = [(doc, 0.95)]
    result = format_context_block(documents_with_scores, 100)
    
    assert "[single 2023 p.10-10]" in result
    assert "Single document content." in result
    assert result.count('\n\n') == 0  # No double newlines for single doc
