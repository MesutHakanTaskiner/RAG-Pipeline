#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text processing utilities for the RAG application.
"""
import re
from typing import List, Tuple
from langchain_core.documents import Document


def trim_text(text: str, limit: int) -> str:
    """
    Trim text to a specified character limit.
    
    Args:
        text: Input text to trim
        limit: Maximum character limit
        
    Returns:
        Trimmed text with ellipsis if truncated
    """
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", text).strip()
    
    # Truncate if necessary
    if len(normalized) > limit:
        return normalized[:limit] + "…"
    
    return normalized


def format_context_block(
    documents_with_scores: List[Tuple[Document, float]], 
    char_limit: int
) -> str:
    """
    Format retrieved documents into a context block for the LLM.
    
    Args:
        documents_with_scores: List of (Document, score) tuples
        char_limit: Character limit per document chunk
        
    Returns:
        Formatted context string
    """
    blocks = []
    
    for doc, _score in documents_with_scores:
        # Create document tag with metadata
        doc_id = doc.metadata.get('doc_id', 'unknown')
        year = doc.metadata.get('year', 'unknown')
        page_start = doc.metadata.get('page_start', 'unknown')
        page_end = doc.metadata.get('page_end', 'unknown')
        
        tag = f"[{doc_id} {year} p.{page_start}-{page_end}]"
        
        # Trim document content
        trimmed_content = trim_text(doc.page_content, char_limit)
        
        # Combine tag and content
        block = f"{tag}\n{trimmed_content}"
        blocks.append(block)
    
    return "\n\n".join(blocks)


def extract_metadata_safely(document: Document, key: str, default=None):
    """
    Safely extract metadata from a document.
    
    Args:
        document: LangChain Document object
        key: Metadata key to extract
        default: Default value if key not found
        
    Returns:
        Metadata value or default
    """
    return document.metadata.get(key, default)


def validate_text_length(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """
    Validate text length is within acceptable bounds.
    
    Args:
        text: Text to validate
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        
    Returns:
        True if text length is valid
    """
    return min_length <= len(text.strip()) <= max_length
