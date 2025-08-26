#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vector store service for managing embeddings and document retrieval.
"""
from typing import List, Tuple, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.settings import Settings


class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[Chroma] = None
    
    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Get or create OpenAI embeddings instance."""
        if self._embeddings is None:
            if self.settings.embedding_dim:
                self._embeddings = OpenAIEmbeddings(
                    model=self.settings.embedding_model,
                    dimensions=self.settings.embedding_dim
                )
            else:
                self._embeddings = OpenAIEmbeddings(
                    model=self.settings.embedding_model
                )
        return self._embeddings
    
    def get_vectorstore(self) -> Chroma:
        """Get or create Chroma vector store instance."""
        if self._vectorstore is None:
            embeddings = self._get_embeddings()
            self._vectorstore = Chroma(
                collection_name=self.settings.collection,
                embedding_function=embeddings,
                persist_directory=str(self.settings.chroma_dir),
            )
        return self._vectorstore
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter dictionary for metadata
            
        Returns:
            List of (Document, score) tuples
        """
        vectorstore = self.get_vectorstore()
        return vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def get_collection_count(self) -> Optional[int]:
        """Get the number of vectors in the collection."""
        try:
            vectorstore = self.get_vectorstore()
            return vectorstore._collection.count()  # type: ignore[attr-defined]
        except Exception:
            return None


def create_year_filter(year_from: Optional[int], year_to: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma 'where' filter for year range.
    
    Chroma expects exactly ONE operator per field, so we combine bounds via $and.
    
    Args:
        year_from: Start year (inclusive)
        year_to: End year (inclusive)
        
    Returns:
        Filter dictionary or None if no filtering needed
    """
    if year_from is None and year_to is None:
        return None

    # Only lower bound
    if year_from is not None and year_to is None:
        return {"year": {"$gte": year_from}}

    # Only upper bound
    if year_from is None and year_to is not None:
        return {"year": {"$lte": year_to}}

    # Both bounds
    if year_from is not None and year_to is not None:
        # If equal, can use $eq
        if year_from == year_to:
            return {"year": {"$eq": year_from}}
        # Use $and for range
        return {
            "$and": [
                {"year": {"$gte": year_from}},
                {"year": {"$lte": year_to}},
            ]
        }
