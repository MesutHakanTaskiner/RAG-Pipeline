#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Maximal Marginal Relevance (MMR) implementation for document re-ranking.

MMR balances relevance and diversity by selecting documents that are:
1. Relevant to the query
2. Diverse from already selected documents

Formula: MMR = λ * sim(Di, Q) - (1-λ) * max(sim(Di, Dj)) for j in S
Where:
- λ (lambda): relevance vs diversity trade-off parameter (0-1)
- sim(Di, Q): similarity between document Di and query Q
- sim(Di, Dj): similarity between document Di and already selected document Dj
- S: set of already selected documents
"""
import numpy as np
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config.settings import Settings


class MMRService:
    """Service for applying Maximal Marginal Relevance to document ranking."""
    
    def __init__(self, settings: Settings, lambda_param: float = 0.7):
        """
        Initialize MMR service.
        
        Args:
            settings: Application settings
            lambda_param: Trade-off parameter between relevance and diversity (0-1)
                         Higher values favor relevance, lower values favor diversity
        """
        self.settings = settings
        self.lambda_param = lambda_param
        self._embeddings: Optional[OpenAIEmbeddings] = None
    
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
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Ensure similarity is in [0, 1] range
        return max(0.0, min(1.0, float(similarity)))
    
    def _get_document_embeddings(self, documents: List[Document]) -> List[np.ndarray]:
        """
        Get embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings_service = self._get_embeddings()
        texts = [doc.page_content for doc in documents]
        embeddings = embeddings_service.embed_documents(texts)
        return [np.array(emb) for emb in embeddings]
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query string.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        embeddings_service = self._get_embeddings()
        embedding = embeddings_service.embed_query(query)
        return np.array(embedding)
    
    def apply_mmr(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float]],
        k: int,
        lambda_param: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Apply MMR to re-rank documents for better relevance-diversity balance.
        
        Args:
            query: Original search query
            documents_with_scores: List of (Document, similarity_score) tuples
            k: Number of documents to select
            lambda_param: Override default lambda parameter
            
        Returns:
            Re-ranked list of (Document, MMR_score) tuples
        """
        if not documents_with_scores:
            return []
        
        if k <= 0:
            return []
        
        if k >= len(documents_with_scores):
            return documents_with_scores
        
        # Use provided lambda or default
        lambda_val = lambda_param if lambda_param is not None else self.lambda_param
        
        # Extract documents and original scores
        documents = [doc for doc, _ in documents_with_scores]
        original_scores = [score for _, score in documents_with_scores]
        
        # Get embeddings
        query_embedding = self._get_query_embedding(query)
        doc_embeddings = self._get_document_embeddings(documents)
        
        # Calculate query-document similarities
        query_similarities = []
        for doc_emb in doc_embeddings:
            sim = self._cosine_similarity(query_embedding, doc_emb)
            query_similarities.append(sim)
        
        # MMR selection algorithm
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # Select first document (highest query similarity)
        if remaining_indices:
            best_idx = max(remaining_indices, key=lambda i: query_similarities[i])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Select remaining documents using MMR
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for i in remaining_indices:
                # Relevance component: similarity to query
                relevance = query_similarities[i]
                
                # Diversity component: max similarity to already selected docs
                max_similarity = 0.0
                for j in selected_indices:
                    sim = self._cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                    max_similarity = max(max_similarity, sim)
                
                # MMR formula
                mmr_score = lambda_val * relevance - (1 - lambda_val) * max_similarity
                mmr_scores.append((i, mmr_score))
            
            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Build result with MMR scores
        result = []
        for idx in selected_indices:
            doc = documents[idx]
            # Calculate final MMR score for this document
            relevance = query_similarities[idx]
            max_similarity = 0.0
            for other_idx in selected_indices:
                if other_idx != idx:
                    sim = self._cosine_similarity(doc_embeddings[idx], doc_embeddings[other_idx])
                    max_similarity = max(max_similarity, sim)
            
            mmr_score = lambda_val * relevance - (1 - lambda_val) * max_similarity
            result.append((doc, float(mmr_score)))
        
        return result
    
    def apply_mmr_simple(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float]],
        k: int,
        lambda_param: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Simplified MMR that uses original similarity scores instead of re-computing embeddings.
        This is faster but less accurate than the full MMR implementation.
        
        Args:
            query: Original search query
            documents_with_scores: List of (Document, similarity_score) tuples
            k: Number of documents to select
            lambda_param: Override default lambda parameter
            
        Returns:
            Re-ranked list of (Document, adjusted_score) tuples
        """
        if not documents_with_scores:
            return []
        
        if k <= 0:
            return []
        
        if k >= len(documents_with_scores):
            return documents_with_scores
        
        # Use provided lambda or default
        lambda_val = lambda_param if lambda_param is not None else self.lambda_param
        
        # Simple diversity penalty based on text similarity
        selected = []
        remaining = list(documents_with_scores)
        
        # Select first document (highest original score)
        if remaining:
            best_doc = max(remaining, key=lambda x: x[1])
            selected.append(best_doc)
            remaining.remove(best_doc)
        
        # Select remaining documents with diversity penalty
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_doc = None
            
            for doc, score in remaining:
                # Calculate diversity penalty based on text overlap
                diversity_penalty = 0.0
                for selected_doc, _ in selected:
                    # Simple text similarity based on common words
                    doc_words = set(doc.page_content.lower().split())
                    selected_words = set(selected_doc.page_content.lower().split())
                    
                    if doc_words and selected_words:
                        overlap = len(doc_words.intersection(selected_words))
                        total = len(doc_words.union(selected_words))
                        text_sim = overlap / total if total > 0 else 0.0
                        diversity_penalty = max(diversity_penalty, text_sim)
                
                # Apply MMR-like scoring
                adjusted_score = lambda_val * score - (1 - lambda_val) * diversity_penalty
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_doc = (doc, adjusted_score)
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove((best_doc[0], [x[1] for x in documents_with_scores if x[0] == best_doc[0]][0]))
        
        return selected
