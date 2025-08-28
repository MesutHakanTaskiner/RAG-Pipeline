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
            lambda_param: Trade-off parameter between relevance and diversity (0-1).
                          Higher values favor relevance, lower values favor diversity.
        """
        self.settings = settings
        self.lambda_param = lambda_param
        self._embeddings: Optional[OpenAIEmbeddings] = None
    
    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Get or create an OpenAI embeddings instance."""
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
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(vec1 / n1, vec2 / n2))
    
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
    
    def _get_document_embeddings_cached(self, documents: List[Document]) -> List[np.ndarray]:
        """
        Get embeddings for documents, using any cached embedding present in metadata.
        If missing, embed and persist the result into document metadata for reuse.
        """
        embs = [None] * len(documents)
        to_embed, to_idx = [], []
        for i, d in enumerate(documents):
            e = d.metadata.get("_embedding")
            if e is not None:
                embs[i] = np.array(e, dtype=np.float32)
            else:
                to_embed.append(d.page_content)
                to_idx.append(i)

        if to_embed:
            new_embs = self._get_embeddings().embed_documents(to_embed)
            for j, i in enumerate(to_idx):
                # If you want to persist permanently, do it at ingest time
                documents[i].metadata["_embedding"] = new_embs[j]
                embs[i] = np.array(new_embs[j], dtype=np.float32)
        return embs

    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get the embedding for a query string.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        embeddings_service = self._get_embeddings()
        embedding = embeddings_service.embed_query(query)
        return np.array(embedding)
    
    def apply_mmr(self, query: str, documents_with_scores: List[Tuple[Document, float]],
              k: int, lambda_param: Optional[float] = 0.7,
              unique_by: Optional[str] = "doc_id", max_per_doc: int = 1) -> List[Tuple[Document, float]]:

        if not documents_with_scores or k <= 0:
            return []
        if k >= len(documents_with_scores):
            return documents_with_scores

        lam = self.lambda_param if lambda_param is None else lambda_param
        lam = max(0.0, min(1.0, lam))

        docs = [d for d, _ in documents_with_scores]
        q_emb = self._get_query_embedding(query)
        d_embs = self._get_document_embeddings_cached(docs)

        # Query-to-document similarities (cosine)
        q_sims = [self._cosine_similarity(q_emb, e) for e in d_embs]

        selected_idx, remaining_idx = [], list(range(len(docs)))
        # De-duplication counter
        per_doc_counter = {}

        # First selection: pick the doc with highest query similarity
        best = max(remaining_idx, key=lambda i: q_sims[i])
        selected_idx.append(best)
        remaining_idx.remove(best)
        if unique_by:
            key = docs[best].metadata.get(unique_by)
            if key:
                per_doc_counter[key] = 1

        # Subsequent selections
        while len(selected_idx) < k and remaining_idx:
            candidates = []
            for i in remaining_idx:
                # De-duplication quota check
                if unique_by:
                    key = docs[i].metadata.get(unique_by)
                    if key and per_doc_counter.get(key, 0) >= max_per_doc:
                        continue

                rel = q_sims[i]
                div = max(self._cosine_similarity(d_embs[i], d_embs[j]) for j in selected_idx) if selected_idx else 0.0
                mmr_score = lam * rel - (1.0 - lam) * div
                candidates.append((i, mmr_score))

            if not candidates:
                break
            i_best, _ = max(candidates, key=lambda x: x[1])
            selected_idx.append(i_best)
            remaining_idx.remove(i_best)

            if unique_by:
                key = docs[i_best].metadata.get(unique_by)
                if key:
                    per_doc_counter[key] = per_doc_counter.get(key, 0) + 1

        # If we couldn't reach k, top off with highest q_sims from remaining (respecting de-dup)
        if len(selected_idx) < k:
            for i in sorted(remaining_idx, key=lambda x: q_sims[x], reverse=True):
                if unique_by:
                    key = docs[i].metadata.get(unique_by)
                    if key and per_doc_counter.get(key, 0) >= max_per_doc:
                        continue
                selected_idx.append(i)
                if unique_by:
                    per_doc_counter[key] = per_doc_counter.get(key, 0) + 1
                if len(selected_idx) >= k:
                    break

        # Output: (doc, mmr_score)
        out = []
        for idx in selected_idx:
            # Compute final MMR score (optional; you could return q_sims if preferred)
            rel = q_sims[idx]
            div = 0.0
            for j in selected_idx:
                if j == idx: 
                    continue
                div = max(div, self._cosine_similarity(d_embs[idx], d_embs[j]))
            mmr_score = lam * rel - (1.0 - lam) * div
            out.append((docs[idx], float(mmr_score)))
        return out

    
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
