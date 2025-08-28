#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API routes for the RAG application.
"""
import time
from typing import List
from fastapi import APIRouter, HTTPException

from models.schemas import AskRequest, AskResponse, HealthResponse, Source
from services.vector_store import VectorStoreService, create_year_filter
from services.llm import LLMService
from services.mmr import MMRService
from utils.text_processing import format_context_block
from config.settings import Settings


class RAGRoutes:
    """RAG API routes handler."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_service = VectorStoreService(settings)
        self.llm_service = LLMService(settings)
        self.mmr_service = MMRService(settings)
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        self.router.get("/health", response_model=HealthResponse)(self.health_check)
        self.router.post("/ask", response_model=AskResponse)(self.ask_question)
    
    def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        try:
            # Try to initialize services
            vectorstore = self.vector_service.get_vectorstore()
            llm = self.llm_service.get_llm()
            
            # Get collection count
            vector_count = self.vector_service.get_collection_count()
            
            return HealthResponse(
                status="ok",
                collection=self.settings.collection,
                persist_dir=str(self.settings.chroma_dir),
                vectors=vector_count,
                retrieval_k=self.settings.retrieval_k,
                context_k=self.settings.context_k,
                embedding_model=self.settings.embedding_model,
                llm_model=self.settings.llm_model,
            )
        
        except Exception as e:
            return HealthResponse(
                status="error",
                error=str(e)
            )
    
    def ask_question(self, request: AskRequest) -> AskResponse:
        """Process a question and return an answer with sources."""
        start_time = time.time()
        
        try:
            # Determine retrieval count
            k = request.top_k or self.settings.retrieval_k
            
            # Create year filter
            year_filter = create_year_filter(request.year_from, request.year_to)
            
            # Retrieve documents with scores
            results = self.vector_service.similarity_search_with_score(
                query=request.question,
                k=k,
                filter_dict=year_filter
            )

            # Apply MMR if requested
            if request.use_mmr and results:
                # Apply MMR to get diverse results
                mmr_results = self.mmr_service.apply_mmr(
                    query=request.question,
                    documents_with_scores=results,
                    k=self.settings.context_k,
                    lambda_param=request.mmr_lambda
                )
                top_context = mmr_results
            else:
                # Select top context documents without MMR
                top_context = results[:self.settings.context_k]
        
            # Format context for LLM
            context_block = format_context_block(
                top_context, 
                self.settings.context_char_limit
            )
            
            # Create prompts
            system_prompt, user_prompt = self.llm_service.create_rag_prompts(
                request.question, 
                context_block
            )
            
            # Generate answer
            answer = self.llm_service.generate_answer(system_prompt, user_prompt)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Build sources
            sources: List[Source] = []
            for doc, score in top_context:
                sources.append(Source(
                    doc_id=doc.metadata.get("doc_id"),
                    year=doc.metadata.get("year"),
                    score=float(score),
                    page_start=doc.metadata.get("page_start"),
                    page_end=doc.metadata.get("page_end"),
                    section_path=doc.metadata.get("section_path"),
                    text=doc.page_content,
                ))
            
            return AskResponse(
                answer=answer,
                sources=sources,
                latency_ms=latency_ms
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


def create_router(settings: Settings) -> APIRouter:
    """Create and return the API router."""
    rag_routes = RAGRoutes(settings)
    return rag_routes.router
