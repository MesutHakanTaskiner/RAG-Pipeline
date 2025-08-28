#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API routes for the RAG application.
"""
import time
from typing import List
from fastapi import APIRouter, HTTPException

from models.schemas import (
    AskRequest, AskResponse, HealthResponse, Source,
    ReasoningStepResponse, SubQuestionResponse, DecompositionResponse
)
from services.vector_store import VectorStoreService, create_year_filter
from services.llm import LLMService
from services.mmr import MMRService
from services.query_decomposer import QueryDecomposer
from services.reasoning_agent import ReasoningAgent
from utils.text_processing import format_context_block
from config.settings import Settings


class RAGRoutes:
    """RAG API routes handler."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_service = VectorStoreService(settings)
        self.llm_service = LLMService(settings)
        self.mmr_service = MMRService(settings)
        self.query_decomposer = QueryDecomposer(settings)
        self.reasoning_agent = ReasoningAgent(settings)
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
            # Check if agentic reasoning is enabled
            if request.use_reasoning:
                return self._process_with_reasoning(request, start_time)
            else:
                return self._process_simple_rag(request, start_time)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
    def _process_simple_rag(self, request: AskRequest, start_time: float) -> AskResponse:
        """Process question using simple RAG without reasoning."""
        # Determine retrieval count
        k = self.settings.retrieval_k
        
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
                k=self.settings.context_k
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
    
    def _process_with_reasoning(self, request: AskRequest, start_time: float) -> AskResponse:
        """Process question using agentic reasoning and query decomposition."""
        
        # Step 1: Decompose the question
        decomposition = self.query_decomposer.decompose_question(request.question)
        
        # Step 2: Process based on decomposition result
        if not decomposition.requires_decomposition:
            # Simple question - use basic RAG with enhanced prompting
            return self._process_simple_question_with_reasoning(request, decomposition, start_time)
        else:
            # Complex question - use full agentic approach
            return self._process_complex_question_with_reasoning(request, decomposition, start_time)
    
    def _process_simple_question_with_reasoning(
        self, 
        request: AskRequest, 
        decomposition, 
        start_time: float
    ) -> AskResponse:
        """Process simple question with enhanced reasoning."""
        
        # Use standard retrieval
        k = self.settings.retrieval_k
        year_filter = create_year_filter(request.year_from, request.year_to)
        
        results = self.vector_service.similarity_search_with_score(
            query=request.question,
            k=k,
            filter_dict=year_filter
        )
        
        # Apply MMR if requested
        if request.use_mmr and results:
            top_context = self.mmr_service.apply_mmr(
                query=request.question,
                documents_with_scores=results,
                k=self.settings.context_k
            )
        else:
            top_context = results[:self.settings.context_k]
        
        # Use reasoning agent for enhanced analysis
        reasoning_step = self.reasoning_agent.analyze_step_by_step(
            request.question,
            self._format_context_for_reasoning(top_context),
            1
        )
        
        # Generate enhanced answer
        answer = self.reasoning_agent._generate_focused_answer(
            request.question,
            self._format_context_for_reasoning(top_context)
        )
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Build sources
        sources = self._build_sources(top_context)
        
        # Build response
        response = AskResponse(
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            question_type=decomposition.question_type.value,
            overall_confidence=reasoning_step.confidence
        )
        
        # Add reasoning trace if requested
        if request.show_reasoning_trace:
            response.reasoning_trace = f"🔍 SORU ANALİZİ:\nSoru Tipi: {decomposition.question_type.value}\nAçıklama: {decomposition.reasoning}\n\n📊 ANALIZ:\n{reasoning_step.analysis}\n\n✅ SONUÇ:\n{reasoning_step.conclusion}"
        
        return response
    
    def _process_complex_question_with_reasoning(
        self, 
        request: AskRequest, 
        decomposition, 
        start_time: float
    ) -> AskResponse:
        """Process complex question with full agentic reasoning."""
        
        sub_answers = []
        all_sources = []
        
        # Process each sub-question
        for sub_question in decomposition.sub_questions:
            # Create year filter for sub-question
            if sub_question.year_filter:
                year_filter = create_year_filter(
                    sub_question.year_filter.get("year_from"),
                    sub_question.year_filter.get("year_to")
                )
            else:
                year_filter = create_year_filter(request.year_from, request.year_to)
            
            # Retrieve documents for sub-question
            k = self.settings.retrieval_k
            results = self.vector_service.similarity_search_with_score(
                query=sub_question.question,
                k=k,
                filter_dict=year_filter
            )
            
            # Apply MMR if requested
            if request.use_mmr and results:
                context_sources = self.mmr_service.apply_mmr(
                    query=sub_question.question,
                    documents_with_scores=results,
                    k=self.settings.context_k
                )
            else:
                context_sources = results[:self.settings.context_k]
            
            # Answer sub-question with reasoning
            sub_answer = self.reasoning_agent.answer_sub_question(
                sub_question, context_sources
            )
            
            sub_answers.append(sub_answer)
            all_sources.extend(context_sources)
        
        # Synthesize final answer
        reasoning_result = self.reasoning_agent.synthesize_answers(
            request.question, decomposition, sub_answers
        )
        
        # Validate reasoning
        validation_report = self.reasoning_agent.validate_reasoning(reasoning_result)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Remove duplicate sources and build final source list
        unique_sources = self._deduplicate_sources(all_sources)
        sources = self._build_sources(unique_sources)
        
        # Build decomposition response
        decomposition_response = self._build_decomposition_response(
            decomposition, sub_answers
        )
        
        # Build final response
        response = AskResponse(
            answer=reasoning_result.final_answer,
            sources=sources,
            latency_ms=latency_ms,
            question_type=decomposition.question_type.value,
            overall_confidence=reasoning_result.overall_confidence,
            decomposition=decomposition_response,
            validation_report=validation_report
        )
        
        # Add reasoning trace if requested
        if request.show_reasoning_trace:
            response.reasoning_trace = reasoning_result.reasoning_trace
        
        return response
    
    def _format_context_for_reasoning(self, context_sources) -> str:
        """Format context sources for reasoning analysis."""
        context_blocks = []
        for doc, score in context_sources:
            doc_id = doc.metadata.get('doc_id', 'unknown')
            year = doc.metadata.get('year', 'unknown')
            page = doc.metadata.get('page_start', 'unknown')
            
            block = f"[{doc_id} {year} s.{page}] (Skor: {score:.3f})\n{doc.page_content[:400]}..."
            context_blocks.append(block)
        
        return "\n\n".join(context_blocks)
    
    def _build_sources(self, context_sources) -> List[Source]:
        """Build source list from context sources."""
        sources = []
        for doc, score in context_sources:
            sources.append(Source(
                doc_id=doc.metadata.get("doc_id"),
                year=doc.metadata.get("year"),
                score=float(score),
                page_start=doc.metadata.get("page_start"),
                page_end=doc.metadata.get("page_end"),
                section_path=doc.metadata.get("section_path"),
                text=doc.page_content,
            ))
        return sources
    
    def _deduplicate_sources(self, all_sources):
        """Remove duplicate sources based on doc_id and page."""
        seen = set()
        unique_sources = []
        
        for doc, score in all_sources:
            doc_id = doc.metadata.get("doc_id", "unknown")
            page = doc.metadata.get("page_start", 0)
            key = f"{doc_id}_{page}"
            
            if key not in seen:
                seen.add(key)
                unique_sources.append((doc, score))
        
        return unique_sources
    
    def _build_decomposition_response(self, decomposition, sub_answers) -> DecompositionResponse:
        """Build decomposition response from reasoning results."""
        
        sub_question_responses = []
        for sub_answer in sub_answers:
            reasoning_step_responses = []
            for step in sub_answer.reasoning_steps:
                reasoning_step_responses.append(ReasoningStepResponse(
                    step_number=step.step_number,
                    description=step.description,
                    question=step.question,
                    analysis=step.analysis,
                    conclusion=step.conclusion,
                    confidence=step.confidence
                ))
            
            sub_question_responses.append(SubQuestionResponse(
                question=sub_answer.sub_question.question,
                question_type=sub_answer.sub_question.question_type.value,
                priority=sub_answer.sub_question.priority,
                answer=sub_answer.answer,
                sources=self._build_sources(sub_answer.sources),
                confidence=sub_answer.confidence,
                reasoning_steps=reasoning_step_responses
            ))
        
        return DecompositionResponse(
            original_question=decomposition.original_question,
            question_type=decomposition.question_type.value,
            requires_decomposition=decomposition.requires_decomposition,
            reasoning=decomposition.reasoning,
            sub_questions=sub_question_responses
        )


def create_router(settings: Settings) -> APIRouter:
    """Create and return the API router."""
    rag_routes = RAGRoutes(settings)
    return rag_routes.router
