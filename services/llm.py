#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM service for managing language model operations.
"""
import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from config.settings import Settings


class LLMService:
    """Service for managing language model operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm: Optional[ChatOpenAI] = None
    
    def get_llm(self) -> ChatOpenAI:
        """Get or create ChatOpenAI instance."""
        if self._llm is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set; please create a .env or export it")
            
            self._llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=0.2
            )
        return self._llm
    
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate an answer using the language model.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            
        Returns:
            Generated answer as string
        """
        llm = self.get_llm()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response = llm.invoke(messages)
        
        # Extract content from response
        if hasattr(response, "content"):
            return response.content.strip()
        else:
            return str(response).strip()
    
    def create_rag_prompts(self, question: str, context: str) -> tuple[str, str]:
        """
        Create system and user prompts for RAG question answering.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.settings.system_prompt
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in Turkish:"
        
        return system_prompt, user_prompt
