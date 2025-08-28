#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query Decomposition Service for breaking complex questions into simpler sub-questions.
This service analyzes incoming questions and decomposes them into manageable parts
for better retrieval and reasoning.
"""
import re
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from config.settings import Settings


class QuestionType(str, Enum):
    """Types of questions that can be identified."""
    FACTUAL = "factual"              # Simple fact-based questions
    COMPARISON = "comparison"        # Comparing across time/entities
    TREND_ANALYSIS = "trend"         # Analyzing trends over time
    CAUSAL = "causal"               # Cause-effect relationships
    MULTI_ASPECT = "multi_aspect"   # Questions with multiple aspects
    COMPLEX = "complex"             # Complex questions requiring decomposition


class SubQuestion(BaseModel):
    """Model for sub-questions generated from complex queries."""
    question: str
    question_type: QuestionType
    priority: int  # 1 (highest) to 5 (lowest)
    dependencies: List[int] = []  # Indices of sub-questions this depends on
    year_filter: Optional[Dict[str, int]] = None  # Year constraints for this sub-question


class DecompositionResult(BaseModel):
    """Result of query decomposition."""
    original_question: str
    question_type: QuestionType
    requires_decomposition: bool
    sub_questions: List[SubQuestion]
    reasoning: str  # Explanation of decomposition strategy


class QueryDecomposer:
    """Service for decomposing complex queries into simpler sub-questions."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm: Optional[ChatOpenAI] = None
    
    def _get_llm(self) -> ChatOpenAI:
        """Get or create ChatOpenAI instance for decomposition."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=0.1  # Low temperature for consistent decomposition
            )
        return self._llm
    
    def identify_question_type(self, question: str) -> QuestionType:
        """
        Identify the type of question to determine decomposition strategy.
        
        Args:
            question: The input question
            
        Returns:
            QuestionType enum value
        """
        question_lower = question.lower()
        
        # Turkish temporal indicators
        temporal_indicators = [
            "yılında", "yılından", "yılına", "döneminde", "süresince", "arasında",
            "zaman içinde", "yıllara göre", "son", "trend", "eğilim", "gidişat",
            "artış eğilimi", "azalış eğilimi"
        ]
                
        # Turkish comparison indicators
        comparison_indicators = [
            "karşılaştır", "karşılaştırma", "kıyasla", "kıyaslama",
            "fark", "farklar", "benzer", "benzerlik", "farklı",
            "göre", "daha", "az", "çok", "yüksek", "düşük", "en",
            "üstün", "geri", "avantaj", "dezavantaj", "karşısında"
        ]

        
        # Turkish causal indicators
        causal_indicators = [
            "neden", "nedenleri", "sebep", "sebebi", "etken", "faktör",
            "sonucu", "sonuçları", "etkisi", "etkile", "etkiledi",
            "nedeniyle", "sebebiyle", "dolayısıyla", "bu yüzden", "bu nedenle",
            "yüzünden", "kaynaklı", "tetikledi", "nasıl etkiledi", "etkileyen faktörler"
        ]

        
        # Multi-aspect indicators (multiple questions in one)
        multi_aspect_indicators = [
            "hem", "ayrıca", "bunun yanında", "yanı sıra",
            "boyutları", "başlıkları", "unsurları", "ölçütleri", "metrikleri",
            "hangi alanlarda", "hangi başlıklarda", "hangi açılardan"
        ]

        
        # Check for year patterns (2020, 2021, etc.)
        year_pattern = r'\b(20\d{2})\b'
        years_found = re.findall(year_pattern, question)
        
        # Decision logic
        if len(years_found) >= 2 or any(indicator in question_lower for indicator in temporal_indicators):
            if any(indicator in question_lower for indicator in comparison_indicators):
                return QuestionType.COMPARISON
            else:
                return QuestionType.TREND_ANALYSIS
        
        elif any(indicator in question_lower for indicator in causal_indicators):
            return QuestionType.CAUSAL
        
        elif (len(question.split()) > 15 or 
              question.count('?') > 1 or 
              any(indicator in question_lower for indicator in multi_aspect_indicators)):
            return QuestionType.MULTI_ASPECT
        
        elif (len(question.split()) > 20 or 
              any(indicator in question_lower for indicator in causal_indicators + comparison_indicators)):
            return QuestionType.COMPLEX
        
        else:
            return QuestionType.FACTUAL
    
    def _create_decomposition_prompt(self, question: str, question_type: QuestionType) -> str:
        """Create a prompt for LLM-based question decomposition."""
        
        system_prompt = """Sen bir soru analizi uzmanısın. Karmaşık soruları daha basit alt sorulara ayırman gerekiyor.

                        GÖREVIN:
                        1. Verilen soruyu analiz et
                        2. Eğer soru karmaşıksa, onu 2-5 alt soruya böl
                        3. Her alt soru bağımsız olarak cevaplanabilir olmalı
                        4. Alt sorular mantıklı bir sıra takip etmeli

                        KURALLAR:
                        - Alt sorular Türkçe olmalı
                        - Her alt soru net ve spesifik olmalı
                        - Yıl filtreleri varsa belirt
                        - Bağımlılıkları işaretle

                        ÇIKTI FORMATI:
                        ```
                        SORU_TIPI: [tip]
                        AYIRMA_GEREKLI: [evet/hayır]
                        ALT_SORULAR:
                        1. [alt soru 1] (Öncelik: 1, Yıl: 2022)
                        2. [alt soru 2] (Öncelik: 2, Bağımlı: 1)
                        ...
                        AÇIKLAMA: [strateji açıklaması]
                        ```"""

        user_prompt = f"""Soru: "{question}"
        Soru Tipi: {question_type.value}

        Bu soruyu analiz et ve gerekirse alt sorulara ayır."""

        return system_prompt, user_prompt
    
    def _parse_llm_response(self, response: str, original_question: str) -> DecompositionResult:
        """Parse LLM response into structured decomposition result."""
        
        lines = response.strip().split('\n')
        
        # Default values
        question_type = QuestionType.FACTUAL
        requires_decomposition = False
        sub_questions = []
        reasoning = "Basit soru, ayrıştırma gerekmiyor."
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('SORU_TIPI:'):
                type_str = line.split(':', 1)[1].strip()
                try:
                    question_type = QuestionType(type_str)
                except ValueError:
                    question_type = QuestionType.COMPLEX
            
            elif line.startswith('AYIRMA_GEREKLI:'):
                requires_decomposition = 'evet' in line.lower()
            
            elif line.startswith('ALT_SORULAR:'):
                current_section = 'sub_questions'
                continue
            
            elif line.startswith('AÇIKLAMA:'):
                reasoning = line.split(':', 1)[1].strip()
                current_section = None
            
            elif current_section == 'sub_questions' and re.match(r'^\d+\.', line):
                # Parse sub-question
                match = re.match(r'^(\d+)\.\s*(.+?)(?:\s*\((.+?)\))?$', line)
                if match:
                    priority = int(match.group(1))
                    question_text = match.group(2).strip()
                    metadata = match.group(3) if match.group(3) else ""
                    
                    # Parse metadata
                    year_filter = None
                    dependencies = []
                    
                    if metadata:
                        # Extract year
                        year_match = re.search(r'Yıl:\s*(\d{4})', metadata)
                        if year_match:
                            year = int(year_match.group(1))
                            year_filter = {"year_from": year, "year_to": year}
                        
                        # Extract dependencies
                        dep_match = re.search(r'Bağımlı:\s*([\d,\s]+)', metadata)
                        if dep_match:
                            deps = [int(d.strip()) for d in dep_match.group(1).split(',') if d.strip().isdigit()]
                            dependencies = deps
                    
                    sub_question = SubQuestion(
                        question=question_text,
                        question_type=QuestionType.FACTUAL,  # Sub-questions are typically factual
                        priority=priority,
                        dependencies=dependencies,
                        year_filter=year_filter
                    )
                    sub_questions.append(sub_question)
        
        return DecompositionResult(
            original_question=original_question,
            question_type=question_type,
            requires_decomposition=requires_decomposition,
            sub_questions=sub_questions,
            reasoning=reasoning
        )
    
    def decompose_question(self, question: str) -> DecompositionResult:
        """
        Decompose a complex question into simpler sub-questions.
        
        Args:
            question: The original complex question
            
        Returns:
            DecompositionResult with sub-questions and metadata
        """
        # First, identify question type
        question_type = self.identify_question_type(question)
        
        # Simple questions don't need decomposition
        if question_type == QuestionType.FACTUAL:
            return DecompositionResult(
                original_question=question,
                question_type=question_type,
                requires_decomposition=False,
                sub_questions=[],
                reasoning="Basit faktual soru, ayrıştırma gerekmiyor."
            )
        
        # Use LLM for complex decomposition
        try:
            llm = self._get_llm()
            system_prompt, user_prompt = self._create_decomposition_prompt(question, question_type)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return self._parse_llm_response(response_text, question)
            
        except Exception as e:
            # Fallback to rule-based decomposition
            return self._fallback_decomposition(question, question_type)
    
    def _fallback_decomposition(self, question: str, question_type: QuestionType) -> DecompositionResult:
        """Fallback rule-based decomposition when LLM fails."""
        
        sub_questions = []
        
        if question_type == QuestionType.COMPARISON:
            # Extract years if present
            years = re.findall(r'\b(20\d{2})\b', question)
            if len(years) >= 2:
                year1, year2 = years[0], years[-1]
                sub_questions = [
                    SubQuestion(
                        question=f"{year1} yılındaki durumu nedir?",
                        question_type=QuestionType.FACTUAL,
                        priority=1,
                        year_filter={"year_from": int(year1), "year_to": int(year1)}
                    ),
                    SubQuestion(
                        question=f"{year2} yılındaki durumu nedir?",
                        question_type=QuestionType.FACTUAL,
                        priority=2,
                        year_filter={"year_from": int(year2), "year_to": int(year2)}
                    ),
                    SubQuestion(
                        question=f"{year1} ve {year2} arasındaki farklar nelerdir?",
                        question_type=QuestionType.FACTUAL,
                        priority=3,
                        dependencies=[1, 2]
                    )
                ]
        
        return DecompositionResult(
            original_question=question,
            question_type=question_type,
            requires_decomposition=len(sub_questions) > 0,
            sub_questions=sub_questions,
            reasoning="Kural tabanlı ayrıştırma uygulandı."
        )
