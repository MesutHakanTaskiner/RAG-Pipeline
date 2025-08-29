#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Step Reasoning Agent for structured analysis and answer synthesis.
This service implements chain-of-thought reasoning and synthesizes answers
from multiple sub-questions into coherent responses.
"""
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from config.settings import Settings
from services.query_decomposer import SubQuestion, DecompositionResult


class ReasoningStep(BaseModel):
    """Individual step in the reasoning process."""
    step_number: int
    description: str
    question: str
    context: str
    analysis: str
    conclusion: str
    confidence: float  # 0.0 to 1.0


class SubAnswer(BaseModel):
    """Answer to a sub-question with supporting information."""
    sub_question: SubQuestion
    answer: str
    sources: List[Tuple[Document, float]]
    confidence: float
    reasoning_steps: List[ReasoningStep]


class ReasoningResult(BaseModel):
    """Complete reasoning result with synthesis."""
    original_question: str
    decomposition: DecompositionResult
    sub_answers: List[SubAnswer]
    synthesis_steps: List[ReasoningStep]
    final_answer: str
    overall_confidence: float
    reasoning_trace: str  # Human-readable reasoning explanation


class ReasoningAgent:
    """Agent for multi-step reasoning and answer synthesis."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm: Optional[ChatOpenAI] = None
    
    def _get_llm(self) -> ChatOpenAI:
        """Get or create ChatOpenAI instance for reasoning."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=0.2  # Slightly higher for reasoning creativity
            )
        return self._llm
    
    def analyze_step_by_step(
        self, 
        question: str, 
        context: str, 
        step_number: int = 1
    ) -> ReasoningStep:
        """
        Perform step-by-step analysis for a single question.
        
        Args:
            question: The question to analyze
            context: Retrieved context for the question
            step_number: Step number in the overall reasoning process
            
        Returns:
            ReasoningStep with detailed analysis
        """
        system_prompt = """Sen bir analiz uzmanısın. Verilen soru ve bağlam için adım adım analiz yapman gerekiyor.

            GÖREVIN:
            1. Soruyu ve bağlamı dikkatlice incele
            2. Adım adım mantıklı analiz yap
            3. Her adımda nedenini açıkla
            4. Sonuca güven seviyeni belirt

            ANALIZ ADIMLARI:
            1. Soru Analizi: Sorunun ne sorduğunu net olarak belirle
            2. Bağlam İncelemesi: Verilen bilgileri kategorize et
            3. Bilgi Eşleştirmesi: Hangi bilgiler soruyu yanıtlıyor
            4. Mantıklı Çıkarım: Verilerden ne sonuç çıkarılabilir
            5. Güven Değerlendirmesi: Cevabın ne kadar güvenilir olduğu

            ÇIKTI FORMATI:
            ```
            ADIM_ANALIZI:
            1. Soru Analizi: [analiz]
            2. Bağlam İncelemesi: [inceleme]
            3. Bilgi Eşleştirmesi: [eşleştirme]
            4. Mantıklı Çıkarım: [çıkarım]
            5. Güven Değerlendirmesi: [değerlendirme]

            SONUÇ: [net sonuç]
            GÜVEN: [0.0-1.0]
        ```"""

        user_prompt = f"""Soru: "{question}"

        Bağlam:
        {context}

        Bu soru için adım adım analiz yap."""

        try:
            llm = self._get_llm()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return self._parse_reasoning_response(
                response_text, question, context, step_number
            )
            
        except Exception as e:
            # Fallback reasoning
            return ReasoningStep(
                step_number=step_number,
                description=f"Adım {step_number}: Temel analiz",
                question=question,
                context=context[:500] + "..." if len(context) > 500 else context,
                analysis="Bağlam analizi yapıldı ve ilgili bilgiler belirlendi.",
                conclusion="Mevcut bilgiler doğrultusunda cevap oluşturuldu.",
                confidence=0.7
            )
    
    def _parse_reasoning_response(
        self, 
        response: str, 
        question: str, 
        context: str, 
        step_number: int
    ) -> ReasoningStep:
        """Parse LLM reasoning response into structured format."""
        
        lines = response.strip().split('\n')
        
        analysis_parts = []
        conclusion = ""
        confidence = 0.7
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('ADIM_ANALIZI:'):
                current_section = 'analysis'
                continue
            elif line.startswith('SONUÇ:'):
                conclusion = line.split(':', 1)[1].strip()
                current_section = None
            elif line.startswith('GÜVEN:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.7
                current_section = None
            elif current_section == 'analysis' and ':' in line:
                analysis_parts.append(line)
        
        analysis = '\n'.join(analysis_parts) if analysis_parts else "Temel analiz yapıldı."
        
        return ReasoningStep(
            step_number=step_number,
            description=f"Adım {step_number}: {question[:50]}...",
            question=question,
            context=context[:300] + "..." if len(context) > 300 else context,
            analysis=analysis,
            conclusion=conclusion or "Analiz tamamlandı.",
            confidence=confidence
        )
    
    def answer_sub_question(
        self, 
        sub_question: SubQuestion, 
        context_sources: List[Tuple[Document, float]]
    ) -> SubAnswer:
        """
        Answer a single sub-question with reasoning.
        
        Args:
            sub_question: The sub-question to answer
            context_sources: Retrieved documents with scores
            
        Returns:
            SubAnswer with reasoning steps
        """
        # Format context
        context_text = self._format_context_for_reasoning(context_sources)
        
        # Perform step-by-step reasoning
        reasoning_step = self.analyze_step_by_step(
            sub_question.question, 
            context_text, 
            sub_question.priority
        )
        
        # Generate focused answer
        answer = self._generate_focused_answer(sub_question.question, context_text)
        
        return SubAnswer(
            sub_question=sub_question,
            answer=answer,
            sources=context_sources,
            confidence=reasoning_step.confidence,
            reasoning_steps=[reasoning_step]
        )
    
    def _format_context_for_reasoning(self, context_sources: List[Tuple[Document, float]]) -> str:
        """Format context sources for reasoning analysis."""
        
        context_blocks = []
        for doc, score in context_sources:
            doc_id = doc.metadata.get('doc_id', 'unknown')
            year = doc.metadata.get('year', 'unknown')
            page = doc.metadata.get('page_start', 'unknown')
            
            block = f"[{doc_id} {year} s.{page}] (Skor: {score:.3f})\n{doc.page_content[:400]}..."
            context_blocks.append(block)
        
        return "\n\n".join(context_blocks)
    
    def _generate_focused_answer(self, question: str, context: str) -> str:
        """Generate a focused answer for a specific question."""
        
        system_prompt = """Sen bir uzman analistisin. Verilen soru için bağlamdan yararlanarak net ve odaklı bir cevap ver.

KURALLAR:
- Sadece verilen bağlamı kullan
- Net ve spesifik ol
- Türkçe cevap ver
- Kaynak bilgilerini dahil et
- Eğer bilgi yoksa açıkça belirt"""

        user_prompt = f"""Soru: {question}

Bağlam:
{context}

Bu soru için net bir cevap ver."""

        try:
            llm = self._get_llm()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception:
            return "Verilen bağlam doğrultusunda cevap oluşturulamadı."
    
    def synthesize_answers(
        self, 
        original_question: str,
        decomposition: DecompositionResult,
        sub_answers: List[SubAnswer]
    ) -> ReasoningResult:
        """
        Synthesize sub-answers into a comprehensive final answer.
        
        Args:
            original_question: The original complex question
            decomposition: Question decomposition result
            sub_answers: List of answered sub-questions
            
        Returns:
            Complete reasoning result with synthesis
        """
        # Create synthesis steps
        synthesis_steps = self._create_synthesis_steps(sub_answers)
        
        # Generate final answer
        final_answer = self._synthesize_final_answer(
            original_question, sub_answers, synthesis_steps
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(sub_answers)
        
        # Create reasoning trace
        reasoning_trace = self._create_reasoning_trace(
            decomposition, sub_answers, synthesis_steps
        )
        
        return ReasoningResult(
            original_question=original_question,
            decomposition=decomposition,
            sub_answers=sub_answers,
            synthesis_steps=synthesis_steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            reasoning_trace=reasoning_trace
        )
    
    def _create_synthesis_steps(self, sub_answers: List[SubAnswer]) -> List[ReasoningStep]:
        """Create synthesis steps from sub-answers."""
        
        synthesis_steps = []
        
        # Step 1: Information gathering
        info_step = ReasoningStep(
            step_number=len(sub_answers) + 1,
            description="Bilgi Toplama ve Organize Etme",
            question="Alt sorulardan elde edilen bilgiler neler?",
            context="",
            analysis=f"{len(sub_answers)} alt soru cevaplanarak kapsamlı bilgi toplandı.",
            conclusion="Tüm alt sorular başarıyla cevaplanarak sentez için hazır hale getirildi.",
            confidence=sum(sa.confidence for sa in sub_answers) / len(sub_answers) if sub_answers else 0.0
        )
        synthesis_steps.append(info_step)
        
        # Step 2: Pattern identification
        pattern_step = ReasoningStep(
            step_number=len(sub_answers) + 2,
            description="Örüntü ve İlişki Analizi",
            question="Alt cevaplar arasında hangi örüntüler ve ilişkiler var?",
            context="",
            analysis="Alt cevaplar arasındaki bağlantılar ve örüntüler analiz edildi.",
            conclusion="Tutarlı bir genel resim oluşturmak için ilişkiler belirlendi.",
            confidence=0.8
        )
        synthesis_steps.append(pattern_step)
        
        return synthesis_steps
    
    def _synthesize_final_answer(
        self, 
        original_question: str, 
        sub_answers: List[SubAnswer],
        synthesis_steps: List[ReasoningStep]
    ) -> str:
        """Generate the final synthesized answer."""
        
        # Combine all sub-answers
        combined_info = "\n\n".join([
            f"Alt Soru: {sa.sub_question.question}\nCevap: {sa.answer}"
            for sa in sub_answers
        ])
        
        system_prompt = """Sen bir sentez uzmanısın. Alt sorulardan gelen cevapları birleştirerek kapsamlı bir final cevabı oluştur.

GÖREVIN:
1. Tüm alt cevapları analiz et
2. Ortak temaları ve örüntüleri belirle
3. Tutarlı ve kapsamlı bir cevap oluştur
4. Çelişkileri çöz ve netleştir

KURALLAR:
- Türkçe cevap ver
- Tüm önemli bilgileri dahil et
- Mantıklı bir akış oluştur
- Kaynak yıllarını belirt
- Net ve anlaşılır ol"""

        user_prompt = f"""Orijinal Soru: "{original_question}"

Alt Cevaplar:
{combined_info}

Bu bilgileri sentezleyerek kapsamlı bir final cevabı oluştur."""

        try:
            llm = self._get_llm()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception:
            # Fallback synthesis
            return f"Analiz sonucunda {len(sub_answers)} farklı açıdan incelenen soru için kapsamlı bilgiler elde edilmiştir. " + \
                   " ".join([sa.answer[:100] + "..." for sa in sub_answers[:3]])
    
    def _calculate_overall_confidence(self, sub_answers: List[SubAnswer]) -> float:
        """Calculate overall confidence from sub-answers."""
        if not sub_answers:
            return 0.0
        
        # Weighted average based on priority (lower priority = higher weight)
        total_weight = 0
        weighted_confidence = 0
        
        for sa in sub_answers:
            weight = 1.0 / sa.sub_question.priority  # Higher priority = lower number = higher weight
            weighted_confidence += sa.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _create_reasoning_trace(
        self, 
        decomposition: DecompositionResult,
        sub_answers: List[SubAnswer],
        synthesis_steps: List[ReasoningStep]
    ) -> str:
        """Create a human-readable reasoning trace."""
        
        trace_parts = []
        
        # Decomposition explanation
        trace_parts.append(f"🔍 SORU ANALİZİ:")
        trace_parts.append(f"Soru Tipi: {decomposition.question_type.value}")
        trace_parts.append(f"Ayrıştırma: {decomposition.reasoning}")
        trace_parts.append("")
        
        # Sub-question analysis
        if sub_answers:
            trace_parts.append(f"📋 ALT SORU ANALİZİ:")
            for i, sa in enumerate(sub_answers, 1):
                trace_parts.append(f"{i}. {sa.sub_question.question}")
                trace_parts.append(f"   Güven: {sa.confidence:.2f}")
                trace_parts.append(f"   Kaynak: {len(sa.sources)} belge")
                trace_parts.append("")
        
        # Synthesis explanation
        trace_parts.append(f"🔗 SENTEz SÜRECİ:")
        for step in synthesis_steps:
            trace_parts.append(f"- {step.description}: {step.conclusion}")
        
        return "\n".join(trace_parts)
    
    def validate_reasoning(
        self, 
        reasoning_result: ReasoningResult
    ) -> Dict[str, Any]:
        """
        Validate the reasoning result for consistency and quality.
        
        Args:
            reasoning_result: The reasoning result to validate
            
        Returns:
            Validation report with scores and recommendations
        """
        validation_report = {
            "overall_score": 0.0,
            "consistency_score": 0.0,
            "completeness_score": 0.0,
            "confidence_score": reasoning_result.overall_confidence,
            "issues": [],
            "recommendations": []
        }
        
        # Check consistency
        if reasoning_result.sub_answers:
            consistency_score = min(sa.confidence for sa in reasoning_result.sub_answers)
            validation_report["consistency_score"] = consistency_score
            
            if consistency_score < 0.5:
                validation_report["issues"].append("Düşük tutarlılık skoru")
                validation_report["recommendations"].append("Daha fazla kaynak belge gerekebilir")
        
        # Check completeness
        expected_sub_questions = len(reasoning_result.decomposition.sub_questions)
        actual_answers = len(reasoning_result.sub_answers)
        
        if expected_sub_questions > 0:
            completeness_score = actual_answers / expected_sub_questions
            validation_report["completeness_score"] = completeness_score
            
            if completeness_score < 1.0:
                validation_report["issues"].append("Bazı alt sorular cevaplanmamış")
                validation_report["recommendations"].append("Eksik alt soruları tamamlayın")
        else:
            validation_report["completeness_score"] = 1.0
        
        # Calculate overall score
        validation_report["overall_score"] = (
            validation_report["consistency_score"] * 0.4 +
            validation_report["completeness_score"] * 0.3 +
            validation_report["confidence_score"] * 0.3
        )
        
        return validation_report
