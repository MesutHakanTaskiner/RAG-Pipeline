#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Question validation utilities for ensuring relevance to NTT Data sustainability topics.
"""
import re
from typing import List, Tuple


class QuestionValidator:
    """Validator for checking question relevance to NTT Data sustainability topics."""
    
    # Keywords related to NTT Data sustainability topics
    SUSTAINABILITY_KEYWORDS = {
        # Turkish keywords
        'sürdürülebilirlik', 'çevre', 'çevresel', 'sosyal', 'yönetişim', 'karbon', 'emisyon',
        'enerji', 'atık', 'geri dönüşüm', 'yeşil', 'iklim', 'sera gazı', 'çevre dostu',
        'toplumsal', 'etik', 'sorumluluk', 'paydaş', 'şeffaflık', 'raporlama',
        'ntt data', 'ntt', 'teknoloji', 'dijital', 'inovasyon', 'veri', 'çözüm',
        
        # English keywords (in case of mixed language questions)
        'sustainability', 'environmental', 'social', 'governance', 'esg', 'carbon',
        'emission', 'energy', 'waste', 'recycling', 'green', 'climate', 'greenhouse',
        'responsible', 'stakeholder', 'transparency', 'reporting', 'impact',
        'ntt data', 'technology', 'digital', 'innovation', 'solution', 'data'
    }
    
    # Topics that are clearly irrelevant
    IRRELEVANT_KEYWORDS = {
        # Turkish
        'yemek', 'tarif', 'spor', 'futbol', 'müzik', 'film', 'oyun', 'seyahat',
        'moda', 'güzellik', 'sağlık', 'hastalık', 'ilaç', 'doktor', 'hastane',
        'siyaset', 'seçim', 'parti', 'haber', 'gündem', 'magazin',
        
        # English
        'recipe', 'cooking', 'sports', 'football', 'music', 'movie', 'game', 'travel',
        'fashion', 'beauty', 'health', 'disease', 'medicine', 'doctor', 'hospital',
        'politics', 'election', 'party', 'news', 'celebrity', 'entertainment'
    }
    
    @classmethod
    def is_relevant_question(cls, question: str) -> Tuple[bool, str]:
        """
        Check if a question is relevant to NTT Data sustainability topics.
        
        Args:
            question: The question to validate
            
        Returns:
            Tuple of (is_relevant: bool, reason: str)
        """
        question_lower = question.lower()
        
        # Check for clearly irrelevant topics
        for keyword in cls.IRRELEVANT_KEYWORDS:
            if keyword in question_lower:
                return False, f"Soru NTT Data sürdürülebilirlik raporları ile ilgili değil. Bu sistem sadece NTT Data'nın sürdürülebilirlik, çevresel, sosyal ve yönetişim konularındaki sorularınızı yanıtlayabilir."
        
        # Check for sustainability-related keywords
        sustainability_score = 0
        for keyword in cls.SUSTAINABILITY_KEYWORDS:
            if keyword in question_lower:
                sustainability_score += 1
        
        # If question contains sustainability keywords, it's likely relevant
        if sustainability_score > 0:
            return True, "Soru NTT Data sürdürülebilirlik konularıyla ilgili görünüyor."
        
        # For questions without clear keywords, allow them but with lower confidence
        # The LLM will handle the final relevance check
        if len(question.strip()) < 10:
            return False, "Lütfen daha detaylı bir soru sorun."
        
        return True, "Soru değerlendiriliyor."
    
    @classmethod
    def get_relevance_message(cls) -> str:
        """Get a message explaining what topics are relevant."""
        return (
            "Bu sistem NTT Data Solutions'ın sürdürülebilirlik raporları ve case book'ları hakkında sorularınızı yanıtlar. "
            "Şu konularda sorular sorabilirsiniz:\n"
            "• Sürdürülebilirlik stratejileri ve hedefleri\n"
            "• Çevresel etki ve karbon emisyonları\n"
            "• Sosyal sorumluluk projeleri\n"
            "• Kurumsal yönetişim uygulamaları\n"
            "• Teknoloji ve inovasyon çözümleri\n"
            "• ESG (Çevresel, Sosyal, Yönetişim) performansı"
        )


def validate_question_relevance(question: str) -> Tuple[bool, str]:
    """
    Convenience function to validate question relevance.
    
    Args:
        question: Question to validate
        
    Returns:
        Tuple of (is_relevant: bool, message: str)
    """
    return QuestionValidator.is_relevant_question(question)
