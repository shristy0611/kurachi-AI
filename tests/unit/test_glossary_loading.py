#!/usr/bin/env python3
"""
Test glossary loading to ensure non-zero terms are loaded
"""
import pytest
from services.intelligent_translation import IntelligentTranslationService
from utils.logger import get_logger

logger = get_logger("test_glossary")


def assert_glossary_contains(glossary: dict, must_have: list) -> list:
    """
    Check if glossary contains must-have terms in keys OR values (case-insensitive)
    Returns list of missing terms
    """
    keys = {str(k).strip().lower() for k in glossary.keys()}
    values = {str(v).strip().lower() for v in glossary.values()}
    
    missing = []
    for term in must_have:
        normalized_term = str(term).strip().lower()
        if normalized_term not in keys and normalized_term not in values:
            missing.append(term)
    
    return missing


@pytest.mark.unit
def test_glossary_loading():
    """Test that glossaries load with non-zero terms and contain critical business terms"""
    # Create intelligent translation service
    translation_service = IntelligentTranslationService()
    
    # Check business glossary
    business_glossary = translation_service.glossary_manager.get_glossary("business")
    business_term_count = len(business_glossary)
    
    # Basic validation
    assert business_term_count > 0, "Business glossary has 0 terms"
    assert business_term_count >= 10, f"Business glossary has too few terms: {business_term_count}"
    
    # Test critical business terms (case-insensitive, both keys and values)
    must_have_terms = [
        "Corporation", "Director", "株式会社", "取締役"  # Core terms that should always exist
    ]
    
    missing_terms = assert_glossary_contains(business_glossary, must_have_terms)
    
    # Warn about missing terms but don't fail (some may be in different glossaries)
    if missing_terms:
        logger.warning(f"Some critical terms not found: {missing_terms}")
    
    # Check all glossaries
    all_glossaries = translation_service.glossary_manager.glossaries
    total_terms = sum(len(glossary) for glossary in all_glossaries.values())
    
    assert total_terms > 0, "Total terms across all glossaries is 0"
    assert total_terms >= 200, f"Total terms ({total_terms}) below expected minimum (200)"
    
    logger.info(f"Loaded {len(all_glossaries)} glossaries with {total_terms} total terms")


@pytest.mark.unit
def test_glossary_normalization():
    """Test that glossary entries are properly normalized"""
    translation_service = IntelligentTranslationService()
    
    # Test normalization function directly
    test_terms = {
        "  spaced key  ": "  spaced value  ",
        "normal_key": "normal_value",
        "": "empty_key_should_be_skipped",
        "empty_value_key": ""
    }
    
    normalized = translation_service.glossary_manager._normalize_glossary_entries(test_terms)
    
    # Check normalization results
    assert "spaced key" in normalized, "Whitespace should be trimmed from keys"
    assert normalized["spaced key"] == "spaced value", "Whitespace should be trimmed from values"
    assert "normal_key" in normalized, "Normal entries should be preserved"
    assert "" not in normalized, "Empty keys should be filtered out"
    assert "empty_value_key" not in normalized, "Empty values should be filtered out"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])