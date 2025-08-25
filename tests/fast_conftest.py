# tests/fast_conftest.py
"""
Fast test configuration that mocks expensive operations
Use with: pytest --confcutdir=tests/fast_conftest.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.fixture(autouse=True, scope="session")
def mock_expensive_operations():
    """Automatically mock expensive operations for fast tests"""
    
    # Mock Ollama LLM calls
    with patch('services.translation_service.Ollama') as mock_ollama, \
         patch('services.intelligent_translation.Ollama') as mock_intelligent_ollama, \
         patch('services.advanced_content_extraction.Ollama') as mock_extraction_ollama:
        
        # Configure mock responses
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Mock translation result"
        
        mock_ollama.return_value = mock_llm
        mock_intelligent_ollama.return_value = mock_llm
        mock_extraction_ollama.return_value = mock_llm
        
        # Mock document processing that might be slow
        with patch('services.document_processors.PDFProcessor.process') as mock_pdf, \
             patch('services.document_processors.WordProcessor.process') as mock_word:
            
            # Fast mock document processing
            mock_pdf.return_value = {
                "content": "Mock PDF content for testing",
                "metadata": {"pages": 1, "title": "Mock Document"},
                "chunks": ["Mock chunk 1", "Mock chunk 2"]
            }
            
            mock_word.return_value = {
                "content": "Mock Word content for testing", 
                "metadata": {"pages": 1, "title": "Mock Document"},
                "chunks": ["Mock chunk 1", "Mock chunk 2"]
            }
            
            yield


@pytest.fixture(autouse=True, scope="session")
def fast_database_operations():
    """Mock database operations for speed"""
    
    # Mock database calls that might be slow
    with patch('models.database.db_manager') as mock_db:
        mock_db.get_metric.return_value = 0
        mock_db.increment_metric.return_value = None
        mock_db.save_preferences.return_value = True
        mock_db.load_preferences.return_value = {}
        
        yield mock_db


@pytest.fixture(scope="session")
def fast_translation_service():
    """Fast mock translation service"""
    mock_service = Mock()
    mock_service.translate_with_context.return_value = Mock(
        translated_text="Mock translation",
        source_language="en",
        target_language="ja", 
        quality_score=Mock(overall_score=0.85, confidence_level="high"),
        cached=False,
        processing_time=0.001
    )
    return mock_service


@pytest.fixture
def fast_file_path(tmp_path: Path) -> Path:
    """Ultra-fast file fixture with minimal content"""
    content = "Fast test content"
    p = tmp_path / "fast_test.txt"
    p.write_text(content, encoding="utf-8")
    return p