# tests/conftest.py
"""
Pytest configuration and fixtures for multilingual pipeline tests
"""
from pathlib import Path
import os
import pytest
import tempfile

FAST = os.getenv("FAST_TESTS") == "1"


@pytest.fixture(scope="session")
def tests_root(request) -> Path:
    """Root directory of tests"""
    return Path(request.config.rootpath) / "tests"


@pytest.fixture(scope="session") 
def data_dir(tests_root: Path) -> Path:
    """Test data directory"""
    d = tests_root / "data"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def file_path(tmp_path: Path) -> Path:
    """
    Provides a real, readable file path for tests that expect `file_path: Path`.
    Creates a sample document that can be processed by the document pipeline.
    """
    # Create a sample markdown file that the processor can handle
    sample_content = """# Sample Document for Testing

## Introduction

This is a sample document used for testing the document processing pipeline.
It contains multiple sections and various content types to ensure proper processing.

## Key Features

- **Multilingual Support**: The system supports multiple languages
- **Document Processing**: Various file formats are supported
- **Quality Assurance**: Comprehensive testing ensures reliability

## Technical Details

The document processing system uses intelligent chunking and analysis to:

1. Extract meaningful content
2. Maintain document structure
3. Preserve formatting and context
4. Enable efficient search and retrieval

## Conclusion

This sample document provides sufficient content for testing document processing
capabilities while remaining lightweight for fast test execution.

---

*Generated for automated testing purposes*
"""
    
    # Create the file
    p = tmp_path / "sample_document.md"
    p.write_text(sample_content, encoding="utf-8")
    return p


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Simple text file for basic processing tests"""
    content = "This is a simple text file for testing basic document processing functionality."
    p = tmp_path / "simple.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """JSON file for structured data tests"""
    import json
    
    data = {
        "title": "Sample JSON Document",
        "content": "This is sample content for JSON processing tests",
        "metadata": {
            "language": "en",
            "type": "test_document",
            "version": "1.0"
        }
    }
    
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# Performance optimization fixtures
@pytest.fixture(scope="session")
def shared_translation_service():
    """Shared translation service to avoid repeated initialization"""
    from services.intelligent_translation import IntelligentTranslationService
    return IntelligentTranslationService()


@pytest.fixture(scope="session") 
def shared_preference_manager():
    """Shared preference manager for faster tests"""
    from services.preference_manager import preference_manager
    return preference_manager


@pytest.fixture(autouse=True)
def _fast_mode_monkeypatch(monkeypatch):
    """Stub heavy services gracefully in FAST mode"""
    if not FAST:
        return
    
    # Stub heavy services gracefully
    try:
        from services import intelligent_translation as it
        monkeypatch.setattr(it.IntelligentTranslationService, "__init__", lambda self, *a, **k: None)
        monkeypatch.setattr(it.IntelligentTranslationService, "translate", lambda *a, **k: "stub")
    except Exception:
        pass
    
    try:
        from services import chat_service as cs
        monkeypatch.setattr(cs, "create_conversation", lambda *a, **k: None)
    except Exception:
        pass