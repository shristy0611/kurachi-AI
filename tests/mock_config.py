#!/usr/bin/env python3
"""
Aggressive mocking configuration for ultra-fast tests
Patches expensive operations at import time
"""
import os
from unittest.mock import Mock, patch, MagicMock

# Set fast test mode
os.environ['KURACHI_FAST_TEST'] = '1'

# Mock expensive imports before they're loaded
def setup_fast_mocks():
    """Setup mocks for expensive operations"""
    
    # Mock Ollama at import time
    mock_ollama = Mock()
    mock_llm = Mock()
    mock_llm.invoke.return_value = "Fast mock response"
    mock_ollama.return_value = mock_llm
    
    # Patch all LLM-related imports
    patches = [
        patch('langchain_community.llms.Ollama', mock_ollama),
        patch('services.translation_service.Ollama', mock_ollama),
        patch('services.intelligent_translation.Ollama', mock_ollama),
        patch('services.advanced_content_extraction.Ollama', mock_ollama),
    ]
    
    # Start all patches
    for p in patches:
        p.start()
    
    # Mock database operations
    mock_db = Mock()
    mock_db.get_metric.return_value = 0
    mock_db.increment_metric.return_value = None
    mock_db.save_preferences.return_value = True
    mock_db.load_preferences.return_value = {}
    
    db_patches = [
        patch('models.database.db_manager', mock_db),
        patch('services.preference_manager.db_manager', mock_db),
    ]
    
    for p in db_patches:
        p.start()
    
    # Mock file operations that might be slow
    mock_file_ops = [
        patch('pathlib.Path.glob', return_value=[]),
        patch('pathlib.Path.exists', return_value=True),
        patch('pathlib.Path.is_file', return_value=True),
    ]
    
    # Don't start file ops patches as they might break legitimate file operations
    
    print("⚡ Fast mocks activated")

# Auto-setup when imported
if os.environ.get('KURACHI_FAST_TEST'):
    setup_fast_mocks()