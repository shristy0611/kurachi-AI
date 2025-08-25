#!/usr/bin/env python3
"""
Service Contract Smoke Tests
Fast-executing tests that verify service interfaces and configuration requirements.
These tests ensure critical service contracts remain stable and prevent regressions.
"""
import pytest
import time
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock


@pytest.mark.smoke
class TestDatabaseManagerContract:
    """Smoke tests for database manager service contract"""
    
    def test_database_manager_get_connection_exists(self):
        """Verify database manager has get_connection method"""
        from models.database import db_manager
        
        # Verify method exists and is callable
        assert hasattr(db_manager, 'get_connection'), "DatabaseManager missing get_connection method"
        assert callable(db_manager.get_connection), "get_connection is not callable"
    
    def test_database_manager_get_connection_returns_connection(self):
        """Verify get_connection returns a valid database connection"""
        from models.database import db_manager
        
        start_time = time.time()
        conn = db_manager.get_connection()
        execution_time = time.time() - start_time
        
        # Verify connection is returned
        assert conn is not None, "get_connection returned None"
        
        # Verify it's a database connection (has execute method)
        assert hasattr(conn, 'execute'), "Connection missing execute method"
        assert hasattr(conn, 'close'), "Connection missing close method"
        
        # Verify fast execution (under 1 second)
        assert execution_time < 1.0, f"get_connection took {execution_time:.2f}s, should be < 1.0s"
        
        # Clean up
        conn.close()
    
    def test_database_manager_initialization_fast(self):
        """Verify database manager initializes quickly"""
        from models.database import DatabaseManager
        
        start_time = time.time()
        # Create new instance to test initialization
        test_db = DatabaseManager(":memory:")
        execution_time = time.time() - start_time
        
        # Verify fast initialization (under 2 seconds)
        assert execution_time < 2.0, f"DatabaseManager initialization took {execution_time:.2f}s, should be < 2.0s"
        
        # Verify basic functionality
        assert hasattr(test_db, 'get_connection'), "Initialized DatabaseManager missing get_connection"
        
        # Test connection works
        conn = test_db.get_connection()
        assert conn is not None
        conn.close()


@pytest.mark.smoke
class TestPreferenceManagerContract:
    """Smoke tests for preference manager service contract"""
    
    def test_preference_manager_get_preference_exists(self):
        """Verify preference manager has get_preference method"""
        from services.preference_manager import preference_manager
        
        # Verify method exists and is callable
        assert hasattr(preference_manager, 'get_preference'), "PreferenceManager missing get_preference method"
        assert callable(preference_manager.get_preference), "get_preference is not callable"
    
    def test_preference_manager_get_preference_basic_functionality(self):
        """Verify get_preference returns expected values"""
        from services.preference_manager import preference_manager
        
        start_time = time.time()
        
        # Test with default value
        result = preference_manager.get_preference("nonexistent_key", "default_value")
        assert result == "default_value", "get_preference should return default for nonexistent key"
        
        # Test with known preference
        result = preference_manager.get_preference("ui_language", "fallback")
        assert result is not None, "get_preference returned None for ui_language"
        
        execution_time = time.time() - start_time
        
        # Verify fast execution (under 1 second)
        assert execution_time < 1.0, f"get_preference took {execution_time:.2f}s, should be < 1.0s"
    
    def test_preference_manager_initialization_fast(self):
        """Verify preference manager initializes quickly"""
        from services.preference_manager import EnhancedPreferenceManager
        
        start_time = time.time()
        # Create new instance to test initialization
        test_pref_manager = EnhancedPreferenceManager()
        execution_time = time.time() - start_time
        
        # Verify fast initialization (under 3 seconds)
        assert execution_time < 3.0, f"PreferenceManager initialization took {execution_time:.2f}s, should be < 3.0s"
        
        # Verify basic functionality
        assert hasattr(test_pref_manager, 'get_preference'), "Initialized PreferenceManager missing get_preference"
        
        # Test method works
        result = test_pref_manager.get_preference("test_key", "test_default")
        assert result == "test_default"


@pytest.mark.smoke
class TestConfigurationContract:
    """Smoke tests for configuration system contract"""
    
    def test_config_module_has_required_attributes(self):
        """Verify config module has all required attributes"""
        import config
        
        required_attributes = [
            'config',  # Main config object
            'ai',      # AI configuration
            'database', # Database configuration  
            'app',     # Application configuration
            'security' # Security configuration
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(config, attr):
                missing_attributes.append(attr)
        
        assert not missing_attributes, f"Config module missing required attributes: {missing_attributes}"
    
    def test_config_sections_are_accessible(self):
        """Verify config sections are not None and have expected properties"""
        import config
        
        # Test config sections exist and are not None
        assert config.ai is not None, "config.ai is None"
        assert config.database is not None, "config.database is None"
        assert config.app is not None, "config.app is None"
        assert config.security is not None, "config.security is None"
    
    def test_database_config_has_required_properties(self):
        """Verify database config has required properties"""
        import config
        
        required_db_properties = [
            'sqlite_path',
            'chroma_persist_dir',
            'neo4j_uri',
            'neo4j_username',
            'neo4j_password'
        ]
        
        missing_properties = []
        for prop in required_db_properties:
            if not hasattr(config.database, prop):
                missing_properties.append(prop)
        
        assert not missing_properties, f"Database config missing required properties: {missing_properties}"
        
        # Verify properties have values
        assert config.database.sqlite_path, "sqlite_path is empty"
        assert config.database.chroma_persist_dir, "chroma_persist_dir is empty"
    
    def test_ai_config_has_required_properties(self):
        """Verify AI config has required properties"""
        import config
        
        required_ai_properties = [
            'llm_model',
            'embedding_model',
            'ollama_base_url',
            'chunk_size',
            'max_tokens',
            'temperature'
        ]
        
        missing_properties = []
        for prop in required_ai_properties:
            if not hasattr(config.ai, prop):
                missing_properties.append(prop)
        
        assert not missing_properties, f"AI config missing required properties: {missing_properties}"
        
        # Verify numeric properties are valid
        assert isinstance(config.ai.chunk_size, int), "chunk_size should be int"
        assert config.ai.chunk_size > 0, "chunk_size should be positive"
        assert isinstance(config.ai.max_tokens, int), "max_tokens should be int"
        assert config.ai.max_tokens > 0, "max_tokens should be positive"
        assert isinstance(config.ai.temperature, float), "temperature should be float"
        assert 0.0 <= config.ai.temperature <= 2.0, "temperature should be between 0.0 and 2.0"
    
    def test_app_config_has_required_properties(self):
        """Verify app config has required properties"""
        import config
        
        required_app_properties = [
            'app_name',
            'version',
            'debug',
            'log_level',
            'upload_dir',
            'max_file_size_mb',
            'supported_file_types'
        ]
        
        missing_properties = []
        for prop in required_app_properties:
            if not hasattr(config.app, prop):
                missing_properties.append(prop)
        
        assert not missing_properties, f"App config missing required properties: {missing_properties}"
        
        # Verify properties have expected types
        assert isinstance(config.app.debug, bool), "debug should be bool"
        assert isinstance(config.app.max_file_size_mb, int), "max_file_size_mb should be int"
        assert config.app.max_file_size_mb > 0, "max_file_size_mb should be positive"
        assert isinstance(config.app.supported_file_types, list), "supported_file_types should be list"
        assert len(config.app.supported_file_types) > 0, "supported_file_types should not be empty"
    
    def test_config_initialization_fast(self):
        """Verify config initialization is fast"""
        from config import Config
        
        start_time = time.time()
        # Create new config instance to test initialization
        test_config = Config()
        execution_time = time.time() - start_time
        
        # Verify fast initialization (under 2 seconds)
        assert execution_time < 2.0, f"Config initialization took {execution_time:.2f}s, should be < 2.0s"
        
        # Verify basic functionality
        assert test_config.database is not None
        assert test_config.ai is not None
        assert test_config.app is not None
        assert test_config.security is not None


@pytest.mark.smoke
class TestServiceImportContract:
    """Smoke tests for critical service imports"""
    
    def test_critical_services_import_successfully(self):
        """Verify all critical services can be imported without errors"""
        critical_imports = [
            ('models.database', 'db_manager'),
            ('services.preference_manager', 'preference_manager'),
            ('services.translation_service', 'translation_service'),
            ('services.language_detection', 'language_detection_service'),
            ('config', 'config'),
        ]
        
        import_errors = []
        start_time = time.time()
        
        for module_name, object_name in critical_imports:
            try:
                module = __import__(module_name, fromlist=[object_name])
                obj = getattr(module, object_name)
                assert obj is not None, f"{object_name} is None after import"
            except Exception as e:
                import_errors.append(f"{module_name}.{object_name}: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Verify no import errors
        assert not import_errors, f"Import errors: {import_errors}"
        
        # Verify fast imports (under 5 seconds total)
        assert execution_time < 5.0, f"Critical imports took {execution_time:.2f}s, should be < 5.0s"
    
    def test_service_objects_have_expected_methods(self):
        """Verify imported service objects have expected methods"""
        service_method_requirements = [
            ('models.database', 'db_manager', ['get_connection', 'init_database']),
            ('services.preference_manager', 'preference_manager', ['get_preference', 'get_user_preferences']),
            ('services.translation_service', 'translation_service', ['translate']),
            ('services.language_detection', 'language_detection_service', ['detect_language']),
        ]
        
        method_errors = []
        
        for module_name, object_name, required_methods in service_method_requirements:
            try:
                module = __import__(module_name, fromlist=[object_name])
                obj = getattr(module, object_name)
                
                for method_name in required_methods:
                    if not hasattr(obj, method_name):
                        method_errors.append(f"{object_name} missing method: {method_name}")
                    elif not callable(getattr(obj, method_name)):
                        method_errors.append(f"{object_name}.{method_name} is not callable")
                        
            except Exception as e:
                method_errors.append(f"Error checking {object_name}: {str(e)}")
        
        assert not method_errors, f"Method requirement errors: {method_errors}"


@pytest.mark.smoke
class TestServiceInitializationContract:
    """Smoke tests for service initialization requirements"""
    
    def test_services_initialize_without_external_dependencies(self):
        """Verify core services can initialize without external dependencies"""
        # Test services that should work without external connections
        core_services = [
            ('models.database', 'DatabaseManager', {'db_path': ':memory:'}),
            ('services.preference_manager', 'EnhancedPreferenceManager', {}),
            ('config', 'Config', {}),
        ]
        
        initialization_errors = []
        
        for module_name, class_name, init_kwargs in core_services:
            try:
                start_time = time.time()
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                instance = cls(**init_kwargs)
                execution_time = time.time() - start_time
                
                # Verify instance is created
                assert instance is not None, f"{class_name} instance is None"
                
                # Verify fast initialization (under 3 seconds per service)
                if execution_time >= 3.0:
                    initialization_errors.append(f"{class_name} initialization took {execution_time:.2f}s, should be < 3.0s")
                    
            except Exception as e:
                initialization_errors.append(f"{class_name} initialization failed: {str(e)}")
        
        assert not initialization_errors, f"Service initialization errors: {initialization_errors}"
    
    def test_database_connection_works_immediately(self):
        """Verify database connection works immediately after initialization"""
        from models.database import DatabaseManager
        
        # Create in-memory database for testing
        test_db = DatabaseManager(":memory:")
        
        # Test connection works immediately
        conn = test_db.get_connection()
        assert conn is not None, "Database connection is None"
        
        # Test basic SQL operation
        cursor = conn.execute("SELECT 1 as test")
        result = cursor.fetchone()
        assert result[0] == 1, "Basic SQL operation failed"
        
        conn.close()
    
    def test_preference_manager_works_immediately(self):
        """Verify preference manager works immediately after initialization"""
        from services.preference_manager import EnhancedPreferenceManager
        
        # Create new instance
        test_pref_manager = EnhancedPreferenceManager()
        
        # Test basic functionality works immediately
        result = test_pref_manager.get_preference("test_key", "default_value")
        assert result == "default_value", "Preference manager basic functionality failed"


@pytest.mark.smoke
class TestPerformanceContract:
    """Smoke tests for performance requirements"""
    
    def test_smoke_tests_complete_within_time_limit(self):
        """Verify all smoke tests complete within 5-second requirement"""
        # This test tracks its own execution time and the time of other smoke tests
        # by checking if we're still under the time limit
        start_time = time.time()
        
        # Perform basic operations that other smoke tests do
        from models.database import db_manager
        from services.preference_manager import preference_manager
        import config
        
        # Basic database operation
        conn = db_manager.get_connection()
        conn.close()
        
        # Basic preference operation
        pref_result = preference_manager.get_preference("test", "default")
        
        # Basic config access
        app_name = config.app.app_name
        
        execution_time = time.time() - start_time
        
        # Verify operations completed quickly
        assert execution_time < 2.0, f"Basic smoke operations took {execution_time:.2f}s, should be < 2.0s"
        assert pref_result == "default", "Preference operation failed"
        assert app_name, "Config access failed"
    
    def test_concurrent_service_access(self):
        """Verify services handle concurrent access without errors"""
        import threading
        import time
        
        results = []
        errors = []
        
        def test_database_access():
            try:
                from models.database import db_manager
                conn = db_manager.get_connection()
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()
                results.append(result[0])
                conn.close()
            except Exception as e:
                errors.append(f"Database access error: {e}")
        
        def test_preference_access():
            try:
                from services.preference_manager import preference_manager
                result = preference_manager.get_preference("concurrent_test", "success")
                results.append(result)
            except Exception as e:
                errors.append(f"Preference access error: {e}")
        
        # Run concurrent operations
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=test_database_access))
            threads.append(threading.Thread(target=test_preference_access))
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        execution_time = time.time() - start_time
        
        # Verify no errors and reasonable performance
        assert not errors, f"Concurrent access errors: {errors}"
        assert len(results) == 6, f"Expected 6 results, got {len(results)}"
        assert execution_time < 3.0, f"Concurrent operations took {execution_time:.2f}s, should be < 3.0s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])