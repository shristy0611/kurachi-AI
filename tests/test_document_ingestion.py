"""
Test suite for document ingestion pipeline
Tests file type detection, processing, and progress tracking
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.document_ingestion import enhanced_ingestion_service, ProcessingStatus
from services.document_processors import processor_factory


class TestDocumentIngestion(unittest.TestCase):
    """Test cases for document ingestion service"""
    
    def setUp(self):
        """Set up test environment"""
        self.service = enhanced_ingestion_service
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str = "Test content") -> str:
        """Create a test file with given content"""
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(file_path)
    
    def test_supported_file_types(self):
        """Test that supported file types are correctly identified"""
        supported_types = self.service.get_supported_file_types()
        
        # Check that we have the expected categories
        expected_categories = ["PDF", "Word Documents", "Spreadsheets", "Presentations", 
                             "Images", "Audio", "Video", "Text"]
        
        for category in expected_categories:
            self.assertIn(category, supported_types)
            self.assertIsInstance(supported_types[category], list)
            self.assertGreater(len(supported_types[category]), 0)
    
    def test_file_type_detection(self):
        """Test file type detection"""
        # Test text file
        text_file = self.create_test_file("test.txt", "Hello world")
        file_info = self.service.detect_file_type(text_file)
        
        self.assertEqual(file_info["extension"], ".txt")
        self.assertEqual(file_info["mime_type"], "text/plain")
        self.assertTrue(file_info["supported"])
        self.assertEqual(file_info["processor_type"], "TextProcessor")
    
    def test_is_supported_file_type(self):
        """Test file type support checking"""
        # Supported types
        self.assertTrue(self.service.is_supported_file_type("document.pdf"))
        self.assertTrue(self.service.is_supported_file_type("spreadsheet.xlsx"))
        self.assertTrue(self.service.is_supported_file_type("image.jpg"))
        self.assertTrue(self.service.is_supported_file_type("text.txt"))
        
        # Unsupported types (should return False for unknown extensions)
        self.assertFalse(self.service.is_supported_file_type("unknown.xyz"))
    
    def test_file_validation(self):
        """Test file validation"""
        # Valid file
        text_file = self.create_test_file("test.txt", "Hello world")
        validation = self.service.validate_file(text_file)
        
        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["errors"]), 0)
        self.assertIn("file_info", validation)
    
    def test_file_validation_empty_file(self):
        """Test validation of empty file"""
        empty_file = self.create_test_file("empty.txt", "")
        validation = self.service.validate_file(empty_file)
        
        self.assertFalse(validation["valid"])
        self.assertIn("File is empty", validation["errors"])
    
    def test_file_validation_nonexistent_file(self):
        """Test validation of non-existent file"""
        validation = self.service.validate_file("/nonexistent/file.txt")
        
        self.assertFalse(validation["valid"])
        self.assertIn("File does not exist", validation["errors"])
    
    def test_processor_factory(self):
        """Test processor factory functionality"""
        # Test PDF processor selection
        pdf_processor = processor_factory.get_processor("test.pdf")
        self.assertIsNotNone(pdf_processor)
        self.assertEqual(pdf_processor.__class__.__name__, "PDFProcessor")
        
        # Test text processor selection
        text_processor = processor_factory.get_processor("test.txt")
        self.assertIsNotNone(text_processor)
        self.assertEqual(text_processor.__class__.__name__, "TextProcessor")
        
        # Test unsupported file
        unknown_processor = processor_factory.get_processor("test.unknown")
        self.assertIsNone(unknown_processor)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality"""
        document_id = "test-doc-123"
        tracker = self.service.progress_tracker
        
        # Start tracking
        tracker.start_tracking(document_id, total_steps=3)
        progress = tracker.get_progress(document_id)
        
        self.assertIsNotNone(progress)
        self.assertEqual(progress.document_id, document_id)
        self.assertEqual(progress.status, ProcessingStatus.PENDING)
        self.assertEqual(progress.progress_percentage, 0.0)
        
        # Update progress
        tracker.update_progress(document_id, ProcessingStatus.PROCESSING, "Processing", 1)
        progress = tracker.get_progress(document_id)
        
        self.assertEqual(progress.status, ProcessingStatus.PROCESSING)
        self.assertEqual(progress.current_step, "Processing")
        self.assertAlmostEqual(progress.progress_percentage, 33.33, places=1)
        
        # Complete processing
        tracker.complete_processing(document_id)
        progress = tracker.get_progress(document_id)
        
        self.assertEqual(progress.status, ProcessingStatus.COMPLETED)
        self.assertEqual(progress.progress_percentage, 100.0)
        
        # Clean up
        tracker.remove_progress(document_id)
    
    def test_progress_callback(self):
        """Test progress callback functionality"""
        document_id = "test-doc-callback"
        tracker = self.service.progress_tracker
        callback_calls = []
        
        def test_callback(progress):
            callback_calls.append(progress.status)
        
        # Start tracking with callback
        tracker.start_tracking(document_id)
        tracker.add_callback(document_id, test_callback)
        
        # Update progress - should trigger callback
        tracker.update_progress(document_id, ProcessingStatus.PROCESSING, "Processing", 1)
        tracker.complete_processing(document_id)
        
        # Check that callbacks were called
        self.assertIn(ProcessingStatus.PROCESSING, callback_calls)
        self.assertIn(ProcessingStatus.COMPLETED, callback_calls)
        
        # Clean up
        tracker.remove_progress(document_id)
    
    @patch('services.document_ingestion.db_manager')
    def test_mock_document_processing(self, mock_db_manager):
        """Test document processing with mocked database"""
        # Create a mock document
        mock_document = Mock()
        mock_document.id = "test-doc"
        mock_document.file_path = self.create_test_file("test.txt", "Test content for processing")
        mock_document.original_filename = "test.txt"
        mock_document.file_type = ".txt"
        mock_document.created_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_document.user_id = "test-user"
        
        # Mock database methods
        mock_db_manager.get_document.return_value = mock_document
        mock_db_manager.update_document_status.return_value = True
        
        # Test processing
        result = self.service.process_document_with_progress("test-doc")
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify database calls
        mock_db_manager.get_document.assert_called_with("test-doc")
        self.assertTrue(mock_db_manager.update_document_status.called)


class TestDocumentProcessors(unittest.TestCase):
    """Test cases for individual document processors"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str = "Test content") -> str:
        """Create a test file with given content"""
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(file_path)
    
    def test_text_processor(self):
        """Test text file processing"""
        from services.document_processors import TextProcessor
        
        processor = TextProcessor()
        
        # Test file type detection
        self.assertTrue(processor.can_process("test.txt", "text/plain"))
        self.assertTrue(processor.can_process("test.md", "text/markdown"))
        self.assertFalse(processor.can_process("test.pdf", "application/pdf"))
        
        # Test processing
        text_file = self.create_test_file("test.txt", "Hello, this is a test document.")
        result = processor.process(text_file)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.documents), 0)
        self.assertIn("Hello, this is a test document.", result.documents[0].page_content)
        self.assertEqual(result.documents[0].metadata["document_type"], "text")
    
    def test_processor_file_info(self):
        """Test processor file info extraction"""
        from services.document_processors import TextProcessor
        
        processor = TextProcessor()
        text_file = self.create_test_file("test.txt", "Test content")
        
        file_info = processor.get_file_info(text_file)
        
        self.assertEqual(file_info["filename"], "test.txt")
        self.assertEqual(file_info["mime_type"], "text/plain")
        self.assertEqual(file_info["file_extension"], ".txt")
        self.assertGreater(file_info["file_size"], 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)