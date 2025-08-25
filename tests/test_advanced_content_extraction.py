"""
Test suite for advanced content extraction with llava:7b integration
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Guard optional dependencies
pytest.importorskip("fitz", reason="PyMuPDF not installed")

# Mark entire module as slow due to PyMuPDF dependency and heavy processing
pytestmark = pytest.mark.slow

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.advanced_content_extraction import (
    AdvancedContentExtractionService,
    LlavaVisionAnalyzer,
    AdvancedPDFProcessor,
    AdvancedExcelProcessor,
    AdvancedPowerPointProcessor,
    AdvancedAudioVideoProcessor,
    StructuralElement,
    AdvancedProcessingResult
)
from services.document_processors import processor_factory
from config import config


class TestLlavaVisionAnalyzer(unittest.TestCase):
    """Test llava:7b vision analysis functionality"""
    
    def setUp(self):
        self.analyzer = LlavaVisionAnalyzer()
    
    @patch('services.advanced_content_extraction.Ollama')
    def test_analyze_document_structure(self, mock_ollama):
        """Test document structure analysis with llava:7b"""
        # Mock llava response
        mock_response = """{
            "document_type": "business_report",
            "layout": "multi-column with charts",
            "elements": [
                {
                    "type": "table",
                    "description": "Financial data table with quarterly results",
                    "position": "center of page",
                    "content_summary": "Q1-Q4 revenue and profit data"
                },
                {
                    "type": "chart",
                    "description": "Bar chart showing revenue growth",
                    "position": "bottom right",
                    "content_summary": "Revenue increased 15% year over year"
                }
            ],
            "tables_detected": 1,
            "charts_detected": 1,
            "text_regions": 3,
            "quality_assessment": "High quality, clearly readable"
        }"""
        
        mock_ollama.return_value.invoke.return_value = mock_response
        
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img.write(b"fake image data")
            temp_img_path = temp_img.name
        
        try:
            result = self.analyzer.analyze_document_structure(temp_img_path, "Test document")
            
            self.assertEqual(result["document_type"], "business_report")
            self.assertEqual(result["tables_detected"], 1)
            self.assertEqual(result["charts_detected"], 1)
            self.assertEqual(len(result["elements"]), 2)
            self.assertEqual(result["elements"][0]["type"], "table")
            
        finally:
            os.unlink(temp_img_path)
    
    @patch('services.advanced_content_extraction.Ollama')
    def test_analyze_visual_elements(self, mock_ollama):
        """Test visual element analysis"""
        mock_response = "This image contains a bar chart showing quarterly sales data with an upward trend. There is also a data table below the chart with specific numerical values for each quarter."
        
        mock_ollama.return_value.invoke.return_value = mock_response
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img.write(b"fake image data")
            temp_img_path = temp_img.name
        
        try:
            result = self.analyzer.analyze_visual_elements(temp_img_path)
            
            self.assertIn("visual_description", result)
            self.assertEqual(result["analysis_method"], "llava_vision")
            self.assertIn("chart", result["visual_description"])
            self.assertIn("table", result["visual_description"])
            
        finally:
            os.unlink(temp_img_path)


class TestAdvancedPDFProcessor(unittest.TestCase):
    """Test advanced PDF processing with llava:7b"""
    
    def setUp(self):
        self.processor = AdvancedPDFProcessor()
    
    @patch('services.advanced_content_extraction.fitz')
    @patch('services.advanced_content_extraction.LlavaVisionAnalyzer')
    def test_process_pdf_with_structure_analysis(self, mock_analyzer, mock_fitz):
        """Test PDF processing with structure analysis"""
        # Mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample PDF text content"
        mock_page.get_pixmap.return_value.tobytes.return_value = b"fake image data"
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz.open.return_value = mock_doc
        
        # Mock vision analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_document_structure.return_value = {
            "document_type": "report",
            "layout": "single column",
            "elements": [{"type": "paragraph", "description": "Main text content"}],
            "tables_detected": 0,
            "charts_detected": 0
        }
        mock_analyzer_instance.analyze_visual_elements.return_value = {
            "visual_description": "Text document with no visual elements"
        }
        mock_analyzer.return_value = mock_analyzer_instance
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(b"fake pdf data")
            temp_pdf_path = temp_pdf.name
        
        try:
            result = self.processor.process_pdf_with_structure_analysis(temp_pdf_path)
            
            self.assertTrue(result.success)
            self.assertEqual(len(result.documents), 1)
            self.assertIn("TEXT CONTENT", result.documents[0].page_content)
            self.assertIn("DOCUMENT STRUCTURE", result.documents[0].page_content)
            self.assertEqual(result.documents[0].metadata["extraction_method"], "advanced_llava_analysis")
            
        finally:
            os.unlink(temp_pdf_path)


class TestAdvancedExcelProcessor(unittest.TestCase):
    """Test advanced Excel processing"""
    
    def setUp(self):
        self.processor = AdvancedExcelProcessor()
    
    @patch('services.advanced_content_extraction.pd')
    def test_process_excel_with_relationships(self, mock_pd):
        """Test Excel processing with relationship preservation"""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.__len__.return_value = 10
        mock_df.columns = ['Name', 'Age', 'Salary']
        mock_df.dtypes = {'Name': 'object', 'Age': 'int64', 'Salary': 'float64'}
        mock_df.head.return_value.to_string.return_value = "Sample data preview"
        mock_df.select_dtypes.return_value.columns = ['Age', 'Salary']
        mock_df.describe.return_value.to_string.return_value = "Statistical summary"
        
        mock_excel_file = Mock()
        mock_excel_file.sheet_names = ['Sheet1']
        
        mock_pd.ExcelFile.return_value = mock_excel_file
        mock_pd.read_excel.return_value = mock_df
        
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_excel:
            temp_excel.write(b"fake excel data")
            temp_excel_path = temp_excel.name
        
        try:
            result = self.processor.process_excel_with_relationships(temp_excel_path)
            
            self.assertTrue(result.success)
            self.assertEqual(len(result.documents), 1)
            self.assertIn("EXCEL SHEET", result.documents[0].page_content)
            self.assertIn("COLUMNS", result.documents[0].page_content)
            self.assertIn("DATA SAMPLE", result.documents[0].page_content)
            
        finally:
            os.unlink(temp_excel_path)


class TestAdvancedPowerPointProcessor(unittest.TestCase):
    """Test advanced PowerPoint processing"""
    
    def setUp(self):
        self.processor = AdvancedPowerPointProcessor()
    
    @patch('services.advanced_content_extraction.Presentation')
    def test_process_powerpoint_with_analysis(self, mock_presentation):
        """Test PowerPoint processing with slide analysis"""
        # Mock presentation and slides
        mock_slide = Mock()
        mock_shape = Mock()
        mock_shape.text = "Sample slide text"
        mock_shape.shape_type = 1  # Text shape
        mock_shape.left = 100
        mock_shape.top = 200
        mock_shape.width = 300
        mock_shape.height = 400
        
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = False
        mock_slide.slide_layout.name = "Title and Content"
        
        mock_pres = Mock()
        mock_pres.slides = [mock_slide]
        mock_presentation.return_value = mock_pres
        
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp_ppt:
            temp_ppt.write(b"fake powerpoint data")
            temp_ppt_path = temp_ppt.name
        
        try:
            result = self.processor.process_powerpoint_with_analysis(temp_ppt_path)
            
            self.assertTrue(result.success)
            self.assertEqual(len(result.documents), 1)
            self.assertIn("SLIDE 1", result.documents[0].page_content)
            self.assertIn("SLIDE CONTENT", result.documents[0].page_content)
            self.assertEqual(result.documents[0].metadata["slide_number"], 1)
            
        finally:
            os.unlink(temp_ppt_path)


class TestAdvancedAudioVideoProcessor(unittest.TestCase):
    """Test advanced audio/video processing"""
    
    def setUp(self):
        self.processor = AdvancedAudioVideoProcessor()
    
    @patch('services.advanced_content_extraction.whisper')
    def test_process_audio_with_whisper(self, mock_whisper):
        """Test audio processing with Whisper transcription"""
        # Mock Whisper model and result
        mock_model = Mock()
        mock_result = {
            "text": "This is a sample audio transcription with multiple segments.",
            "language": "en",
            "duration": 30.5,
            "segments": [
                {
                    "start": 0.0,
                    "end": 15.0,
                    "text": "This is a sample audio transcription",
                    "avg_logprob": -0.2
                },
                {
                    "start": 15.0,
                    "end": 30.5,
                    "text": "with multiple segments.",
                    "avg_logprob": -0.3
                }
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(b"fake audio data")
            temp_audio_path = temp_audio.name
        
        try:
            result = self.processor.process_audio_with_whisper(temp_audio_path)
            
            self.assertTrue(result.success)
            self.assertEqual(len(result.documents), 1)
            self.assertIn("AUDIO TRANSCRIPTION", result.documents[0].page_content)
            self.assertIn("TIMESTAMPED SEGMENTS", result.documents[0].page_content)
            self.assertEqual(result.documents[0].metadata["detected_language"], "en")
            self.assertEqual(len(result.structural_elements), 2)  # Two segments
            
        finally:
            os.unlink(temp_audio_path)


class TestAdvancedContentExtractionService(unittest.TestCase):
    """Test the main advanced content extraction service"""
    
    def setUp(self):
        self.service = AdvancedContentExtractionService()
    
    def test_service_initialization(self):
        """Test service initializes all processors"""
        self.assertIsNotNone(self.service.pdf_processor)
        self.assertIsNotNone(self.service.excel_processor)
        self.assertIsNotNone(self.service.powerpoint_processor)
        self.assertIsNotNone(self.service.audio_video_processor)
        self.assertIsNotNone(self.service.vision_analyzer)
    
    @patch('services.advanced_content_extraction.AdvancedPDFProcessor')
    def test_process_document_advanced_pdf(self, mock_pdf_processor):
        """Test advanced processing routes to correct processor"""
        mock_result = AdvancedProcessingResult(
            success=True,
            documents=[],
            structural_elements=[],
            visual_analysis={},
            metadata={"test": "data"}
        )
        mock_pdf_processor.return_value.process_pdf_with_structure_analysis.return_value = mock_result
        
        result = self.service.process_document_advanced("test.pdf", ".pdf")
        
        self.assertTrue(result.success)
        mock_pdf_processor.return_value.process_pdf_with_structure_analysis.assert_called_once()


class TestIntegrationWithExistingProcessors(unittest.TestCase):
    """Test integration with existing document processors"""
    
    def test_processor_factory_integration(self):
        """Test that existing processors can use advanced extraction"""
        # Test that PDF processor exists and can be enhanced
        pdf_processor = processor_factory.get_processor("test.pdf")
        self.assertIsNotNone(pdf_processor)
        
        # Test that Excel processor exists
        excel_processor = processor_factory.get_processor("test.xlsx")
        self.assertIsNotNone(excel_processor)
        
        # Test that PowerPoint processor exists
        ppt_processor = processor_factory.get_processor("test.pptx")
        self.assertIsNotNone(ppt_processor)
        
        # Test that Image processor exists
        img_processor = processor_factory.get_processor("test.jpg")
        self.assertIsNotNone(img_processor)
        
        # Test that Audio processor exists
        audio_processor = processor_factory.get_processor("test.mp3")
        self.assertIsNotNone(audio_processor)


if __name__ == '__main__':
    # Set up test environment
    os.environ['KURACHI_ENV'] = 'test'
    
    # Run tests
    unittest.main(verbosity=2)