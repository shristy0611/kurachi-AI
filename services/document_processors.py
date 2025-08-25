"""
Document processors for different file types
Handles extraction and processing of various document formats
"""
import os
import io
import mimetypes
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import tempfile
import subprocess

# Core processing libraries
from langchain.schema import Document as LangChainDocument
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader
)

# Additional libraries for advanced processing
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger("document_processors")

# Import advanced extraction service
try:
    from services.advanced_content_extraction import advanced_extraction_service, AdvancedProcessingResult
    ADVANCED_EXTRACTION_AVAILABLE = True
except ImportError:
    ADVANCED_EXTRACTION_AVAILABLE = False
    logger.warning("Advanced extraction service not available")


@dataclass
class ProcessingResult:
    """Result of document processing"""
    success: bool
    documents: List[LangChainDocument]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    extracted_elements: Optional[Dict[str, Any]] = None


class DocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    @abstractmethod
    def can_process(self, file_path: str, mime_type: str) -> bool:
        """Check if this processor can handle the given file"""
        pass
    
    @abstractmethod
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process the document and return extracted content"""
        pass
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information"""
        file_path = Path(file_path)
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return {
            "filename": file_path.name,
            "file_size": stat.st_size,
            "mime_type": mime_type or "application/octet-stream",
            "file_extension": file_path.suffix.lower(),
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime
        }


class PDFProcessor(DocumentProcessor):
    """Processor for PDF documents with OCR support"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        return mime_type == "application/pdf" or file_path.lower().endswith('.pdf')
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process PDF with advanced llava:7b integration and OCR fallback"""
        import time
        start_time = time.time()
        
        try:
            # Try advanced processing with llava:7b first
            if ADVANCED_EXTRACTION_AVAILABLE and kwargs.get("use_advanced", True):
                try:
                    logger.info(f"Using advanced llava:7b processing for PDF: {file_path}")
                    advanced_result = advanced_extraction_service.process_document_advanced(file_path, ".pdf")
                    
                    if advanced_result.success and advanced_result.documents:
                        # Convert AdvancedProcessingResult to ProcessingResult
                        return ProcessingResult(
                            success=True,
                            documents=advanced_result.documents,
                            metadata=advanced_result.metadata,
                            processing_time=advanced_result.processing_time,
                            extracted_elements={
                                "pages": len(advanced_result.documents),
                                "method": "llava_advanced",
                                "structural_elements": len(advanced_result.structural_elements),
                                "has_visual_analysis": bool(advanced_result.visual_analysis),
                                "advanced_processing": True
                            }
                        )
                except Exception as e:
                    logger.warning(f"Advanced processing failed, falling back to standard: {e}")
            
            # Fallback to standard processing
            documents = []
            metadata = self.get_file_info(file_path)
            
            # Try PyPDFLoader first for better text extraction
            try:
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                
                # Check if we got meaningful text
                total_text = "".join([doc.page_content for doc in pdf_documents])
                if len(total_text.strip()) > 50:  # Arbitrary threshold for meaningful text
                    documents = pdf_documents
                    metadata["extraction_method"] = "text_based"
                    logger.info(f"Successfully extracted text from PDF: {file_path}")
                else:
                    raise ValueError("Insufficient text extracted, trying OCR")
                    
            except Exception as e:
                logger.warning(f"Text extraction failed for {file_path}, trying OCR: {e}")
                
                # Fallback to OCR if text extraction fails or yields poor results
                if TESSERACT_AVAILABLE:
                    documents = self._process_with_ocr(file_path)
                    metadata["extraction_method"] = "ocr"
                else:
                    # Final fallback to UnstructuredFileLoader
                    loader = UnstructuredFileLoader(file_path)
                    documents = loader.load()
                    metadata["extraction_method"] = "unstructured"
            
            # Add page numbers and source information
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "page_number": i + 1,
                    "source": file_path,
                    "document_type": "pdf",
                    **metadata
                })
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={"pages": len(documents), "method": metadata.get("extraction_method")}
            )
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                metadata=self.get_file_info(file_path),
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _process_with_ocr(self, file_path: str) -> List[LangChainDocument]:
        """Process PDF using OCR"""
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not available for OCR processing")
        
        try:
            # Convert PDF to images and process with OCR
            import fitz  # PyMuPDF
            
            documents = []
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                text = pytesseract.image_to_string(image, lang='eng+jpn')  # Support English and Japanese
                
                if text.strip():
                    doc = LangChainDocument(
                        page_content=text,
                        metadata={
                            "page_number": page_num + 1,
                            "extraction_method": "ocr"
                        }
                    )
                    documents.append(doc)
            
            pdf_document.close()
            return documents
            
        except ImportError:
            logger.error("PyMuPDF (fitz) not available for PDF to image conversion")
            raise
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise


class WordProcessor(DocumentProcessor):
    """Processor for Word documents (.docx)"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        return (mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" 
                or file_path.lower().endswith('.docx'))
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            metadata = self.get_file_info(file_path)
            metadata["extraction_method"] = "docx2txt"
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "document_type": "docx",
                    **metadata
                })
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={"sections": len(documents)}
            )
            
        except Exception as e:
            logger.error(f"Failed to process Word document {file_path}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                metadata=self.get_file_info(file_path),
                error_message=str(e),
                processing_time=time.time() - start_time
            )


class ExcelProcessor(DocumentProcessor):
    """Processor for Excel documents (.xlsx, .csv)"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        excel_types = [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "text/csv"
        ]
        excel_extensions = ['.xlsx', '.xls', '.csv']
        
        return mime_type in excel_types or any(file_path.lower().endswith(ext) for ext in excel_extensions)
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            # Try advanced processing for Excel files (not CSV)
            if (ADVANCED_EXTRACTION_AVAILABLE and 
                not file_path.lower().endswith('.csv') and 
                kwargs.get("use_advanced", True)):
                try:
                    logger.info(f"Using advanced Excel processing: {file_path}")
                    advanced_result = advanced_extraction_service.process_document_advanced(
                        file_path, Path(file_path).suffix.lower()
                    )
                    
                    if advanced_result.success and advanced_result.documents:
                        return ProcessingResult(
                            success=True,
                            documents=advanced_result.documents,
                            metadata=advanced_result.metadata,
                            processing_time=advanced_result.processing_time,
                            extracted_elements={
                                "sheets": len(advanced_result.documents),
                                "method": "advanced_excel",
                                "structural_elements": len(advanced_result.structural_elements),
                                "advanced_processing": True
                            }
                        )
                except Exception as e:
                    logger.warning(f"Advanced Excel processing failed, falling back: {e}")
            
            # Fallback to standard processing
            if file_path.lower().endswith('.csv'):
                loader = CSVLoader(file_path)
                documents = loader.load()
                extraction_method = "csv_loader"
            else:
                loader = UnstructuredExcelLoader(file_path)
                documents = loader.load()
                extraction_method = "unstructured_excel"
            
            metadata = self.get_file_info(file_path)
            metadata["extraction_method"] = extraction_method
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "document_type": "spreadsheet",
                    **metadata
                })
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={"sheets": len(documents)}
            )
            
        except Exception as e:
            logger.error(f"Failed to process Excel document {file_path}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                metadata=self.get_file_info(file_path),
                error_message=str(e),
                processing_time=time.time() - start_time
            )


class PowerPointProcessor(DocumentProcessor):
    """Processor for PowerPoint presentations (.pptx)"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        return (mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                or file_path.lower().endswith('.pptx'))
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            # Try advanced processing with speaker notes extraction
            if ADVANCED_EXTRACTION_AVAILABLE and kwargs.get("use_advanced", True):
                try:
                    logger.info(f"Using advanced PowerPoint processing: {file_path}")
                    advanced_result = advanced_extraction_service.process_document_advanced(
                        file_path, Path(file_path).suffix.lower()
                    )
                    
                    if advanced_result.success and advanced_result.documents:
                        return ProcessingResult(
                            success=True,
                            documents=advanced_result.documents,
                            metadata=advanced_result.metadata,
                            processing_time=advanced_result.processing_time,
                            extracted_elements={
                                "slides": len(advanced_result.documents),
                                "method": "advanced_powerpoint",
                                "structural_elements": len(advanced_result.structural_elements),
                                "has_visual_analysis": bool(advanced_result.visual_analysis),
                                "advanced_processing": True
                            }
                        )
                except Exception as e:
                    logger.warning(f"Advanced PowerPoint processing failed, falling back: {e}")
            
            # Fallback to standard processing
            loader = UnstructuredPowerPointLoader(file_path)
            documents = loader.load()
            
            metadata = self.get_file_info(file_path)
            metadata["extraction_method"] = "unstructured_powerpoint"
            
            # Add metadata to documents
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "slide_number": i + 1,
                    "source": file_path,
                    "document_type": "presentation",
                    **metadata
                })
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={"slides": len(documents)}
            )
            
        except Exception as e:
            logger.error(f"Failed to process PowerPoint document {file_path}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                metadata=self.get_file_info(file_path),
                error_message=str(e),
                processing_time=time.time() - start_time
            )


class ImageProcessor(DocumentProcessor):
    """Processor for image files with OCR"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        image_types = ["image/jpeg", "image/png", "image/tiff", "image/bmp", "image/gif"]
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']
        
        return mime_type in image_types or any(file_path.lower().endswith(ext) for ext in image_extensions)
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            # Try advanced processing with llava:7b vision analysis first
            if ADVANCED_EXTRACTION_AVAILABLE and kwargs.get("use_advanced", True):
                try:
                    logger.info(f"Using llava:7b vision analysis for image: {file_path}")
                    advanced_result = advanced_extraction_service.process_document_advanced(
                        file_path, Path(file_path).suffix.lower()
                    )
                    
                    if advanced_result.success and advanced_result.documents:
                        return ProcessingResult(
                            success=True,
                            documents=advanced_result.documents,
                            metadata=advanced_result.metadata,
                            processing_time=advanced_result.processing_time,
                            extracted_elements={
                                "method": "llava_vision_analysis",
                                "structural_elements": len(advanced_result.structural_elements),
                                "has_visual_analysis": bool(advanced_result.visual_analysis),
                                "vision_model": "llava:7b",
                                "advanced_processing": True
                            }
                        )
                except Exception as e:
                    logger.warning(f"Advanced image processing failed, falling back to OCR: {e}")
            
            # Fallback to OCR processing
            documents = []
            metadata = self.get_file_info(file_path)
            
            if TESSERACT_AVAILABLE:
                # Use OCR to extract text from image
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image, lang='eng+jpn')
                
                if text.strip():
                    doc = LangChainDocument(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "document_type": "image",
                            "extraction_method": "ocr",
                            **metadata
                        }
                    )
                    documents.append(doc)
                    metadata["extraction_method"] = "tesseract_ocr"
                else:
                    # If no text found, create a document with image description
                    doc = LangChainDocument(
                        page_content=f"Image file: {Path(file_path).name}. No text content detected.",
                        metadata={
                            "source": file_path,
                            "document_type": "image",
                            "extraction_method": "metadata_only",
                            **metadata
                        }
                    )
                    documents.append(doc)
                    metadata["extraction_method"] = "metadata_only"
            else:
                # Fallback to UnstructuredImageLoader
                try:
                    loader = UnstructuredImageLoader(file_path)
                    documents = loader.load()
                    metadata["extraction_method"] = "unstructured_image"
                    
                    for doc in documents:
                        doc.metadata.update({
                            "source": file_path,
                            "document_type": "image",
                            **metadata
                        })
                except Exception:
                    # Final fallback - just metadata
                    doc = LangChainDocument(
                        page_content=f"Image file: {Path(file_path).name}",
                        metadata={
                            "source": file_path,
                            "document_type": "image",
                            "extraction_method": "filename_only",
                            **metadata
                        }
                    )
                    documents.append(doc)
                    metadata["extraction_method"] = "filename_only"
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={"text_detected": len(documents[0].page_content) > 50 if documents else False}
            )
            
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                metadata=self.get_file_info(file_path),
                error_message=str(e),
                processing_time=time.time() - start_time
            )


class AudioProcessor(DocumentProcessor):
    """Processor for audio files with speech-to-text"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        audio_types = ["audio/mpeg", "audio/wav", "audio/mp4", "audio/m4a", "audio/flac"]
        audio_extensions = ['.mp3', '.wav', '.mp4', '.m4a', '.flac', '.ogg']
        
        return mime_type in audio_types or any(file_path.lower().endswith(ext) for ext in audio_extensions)
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            # Try advanced processing with enhanced Whisper transcription
            if ADVANCED_EXTRACTION_AVAILABLE and kwargs.get("use_advanced", True):
                try:
                    logger.info(f"Using advanced Whisper processing for audio: {file_path}")
                    advanced_result = advanced_extraction_service.process_document_advanced(
                        file_path, Path(file_path).suffix.lower()
                    )
                    
                    if advanced_result.success and advanced_result.documents:
                        return ProcessingResult(
                            success=True,
                            documents=advanced_result.documents,
                            metadata=advanced_result.metadata,
                            processing_time=advanced_result.processing_time,
                            extracted_elements={
                                "method": "whisper_enhanced",
                                "structural_elements": len(advanced_result.structural_elements),
                                "has_timestamps": True,
                                "transcription_model": "whisper",
                                "advanced_processing": True
                            }
                        )
                except Exception as e:
                    logger.warning(f"Advanced audio processing failed, falling back: {e}")
            
            # Fallback to standard processing
            documents = []
            metadata = self.get_file_info(file_path)
            
            # Check if transcription is enabled in config
            from config import config
            if config.app.enable_audio_transcription and WHISPER_AVAILABLE:
                try:
                    # Use Whisper for speech-to-text (external model, not our local Ollama models)
                    logger.info(f"Using OpenAI Whisper for audio transcription: {file_path}")
                    model = whisper.load_model("base")  # External Whisper model
                    result = model.transcribe(file_path)
                    
                    text = result["text"]
                    language = result.get("language", "unknown")
                    
                    doc = LangChainDocument(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "document_type": "audio",
                            "extraction_method": "whisper_external",
                            "detected_language": language,
                            "transcription_model": "whisper-base",
                            **metadata
                        }
                    )
                    documents.append(doc)
                    metadata["extraction_method"] = "whisper_external"
                    metadata["detected_language"] = language
                    metadata["transcription_model"] = "whisper-base"
                    
                except Exception as whisper_error:
                    logger.warning(f"Whisper transcription failed: {whisper_error}")
                    # Fall through to metadata-only processing
                    raise whisper_error
                    
            else:
                # Fallback - create metadata-only document
                logger.info(f"Audio transcription disabled or Whisper unavailable for: {file_path}")
                doc = LangChainDocument(
                    page_content=f"Audio file: {Path(file_path).name}. Contains audio content but transcription is not available. Enable audio transcription and install Whisper to extract speech content.",
                    metadata={
                        "source": file_path,
                        "document_type": "audio",
                        "extraction_method": "metadata_only",
                        "transcription_available": False,
                        "note": "Install OpenAI Whisper for speech-to-text capabilities",
                        **metadata
                    }
                )
                documents.append(doc)
                metadata["extraction_method"] = "metadata_only"
                metadata["transcription_available"] = False
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={
                    "transcription_available": WHISPER_AVAILABLE and config.app.enable_audio_transcription,
                    "uses_external_model": True,
                    "model_type": "whisper" if WHISPER_AVAILABLE else "none"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process audio {file_path}: {e}")
            # Create fallback document even on error
            fallback_doc = LangChainDocument(
                page_content=f"Audio file: {Path(file_path).name}. Processing failed: {str(e)}",
                metadata={
                    "source": file_path,
                    "document_type": "audio",
                    "extraction_method": "error_fallback",
                    "error": str(e),
                    **self.get_file_info(file_path)
                }
            )
            
            return ProcessingResult(
                success=True,  # Still successful as we have metadata
                documents=[fallback_doc],
                metadata=self.get_file_info(file_path),
                error_message=f"Transcription failed but file metadata preserved: {str(e)}",
                processing_time=time.time() - start_time
            )


class VideoProcessor(DocumentProcessor):
    """Processor for video files with audio extraction and transcription"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        video_types = ["video/mp4", "video/avi", "video/mov", "video/wmv", "video/flv"]
        video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']
        
        return mime_type in video_types or any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            documents = []
            metadata = self.get_file_info(file_path)
            
            # Check if video processing is enabled in config
            from config import config
            if config.app.enable_video_processing and WHISPER_AVAILABLE:
                try:
                    logger.info(f"Processing video with audio extraction: {file_path}")
                    
                    # Extract audio from video and transcribe
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                    
                    try:
                        # Use ffmpeg to extract audio (requires ffmpeg to be installed)
                        subprocess.run([
                            "ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", 
                            "-ar", "16000", "-ac", "1", temp_audio_path, "-y"
                        ], check=True, capture_output=True)
                        
                        # Transcribe the extracted audio using external Whisper model
                        model = whisper.load_model("base")  # External Whisper model
                        result = model.transcribe(temp_audio_path)
                        
                        text = result["text"]
                        language = result.get("language", "unknown")
                        
                        doc = LangChainDocument(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "document_type": "video",
                                "extraction_method": "whisper_from_video_external",
                                "detected_language": language,
                                "transcription_model": "whisper-base",
                                "audio_extracted": True,
                                **metadata
                            }
                        )
                        documents.append(doc)
                        metadata["extraction_method"] = "whisper_from_video_external"
                        metadata["detected_language"] = language
                        metadata["transcription_model"] = "whisper-base"
                        
                    except subprocess.CalledProcessError as ffmpeg_error:
                        logger.warning(f"Failed to extract audio from video {file_path}: {ffmpeg_error}")
                        logger.info("FFmpeg may not be installed or video format not supported")
                        raise ffmpeg_error
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                            
                except Exception as processing_error:
                    logger.warning(f"Video processing failed: {processing_error}")
                    # Fall through to metadata-only processing
                    raise processing_error
                    
            else:
                # Fallback - create metadata-only document
                logger.info(f"Video processing disabled or dependencies unavailable for: {file_path}")
                doc = LangChainDocument(
                    page_content=f"Video file: {Path(file_path).name}. Contains video content but audio transcription is not available. Enable video processing and install FFmpeg + Whisper to extract speech content from video audio track.",
                    metadata={
                        "source": file_path,
                        "document_type": "video",
                        "extraction_method": "metadata_only",
                        "transcription_available": False,
                        "note": "Install FFmpeg and OpenAI Whisper for video audio transcription",
                        **metadata
                    }
                )
                documents.append(doc)
                metadata["extraction_method"] = "metadata_only"
                metadata["transcription_available"] = False
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={
                    "transcription_available": WHISPER_AVAILABLE and config.app.enable_video_processing,
                    "uses_external_model": True,
                    "model_type": "whisper" if WHISPER_AVAILABLE else "none",
                    "requires_ffmpeg": True
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process video {file_path}: {e}")
            # Create fallback document even on error
            fallback_doc = LangChainDocument(
                page_content=f"Video file: {Path(file_path).name}. Processing failed: {str(e)}. Video contains visual and audio content but extraction failed.",
                metadata={
                    "source": file_path,
                    "document_type": "video",
                    "extraction_method": "error_fallback",
                    "error": str(e),
                    **self.get_file_info(file_path)
                }
            )
            
            return ProcessingResult(
                success=True,  # Still successful as we have metadata
                documents=[fallback_doc],
                metadata=self.get_file_info(file_path),
                error_message=f"Video processing failed but file metadata preserved: {str(e)}",
                processing_time=time.time() - start_time
            )


class TextProcessor(DocumentProcessor):
    """Processor for plain text files"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        text_types = ["text/plain", "text/markdown", "application/json", "application/xml"]
        text_extensions = ['.txt', '.md', '.json', '.xml', '.log', '.py', '.js', '.html', '.css']
        
        # Handle file extension check first
        file_ext = Path(file_path).suffix.lower()
        if file_ext in text_extensions:
            return True
        
        # Handle known text mime types
        if mime_type in text_types:
            return True
        
        # Handle octet-stream for common text extensions
        if mime_type in {None, "", "application/octet-stream"} and file_ext in text_extensions:
            return True
        
        return False
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            # Try UnstructuredFileLoader first
            try:
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()
                extraction_method = "unstructured_text"
            except Exception as e:
                logger.warning(f"UnstructuredFileLoader failed for {file_path}: {e}, trying direct read")
                # Fallback to direct file reading
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                documents = [LangChainDocument(
                    page_content=text,
                    metadata={"source": file_path}
                )]
                extraction_method = "direct_read"
            
            metadata = self.get_file_info(file_path)
            metadata["extraction_method"] = extraction_method
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "document_type": "text",
                    **metadata
                })
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=documents,
                metadata=metadata,
                processing_time=processing_time,
                extracted_elements={"character_count": len(documents[0].page_content) if documents else 0}
            )
            
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                metadata=self.get_file_info(file_path),
                error_message=str(e),
                processing_time=time.time() - start_time
            )


class DocumentProcessorFactory:
    """Factory class to get appropriate processor for a file"""
    
    def __init__(self):
        self.processors = [
            PDFProcessor(),
            WordProcessor(),
            ExcelProcessor(),
            PowerPointProcessor(),
            ImageProcessor(),
            AudioProcessor(),
            VideoProcessor(),
            TextProcessor()  # Keep as last fallback
        ]
    
    def get_processor(self, file_path: str) -> Optional[DocumentProcessor]:
        """Get the appropriate processor for a file"""
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        for processor in self.processors:
            if processor.can_process(file_path, mime_type):
                logger.info(f"Selected {processor.__class__.__name__} for {file_path}")
                return processor
        
        logger.warning(f"No suitable processor found for {file_path} (mime_type: {mime_type})")
        return None
    
    def get_supported_types(self) -> Dict[str, List[str]]:
        """Get all supported file types by category"""
        supported = {
            "PDF": [".pdf"],
            "Word Documents": [".docx"],
            "Spreadsheets": [".xlsx", ".xls", ".csv"],
            "Presentations": [".pptx"],
            "Images": [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"],
            "Audio": [".mp3", ".wav", ".mp4", ".m4a", ".flac", ".ogg"],
            "Video": [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"],
            "Text": [".txt", ".md", ".json", ".xml", ".log", ".py", ".js", ".html", ".css"]
        }
        return supported


# Global processor factory instance
processor_factory = DocumentProcessorFactory()