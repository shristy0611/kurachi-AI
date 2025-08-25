"""
Enhanced document ingestion service with comprehensive processing pipeline
Handles file type detection, routing, progress tracking, and status updates
"""
import os
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import mimetypes
import threading
import concurrent.futures
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import OllamaEmbeddings

from config import config
from models.database import Document, db_manager
from services.document_processors import processor_factory, ProcessingResult
from services.intelligent_chunking import intelligent_chunking_service
from services.language_detection import language_detection_service, DocumentLanguage
from services.language_processing_pipelines import language_pipeline_factory
from services.analytics_service import analytics_service
from services.knowledge_graph import knowledge_graph
from utils.logger import get_logger, audit_logger

logger = get_logger("document_ingestion")


class ProcessingStatus(Enum):
    """Document processing status enumeration"""
    PENDING = "pending"
    VALIDATING = "validating"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingProgress:
    """Progress tracking for document processing"""
    document_id: str
    status: ProcessingStatus
    progress_percentage: float
    current_step: str
    total_steps: int
    current_step_number: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['start_time'] = self.start_time.isoformat()
        if self.estimated_completion:
            data['estimated_completion'] = self.estimated_completion.isoformat()
        return data


class ProgressTracker:
    """Thread-safe progress tracking for document processing"""
    
    def __init__(self):
        self._progress: Dict[str, ProcessingProgress] = {}
        self._lock = threading.Lock()
        self._callbacks: Dict[str, List[Callable]] = {}
    
    def start_tracking(self, document_id: str, total_steps: int = 5) -> None:
        """Start tracking progress for a document"""
        with self._lock:
            self._progress[document_id] = ProcessingProgress(
                document_id=document_id,
                status=ProcessingStatus.PENDING,
                progress_percentage=0.0,
                current_step="Initializing",
                total_steps=total_steps,
                current_step_number=0,
                start_time=datetime.utcnow()
            )
    
    def update_progress(self, document_id: str, status: ProcessingStatus, 
                       step_name: str, step_number: int, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update progress for a document"""
        with self._lock:
            if document_id not in self._progress:
                return
            
            progress = self._progress[document_id]
            progress.status = status
            progress.current_step = step_name
            progress.current_step_number = step_number
            progress.progress_percentage = (step_number / progress.total_steps) * 100
            progress.metadata = metadata or {}
            
            # Estimate completion time based on current progress
            if step_number > 0:
                elapsed = datetime.utcnow() - progress.start_time
                estimated_total = elapsed * (progress.total_steps / step_number)
                progress.estimated_completion = progress.start_time + estimated_total
        
        # Notify callbacks
        self._notify_callbacks(document_id)
    
    def set_error(self, document_id: str, error_message: str) -> None:
        """Set error status for a document"""
        with self._lock:
            if document_id not in self._progress:
                return
            
            progress = self._progress[document_id]
            progress.status = ProcessingStatus.FAILED
            progress.error_message = error_message
            progress.progress_percentage = 0.0
        
        self._notify_callbacks(document_id)
    
    def complete_processing(self, document_id: str) -> None:
        """Mark processing as completed"""
        with self._lock:
            if document_id not in self._progress:
                return
            
            progress = self._progress[document_id]
            progress.status = ProcessingStatus.COMPLETED
            progress.progress_percentage = 100.0
            progress.current_step = "Completed"
            progress.current_step_number = progress.total_steps
        
        self._notify_callbacks(document_id)
    
    def get_progress(self, document_id: str) -> Optional[ProcessingProgress]:
        """Get current progress for a document"""
        with self._lock:
            return self._progress.get(document_id)
    
    def remove_progress(self, document_id: str) -> None:
        """Remove progress tracking for a document"""
        with self._lock:
            self._progress.pop(document_id, None)
            self._callbacks.pop(document_id, None)
    
    def add_callback(self, document_id: str, callback: Callable[[ProcessingProgress], None]) -> None:
        """Add a callback for progress updates"""
        with self._lock:
            if document_id not in self._callbacks:
                self._callbacks[document_id] = []
            self._callbacks[document_id].append(callback)
    
    def _notify_callbacks(self, document_id: str) -> None:
        """Notify all callbacks for a document"""
        callbacks = self._callbacks.get(document_id, [])
        progress = self._progress.get(document_id)
        
        if progress:
            for callback in callbacks:
                try:
                    callback(progress)
                except Exception as e:
                    logger.error(f"Callback error for document {document_id}: {e}")


class EnhancedDocumentIngestionService:
    """Enhanced document ingestion service with comprehensive processing"""
    
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            model=config.ai.embedding_model,
            base_url=config.ai.ollama_base_url
        )
        # Keep fallback text splitter for compatibility
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.ai.chunk_size,
            chunk_overlap=config.ai.chunk_overlap
        )
        # Use intelligent chunking service as primary method
        self.intelligent_chunker = intelligent_chunking_service
        self.vector_store = None
        self.progress_tracker = ProgressTracker()
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize or load existing vector store"""
        try:
            if Path(config.database.chroma_persist_dir).exists():
                self.vector_store = Chroma(
                    persist_directory=config.database.chroma_persist_dir,
                    embedding_function=self.embedding_model
                )
                logger.info("Loaded existing vector store")
            else:
                logger.info("Vector store will be created on first document upload")
        except Exception as e:
            logger.error("Failed to initialize vector store", error=e)
            raise
    
    def get_supported_file_types(self) -> Dict[str, List[str]]:
        """Get all supported file types"""
        return processor_factory.get_supported_types()
    
    def is_supported_file_type(self, filename: str) -> bool:
        """Check if file type is supported"""
        processor = processor_factory.get_processor(filename)
        return processor is not None
    
    def detect_file_type(self, file_path: str) -> Dict[str, Any]:
        """Detect file type and get detailed information"""
        file_path = Path(file_path)
        mime_type, encoding = mimetypes.guess_type(str(file_path))
        
        processor = processor_factory.get_processor(str(file_path))
        
        return {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "mime_type": mime_type or "application/octet-stream",
            "encoding": encoding,
            "size": file_path.stat().st_size if file_path.exists() else 0,
            "supported": processor is not None,
            "processor_type": processor.__class__.__name__ if processor else None
        }
    
    def validate_file(self, file_path: str, max_size_mb: Optional[int] = None) -> Dict[str, Any]:
        """Validate file before processing"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                validation_result["valid"] = False
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Get file info
            file_info = self.detect_file_type(str(file_path))
            validation_result["file_info"] = file_info
            
            # Check if file type is supported
            if not file_info["supported"]:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Unsupported file type: {file_info['extension']}")
            
            # Check file size
            max_size = max_size_mb or config.app.max_file_size_mb
            if file_info["size"] > max_size * 1024 * 1024:
                validation_result["valid"] = False
                validation_result["errors"].append(f"File too large: {file_info['size']} bytes (max: {max_size}MB)")
            
            # Check for empty files
            if file_info["size"] == 0:
                validation_result["valid"] = False
                validation_result["errors"].append("File is empty")
            
            # Add warnings for large files
            if file_info["size"] > 50 * 1024 * 1024:  # 50MB
                validation_result["warnings"].append("Large file may take longer to process")
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def save_uploaded_file(self, uploaded_file, user_id: Optional[str] = None) -> Optional[Document]:
        """Save uploaded file and create document record with validation"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_ext = Path(uploaded_file.name).suffix
            filename = f"{file_id}{file_ext}"
            file_path = Path(config.app.upload_dir) / filename
            
            # Save file
            with open(file_path, "wb") as f:
                if hasattr(uploaded_file, 'read'):
                    f.write(uploaded_file.read())
                else:
                    # Handle Streamlit uploaded file
                    f.write(uploaded_file.getvalue())
            
            # Validate the saved file
            validation = self.validate_file(str(file_path))
            if not validation["valid"]:
                # Clean up file and return None
                file_path.unlink(missing_ok=True)
                logger.error(f"File validation failed: {validation['errors']}")
                return None
            
            file_info = validation["file_info"]
            
            # Create document record
            document = Document(
                id=file_id,
                filename=filename,
                original_filename=uploaded_file.name,
                file_path=str(file_path),
                file_type=file_info["extension"],
                file_size=file_info["size"],
                content_type=file_info["mime_type"],
                processing_status=ProcessingStatus.PENDING.value,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                user_id=user_id,
                metadata={
                    "processor_type": file_info["processor_type"],
                    "validation_warnings": validation["warnings"]
                }
            )
            
            # Save to database
            if db_manager.create_document(document):
                # Track document upload
                analytics_service.track_usage(
                    user_id=user_id or "anonymous",
                    action_type="document_upload",
                    metadata={
                        "file_type": document.file_type,
                        "file_size": document.file_size,
                        "processor_type": file_info["processor_type"]
                    }
                )
                
                logger.info(f"Document saved successfully", extra={
                    "document_id": document.id,
                    "filename": document.original_filename,
                    "user_id": user_id,
                    "file_size": document.file_size
                })
                return document
            else:
                # Clean up file if database save failed
                file_path.unlink(missing_ok=True)
                return None
                
        except Exception as e:
            logger.error("Failed to save uploaded file", error=e, extra={
                "filename": uploaded_file.name,
                "user_id": user_id
            })
            return None
    
    def process_document_async(self, document_id: str, 
                              progress_callback: Optional[Callable[[ProcessingProgress], None]] = None) -> None:
        """Process document asynchronously with progress tracking"""
        def process_in_thread():
            try:
                self.process_document_with_progress(document_id, progress_callback)
            except Exception as e:
                logger.error(f"Async processing failed for document {document_id}: {e}")
                self.progress_tracker.set_error(document_id, str(e))
        
        thread = threading.Thread(target=process_in_thread, daemon=True)
        thread.start()
    
    def process_document_with_progress(self, document_id: str,
                                     progress_callback: Optional[Callable[[ProcessingProgress], None]] = None) -> bool:
        """Process document with detailed progress tracking and performance logging."""
        total_start_time = time.time()
        timings = {}
        logger.info(f"Starting processing for document {document_id}")
        try:
            # Get document from database
            document = db_manager.get_document(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                return False
            
            # Start progress tracking
            self.progress_tracker.start_tracking(document_id, total_steps=7)
            if progress_callback:
                self.progress_tracker.add_callback(document_id, progress_callback)
            
            # Step 1: Validation
            self.progress_tracker.update_progress(
                document_id, ProcessingStatus.VALIDATING, "Validating file", 1
            )
            db_manager.update_document_status(document_id, ProcessingStatus.VALIDATING.value)
            # Start timing for Validation step
            step_start_time = time.time()
            
            validation = self.validate_file(document.file_path)
            if not validation["valid"]:
                error_msg = "; ".join(validation["errors"])
                self.progress_tracker.set_error(document_id, error_msg)
                db_manager.update_document_status(document_id, ProcessingStatus.FAILED.value, error_msg)
                return False
            timings["validation"] = time.time() - step_start_time
            logger.info(f"Step 'Validation' for doc {document_id} took {timings['validation']:.2f}s")
            
            # Step 2: Content extraction
            self.progress_tracker.update_progress(
                document_id, ProcessingStatus.PROCESSING, "Extracting content", 2
            )
            db_manager.update_document_status(document_id, ProcessingStatus.PROCESSING.value)
            # Start timing for Content Extraction step
            step_start_time = time.time()
            
            processor = processor_factory.get_processor(document.file_path)
            if not processor:
                error_msg = f"No processor available for file type: {document.file_type}"
                self.progress_tracker.set_error(document_id, error_msg)
                db_manager.update_document_status(document_id, ProcessingStatus.FAILED.value, error_msg)
                return False
            
            processing_result = processor.process(document.file_path)
            if not processing_result.success:
                self.progress_tracker.set_error(document_id, processing_result.error_message)
                db_manager.update_document_status(document_id, ProcessingStatus.FAILED.value, processing_result.error_message)
                return False
            
            timings["content_extraction"] = time.time() - step_start_time
            logger.info(f"Step 'Content Extraction' for doc {document_id} took {timings['content_extraction']:.2f}s")
            
            # Add document metadata to extracted documents
            for doc in processing_result.documents:
                doc.metadata.update({
                    "document_id": document.id,
                    "filename": document.original_filename,
                    "file_type": document.file_type,
                    "upload_date": document.created_at.isoformat(),
                    "user_id": document.user_id,
                    "processing_time": processing_result.processing_time,
                    "extraction_method": processing_result.metadata.get("extraction_method")
                })
            
            # Step 2.5: Advanced language detection and processing
            self.progress_tracker.update_progress(
                document_id, ProcessingStatus.PROCESSING, "Detecting language and applying language-specific processing", 3,
                {"extracted_documents": len(processing_result.documents)}
            )
            # Start timing for Language Processing step
            step_start_time = time.time()
            
            # Apply language detection and processing pipelines
            if processing_result.documents:
                # Use advanced language detection service
                language_processed_result = language_pipeline_factory.process_documents_with_language_awareness(
                    processing_result.documents
                )
                
                if language_processed_result.success:
                    # Update documents with language-processed versions
                    processing_result.documents = language_processed_result.documents
                    
                    # Extract language information
                    language_detection_info = language_processed_result.processing_metadata.get('language_detection', {})
                    primary_language = language_detection_info.get('primary_language', 'unknown')
                    is_mixed_language = language_detection_info.get('is_mixed_language', False)
                    language_confidence = language_detection_info.get('confidence', 0.0)
                    
                    # Add comprehensive language metadata to all documents
                    for doc in processing_result.documents:
                        doc.metadata.update({
                            "detected_language": primary_language,
                            "language_confidence": language_confidence,
                            "is_mixed_language": is_mixed_language,
                            "language_distribution": language_detection_info.get('language_distribution', {}),
                            "supports_translation": primary_language in ['ja', 'en'],
                            "original_language": primary_language,
                            "language_processing_applied": True,
                            "processing_pipeline": language_processed_result.processing_metadata.get('pipeline', 'unknown')
                        })
                        
                        # Add language-specific features if available
                        if language_processed_result.language_specific_features:
                            doc.metadata["language_specific_features"] = language_processed_result.language_specific_features
                    
                    logger.info(f"Advanced language processing completed", extra={
                        "document_id": document_id,
                        "primary_language": primary_language,
                        "confidence": language_confidence,
                        "is_mixed_language": is_mixed_language,
                        "pipeline_used": language_processed_result.processing_metadata.get('pipeline', 'unknown')
                    })
                else:
                    # Fallback to basic language detection
                    logger.warning(f"Advanced language processing failed, using basic detection: {language_processed_result.error_message}")
                    primary_content = processing_result.documents[0].page_content[:1000]
                    detected_lang_code = language_detection_service.detect_language(primary_content)
                    
                    for doc in processing_result.documents:
                        doc.metadata.update({
                            "detected_language": detected_lang_code,
                            "language_confidence": 0.5,  # Lower confidence for fallback
                            "supports_translation": True,
                            "original_language": detected_lang_code,
                            "language_processing_applied": False,
                            "processing_pipeline": "fallback"
                        })
            else:
                logger.warning(f"No content found for language detection in document {document_id}")
            
            timings["language_processing"] = time.time() - step_start_time
            logger.info(f"Step 'Language Processing' for doc {document_id} took {timings['language_processing']:.2f}s")
            
            # Step 3: Intelligent chunking
            self.progress_tracker.update_progress(
                document_id, ProcessingStatus.CHUNKING, "Creating intelligent chunks", 4,
                {"extracted_documents": len(processing_result.documents)}
            )
            # Start timing for Chunking step
            step_start_time = time.time()
            
            # Use intelligent chunking service
            try:
                split_docs = self.intelligent_chunker.chunk_documents(processing_result.documents)
                chunking_method = "intelligent"
                logger.info(f"Used intelligent chunking for document {document_id}")
            except Exception as e:
                logger.warning(f"Intelligent chunking failed for document {document_id}, using fallback: {e}")
                split_docs = self.text_splitter.split_documents(processing_result.documents)
                chunking_method = "fallback"
            
            timings["chunking"] = time.time() - step_start_time
            logger.info(f"Step 'Chunking' for doc {document_id} took {timings['chunking']:.2f}s")
            
            # Step 4: Vector embedding
            self.progress_tracker.update_progress(
                document_id, ProcessingStatus.EMBEDDING, "Creating embeddings", 5,
                {"chunks_created": len(split_docs)}
            )
            # Start timing for Embedding & Persistence step
            step_start_time = time.time()

            # Filter complex metadata that Chroma cannot handle
            filtered_docs = filter_complex_metadata(split_docs)

            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=filtered_docs,
                    embedding=self.embedding_model,
                    persist_directory=config.database.chroma_persist_dir
                )
                logger.info("Created new vector store")
                self._persist_vector_store_with_timeout()
            else:
                self.vector_store.add_documents(filtered_docs)
                logger.info(f"Added {len(filtered_docs)} chunks to vector store")
                self._persist_vector_store_with_timeout()

            timings["embedding_and_persistence"] = time.time() - step_start_time
            logger.info(f"Step 'Embedding & Persistence' for doc {document_id} took {timings['embedding_and_persistence']:.2f}s")

            # Get chunking statistics
            chunk_stats = self.intelligent_chunker.get_chunk_statistics(split_docs)

            # Step 5: Knowledge graph construction
            self.progress_tracker.update_progress(
                document_id, ProcessingStatus.PROCESSING, "Building knowledge graph", 6,
                {"chunks_created": len(split_docs)}
            )
            # Start timing for Knowledge Graph step
            step_start_time = time.time()

            # Build knowledge graph from the document
            kg_result = knowledge_graph.process_document(document_id)
            if not kg_result.get("success"):
                logger.warning(f"Knowledge graph construction failed for document {document_id}: {kg_result.get('error')}")
            else:
                logger.info(f"Knowledge graph updated: {kg_result.get('entities_extracted', 0)} entities, {kg_result.get('relationships_extracted', 0)} relationships")

            timings["knowledge_graph"] = time.time() - step_start_time
            logger.info(f"Step 'Knowledge Graph' for doc {document_id} took {timings['knowledge_graph']:.2f}s")
            
            # Step 6: Completion
            self.progress_tracker.update_progress(
                document_id, ProcessingStatus.COMPLETED, "Processing completed", 7,
                {
                    "total_chunks": len(split_docs),
                    "processing_time": processing_result.processing_time,
                    "extracted_elements": processing_result.extracted_elements,
                    "chunking_method": chunking_method,
                    "chunk_statistics": chunk_stats,
                    "knowledge_graph": kg_result if kg_result.get("success") else None
                }
            )
            
            # Update database status
            db_manager.update_document_status(document_id, ProcessingStatus.COMPLETED.value)
            
            # Track document processing completion
            analytics_service.track_usage(
                user_id=document.user_id or "anonymous",
                action_type="document_processing",
                duration_ms=int(processing_result.processing_time * 1000) if processing_result.processing_time else None,
                metadata={
                    "chunks_created": len(split_docs),
                    "chunking_method": chunking_method,
                    "processor_type": processor.__class__.__name__,
                    "detected_language": processing_result.documents[0].metadata.get("detected_language") if processing_result.documents else None,
                    "file_type": document.file_type
                }
            )
            
            # Log audit event
            audit_logger.log_user_action(
                document.user_id or "system",
                "process_document",
                document_id,
                success=True,
                details={
                    "filename": document.original_filename,
                    "chunks_created": len(split_docs),
                    "processing_time": processing_result.processing_time,
                    "processor_type": processor.__class__.__name__,
                    "chunking_method": chunking_method,
                    "chunk_statistics": chunk_stats
                }
            )
            
            total_duration = time.time() - total_start_time
            timings["total_duration"] = total_duration

            logger.info(f"Document processed successfully", extra={
                "document_id": document_id,
                "chunks_created": len(split_docs),
                "processing_time": total_duration,
                "chunking_method": chunking_method,
                "chunk_types": chunk_stats.get("chunk_types", {}),
                "timings": timings
            })
            
            # Complete progress tracking
            self.progress_tracker.complete_processing(document_id)
            
            return True
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.progress_tracker.set_error(document_id, error_msg)
            db_manager.update_document_status(document_id, ProcessingStatus.FAILED.value, error_msg)
            logger.error("Failed to process document", error=e, extra={"document_id": document_id})
            return False
    
    def _persist_vector_store_with_timeout(self, timeout: int = 60):
        """Persists the vector store with a timeout using a thread pool."""
        if self.vector_store is None:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.vector_store.persist)
            try:
                future.result(timeout=timeout)
                logger.info("Vector store persisted successfully.")
            except concurrent.futures.TimeoutError:
                logger.error(f"Vector store persist timed out after {timeout} seconds. The operation will continue in the background.")
            except Exception as e:
                logger.error(f"An error occurred during vector store persistence: {e}")

    def get_processing_progress(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing progress for a document"""
        progress = self.progress_tracker.get_progress(document_id)
        return progress.to_dict() if progress else None
    
    def cancel_processing(self, document_id: str) -> bool:
        """Cancel document processing (if possible)"""
        try:
            progress = self.progress_tracker.get_progress(document_id)
            if progress and progress.status not in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                self.progress_tracker.update_progress(
                    document_id, ProcessingStatus.CANCELLED, "Processing cancelled", 0
                )
                db_manager.update_document_status(document_id, ProcessingStatus.CANCELLED.value)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel processing for document {document_id}: {e}")
            return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get overall processing statistics"""
        try:
            # This would typically query the database for statistics
            # For now, return basic stats
            return {
                "total_documents_processed": 0,  # Implement database query
                "average_processing_time": 0.0,
                "success_rate": 0.0,
                "supported_file_types": len(self.get_supported_file_types()),
                "active_processing_jobs": len([p for p in self.progress_tracker._progress.values() 
                                             if p.status == ProcessingStatus.PROCESSING])
            }
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {}


# Global enhanced ingestion service instance
enhanced_ingestion_service = EnhancedDocumentIngestionService()