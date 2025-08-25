"""
Document processing service for Kurachi AI
Handles document ingestion, processing, and management
"""
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import mimetypes

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from config import config
from models.database import Document, db_manager
from services.document_ingestion import enhanced_ingestion_service
from services.intelligent_chunking import intelligent_chunking_service
from utils.logger import get_logger, audit_logger

logger = get_logger("document_service")


class DocumentService:
    """Service for handling document operations"""
    
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
        # Use intelligent chunking service
        self.intelligent_chunker = intelligent_chunking_service
        self.vector_store = None
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
                # Will be created when first document is processed
                logger.info("Vector store will be created on first document upload")
        except Exception as e:
            logger.error("Failed to initialize vector store", error=e)
            raise
    
    def is_supported_file_type(self, filename: str) -> bool:
        """Check if file type is supported"""
        return enhanced_ingestion_service.is_supported_file_type(filename)
    
    def get_file_info(self, file_path: str) -> Tuple[str, int]:
        """Get file type and size"""
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return mime_type, file_size
    
    def save_uploaded_file(self, uploaded_file, user_id: Optional[str] = None) -> Optional[Document]:
        """Save uploaded file and create document record with enhanced validation"""
        return enhanced_ingestion_service.save_uploaded_file(uploaded_file, user_id)
    
    def process_document(self, document_id: str) -> bool:
        """Process document for vector storage with enhanced processing"""
        return enhanced_ingestion_service.process_document_with_progress(document_id)
    
    def process_document_async(self, document_id: str, progress_callback=None) -> None:
        """Process document asynchronously with progress tracking"""
        enhanced_ingestion_service.process_document_async(document_id, progress_callback)
    
    def get_processing_progress(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing progress for a document"""
        return enhanced_ingestion_service.get_processing_progress(document_id)
    
    def get_supported_file_types(self) -> Dict[str, List[str]]:
        """Get all supported file types by category"""
        return enhanced_ingestion_service.get_supported_file_types()
    
    def detect_file_type(self, file_path: str) -> Dict[str, Any]:
        """Detect file type and get detailed information"""
        return enhanced_ingestion_service.detect_file_type(file_path)
    
    def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Delete document and its associated data"""
        try:
            # Get document from database
            document = db_manager.get_document(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                return False
            
            # Check user permissions (if user_id provided)
            if user_id and document.user_id != user_id:
                logger.warning(f"User {user_id} attempted to delete document {document_id} owned by {document.user_id}")
                return False
            
            # Delete physical file
            file_path = Path(document.file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
            
            # TODO: Remove from vector store (ChromaDB doesn't have easy document removal by metadata)
            # This would require rebuilding the vector store or implementing a custom solution
            
            # Delete from database (implement this in database manager)
            # For now, we'll mark as deleted
            db_manager.update_document_status(document_id, "deleted")
            
            audit_logger.log_user_action(
                user_id or "system",
                "delete_document",
                document_id,
                success=True,
                details={"filename": document.original_filename}
            )
            
            logger.info(f"Document deleted successfully", extra={
                "document_id": document_id,
                "filename": document.original_filename
            })
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete document", error=e, extra={"document_id": document_id})
            return False
    
    def get_user_documents(self, user_id: str) -> List[Document]:
        """Get all documents for a user"""
        try:
            documents = db_manager.get_user_documents(user_id)
            # Filter out deleted documents
            return [doc for doc in documents if doc.processing_status != "deleted"]
        except Exception as e:
            logger.error("Failed to get user documents", error=e, extra={"user_id": user_id})
            return []
    
    def get_document_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get document statistics"""
        try:
            if user_id:
                documents = self.get_user_documents(user_id)
            else:
                # Get all documents (admin view)
                documents = []  # Implement get_all_documents if needed
            
            stats = {
                "total_documents": len(documents),
                "by_status": {},
                "by_file_type": {},
                "total_size": 0
            }
            
            for doc in documents:
                # Count by status
                status = doc.processing_status
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                
                # Count by file type
                file_type = doc.file_type
                stats["by_file_type"][file_type] = stats["by_file_type"].get(file_type, 0) + 1
                
                # Sum file sizes
                stats["total_size"] += doc.file_size
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get document stats", error=e, extra={"user_id": user_id})
            return {"total_documents": 0, "by_status": {}, "by_file_type": {}, "total_size": 0}
    
    def search_documents(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=limit)
            
            # Filter by user if specified
            filtered_results = []
            for doc, score in results:
                if user_id and doc.metadata.get("user_id") != user_id:
                    continue
                
                filtered_results.append({
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata,
                    "document_id": doc.metadata.get("document_id"),
                    "filename": doc.metadata.get("filename")
                })
            
            return filtered_results
            
        except Exception as e:
            logger.error("Failed to search documents", error=e, extra={"query": query, "user_id": user_id})
            return []


# Global document service instance
document_service = DocumentService()