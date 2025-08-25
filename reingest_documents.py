#!/usr/bin/env python3
"""
Script to re-ingest documents from the documents directory
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.document_service import DocumentService
from config import config
import glob
from pathlib import Path
import io

class MockUploadedFile:
    """Mock uploaded file to simulate Streamlit uploaded file"""
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.name = self.file_path.name
        with open(file_path, 'rb') as f:
            self._content = f.read()
    
    def getvalue(self):
        return self._content

def main():
    """Re-ingest all documents from the documents directory"""
    print("🔄 Starting document re-ingestion...")
    
    # Initialize document service
    doc_service = DocumentService()
    
    # Get all files from documents directory
    documents_dir = "documents"
    file_patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.txt", "*.docx", "*.xlsx", "*.pptx"]
    
    files_to_process = []
    for pattern in file_patterns:
        files_to_process.extend(glob.glob(os.path.join(documents_dir, pattern)))
    
    print(f"📁 Found {len(files_to_process)} files to process:")
    for file_path in files_to_process:
        print(f"  - {file_path}")
    
    # Process each file
    for file_path in files_to_process:
        try:
            print(f"\n📄 Processing: {file_path}")
            
            # Create mock uploaded file
            mock_file = MockUploadedFile(file_path)
            
            # Save uploaded file (creates document record)
            document = doc_service.save_uploaded_file(
                mock_file,
                user_id="system_reingest"
            )
            
            if document:
                print(f"✅ File saved: {document.original_filename}")
                print(f"   Document ID: {document.id}")
                
                # Process the document
                print(f"🔄 Processing document...")
                success = doc_service.process_document(document.id)
                
                if success:
                    print(f"✅ Successfully processed: {document.original_filename}")
                else:
                    print(f"❌ Failed to process: {document.original_filename}")
            else:
                print(f"❌ Failed to save file: {file_path}")
                    
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
    
    print("\n🎉 Document re-ingestion completed!")
    print("💬 You can now chat with your real documents!")

if __name__ == "__main__":
    main()