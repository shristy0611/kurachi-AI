
def process_document(file_path, user_id=None):
    """
    Process a document using the enhanced ingestion pipeline
    """
    try:
        # Validate file
        validation = service.validate_file(file_path)
        if not validation["valid"]:
            return False, validation["errors"]
        
        # Process with progress tracking
        result = service.process_document_with_progress(document_id)
        return result, None
        
    except Exception as e:
        return False, str(e)

class DocumentProcessor:
    """Enhanced document processor with multi-format support"""
    
    def __init__(self):
        self.supported_types = [
            ".pdf", ".docx", ".xlsx", ".pptx", 
            ".jpg", ".png", ".mp3", ".mp4", ".txt"
        ]
    
    def can_process(self, file_path):
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_types
