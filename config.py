"""
Configuration management for Kurachi AI
Handles environment-specific settings and deployment scenarios
"""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    sqlite_path: str = "kurachi.db"
    chroma_persist_dir: str = "./chroma_db"
    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "kurachi"
    
    
@dataclass
class AIConfig:
    """AI model configuration settings"""
    # Local Ollama models
    llm_model: str = "qwen3:4b"  # Local LLM for text generation
    vision_model: str = "llava:7b"  # Local vision model for image understanding
    embedding_model: str = "nomic-embed-text"  # Local model for text embeddings (Ollama embedding model)
    ollama_base_url: str = "http://localhost:11434"
    
    # External models (not running on Ollama)
    whisper_model: str = "base"  # OpenAI Whisper for speech-to-text (external)
    ocr_language: str = "eng+jpn"  # Tesseract OCR languages
    
    # Processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class AppConfig:
    """Application configuration settings"""
    app_name: str = "Kurachi AI"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 100
    supported_file_types: list = None
    enable_ocr: bool = True
    enable_audio_transcription: bool = True
    enable_video_processing: bool = True
    
    # Optional features (can be disabled via environment variables)
    enable_neo4j: bool = False
    enable_spacy: bool = True
    enable_performance_monitoring: bool = True
    validation_timeout: int = 120
    
    def __post_init__(self):
        if self.supported_file_types is None:
            self.supported_file_types = [
                # Documents
                ".pdf", ".docx", ".txt", ".md", ".rtf",
                # Spreadsheets
                ".csv", ".xlsx", ".xls",
                # Presentations
                ".pptx", ".ppt",
                # Images
                ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif",
                # Audio
                ".mp3", ".wav", ".m4a", ".flac", ".ogg",
                # Video
                ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv",
                # Code and markup
                ".py", ".js", ".html", ".css", ".json", ".xml", ".yaml", ".yml",
                # Other text formats
                ".log", ".ini", ".cfg", ".conf"
            ]


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = "kurachi-ai-secret-key-change-in-production"
    session_timeout_hours: int = 24
    max_login_attempts: int = 5
    enable_audit_logging: bool = True


class Config:
    """Main configuration class that loads settings from environment variables"""
    
    def __init__(self):
        self.environment = os.getenv("KURACHI_ENV", "development")
        self.database = self._load_database_config()
        self.ai = self._load_ai_config()
        self.app = self._load_app_config()
        self.security = self._load_security_config()
        
        # Ensure required directories exist
        self._create_directories()
    
    def _load_database_config(self) -> DatabaseConfig:
        return DatabaseConfig(
            sqlite_path=os.getenv("KURACHI_DB_PATH", "data/kurachi.db"),
            chroma_persist_dir=os.getenv("KURACHI_CHROMA_DIR", "./chroma_db"),
            # Neo4j configuration
            neo4j_uri=os.getenv("KURACHI_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("KURACHI_NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("KURACHI_NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("KURACHI_NEO4J_DATABASE", "kurachi")
        )
    
    def _load_ai_config(self) -> AIConfig:
        return AIConfig(
            # Local Ollama models
            llm_model=os.getenv("KURACHI_LLM_MODEL", "qwen3:4b"),
            vision_model=os.getenv("KURACHI_VISION_MODEL", "llava:7b"),
            embedding_model=os.getenv("KURACHI_EMBEDDING_MODEL", "nomic-embed-text"),
            ollama_base_url=os.getenv("KURACHI_OLLAMA_URL", "http://localhost:11434"),
            
            # External models
            whisper_model=os.getenv("KURACHI_WHISPER_MODEL", "base"),
            ocr_language=os.getenv("KURACHI_OCR_LANGUAGE", "eng+jpn"),
            
            # Processing settings
            chunk_size=int(os.getenv("KURACHI_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("KURACHI_CHUNK_OVERLAP", "200")),
            max_tokens=int(os.getenv("KURACHI_MAX_TOKENS", "2048")),
            temperature=float(os.getenv("KURACHI_TEMPERATURE", "0.7"))
        )
    
    def _load_app_config(self) -> AppConfig:
        return AppConfig(
            app_name=os.getenv("KURACHI_APP_NAME", "Kurachi AI"),
            version=os.getenv("KURACHI_VERSION", "1.0.0"),
            debug=os.getenv("KURACHI_DEBUG", "false").lower() == "true",
            log_level=os.getenv("KURACHI_LOG_LEVEL", "INFO"),
            upload_dir=os.getenv("KURACHI_UPLOAD_DIR", "./uploads"),
            max_file_size_mb=int(os.getenv("KURACHI_MAX_FILE_SIZE_MB", "100")),
            enable_ocr=os.getenv("ENABLE_OCR", "true").lower() == "true",
            enable_audio_transcription=os.getenv("ENABLE_AUDIO_TRANSCRIPTION", "true").lower() == "true",
            enable_video_processing=os.getenv("ENABLE_VIDEO_PROCESSING", "true").lower() == "true",
            enable_neo4j=os.getenv("ENABLE_NEO4J", "false").lower() == "true",
            enable_spacy=os.getenv("ENABLE_SPACY", "true").lower() == "true",
            enable_performance_monitoring=os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true",
            validation_timeout=int(os.getenv("VALIDATION_TIMEOUT", "120"))
        )
    
    def _load_security_config(self) -> SecurityConfig:
        return SecurityConfig(
            secret_key=os.getenv("KURACHI_SECRET_KEY", "kurachi-ai-secret-key-change-in-production"),
            session_timeout_hours=int(os.getenv("KURACHI_SESSION_TIMEOUT_HOURS", "24")),
            max_login_attempts=int(os.getenv("KURACHI_MAX_LOGIN_ATTEMPTS", "5")),
            enable_audit_logging=os.getenv("KURACHI_ENABLE_AUDIT_LOGGING", "true").lower() == "true"
        )
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.app.upload_dir,
            self.database.chroma_persist_dir,
            "logs",
            "data",
            "tests"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


# Global configuration instance
config = Config()

# Export commonly checked attributes for validators
ai = config.ai
database = config.database
app = config.app
security = config.security