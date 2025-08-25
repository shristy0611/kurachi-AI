"""
SOTA Configuration System for Kurachi AI
Pydantic-based type-safe configuration with validation

Features:
- Type-safe configuration with Pydantic
- Environment variable support
- Configuration validation
- Hot reloading capabilities
- Nested configuration sections
- Default value management
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import timedelta
import json

try:
    from pydantic import BaseSettings, Field, validator, root_validator
    from pydantic.types import DirectoryPath, FilePath
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without Pydantic
    PYDANTIC_AVAILABLE = False
    class BaseSettings:
        pass
    def Field(*args, **kwargs):
        return kwargs.get('default')
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def root_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from utils.logger import get_logger

logger = get_logger("sota_config")


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    sqlite_path: Path = Field(default=Path("data/kurachi.db"), description="SQLite database path")
    chroma_persist_dir: Path = Field(default=Path("./chroma_db"), description="ChromaDB persistence directory")
    connection_pool_size: int = Field(default=5, ge=1, le=20, description="Database connection pool size")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Database operation timeout")
    backup_enabled: bool = Field(default=True, description="Enable automatic database backups")
    backup_interval_hours: int = Field(default=24, ge=1, le=168, description="Backup interval in hours")
    
    @validator('sqlite_path')
    def validate_sqlite_path(cls, v):
        """Ensure SQLite path parent directory exists"""
        if isinstance(v, str):
            v = Path(v)
        v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('chroma_persist_dir')
    def validate_chroma_dir(cls, v):
        """Ensure ChromaDB directory exists"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class AIConfig(BaseSettings):
    """AI and LLM configuration"""
    llm_model: str = Field(default="qwen3:4b", description="Primary LLM model")
    embedding_model: str = Field(default="nomic-embed-text", description="Embedding model")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum tokens per response")
    timeout_seconds: int = Field(default=60, ge=1, le=300, description="LLM request timeout")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Number of retry attempts")
    
    # Translation settings
    translation_cache_size: int = Field(default=1000, ge=100, le=10000, description="Translation cache size")
    translation_quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum translation quality")
    
    # Document processing
    max_chunk_size: int = Field(default=800, ge=100, le=2000, description="Maximum chunk size for documents")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="Chunk overlap size")
    
    @validator('llm_model')
    def validate_model_format(cls, v):
        """Validate LLM model format"""
        if ':' not in v:
            raise ValueError('Model must include version (e.g., "qwen3:4b")')
        return v
    
    @validator('ollama_base_url')
    def validate_ollama_url(cls, v):
        """Validate Ollama URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Ollama URL must start with http:// or https://')
        return v.rstrip('/')
    
    class Config:
        env_prefix = "AI_"
        case_sensitive = False


class SecurityConfig(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Application secret key")
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440, description="Session timeout in minutes")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file upload size in MB")
    allowed_file_types: List[str] = Field(
        default=["pdf", "txt", "md", "docx", "xlsx", "pptx"], 
        description="Allowed file types for upload"
    )
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    rate_limit_requests_per_minute: int = Field(default=100, ge=1, le=1000, description="Rate limit per minute")
    
    @validator('secret_key')
    def validate_secret_key(cls, v, values):
        """Validate secret key strength"""
        if len(v) < 16:
            raise ValueError('Secret key must be at least 16 characters long')
        if v == "dev-secret-key-change-in-production":
            # In production, this should be changed
            environment = values.get('environment', Environment.DEVELOPMENT)
            if environment == Environment.PRODUCTION:
                raise ValueError('Default secret key cannot be used in production')
        return v
    
    @validator('allowed_file_types')
    def validate_file_types(cls, v):
        """Validate file types format"""
        return [ft.lower().lstrip('.') for ft in v]
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class PerformanceConfig(BaseSettings):
    """Performance and optimization configuration"""
    max_workers: int = Field(default=5, ge=1, le=20, description="Maximum worker threads")
    max_concurrent_requests: int = Field(default=10, ge=1, le=100, description="Maximum concurrent requests")
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Cache TTL in seconds")
    enable_lazy_loading: bool = Field(default=True, description="Enable lazy service loading")
    enable_async_processing: bool = Field(default=True, description="Enable async processing")
    
    # Memory management
    max_memory_usage_percent: float = Field(default=80.0, ge=50.0, le=95.0, description="Maximum memory usage threshold")
    garbage_collection_interval_minutes: int = Field(default=30, ge=5, le=120, description="GC interval in minutes")
    
    # Monitoring
    metrics_collection_enabled: bool = Field(default=True, description="Enable metrics collection")
    tracing_enabled: bool = Field(default=True, description="Enable distributed tracing")
    health_check_interval_seconds: int = Field(default=30, ge=10, le=300, description="Health check interval")
    
    class Config:
        env_prefix = "PERFORMANCE_"
        case_sensitive = False


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration"""
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Application log level")
    log_file_path: Optional[Path] = Field(default=Path("logs/kurachi.log"), description="Log file path")
    log_rotation_size_mb: int = Field(default=100, ge=10, le=1000, description="Log rotation size in MB")
    log_retention_days: int = Field(default=30, ge=1, le=365, description="Log retention in days")
    
    # OpenTelemetry settings
    jaeger_endpoint: Optional[str] = Field(default=None, description="Jaeger collector endpoint")
    prometheus_port: int = Field(default=8090, ge=1024, le=65535, description="Prometheus metrics port")
    enable_stdout_logging: bool = Field(default=True, description="Enable stdout logging")
    enable_structured_logging: bool = Field(default=True, description="Enable structured JSON logging")
    
    # Alert settings
    alert_email_enabled: bool = Field(default=False, description="Enable email alerts")
    alert_email_recipients: List[str] = Field(default=[], description="Alert email recipients")
    alert_webhook_url: Optional[str] = Field(default=None, description="Alert webhook URL")
    
    @validator('log_file_path')
    def validate_log_path(cls, v):
        """Ensure log directory exists"""
        if v and isinstance(v, (str, Path)):
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            return log_path
        return v
    
    @validator('jaeger_endpoint')
    def validate_jaeger_endpoint(cls, v):
        """Validate Jaeger endpoint format"""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Jaeger endpoint must start with http:// or https://')
        return v
    
    class Config:
        env_prefix = "OBSERVABILITY_"
        case_sensitive = False


class UIConfig(BaseSettings):
    """User interface configuration"""
    streamlit_port: int = Field(default=8501, ge=1024, le=65535, description="Streamlit server port")
    streamlit_host: str = Field(default="localhost", description="Streamlit server host")
    enable_theme_switching: bool = Field(default=True, description="Enable theme switching")
    default_theme: str = Field(default="light", description="Default UI theme")
    max_chat_history: int = Field(default=100, ge=10, le=1000, description="Maximum chat history entries")
    enable_file_upload: bool = Field(default=True, description="Enable file upload functionality")
    upload_directory: Path = Field(default=Path("uploads"), description="File upload directory")
    
    @validator('default_theme')
    def validate_theme(cls, v):
        """Validate theme name"""
        valid_themes = ["light", "dark", "auto"]
        if v not in valid_themes:
            raise ValueError(f'Theme must be one of: {valid_themes}')
        return v
    
    @validator('upload_directory')
    def validate_upload_dir(cls, v):
        """Ensure upload directory exists"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "UI_"
        case_sensitive = False


class SOTAConfig(BaseSettings):
    """
    SOTA Configuration System
    
    Comprehensive type-safe configuration with validation
    """
    
    # Environment settings
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=True, description="Enable debug mode")
    version: str = Field(default="1.0.0", description="Application version")
    
    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    # Additional settings
    data_directory: Path = Field(default=Path("data"), description="Data storage directory")
    temp_directory: Path = Field(default=Path("temp"), description="Temporary files directory")
    
    @root_validator
    def validate_environment_consistency(cls, values):
        """Validate configuration consistency across environment"""
        env = values.get('environment', Environment.DEVELOPMENT)
        
        if env == Environment.PRODUCTION:
            # Production-specific validations
            if values.get('debug', True):
                logger.warning("Debug mode should be disabled in production")
            
            security = values.get('security')
            if security and hasattr(security, 'secret_key'):
                if security.secret_key == "dev-secret-key-change-in-production":
                    raise ValueError("Default secret key cannot be used in production")
        
        return values
    
    @validator('data_directory', 'temp_directory')
    def ensure_directory_exists(cls, v):
        """Ensure directories exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return f"sqlite:///{self.database.sqlite_path}"
    
    def get_cache_settings(self) -> Dict[str, Any]:
        """Get cache configuration"""
        return {
            "ttl_seconds": self.performance.cache_ttl_seconds,
            "max_size": self.ai.translation_cache_size,
            "enabled": True
        }
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for web interface"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export configuration as dictionary
        
        Args:
            include_secrets: Whether to include sensitive data
            
        Returns:
            Configuration dictionary
        """
        config_dict = self.dict()
        
        if not include_secrets:
            # Remove sensitive information
            if 'security' in config_dict and 'secret_key' in config_dict['security']:
                config_dict['security']['secret_key'] = "***HIDDEN***"
        
        return config_dict
    
    def validate_runtime_requirements(self) -> List[str]:
        """
        Validate runtime requirements and return any issues
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check Ollama connectivity
        try:
            import requests
            response = requests.get(f"{self.ai.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                issues.append(f"Ollama not accessible at {self.ai.ollama_base_url}")
        except Exception as e:
            issues.append(f"Failed to connect to Ollama: {e}")
        
        # Check required directories
        required_dirs = [
            self.data_directory,
            self.temp_directory,
            self.ui.upload_directory,
            self.database.chroma_persist_dir.parent
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                issues.append(f"Required directory does not exist: {directory}")
        
        # Check disk space
        try:
            import shutil
            free_space_gb = shutil.disk_usage(self.data_directory).free / (1024**3)
            if free_space_gb < 1.0:  # Less than 1GB free
                issues.append(f"Low disk space: {free_space_gb:.1f}GB free")
        except Exception:
            pass
        
        return issues
    
    @classmethod
    def load_from_file(cls, config_file: Union[str, Path]) -> 'SOTAConfig':
        """
        Load configuration from file
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            SOTAConfig instance
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return cls()
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                # Assume environment file format
                config_data = {}
                with open(config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config_data[key.strip()] = value.strip()
            
            return cls(**config_data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return cls()
    
    def save_to_file(self, config_file: Union[str, Path], format: str = "json"):
        """
        Save configuration to file
        
        Args:
            config_file: Path to save configuration
            format: File format ('json' or 'env')
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "json":
                with open(config_path, 'w') as f:
                    json.dump(self.export_config(include_secrets=False), f, indent=2, default=str)
            else:
                # Environment file format
                with open(config_path, 'w') as f:
                    f.write("# SOTA Configuration\\n")
                    f.write(f"# Generated on {datetime.now()}\\n\\n")
                    
                    config_dict = self.export_config(include_secrets=False)
                    for section, values in config_dict.items():
                        if isinstance(values, dict):
                            f.write(f"# {section.upper()} Configuration\\n")
                            for key, value in values.items():
                                env_key = f"{section.upper()}_{key.upper()}"
                                f.write(f"{env_key}={value}\\n")
                            f.write("\\n")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        validate_assignment = True


# Global configuration instance
if PYDANTIC_AVAILABLE:
    sota_config = SOTAConfig()
    logger.info("SOTA Configuration system initialized with Pydantic validation")
else:
    # Fallback configuration for environments without Pydantic
    class FallbackConfig:
        def __init__(self):
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.debug = os.getenv("DEBUG", "true").lower() == "true"
            
            # Simple nested config
            class DatabaseConfig:
                sqlite_path = Path(os.getenv("DB_SQLITE_PATH", "data/kurachi.db"))
                chroma_persist_dir = Path(os.getenv("DB_CHROMA_PERSIST_DIR", "./chroma_db"))
            
            class AIConfig:
                llm_model = os.getenv("AI_LLM_MODEL", "qwen3:4b")
                ollama_base_url = os.getenv("AI_OLLAMA_BASE_URL", "http://localhost:11434")
                temperature = float(os.getenv("AI_TEMPERATURE", "0.7"))
            
            self.database = DatabaseConfig()
            self.ai = AIConfig()
    
    sota_config = FallbackConfig()
    logger.warning("Using fallback configuration (Pydantic not available)")

# Convenience functions
def get_config() -> Union[SOTAConfig, Any]:
    """Get global configuration instance"""
    return sota_config

def reload_config():
    """Reload configuration from environment"""
    global sota_config
    if PYDANTIC_AVAILABLE:
        sota_config = SOTAConfig()
    else:
        sota_config = FallbackConfig()
    logger.info("Configuration reloaded")

def validate_config() -> List[str]:
    """Validate current configuration"""
    if hasattr(sota_config, 'validate_runtime_requirements'):
        return sota_config.validate_runtime_requirements()
    return []