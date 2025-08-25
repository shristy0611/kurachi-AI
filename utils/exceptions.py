"""
Custom exceptions for Kurachi AI
Provides structured error handling across the application
"""


class KurachiError(Exception):
    """Base exception for Kurachi AI"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DocumentProcessingError(KurachiError):
    """Exception raised during document processing"""
    pass


class ChatServiceError(KurachiError):
    """Exception raised in chat service operations"""
    pass


class DatabaseError(KurachiError):
    """Exception raised during database operations"""
    pass


class ConfigurationError(KurachiError):
    """Exception raised for configuration issues"""
    pass


class AuthenticationError(KurachiError):
    """Exception raised for authentication issues"""
    pass


class AuthorizationError(KurachiError):
    """Exception raised for authorization issues"""
    pass


class ValidationError(KurachiError):
    """Exception raised for data validation issues"""
    pass


class ExternalServiceError(KurachiError):
    """Exception raised when external services fail"""
    pass