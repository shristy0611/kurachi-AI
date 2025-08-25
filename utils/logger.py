"""
Logging utilities for Kurachi AI
Provides structured logging with different levels and audit capabilities
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

from config import config


class KurachiLogger:
    """Enhanced logger with audit capabilities and structured logging"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.app.log_level))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up console and file handlers"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = Path("logs") / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with optional extra data"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.info(message)
    
    def error(self, message: str, error: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None):
        """Log error message with optional exception and extra data"""
        if error:
            message = f"{message} | Error: {str(error)}"
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.error(message, exc_info=error is not None)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with optional extra data"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.warning(message)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with optional extra data"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.debug(message)


class AuditLogger:
    """Specialized logger for audit events"""
    
    def __init__(self):
        self.logger = KurachiLogger("audit")
        self.audit_file = Path("logs") / "audit.log"
    
    def log_user_action(self, user_id: str, action: str, resource: str, 
                       success: bool = True, details: Optional[Dict[str, Any]] = None):
        """Log user actions for audit purposes"""
        if not config.security.enable_audit_logging:
            return
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "details": details or {}
        }
        
        # Write to audit file
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
        
        # Also log to regular logger
        level = "info" if success else "error"
        message = f"User {user_id} {action} on {resource}"
        getattr(self.logger, level)(message, extra=audit_entry)
    
    def log_system_event(self, event_type: str, description: str, 
                        details: Optional[Dict[str, Any]] = None):
        """Log system events for audit purposes"""
        if not config.security.enable_audit_logging:
            return
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "description": description,
            "details": details or {}
        }
        
        # Write to audit file
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
        
        self.logger.info(f"System event: {event_type} - {description}", extra=audit_entry)


# Global logger instances
def get_logger(name: str) -> KurachiLogger:
    """Get a logger instance for the given name"""
    return KurachiLogger(name)


# Global audit logger
audit_logger = AuditLogger()