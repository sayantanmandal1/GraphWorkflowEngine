"""Enhanced logging configuration for the workflow engine."""

import logging
import sys
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, default=str)


class WorkflowContextFilter(logging.Filter):
    """Filter to add workflow context to log records."""
    
    def __init__(self):
        super().__init__()
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set context fields for logging."""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear all context fields."""
        self._context.clear()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to log record."""
        if not hasattr(record, 'extra_fields'):
            record.extra_fields = {}
        record.extra_fields.update(self._context)
        return True


# Global context filter instance
_context_filter = WorkflowContextFilter()


def setup_logging(
    level: str = "INFO", 
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    structured: bool = False,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure enhanced logging for the workflow engine.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Custom log format string
        structured: Whether to use structured JSON logging
        max_size: Maximum log file size in bytes before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Root logger instance
    """
    # Create formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        # Use custom format if provided, otherwise use default
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
        
        formatter = logging.Formatter(
            fmt=log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_context_filter)
    root_logger.addHandler(console_handler)
    
    # Add file handler with rotation if specified
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_context_filter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers with appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Set workflow engine loggers to appropriate levels
    logging.getLogger("app.core").setLevel(logging.DEBUG if level == "DEBUG" else logging.INFO)
    logging.getLogger("app.api").setLevel(logging.INFO)
    logging.getLogger("app.tools").setLevel(logging.INFO)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)


def set_logging_context(**kwargs):
    """Set context fields for all subsequent log messages."""
    _context_filter.set_context(**kwargs)


def clear_logging_context():
    """Clear all logging context fields."""
    _context_filter.clear_context()


def log_with_context(logger: logging.Logger, level: int, message: str, **context):
    """Log a message with additional context fields."""
    extra = {"extra_fields": context}
    logger.log(level, message, extra=extra)


class ErrorRecoveryLogger:
    """Specialized logger for error recovery operations."""
    
    def __init__(self, component_name: str):
        self.logger = get_logger(f"recovery.{component_name}")
        self.component_name = component_name
    
    def log_recovery_attempt(self, operation: str, error: Exception, attempt: int, max_attempts: int):
        """Log a recovery attempt."""
        log_with_context(
            self.logger, logging.WARNING,
            f"Recovery attempt {attempt}/{max_attempts} for {operation}",
            component=self.component_name,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            attempt=attempt,
            max_attempts=max_attempts
        )
    
    def log_recovery_success(self, operation: str, attempts_used: int):
        """Log successful recovery."""
        log_with_context(
            self.logger, logging.INFO,
            f"Successfully recovered from {operation} after {attempts_used} attempts",
            component=self.component_name,
            operation=operation,
            attempts_used=attempts_used,
            recovery_status="success"
        )
    
    def log_recovery_failure(self, operation: str, final_error: Exception, attempts_used: int):
        """Log failed recovery."""
        log_with_context(
            self.logger, logging.ERROR,
            f"Failed to recover from {operation} after {attempts_used} attempts",
            component=self.component_name,
            operation=operation,
            error_type=type(final_error).__name__,
            error_message=str(final_error),
            attempts_used=attempts_used,
            recovery_status="failed"
        )