"""Enhanced custom exceptions for the workflow engine with detailed error information."""

import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better classification."""
    VALIDATION = "validation"
    EXECUTION = "execution"
    STORAGE = "storage"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    SECURITY = "security"
    BUSINESS_LOGIC = "business_logic"


class WorkflowEngineError(Exception):
    """Enhanced base exception for all workflow engine errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.EXECUTION,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        self.traceback_info = traceback.format_stack()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "exception_type": self.__class__.__name__
        }
    
    def add_context(self, **kwargs):
        """Add additional context to the exception."""
        self.context.update(kwargs)
        return self
    
    def add_details(self, **kwargs):
        """Add additional details to the exception."""
        self.details.update(kwargs)
        return self


class GraphValidationError(WorkflowEngineError):
    """Raised when graph validation fails."""
    
    def __init__(
        self, 
        message: str, 
        validation_errors: Optional[List[str]] = None,
        graph_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        self.validation_errors = validation_errors or []
        if graph_name:
            self.add_context(graph_name=graph_name)
        if validation_errors:
            self.add_details(validation_errors=validation_errors)


class NodeExecutionError(WorkflowEngineError):
    """Raised when node execution fails."""
    
    def __init__(
        self, 
        message: str, 
        node_id: Optional[str] = None,
        run_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXECUTION,
            recoverable=True,
            **kwargs
        )
        if node_id:
            self.add_context(node_id=node_id)
        if run_id:
            self.add_context(run_id=run_id)
        if execution_time:
            self.add_details(execution_time=execution_time)


class StateManagementError(WorkflowEngineError):
    """Raised when state management operations fail."""
    
    def __init__(
        self, 
        message: str, 
        run_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.STORAGE,
            recoverable=True,
            retry_after=5,
            **kwargs
        )
        if run_id:
            self.add_context(run_id=run_id)
        if operation:
            self.add_context(operation=operation)


class ToolRegistryError(WorkflowEngineError):
    """Raised when tool registry operations fail."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )
        if tool_name:
            self.add_context(tool_name=tool_name)
        if operation:
            self.add_context(operation=operation)


class ExecutionEngineError(WorkflowEngineError):
    """Raised when execution engine operations fail."""
    
    def __init__(
        self, 
        message: str, 
        run_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXECUTION,
            **kwargs
        )
        if run_id:
            self.add_context(run_id=run_id)
        if graph_id:
            self.add_context(graph_id=graph_id)


class StorageError(WorkflowEngineError):
    """Raised when storage operations fail."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.STORAGE,
            recoverable=True,
            retry_after=3,
            **kwargs
        )
        if operation:
            self.add_context(operation=operation)
        if table:
            self.add_context(table=table)


class ResourceExhaustionError(WorkflowEngineError):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RESOURCE,
            recoverable=True,
            retry_after=30,
            **kwargs
        )
        if resource_type:
            self.add_context(resource_type=resource_type)
        if current_usage is not None and limit is not None:
            self.add_details(current_usage=current_usage, limit=limit)


class TransientError(WorkflowEngineError):
    """Raised for transient errors that should be retried."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            retry_after=5,
            **kwargs
        )


class ConfigurationError(WorkflowEngineError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )
        if config_key:
            self.add_context(config_key=config_key)


class APIError(WorkflowEngineError):
    """Raised when API operations fail."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            **kwargs
        )
        self.status_code = status_code
        if endpoint:
            self.add_context(endpoint=endpoint)
        self.add_details(status_code=status_code)


class WebSocketError(WorkflowEngineError):
    """Raised when WebSocket operations fail."""
    
    def __init__(
        self, 
        message: str, 
        connection_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recoverable=True,
            **kwargs
        )
        if connection_id:
            self.add_context(connection_id=connection_id)


def create_error_response(error: WorkflowEngineError) -> Dict[str, Any]:
    """Create a standardized error response from a WorkflowEngineError."""
    return {
        "error": error.error_code,
        "message": error.message,
        "details": {
            **error.details,
            "severity": error.severity.value,
            "category": error.category.value,
            "recoverable": error.recoverable,
            "retry_after": error.retry_after,
            "timestamp": error.timestamp.isoformat()
        },
        "context": error.context
    }