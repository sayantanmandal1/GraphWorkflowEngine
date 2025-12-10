"""Custom exceptions for the workflow engine."""


class WorkflowEngineError(Exception):
    """Base exception for all workflow engine errors."""
    pass


class GraphValidationError(WorkflowEngineError):
    """Raised when graph validation fails."""
    pass


class NodeExecutionError(WorkflowEngineError):
    """Raised when node execution fails."""
    pass


class StateManagementError(WorkflowEngineError):
    """Raised when state management operations fail."""
    pass


class ToolRegistryError(WorkflowEngineError):
    """Raised when tool registry operations fail."""
    pass


class ExecutionEngineError(WorkflowEngineError):
    """Raised when execution engine operations fail."""
    pass


class StorageError(WorkflowEngineError):
    """Raised when storage operations fail."""
    pass


class APIError(WorkflowEngineError):
    """Raised when API operations fail."""
    
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code