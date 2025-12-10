"""Core Pydantic models for the workflow engine."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionStatusEnum(str, Enum):
    """Enumeration of workflow execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LogEventType(str, Enum):
    """Enumeration of log event types."""
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_ERROR = "node_error"
    STATE_UPDATE = "state_update"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"


class NodeDefinition(BaseModel):
    """Definition of a workflow node."""
    id: str = Field(..., description="Unique identifier for the node")
    function_name: str = Field(..., description="Name of the function to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the function")
    timeout: Optional[int] = Field(None, description="Timeout in seconds for node execution")


class EdgeDefinition(BaseModel):
    """Definition of an edge between workflow nodes."""
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    condition: Optional[str] = Field(None, description="Condition for edge traversal")


class GraphDefinition(BaseModel):
    """Complete definition of a workflow graph."""
    name: str = Field(..., description="Name of the workflow graph")
    description: str = Field(..., description="Description of the workflow")
    nodes: List[NodeDefinition] = Field(..., description="List of nodes in the graph")
    edges: List[EdgeDefinition] = Field(..., description="List of edges connecting nodes")
    entry_point: str = Field(..., description="ID of the starting node")
    exit_conditions: List[str] = Field(default_factory=list, description="Conditions for workflow termination")


class WorkflowState(BaseModel):
    """State of a workflow execution."""
    data: Dict[str, Any] = Field(default_factory=dict, description="Workflow data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the execution")
    current_node: Optional[str] = Field(None, description="Currently executing node")
    execution_path: List[str] = Field(default_factory=list, description="Path of executed nodes")


class ExecutionStatus(BaseModel):
    """Status of a workflow execution."""
    run_id: str = Field(..., description="Unique identifier for the execution run")
    graph_id: str = Field(..., description="ID of the graph being executed")
    status: ExecutionStatusEnum = Field(..., description="Current execution status")
    current_state: WorkflowState = Field(..., description="Current workflow state")
    started_at: datetime = Field(..., description="Timestamp when execution started")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when execution completed")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")


class LogEntry(BaseModel):
    """Log entry for workflow execution events."""
    timestamp: datetime = Field(..., description="Timestamp of the log entry")
    run_id: str = Field(..., description="ID of the workflow run")
    node_id: str = Field(..., description="ID of the node that generated the log")
    event_type: LogEventType = Field(..., description="Type of event")
    message: str = Field(..., description="Log message")
    state_snapshot: Optional[Dict] = Field(None, description="State snapshot at the time of the event")


class GraphSummary(BaseModel):
    """Summary information about a workflow graph."""
    id: str = Field(..., description="Graph ID")
    name: str = Field(..., description="Graph name")
    description: str = Field(..., description="Graph description")
    created_at: datetime = Field(..., description="Creation timestamp")
    node_count: int = Field(..., description="Number of nodes in the graph")


class ValidationResult(BaseModel):
    """Result of graph validation."""
    is_valid: bool = Field(..., description="Whether the graph is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")