"""Core Pydantic models for the workflow engine."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator


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


class ValidationResult(BaseModel):
    """Result of graph validation."""
    is_valid: bool = Field(..., description="Whether the graph is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")


class NodeDefinition(BaseModel):
    """Definition of a workflow node."""
    id: str = Field(..., description="Unique identifier for the node")
    function_name: str = Field(..., description="Name of the function to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the function")
    timeout: Optional[int] = Field(None, description="Timeout in seconds for node execution")

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, id_value):
        """Ensure node ID follows valid format."""
        if not id_value or not id_value.strip():
            raise ValueError("Node ID cannot be empty")
        
        # Check for valid identifier format (alphanumeric, underscore, hyphen)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', id_value.strip()):
            raise ValueError("Node ID must contain only alphanumeric characters, underscores, and hyphens")
        
        return id_value.strip()

    @field_validator('function_name')
    @classmethod
    def validate_function_name(cls, function_name):
        """Ensure function name is valid."""
        if not function_name or not function_name.strip():
            raise ValueError("Function name cannot be empty")
        return function_name.strip()

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, timeout):
        """Ensure timeout is positive if specified."""
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be a positive integer")
        return timeout


class EdgeDefinition(BaseModel):
    """Definition of an edge between workflow nodes."""
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    condition: Optional[str] = Field(None, description="Condition for edge traversal")

    @field_validator('from_node', 'to_node')
    @classmethod
    def validate_node_ids(cls, node_id):
        """Ensure node IDs are valid."""
        if not node_id or not node_id.strip():
            raise ValueError("Node ID cannot be empty")
        return node_id.strip()

    @model_validator(mode='after')
    def validate_edge(self):
        """Validate edge definition."""
        if self.from_node and self.to_node and self.from_node == self.to_node:
            raise ValueError("Self-referencing edges are not allowed")
        return self


class GraphDefinition(BaseModel):
    """Complete definition of a workflow graph."""
    name: str = Field(..., description="Name of the workflow graph")
    description: str = Field(..., description="Description of the workflow")
    nodes: List[NodeDefinition] = Field(..., description="List of nodes in the graph")
    edges: List[EdgeDefinition] = Field(..., description="List of edges connecting nodes")
    entry_point: str = Field(..., description="ID of the starting node")
    exit_conditions: List[str] = Field(default_factory=list, description="Conditions for workflow termination")

    @field_validator('nodes')
    @classmethod
    def validate_unique_node_ids(cls, nodes):
        """Ensure all node IDs are unique."""
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("All node IDs must be unique")
        return nodes

    @field_validator('name')
    @classmethod
    def validate_name_not_empty(cls, name):
        """Ensure graph name is not empty."""
        if not name.strip():
            raise ValueError("Graph name cannot be empty")
        return name.strip()

    @model_validator(mode='after')
    def validate_graph_structure(self):
        """Validate the overall graph structure."""
        if not self.nodes:
            raise ValueError("Graph must contain at least one node")
        
        # Create set of node IDs for efficient lookup
        node_ids = {node.id for node in self.nodes}
        
        # Validate entry point exists
        if self.entry_point and self.entry_point not in node_ids:
            raise ValueError(f"Entry point '{self.entry_point}' does not exist in nodes")
        
        # Validate edge references
        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(f"Edge references non-existent source node: {edge.from_node}")
            if edge.to_node not in node_ids:
                raise ValueError(f"Edge references non-existent target node: {edge.to_node}")
            if edge.from_node == edge.to_node:
                raise ValueError(f"Self-referencing edge not allowed: {edge.from_node}")
        
        # Check for unreachable nodes (basic reachability analysis)
        if self.entry_point and self.edges:
            reachable = self._find_reachable_nodes(self.entry_point, self.edges)
            unreachable = node_ids - reachable
            if unreachable:
                raise ValueError(f"Unreachable nodes detected: {', '.join(sorted(unreachable))}")
        
        return self

    @staticmethod
    def _find_reachable_nodes(entry_point: str, edges: List[EdgeDefinition]) -> Set[str]:
        """Find all nodes reachable from the entry point."""
        reachable = {entry_point}
        edge_map = {}
        
        # Build adjacency list
        for edge in edges:
            if edge.from_node not in edge_map:
                edge_map[edge.from_node] = []
            edge_map[edge.from_node].append(edge.to_node)
        
        # BFS to find reachable nodes
        queue = [entry_point]
        while queue:
            current = queue.pop(0)
            if current in edge_map:
                for neighbor in edge_map[current]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        queue.append(neighbor)
        
        return reachable

    def validate_structure(self) -> ValidationResult:
        """Perform comprehensive graph validation and return detailed results."""
        errors = []
        warnings = []
        
        try:
            # Basic validation is handled by Pydantic validators
            # Additional custom validations can be added here
            
            # Check for potential infinite loops (cycles without exit conditions)
            if self._has_cycles() and not self.exit_conditions:
                warnings.append("Graph contains cycles but no exit conditions specified")
            
            # Check for isolated nodes (nodes with no incoming or outgoing edges)
            isolated_nodes = self._find_isolated_nodes()
            if isolated_nodes:
                warnings.append(f"Isolated nodes detected: {', '.join(sorted(isolated_nodes))}")
            
        except Exception as e:
            errors.append(str(e))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _has_cycles(self) -> bool:
        """Check if the graph contains cycles using DFS."""
        if not self.edges:
            return False
        
        # Build adjacency list
        graph = {}
        for edge in self.edges:
            if edge.from_node not in graph:
                graph[edge.from_node] = []
            graph[edge.from_node].append(edge.to_node)
        
        # Track visited nodes and recursion stack
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if has_cycle_util(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
            
            rec_stack.remove(node)
            return False
        
        # Check for cycles starting from each unvisited node
        for node_id in {node.id for node in self.nodes}:
            if node_id not in visited:
                if has_cycle_util(node_id):
                    return True
        
        return False

    def _find_isolated_nodes(self) -> Set[str]:
        """Find nodes that have no incoming or outgoing edges."""
        node_ids = {node.id for node in self.nodes}
        connected_nodes = set()
        
        for edge in self.edges:
            connected_nodes.add(edge.from_node)
            connected_nodes.add(edge.to_node)
        
        # Entry point is not considered isolated even if it has no incoming edges
        if self.entry_point:
            connected_nodes.add(self.entry_point)
        
        return node_ids - connected_nodes


class WorkflowState(BaseModel):
    """State of a workflow execution."""
    data: Dict[str, Any] = Field(default_factory=dict, description="Workflow data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the execution")
    current_node: Optional[str] = Field(None, description="Currently executing node")
    execution_path: List[str] = Field(default_factory=list, description="Path of executed nodes")

    @field_validator('current_node')
    @classmethod
    def validate_current_node(cls, current_node):
        """Ensure current node ID is valid if specified."""
        if current_node is not None and not current_node.strip():
            raise ValueError("Current node ID cannot be empty string")
        return current_node.strip() if current_node else None

    @field_validator('execution_path')
    @classmethod
    def validate_execution_path(cls, execution_path):
        """Ensure execution path contains valid node IDs."""
        for node_id in execution_path:
            if not node_id or not node_id.strip():
                raise ValueError("Execution path cannot contain empty node IDs")
        return [node_id.strip() for node_id in execution_path]


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