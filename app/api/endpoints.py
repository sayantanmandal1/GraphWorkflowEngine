"""FastAPI REST endpoints for the workflow engine."""

from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.graph_manager import GraphManager
from ..core.execution_engine import ExecutionEngine
from ..core.tool_registry import ToolRegistry
from ..core.state_manager import StateManager
from ..core.exceptions import (
    GraphValidationError, 
    ExecutionEngineError, 
    StorageError,
    NodeExecutionError
)
from ..models.core import (
    GraphDefinition, 
    ExecutionStatus, 
    LogEntry,
    GraphSummary,
    ValidationResult
)
from ..core.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["workflow"])

# Global instances (will be initialized in main.py)
_graph_manager: Optional[GraphManager] = None
_execution_engine: Optional[ExecutionEngine] = None
_tool_registry: Optional[ToolRegistry] = None
_state_manager: Optional[StateManager] = None


def init_dependencies(
    graph_manager: GraphManager,
    execution_engine: ExecutionEngine,
    tool_registry: ToolRegistry,
    state_manager: StateManager
):
    """Initialize the global dependencies."""
    global _graph_manager, _execution_engine, _tool_registry, _state_manager
    _graph_manager = graph_manager
    _execution_engine = execution_engine
    _tool_registry = tool_registry
    _state_manager = state_manager


def get_graph_manager() -> GraphManager:
    """Dependency to get graph manager."""
    if _graph_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Graph manager not initialized"
        )
    return _graph_manager


def get_execution_engine() -> ExecutionEngine:
    """Dependency to get execution engine."""
    if _execution_engine is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Execution engine not initialized"
        )
    return _execution_engine


def get_tool_registry() -> ToolRegistry:
    """Dependency to get tool registry."""
    if _tool_registry is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tool registry not initialized"
        )
    return _tool_registry


def get_state_manager() -> StateManager:
    """Dependency to get state manager."""
    if _state_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="State manager not initialized"
        )
    return _state_manager


# Request/Response models
class CreateGraphRequest(BaseModel):
    """Request model for creating a graph."""
    graph: GraphDefinition = Field(..., description="Graph definition to create")


class CreateGraphResponse(BaseModel):
    """Response model for graph creation."""
    graph_id: str = Field(..., description="Unique identifier of the created graph")
    message: str = Field(..., description="Success message")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class RunWorkflowRequest(BaseModel):
    """Request model for running a workflow."""
    graph_id: str = Field(..., description="ID of the graph to execute")
    initial_state: Dict[str, Any] = Field(default_factory=dict, description="Initial state data")


class RunWorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    run_id: str = Field(..., description="Unique identifier for the execution run")
    message: str = Field(..., description="Success message")
    status: str = Field(..., description="Initial execution status")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Endpoints

@router.post(
    "/graph/create",
    response_model=CreateGraphResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workflow graph",
    description="Create a new workflow graph with validation and return its unique identifier"
)
async def create_graph(
    request: CreateGraphRequest,
    graph_manager: GraphManager = Depends(get_graph_manager)
) -> CreateGraphResponse:
    """
    Create a new workflow graph.
    
    Args:
        request: Graph creation request containing the graph definition
        graph_manager: Graph manager dependency
        
    Returns:
        Response containing the created graph ID and any validation warnings
        
    Raises:
        HTTPException: If graph validation fails or creation encounters an error
    """
    try:
        logger.info(f"Creating new graph: {request.graph.name}")
        
        # Validate the graph first to get warnings
        validation_result = graph_manager.validate_graph(request.graph)
        
        # Create the graph (this will also validate and raise errors if invalid)
        graph_id = graph_manager.create_graph(request.graph)
        
        logger.info(f"Successfully created graph '{request.graph.name}' with ID: {graph_id}")
        
        return CreateGraphResponse(
            graph_id=graph_id,
            message=f"Graph '{request.graph.name}' created successfully",
            validation_warnings=validation_result.warnings
        )
        
    except GraphValidationError as e:
        logger.warning(f"Graph validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": str(e),
                "details": {"graph_name": request.graph.name}
            }
        )
    except StorageError as e:
        logger.error(f"Storage error during graph creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "StorageError",
                "message": "Failed to store graph",
                "details": {"original_error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during graph creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while creating the graph",
                "details": {"original_error": str(e)}
            }
        )


@router.post(
    "/graph/run",
    response_model=RunWorkflowResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Execute a workflow graph",
    description="Start execution of a workflow graph with the provided initial state"
)
async def run_workflow(
    request: RunWorkflowRequest,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> RunWorkflowResponse:
    """
    Execute a workflow graph.
    
    Args:
        request: Workflow execution request containing graph ID and initial state
        execution_engine: Execution engine dependency
        
    Returns:
        Response containing the run ID and initial status
        
    Raises:
        HTTPException: If graph not found or execution setup fails
    """
    try:
        logger.info(f"Starting workflow execution for graph: {request.graph_id}")
        
        # Start workflow execution
        run_id = execution_engine.execute_workflow(request.graph_id, request.initial_state)
        
        logger.info(f"Successfully started workflow execution: run_id={run_id}")
        
        return RunWorkflowResponse(
            run_id=run_id,
            message="Workflow execution started successfully",
            status="pending"
        )
        
    except StorageError as e:
        # Graph not found or storage issue
        if "not found" in str(e).lower():
            logger.warning(f"Graph not found: {request.graph_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "GraphNotFound",
                    "message": f"Graph with ID '{request.graph_id}' not found",
                    "details": {"graph_id": request.graph_id}
                }
            )
        else:
            logger.error(f"Storage error during workflow execution: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "StorageError",
                    "message": "Failed to access workflow data",
                    "details": {"original_error": str(e)}
                }
            )
    except ExecutionEngineError as e:
        logger.error(f"Execution engine error: {str(e)}")
        if "queue is full" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "ServiceUnavailable",
                    "message": "Execution queue is full, please try again later",
                    "details": {"retry_after": "30 seconds"}
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "ExecutionError",
                    "message": str(e),
                    "details": {"graph_id": request.graph_id}
                }
            )
    except Exception as e:
        logger.error(f"Unexpected error during workflow execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while starting workflow execution",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/graph/state/{run_id}",
    response_model=ExecutionStatus,
    summary="Get workflow execution status",
    description="Retrieve the current execution status and state of a workflow run"
)
async def get_execution_status(
    run_id: str,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> ExecutionStatus:
    """
    Get the execution status of a workflow run.
    
    Args:
        run_id: ID of the workflow run
        execution_engine: Execution engine dependency
        
    Returns:
        Current execution status and state
        
    Raises:
        HTTPException: If run not found or status retrieval fails
    """
    try:
        logger.debug(f"Getting execution status for run: {run_id}")
        
        # Get execution status
        status = execution_engine.get_execution_status(run_id)
        
        logger.debug(f"Retrieved execution status for run {run_id}: {status.status}")
        
        return status
        
    except ExecutionEngineError as e:
        if "not found" in str(e).lower():
            logger.warning(f"Run not found: {run_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "RunNotFound",
                    "message": f"Workflow run with ID '{run_id}' not found",
                    "details": {"run_id": run_id}
                }
            )
        else:
            logger.error(f"Error getting execution status: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "ExecutionError",
                    "message": str(e),
                    "details": {"run_id": run_id}
                }
            )
    except Exception as e:
        logger.error(f"Unexpected error getting execution status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while retrieving execution status",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/graph/logs/{run_id}",
    response_model=List[LogEntry],
    summary="Get workflow execution logs",
    description="Retrieve the execution logs for a workflow run in chronological order"
)
async def get_execution_logs(
    run_id: str,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> List[LogEntry]:
    """
    Get the execution logs of a workflow run.
    
    Args:
        run_id: ID of the workflow run
        execution_engine: Execution engine dependency
        
    Returns:
        List of log entries in chronological order
        
    Raises:
        HTTPException: If run not found or log retrieval fails
    """
    try:
        logger.debug(f"Getting execution logs for run: {run_id}")
        
        # Get execution logs
        logs = execution_engine.get_execution_logs(run_id)
        
        logger.debug(f"Retrieved {len(logs)} log entries for run {run_id}")
        
        return logs
        
    except ExecutionEngineError as e:
        if "not found" in str(e).lower():
            logger.warning(f"Run not found for logs: {run_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "RunNotFound",
                    "message": f"Workflow run with ID '{run_id}' not found",
                    "details": {"run_id": run_id}
                }
            )
        else:
            logger.error(f"Error getting execution logs: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "ExecutionError",
                    "message": str(e),
                    "details": {"run_id": run_id}
                }
            )
    except Exception as e:
        logger.error(f"Unexpected error getting execution logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while retrieving execution logs",
                "details": {"original_error": str(e)}
            }
        )


# Additional utility endpoints

@router.get(
    "/graphs",
    response_model=List[GraphSummary],
    summary="List all workflow graphs",
    description="Retrieve a list of all available workflow graphs with summary information"
)
async def list_graphs(
    graph_manager: GraphManager = Depends(get_graph_manager)
) -> List[GraphSummary]:
    """
    List all available workflow graphs.
    
    Args:
        graph_manager: Graph manager dependency
        
    Returns:
        List of graph summaries
        
    Raises:
        HTTPException: If graph listing fails
    """
    try:
        logger.debug("Listing all graphs")
        
        graphs = graph_manager.list_graphs()
        
        logger.debug(f"Retrieved {len(graphs)} graphs")
        
        return graphs
        
    except StorageError as e:
        logger.error(f"Storage error listing graphs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "StorageError",
                "message": "Failed to retrieve graph list",
                "details": {"original_error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error listing graphs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while listing graphs",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/graph/{graph_id}",
    response_model=GraphDefinition,
    summary="Get workflow graph definition",
    description="Retrieve the complete definition of a workflow graph"
)
async def get_graph(
    graph_id: str,
    graph_manager: GraphManager = Depends(get_graph_manager)
) -> GraphDefinition:
    """
    Get a workflow graph definition.
    
    Args:
        graph_id: ID of the graph to retrieve
        graph_manager: Graph manager dependency
        
    Returns:
        Complete graph definition
        
    Raises:
        HTTPException: If graph not found or retrieval fails
    """
    try:
        logger.debug(f"Getting graph definition: {graph_id}")
        
        graph = graph_manager.get_graph(graph_id)
        
        logger.debug(f"Retrieved graph definition: {graph.name}")
        
        return graph
        
    except StorageError as e:
        if "not found" in str(e).lower():
            logger.warning(f"Graph not found: {graph_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "GraphNotFound",
                    "message": f"Graph with ID '{graph_id}' not found",
                    "details": {"graph_id": graph_id}
                }
            )
        else:
            logger.error(f"Storage error getting graph: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "StorageError",
                    "message": "Failed to retrieve graph",
                    "details": {"original_error": str(e)}
                }
            )
    except Exception as e:
        logger.error(f"Unexpected error getting graph: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while retrieving graph",
                "details": {"original_error": str(e)}
            }
        )


@router.post(
    "/graph/validate",
    response_model=ValidationResult,
    summary="Validate a workflow graph",
    description="Validate a workflow graph definition without creating it"
)
async def validate_graph(
    request: CreateGraphRequest,
    graph_manager: GraphManager = Depends(get_graph_manager)
) -> ValidationResult:
    """
    Validate a workflow graph definition.
    
    Args:
        request: Graph validation request containing the graph definition
        graph_manager: Graph manager dependency
        
    Returns:
        Validation result with errors and warnings
    """
    try:
        logger.debug(f"Validating graph: {request.graph.name}")
        
        validation_result = graph_manager.validate_graph(request.graph)
        
        logger.debug(f"Graph validation completed. Valid: {validation_result.is_valid}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error during graph validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ValidationError",
                "message": "An error occurred during graph validation",
                "details": {"original_error": str(e)}
            }
        )


@router.delete(
    "/graph/{graph_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a workflow graph",
    description="Delete a workflow graph by its ID"
)
async def delete_graph(
    graph_id: str,
    graph_manager: GraphManager = Depends(get_graph_manager)
):
    """
    Delete a workflow graph.
    
    Args:
        graph_id: ID of the graph to delete
        graph_manager: Graph manager dependency
        
    Raises:
        HTTPException: If graph not found or deletion fails
    """
    try:
        logger.info(f"Deleting graph: {graph_id}")
        
        deleted = graph_manager.delete_graph(graph_id)
        
        if not deleted:
            logger.warning(f"Graph not found for deletion: {graph_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "GraphNotFound",
                    "message": f"Graph with ID '{graph_id}' not found",
                    "details": {"graph_id": graph_id}
                }
            )
        
        logger.info(f"Successfully deleted graph: {graph_id}")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except StorageError as e:
        logger.error(f"Storage error deleting graph: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "StorageError",
                "message": "Failed to delete graph",
                "details": {"original_error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting graph: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while deleting graph",
                "details": {"original_error": str(e)}
            }
        )


@router.post(
    "/graph/cancel/{run_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel workflow execution",
    description="Cancel a running workflow execution"
)
async def cancel_execution(
    run_id: str,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> Dict[str, Any]:
    """
    Cancel a workflow execution.
    
    Args:
        run_id: ID of the workflow run to cancel
        execution_engine: Execution engine dependency
        
    Returns:
        Cancellation result
        
    Raises:
        HTTPException: If run not found or cancellation fails
    """
    try:
        logger.info(f"Cancelling workflow execution: {run_id}")
        
        cancelled = execution_engine.cancel_execution(run_id)
        
        if not cancelled:
            logger.warning(f"Could not cancel execution (may not be active): {run_id}")
            return {
                "message": f"Execution {run_id} could not be cancelled (may not be active)",
                "cancelled": False
            }
        
        logger.info(f"Successfully cancelled execution: {run_id}")
        
        return {
            "message": f"Execution {run_id} cancelled successfully",
            "cancelled": True
        }
        
    except Exception as e:
        logger.error(f"Error cancelling execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "CancellationError",
                "message": "An error occurred while cancelling execution",
                "details": {"original_error": str(e)}
            }
        )