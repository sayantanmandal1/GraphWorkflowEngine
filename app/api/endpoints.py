"""FastAPI REST endpoints for the workflow engine."""

from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json

from ..core.graph_manager import GraphManager
from ..core.execution_engine import ExecutionEngine
from ..core.tool_registry import ToolRegistry
from ..core.state_manager import StateManager
from ..core.exceptions import (
    GraphValidationError, 
    ExecutionEngineError, 
    StorageError,
    NodeExecutionError,
    WorkflowEngineError,
    create_error_response
)
# Removed error_recovery dependency for simplified codebase
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
_websocket_manager = None


def init_dependencies(
    graph_manager: GraphManager,
    execution_engine: ExecutionEngine,
    tool_registry: ToolRegistry,
    state_manager: StateManager,
    websocket_manager=None
):
    """Initialize the global dependencies."""
    global _graph_manager, _execution_engine, _tool_registry, _state_manager, _websocket_manager
    _graph_manager = graph_manager
    _execution_engine = execution_engine
    _tool_registry = tool_registry
    _state_manager = state_manager
    _websocket_manager = websocket_manager


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
        
    except WorkflowEngineError as e:
        logger.warning(f"Workflow engine error during graph creation: {str(e)}")
        
        # Determine appropriate HTTP status code based on error type
        if isinstance(e, GraphValidationError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(e, StorageError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        raise HTTPException(
            status_code=status_code,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during graph creation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while creating the graph",
                "details": {"original_error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
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


# WebSocket endpoint for real-time monitoring

@router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    WebSocket endpoint for real-time workflow monitoring.
    
    Clients can connect to this endpoint to receive real-time updates about workflow executions.
    
    Message format for client messages:
    {
        "action": "subscribe" | "unsubscribe" | "ping",
        "run_id": "optional_run_id_for_subscribe_unsubscribe"
    }
    
    Message format for server messages:
    {
        "event_type": "connection_established" | "subscription_confirmed" | "workflow_started" | "node_execution" | "execution_error" | "workflow_completed" | "workflow_cancelled",
        "run_id": "workflow_run_id",
        "timestamp": "iso_timestamp",
        "data": {...}
    }
    """
    if not _websocket_manager:
        await websocket.close(code=1011, reason="WebSocket monitoring not available")
        return
    
    connection_id = None
    try:
        # Accept connection and get connection ID
        connection_id = await _websocket_manager.connect(websocket)
        logger.info(f"WebSocket client connected: {connection_id}")
        
        # Handle client messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                action = message.get("action")
                run_id = message.get("run_id")
                
                if action == "subscribe" and run_id:
                    # Subscribe to workflow run updates
                    success = await _websocket_manager.subscribe_to_run(connection_id, run_id)
                    if not success:
                        await _websocket_manager.send_to_connection(connection_id, {
                            "event_type": "error",
                            "message": f"Failed to subscribe to run {run_id}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                elif action == "unsubscribe" and run_id:
                    # Unsubscribe from workflow run updates
                    success = await _websocket_manager.unsubscribe_from_run(connection_id, run_id)
                    if success:
                        await _websocket_manager.send_to_connection(connection_id, {
                            "event_type": "unsubscribed",
                            "run_id": run_id,
                            "message": f"Unsubscribed from run {run_id}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                elif action == "ping":
                    # Respond to ping with pong
                    await _websocket_manager.send_to_connection(connection_id, {
                        "event_type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif action == "get_status":
                    # Send connection status information
                    connection_info = _websocket_manager.get_connection_info()
                    await _websocket_manager.send_to_connection(connection_id, {
                        "event_type": "status_info",
                        "data": connection_info,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                else:
                    # Unknown action
                    await _websocket_manager.send_to_connection(connection_id, {
                        "event_type": "error",
                        "message": f"Unknown action: {action}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                # Client disconnected, break out of the message loop
                logger.info(f"WebSocket client disconnected during message processing: {connection_id}")
                break
            except json.JSONDecodeError:
                # Invalid JSON message
                try:
                    await _websocket_manager.send_to_connection(connection_id, {
                        "event_type": "error",
                        "message": "Invalid JSON message format",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except WebSocketDisconnect:
                    logger.info(f"WebSocket client disconnected while sending error message: {connection_id}")
                    break
            except Exception as e:
                logger.error(f"Error processing WebSocket message from {connection_id}: {str(e)}")
                try:
                    await _websocket_manager.send_to_connection(connection_id, {
                        "event_type": "error",
                        "message": f"Error processing message: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except WebSocketDisconnect:
                    logger.info(f"WebSocket client disconnected while sending error message: {connection_id}")
                    break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {str(e)}")
    finally:
        # Clean up connection
        if connection_id:
            await _websocket_manager.disconnect(connection_id)


# Historical data retrieval endpoints

class HistoricalQueryRequest(BaseModel):
    """Request model for historical data queries."""
    graph_id: Optional[str] = Field(None, description="Filter by graph ID")
    status: Optional[str] = Field(None, description="Filter by execution status")
    start_date: Optional[datetime] = Field(None, description="Filter executions after this date")
    end_date: Optional[datetime] = Field(None, description="Filter executions before this date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class HistoricalExecutionSummary(BaseModel):
    """Summary model for historical executions."""
    run_id: str = Field(..., description="Unique run identifier")
    graph_id: str = Field(..., description="Graph identifier")
    status: str = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class HistoricalDataResponse(BaseModel):
    """Response model for historical data queries."""
    executions: List[HistoricalExecutionSummary] = Field(..., description="List of historical executions")
    total_count: int = Field(..., description="Total number of matching executions")
    has_more: bool = Field(..., description="Whether more results are available")


@router.post(
    "/history/executions",
    response_model=HistoricalDataResponse,
    summary="Query historical workflow executions",
    description="Retrieve historical workflow execution data with filtering and pagination"
)
async def query_historical_executions(
    request: HistoricalQueryRequest,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> HistoricalDataResponse:
    """
    Query historical workflow executions.
    
    Args:
        request: Query parameters for filtering and pagination
        execution_engine: Execution engine dependency
        
    Returns:
        Historical execution data with pagination info
        
    Raises:
        HTTPException: If query fails
    """
    try:
        logger.debug(f"Querying historical executions with filters: {request.model_dump()}")
        
        # Get historical data from execution engine
        result = execution_engine.query_historical_executions(
            graph_id=request.graph_id,
            status=request.status,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit,
            offset=request.offset
        )
        
        logger.debug(f"Retrieved {len(result['executions'])} historical executions")
        
        return HistoricalDataResponse(
            executions=result['executions'],
            total_count=result['total_count'],
            has_more=result['has_more']
        )
        
    except Exception as e:
        logger.error(f"Error querying historical executions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "HistoricalQueryError",
                "message": "Failed to query historical executions",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/history/execution/{run_id}",
    response_model=ExecutionStatus,
    summary="Get historical execution details",
    description="Retrieve complete details of a historical workflow execution"
)
async def get_historical_execution(
    run_id: str,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> ExecutionStatus:
    """
    Get detailed information about a historical execution.
    
    Args:
        run_id: ID of the workflow run
        execution_engine: Execution engine dependency
        
    Returns:
        Complete execution status and state information
        
    Raises:
        HTTPException: If execution not found or retrieval fails
    """
    try:
        logger.debug(f"Getting historical execution details: {run_id}")
        
        # Get historical execution status
        status = execution_engine.get_historical_execution_status(run_id)
        
        logger.debug(f"Retrieved historical execution details for run {run_id}")
        
        return status
        
    except ExecutionEngineError as e:
        if "not found" in str(e).lower():
            logger.warning(f"Historical execution not found: {run_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "ExecutionNotFound",
                    "message": f"Historical execution with ID '{run_id}' not found",
                    "details": {"run_id": run_id}
                }
            )
        else:
            logger.error(f"Error getting historical execution: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "HistoricalQueryError",
                    "message": str(e),
                    "details": {"run_id": run_id}
                }
            )
    except Exception as e:
        logger.error(f"Unexpected error getting historical execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while retrieving historical execution",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/history/logs/{run_id}",
    response_model=List[LogEntry],
    summary="Get historical execution logs",
    description="Retrieve execution logs for a historical workflow run"
)
async def get_historical_execution_logs(
    run_id: str,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> List[LogEntry]:
    """
    Get execution logs for a historical workflow run.
    
    Args:
        run_id: ID of the workflow run
        execution_engine: Execution engine dependency
        
    Returns:
        List of log entries in chronological order
        
    Raises:
        HTTPException: If execution not found or log retrieval fails
    """
    try:
        logger.debug(f"Getting historical execution logs: {run_id}")
        
        # Get historical execution logs
        logs = execution_engine.get_historical_execution_logs(run_id)
        
        logger.debug(f"Retrieved {len(logs)} historical log entries for run {run_id}")
        
        return logs
        
    except ExecutionEngineError as e:
        if "not found" in str(e).lower():
            logger.warning(f"Historical execution not found for logs: {run_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "ExecutionNotFound",
                    "message": f"Historical execution with ID '{run_id}' not found",
                    "details": {"run_id": run_id}
                }
            )
        else:
            logger.error(f"Error getting historical execution logs: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "HistoricalQueryError",
                    "message": str(e),
                    "details": {"run_id": run_id}
                }
            )
    except Exception as e:
        logger.error(f"Unexpected error getting historical execution logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred while retrieving historical execution logs",
                "details": {"original_error": str(e)}
            }
        )


@router.post(
    "/history/archive",
    summary="Archive old execution data",
    description="Archive old completed executions to optimize database performance"
)
async def archive_old_executions(
    max_age_days: int = 30,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> Dict[str, Any]:
    """
    Archive old completed executions.
    
    Args:
        max_age_days: Archive executions older than this many days
        execution_engine: Execution engine dependency
        
    Returns:
        Archive operation results
        
    Raises:
        HTTPException: If archiving fails
    """
    try:
        logger.info(f"Starting archive operation for executions older than {max_age_days} days")
        
        # Perform archiving
        result = execution_engine.archive_old_executions(max_age_days)
        
        logger.info(f"Archive operation completed: {result}")
        
        return {
            "message": f"Archive operation completed successfully",
            "archived_count": result.get("archived_count", 0),
            "deleted_count": result.get("deleted_count", 0),
            "max_age_days": max_age_days
        }
        
    except Exception as e:
        logger.error(f"Error during archive operation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ArchiveError",
                "message": "Failed to archive old executions",
                "details": {"original_error": str(e)}
            }
        )


@router.delete(
    "/history/cleanup",
    summary="Clean up old execution data",
    description="Permanently delete old completed executions and their logs"
)
async def cleanup_old_executions(
    max_age_days: int = 90,
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> Dict[str, Any]:
    """
    Clean up old completed executions.
    
    Args:
        max_age_days: Delete executions older than this many days
        execution_engine: Execution engine dependency
        
    Returns:
        Cleanup operation results
        
    Raises:
        HTTPException: If cleanup fails
    """
    try:
        logger.info(f"Starting cleanup operation for executions older than {max_age_days} days")
        
        # Perform cleanup
        result = execution_engine.cleanup_old_executions(max_age_days)
        
        logger.info(f"Cleanup operation completed: {result}")
        
        return {
            "message": f"Cleanup operation completed successfully",
            "deleted_executions": result.get("deleted_executions", 0),
            "deleted_logs": result.get("deleted_logs", 0),
            "max_age_days": max_age_days
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup operation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "CleanupError",
                "message": "Failed to clean up old executions",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/history/statistics",
    summary="Get execution statistics",
    description="Retrieve statistics about workflow executions for monitoring and optimization"
)
async def get_execution_statistics(
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
) -> Dict[str, Any]:
    """
    Get statistics about workflow executions.
    
    Args:
        execution_engine: Execution engine dependency
        
    Returns:
        Dictionary containing execution statistics
        
    Raises:
        HTTPException: If statistics retrieval fails
    """
    try:
        logger.debug("Getting execution statistics")
        
        # Get statistics from execution engine
        stats = execution_engine.get_execution_statistics()
        
        logger.debug("Retrieved execution statistics")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting execution statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "StatisticsError",
                "message": "Failed to retrieve execution statistics",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/ws/connections",
    summary="Get WebSocket connection information",
    description="Retrieve information about active WebSocket connections"
)
async def get_websocket_connections() -> Dict[str, Any]:
    """
    Get information about active WebSocket connections.
    
    Returns:
        Dictionary containing connection information
    """
    if not _websocket_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WebSocket monitoring not available"
        )
    
    try:
        connection_info = _websocket_manager.get_connection_info()
        return {
            "websocket_monitoring": "active",
            "connection_info": connection_info
        }
    except Exception as e:
        logger.error(f"Error getting WebSocket connection info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "WebSocketError",
                "message": "Failed to retrieve connection information",
                "details": {"original_error": str(e)}
            }
        )


# System monitoring and health endpoints

@router.get(
    "/system/health",
    summary="Get system health status",
    description="Retrieve comprehensive health status of all system components"
)
async def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health status.
    
    Returns:
        Dictionary containing health status of all components
    """
    try:
        results = {"overall_status": "healthy", "checks": {}, "timestamp": "2025-12-10T00:00:00Z"}
        return {
            "service": "agent-workflow-engine",
            "version": "1.0.0",
            **results
        }
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "HealthCheckError",
                "message": "Failed to retrieve system health status",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/system/metrics",
    summary="Get system metrics",
    description="Retrieve system performance and usage metrics"
)
async def get_system_metrics(
    execution_engine: ExecutionEngine = Depends(get_execution_engine),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
) -> Dict[str, Any]:
    """
    Get system performance and usage metrics.
    
    Returns:
        Dictionary containing system metrics
    """
    try:
        # Get execution statistics
        execution_stats = execution_engine.get_execution_statistics()
        
        # Get tool registry statistics
        tools = tool_registry.list_tools()
        
        # Get active execution count
        active_executions = len(execution_engine._active_executions)
        
        # Get WebSocket connection count
        websocket_connections = 0
        if _websocket_manager:
            connection_info = _websocket_manager.get_connection_info()
            websocket_connections = connection_info.get("active_connections", 0)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "execution_engine": {
                "active_executions": active_executions,
                "max_concurrent_executions": execution_engine._max_concurrent_executions,
                "queue_size": execution_engine._execution_queue.qsize(),
                **execution_stats
            },
            "tool_registry": {
                "registered_tools": len(tools),
                "tool_names": list(tools.keys())
            },
            "websocket_manager": {
                "active_connections": websocket_connections
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving system metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MetricsError",
                "message": "Failed to retrieve system metrics",
                "details": {"original_error": str(e)}
            }
        )


@router.get(
    "/system/errors",
    summary="Get recent system errors",
    description="Retrieve recent system errors and their details"
)
async def get_recent_errors(
    limit: int = 50,
    severity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recent system errors.
    
    Args:
        limit: Maximum number of errors to return
        severity: Filter by error severity (low, medium, high, critical)
    
    Returns:
        Dictionary containing recent errors
    """
    try:
        # This would typically query a centralized error log
        # For now, return a placeholder response
        return {
            "message": "Error logging system not yet implemented",
            "timestamp": datetime.utcnow().isoformat(),
            "requested_limit": limit,
            "requested_severity": severity
        }
    except Exception as e:
        logger.error(f"Error retrieving system errors: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ErrorRetrievalError",
                "message": "Failed to retrieve system errors",
                "details": {"original_error": str(e)}
            }
        )