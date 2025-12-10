"""Execution Engine for workflow processing."""

import asyncio
import uuid
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from sqlalchemy.orm import Session
from queue import Queue, Empty

from ..storage.database import get_db
from ..storage.models import WorkflowRunModel, LogEntryModel
from ..models.core import (
    GraphDefinition, NodeDefinition, EdgeDefinition, WorkflowState,
    ExecutionStatus, ExecutionStatusEnum, LogEntry, LogEventType
)
from .exceptions import ExecutionEngineError, NodeExecutionError, StateManagementError
from .logging import get_logger
from .tool_registry import ToolRegistry
from .state_manager import StateManager
from .graph_manager import GraphManager

logger = get_logger(__name__)


class ExecutionContext:
    """Context for node execution containing state and tools."""
    
    def __init__(self, state: WorkflowState, tool_registry: ToolRegistry, run_id: str):
        self.state = state
        self.tool_registry = tool_registry
        self.run_id = run_id
        self.execution_metadata = {}


class ExecutionEngine:
    """Engine for executing workflow graphs with support for conditional branching, loops, and async execution."""
    
    def __init__(self, tool_registry: ToolRegistry, state_manager: StateManager, graph_manager: GraphManager, max_concurrent_executions: int = 10, websocket_manager=None):
        """Initialize the execution engine.
        
        Args:
            tool_registry: Registry for accessing workflow tools
            state_manager: Manager for workflow state persistence
            graph_manager: Manager for graph definitions
            max_concurrent_executions: Maximum number of concurrent workflow executions
            websocket_manager: Optional WebSocket manager for real-time monitoring
        """
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.graph_manager = graph_manager
        self.websocket_manager = websocket_manager
        
        # Concurrent execution management
        self._active_executions: Dict[str, Future] = {}
        self._execution_contexts: Dict[str, ExecutionContext] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_executions)
        self._max_concurrent_executions = max_concurrent_executions
        
        # Execution queue for resource management
        self._execution_queue: Queue = Queue(maxsize=100)  # Queue for pending executions
        self._queue_processor_running = False
        self._queue_processor_thread: Optional[threading.Thread] = None
        
        # Shared resource synchronization
        self._resource_locks: Dict[str, threading.RLock] = {}
        self._resource_lock_manager = threading.RLock()
        
        # Error containment and isolation
        self._run_isolation_locks: Dict[str, threading.RLock] = {}
        self._isolation_lock_manager = threading.RLock()
        
        # Start queue processor
        self._start_queue_processor()
        
        logger.info(f"ExecutionEngine initialized with max_concurrent_executions={max_concurrent_executions}")
    
    def execute_workflow(self, graph_id: str, initial_state: Dict[str, Any]) -> str:
        """
        Execute a workflow graph with the given initial state.
        
        Args:
            graph_id: ID of the graph to execute
            initial_state: Initial state data for the workflow
            
        Returns:
            Unique run ID for tracking execution
            
        Raises:
            ExecutionEngineError: If execution setup fails
        """
        try:
            # Generate unique run ID
            run_id = str(uuid.uuid4())
            
            # Get graph definition to validate it exists
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                raise ExecutionEngineError(f"Graph {graph_id} not found")
            
            # Create run state
            self.state_manager.create_run_state(run_id, graph_id, initial_state)
            
            # Check if we can start execution immediately or need to queue it
            if len(self._active_executions) < self._max_concurrent_executions:
                # Start execution immediately
                self._start_execution_immediately(run_id, graph_id, initial_state)
                logger.info(f"Started workflow execution immediately: run_id={run_id}, graph_id={graph_id}")
            else:
                # Queue the execution
                try:
                    self._execution_queue.put((run_id, graph_id, initial_state), timeout=1.0)
                    logger.info(f"Queued workflow execution: run_id={run_id}, graph_id={graph_id}")
                except Exception as queue_error:
                    # If queue is full, reject the execution
                    self.state_manager.finalize_run(
                        run_id, ExecutionStatusEnum.FAILED,
                        error_message="Execution queue is full, please try again later"
                    )
                    raise ExecutionEngineError("Execution queue is full, please try again later")
            
            return run_id
            
        except Exception as e:
            if isinstance(e, ExecutionEngineError):
                raise
            raise ExecutionEngineError(f"Failed to start workflow execution: {str(e)}")
    
    def _start_execution_immediately(self, run_id: str, graph_id: str, initial_state: Dict[str, Any]) -> None:
        """
        Start workflow execution immediately (called from queue processor or directly).
        
        Args:
            run_id: Unique run ID
            graph_id: ID of the graph to execute
            initial_state: Initial state data
            
        Raises:
            ExecutionEngineError: If execution setup fails
        """
        try:
            # Get graph definition
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                raise ExecutionEngineError(f"Graph {graph_id} not found")
            
            # Create execution context with isolation
            workflow_state = self.state_manager.get_state(run_id)
            context = ExecutionContext(workflow_state, self.tool_registry, run_id)
            
            # Store context with isolation lock
            with self._get_run_isolation_lock(run_id):
                self._execution_contexts[run_id] = context
            
            # Start asynchronous execution with error containment
            future = self._executor.submit(self._execute_graph_with_isolation, run_id, graph)
            self._active_executions[run_id] = future
            
            # Log workflow start
            self._log_execution_event(
                run_id, None, LogEventType.WORKFLOW_START,
                f"Started workflow execution for graph {graph_id}"
            )
            
            # Broadcast workflow start event
            if self.websocket_manager:
                self.websocket_manager.queue_workflow_event(
                    run_id, "workflow_started",
                    {"graph_id": graph_id, "message": f"Started workflow execution for graph {graph_id}"}
                )
            
        except Exception as e:
            # Clean up on failure
            self._cleanup_execution(run_id)
            raise
    
    def get_execution_status(self, run_id: str) -> ExecutionStatus:
        """
        Get the current execution status of a workflow run.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            Current execution status
            
        Raises:
            ExecutionEngineError: If run not found
        """
        try:
            # Get run from database
            db = next(get_db())
            try:
                run_model = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
                if not run_model:
                    raise ExecutionEngineError(f"Run {run_id} not found")
                
                # Get current state
                current_state = self.state_manager.get_state(run_id)
                
                # Create ExecutionStatus object
                status = ExecutionStatus(
                    run_id=run_id,
                    graph_id=run_model.graph_id,
                    status=ExecutionStatusEnum(run_model.status),
                    current_state=current_state,
                    started_at=run_model.started_at,
                    completed_at=run_model.completed_at,
                    error_message=run_model.error_message
                )
                
                return status
                
            finally:
                db.close()
                
        except Exception as e:
            if isinstance(e, ExecutionEngineError):
                raise
            raise ExecutionEngineError(f"Failed to get execution status: {str(e)}")
    
    def get_execution_logs(self, run_id: str) -> List[LogEntry]:
        """
        Get execution logs for a workflow run.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            List of log entries in chronological order
            
        Raises:
            ExecutionEngineError: If run not found
        """
        try:
            db = next(get_db())
            try:
                log_models = (
                    db.query(LogEntryModel)
                    .filter(LogEntryModel.run_id == run_id)
                    .order_by(LogEntryModel.timestamp)
                    .all()
                )
                
                logs = []
                for log_model in log_models:
                    log_entry = LogEntry(
                        timestamp=log_model.timestamp,
                        run_id=log_model.run_id,
                        node_id=log_model.node_id or "",
                        event_type=LogEventType(log_model.event_type),
                        message=log_model.message,
                        state_snapshot=log_model.state_snapshot
                    )
                    logs.append(log_entry)
                
                return logs
                
            finally:
                db.close()
                
        except Exception as e:
            if isinstance(e, ExecutionEngineError):
                raise
            raise ExecutionEngineError(f"Failed to get execution logs: {str(e)}")
    
    def cancel_execution(self, run_id: str) -> bool:
        """
        Cancel a running workflow execution.
        
        Args:
            run_id: ID of the workflow run to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        try:
            if run_id not in self._active_executions:
                logger.warning(f"Attempted to cancel non-active execution: {run_id}")
                return False
            
            # Cancel the future
            future = self._active_executions[run_id]
            cancelled = future.cancel()
            
            if cancelled:
                # Update run status
                self.state_manager.finalize_run(
                    run_id, ExecutionStatusEnum.CANCELLED, 
                    error_message="Execution cancelled by user"
                )
                
                # Clean up
                self._cleanup_execution(run_id)
                
                # Log cancellation
                self._log_execution_event(
                    run_id, None, LogEventType.WORKFLOW_COMPLETE,
                    f"Workflow execution cancelled"
                )
                
                # Broadcast cancellation
                if self.websocket_manager:
                    self.websocket_manager.queue_workflow_event(
                        run_id, "workflow_cancelled",
                        {"message": "Workflow execution cancelled"}
                    )
                
                logger.info(f"Cancelled workflow execution: {run_id}")
            
            return cancelled
            
        except Exception as e:
            logger.error(f"Failed to cancel execution {run_id}: {str(e)}")
            return False
    
    def _execute_graph_with_isolation(self, run_id: str, graph: GraphDefinition) -> None:
        """
        Execute a workflow graph with proper isolation and error containment.
        
        Args:
            run_id: ID of the workflow run
            graph: Graph definition to execute
        """
        # Acquire isolation lock for this run
        isolation_lock = self._get_run_isolation_lock(run_id)
        
        try:
            with isolation_lock:
                # Execute the graph with error containment
                self._execute_graph(run_id, graph)
                
        except Exception as e:
            # Error containment - ensure this run's failure doesn't affect others
            error_message = f"Isolated execution failed: {str(e)}"
            logger.error(f"Workflow execution failed for run {run_id}: {error_message}")
            
            try:
                # Finalize the failed run in isolation
                with isolation_lock:
                    self.state_manager.finalize_run(
                        run_id, ExecutionStatusEnum.FAILED,
                        error_message=error_message
                    )
            except Exception as finalize_error:
                logger.error(f"Failed to finalize failed run {run_id}: {str(finalize_error)}")
            
            # Log execution failure
            self._log_execution_event(
                run_id, None, LogEventType.WORKFLOW_COMPLETE,
                f"Workflow execution failed: {error_message}"
            )
            
            # Broadcast workflow failure
            if self.websocket_manager:
                self.websocket_manager.queue_error(run_id, error_message)
            
        finally:
            # Always clean up execution resources
            self._cleanup_execution(run_id)
            self._cleanup_run_isolation_lock(run_id)
    
    def _execute_graph(self, run_id: str, graph: GraphDefinition) -> None:
        """
        Execute a workflow graph (runs in background thread).
        
        Args:
            run_id: ID of the workflow run
            graph: Graph definition to execute
        """
        try:
            # Update run status to running
            db = next(get_db())
            try:
                run_model = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
                if run_model:
                    run_model.status = ExecutionStatusEnum.RUNNING.value
                    db.commit()
            finally:
                db.close()
            
            # Get execution context with isolation
            context = None
            isolation_lock = self._get_run_isolation_lock(run_id)
            with isolation_lock:
                context = self._execution_contexts.get(run_id)
                if not context:
                    raise ExecutionEngineError(f"Execution context not found for run {run_id}")
            
            # Start execution from entry point
            current_node_id = graph.entry_point
            visited_nodes = set()
            loop_counters = {}
            
            while current_node_id:
                # Check for infinite loops
                if current_node_id in loop_counters:
                    loop_counters[current_node_id] += 1
                    if loop_counters[current_node_id] > 1000:  # Prevent infinite loops
                        raise ExecutionEngineError(f"Infinite loop detected at node {current_node_id}")
                else:
                    loop_counters[current_node_id] = 1
                
                # Find node definition
                node = self._find_node(graph, current_node_id)
                if not node:
                    raise ExecutionEngineError(f"Node {current_node_id} not found in graph")
                
                # Execute node with isolation
                self._execute_node_with_isolation(run_id, node, context)
                
                # Check exit conditions with isolation
                with isolation_lock:
                    if self._check_exit_conditions(graph.exit_conditions, context.state):
                        logger.info(f"Exit condition met for run {run_id}")
                        break
                    
                    # Determine next node
                    current_node_id = self._get_next_node(graph, current_node_id, context.state)
            
            # Finalize successful execution
            with isolation_lock:
                final_state = context.state.model_dump()
            self.state_manager.finalize_run(run_id, ExecutionStatusEnum.COMPLETED, final_state)
            
            # Broadcast workflow completion
            if self.websocket_manager:
                self.websocket_manager.queue_workflow_event(
                    run_id, "workflow_completed",
                    {"final_state": final_state, "message": "Workflow execution completed successfully"}
                )
            
            logger.info(f"Workflow execution completed successfully: {run_id}")
            
        except Exception as e:
            # Re-raise to be handled by the isolation wrapper
            raise
    
    def _execute_node_with_isolation(self, run_id: str, node: NodeDefinition, context: ExecutionContext) -> None:
        """
        Execute a single workflow node with proper isolation.
        
        Args:
            run_id: ID of the workflow run
            node: Node definition to execute
            context: Execution context
            
        Raises:
            NodeExecutionError: If node execution fails
        """
        isolation_lock = self._get_run_isolation_lock(run_id)
        
        try:
            with isolation_lock:
                # Execute the node with error containment
                self._execute_node(run_id, node, context)
                
        except Exception as e:
            # Error containment for node execution
            error_message = f"Isolated node execution failed: {str(e)}"
            logger.error(f"Node {node.id} execution failed for run {run_id}: {error_message}")
            
            # Log node error with isolation
            with isolation_lock:
                self._log_execution_event(
                    run_id, node.id, LogEventType.NODE_ERROR,
                    error_message
                )
            
            raise NodeExecutionError(error_message)
    
    def _execute_node(self, run_id: str, node: NodeDefinition, context: ExecutionContext) -> None:
        """
        Execute a single workflow node.
        
        Args:
            run_id: ID of the workflow run
            node: Node definition to execute
            context: Execution context
            
        Raises:
            NodeExecutionError: If node execution fails
        """
        try:
            # Log node start
            self._log_execution_event(
                run_id, node.id, LogEventType.NODE_START,
                f"Starting execution of node {node.id}"
            )
            
            # Broadcast node start
            if self.websocket_manager:
                self.websocket_manager.queue_node_execution(
                    run_id, node.id, LogEventType.NODE_START,
                    f"Starting execution of node {node.id}"
                )
            
            # Update current node in state
            context.state.current_node = node.id
            if node.id not in context.state.execution_path:
                context.state.execution_path.append(node.id)
            
            # Update state in manager
            self.state_manager.update_state(run_id, {}, node.id)
            
            # Get the tool function
            tool_function = self.tool_registry.get_tool(node.function_name)
            
            # Prepare function arguments with execution engine reference for resource locking
            kwargs = node.parameters.copy() if node.parameters else {}
            kwargs['state'] = context.state.data
            kwargs['context'] = context
            kwargs['execution_engine'] = self  # Allow nodes to access resource locking
            
            # Execute the function with timeout if specified
            if node.timeout:
                # Use asyncio for timeout handling
                result = self._execute_with_timeout(tool_function, kwargs, node.timeout)
            else:
                result = tool_function(**kwargs)
            
            # Handle function result
            if isinstance(result, dict):
                # Update state with returned data
                context.state.data.update(result)
                self.state_manager.update_state(run_id, result, node.id)
            
            # Log node completion
            self._log_execution_event(
                run_id, node.id, LogEventType.NODE_COMPLETE,
                f"Completed execution of node {node.id}",
                context.state.model_dump()
            )
            
            # Broadcast node completion
            if self.websocket_manager:
                self.websocket_manager.queue_node_execution(
                    run_id, node.id, LogEventType.NODE_COMPLETE,
                    f"Completed execution of node {node.id}",
                    context.state.model_dump()
                )
            
            logger.debug(f"Successfully executed node {node.id} for run {run_id}")
            
        except Exception as e:
            error_message = f"Node {node.id} execution failed: {str(e)}"
            
            # Log node error
            self._log_execution_event(
                run_id, node.id, LogEventType.NODE_ERROR,
                error_message
            )
            
            # Broadcast node error
            if self.websocket_manager:
                self.websocket_manager.queue_node_execution(
                    run_id, node.id, LogEventType.NODE_ERROR,
                    error_message
                )
            
            raise NodeExecutionError(error_message)
    
    def _execute_with_timeout(self, function: Callable, kwargs: Dict[str, Any], timeout: int) -> Any:
        """
        Execute a function with timeout.
        
        Args:
            function: Function to execute
            kwargs: Function arguments
            timeout: Timeout in seconds
            
        Returns:
            Function result
            
        Raises:
            NodeExecutionError: If execution times out or fails
        """
        try:
            # Create a future for the function execution
            future = self._executor.submit(function, **kwargs)
            
            # Wait for completion with timeout
            result = future.result(timeout=timeout)
            return result
            
        except TimeoutError:
            raise NodeExecutionError(f"Function execution timed out after {timeout} seconds")
        except Exception as e:
            raise NodeExecutionError(f"Function execution failed: {str(e)}")
    
    def _find_node(self, graph: GraphDefinition, node_id: str) -> Optional[NodeDefinition]:
        """
        Find a node definition by ID in the graph.
        
        Args:
            graph: Graph definition
            node_id: ID of the node to find
            
        Returns:
            Node definition if found, None otherwise
        """
        for node in graph.nodes:
            if node.id == node_id:
                return node
        return None
    
    def _get_next_node(self, graph: GraphDefinition, current_node_id: str, state: WorkflowState) -> Optional[str]:
        """
        Determine the next node to execute based on edges and conditions.
        
        Args:
            graph: Graph definition
            current_node_id: ID of the current node
            state: Current workflow state
            
        Returns:
            ID of the next node to execute, or None if no next node
        """
        # Find outgoing edges from current node
        outgoing_edges = [edge for edge in graph.edges if edge.from_node == current_node_id]
        
        if not outgoing_edges:
            # No outgoing edges, execution ends
            return None
        
        # Evaluate conditions for conditional branching
        for edge in outgoing_edges:
            if edge.condition:
                # Evaluate condition
                if self._evaluate_condition(edge.condition, state):
                    return edge.to_node
            else:
                # Unconditional edge
                return edge.to_node
        
        # No conditions matched, execution ends
        return None
    
    def _evaluate_condition(self, condition: str, state: WorkflowState) -> bool:
        """
        Evaluate a condition string against the current state.
        
        Args:
            condition: Condition string to evaluate
            state: Current workflow state
            
        Returns:
            True if condition is met, False otherwise
        """
        try:
            # Create a safe evaluation context
            eval_context = {
                'state': state.data,
                'metadata': state.metadata,
                'execution_path': state.execution_path,
                'current_node': state.current_node,
                # Add safe built-in functions
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'isinstance': isinstance,
            }
            
            # Evaluate the condition
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {str(e)}")
            return False
    
    def _check_exit_conditions(self, exit_conditions: List[str], state: WorkflowState) -> bool:
        """
        Check if any exit conditions are met.
        
        Args:
            exit_conditions: List of exit condition strings
            state: Current workflow state
            
        Returns:
            True if any exit condition is met, False otherwise
        """
        if not exit_conditions:
            return False
        
        for condition in exit_conditions:
            if self._evaluate_condition(condition, state):
                return True
        
        return False
    
    def _log_execution_event(self, run_id: str, node_id: Optional[str], 
                           event_type: LogEventType, message: str, 
                           state_snapshot: Optional[Dict] = None) -> None:
        """
        Log an execution event to the database.
        
        Args:
            run_id: ID of the workflow run
            node_id: ID of the node (optional)
            event_type: Type of event
            message: Log message
            state_snapshot: State snapshot (optional)
        """
        try:
            db = next(get_db())
            try:
                log_entry = LogEntryModel(
                    run_id=run_id,
                    node_id=node_id,
                    event_type=event_type.value,
                    message=message,
                    state_snapshot=state_snapshot,
                    timestamp=datetime.utcnow()
                )
                
                db.add(log_entry)
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to log execution event: {str(e)}")
            # Don't raise exception to avoid breaking execution
    
    def _cleanup_execution(self, run_id: str) -> None:
        """
        Clean up resources for a completed execution.
        
        Args:
            run_id: ID of the workflow run
        """
        try:
            # Remove from active executions
            self._active_executions.pop(run_id, None)
            
            # Remove execution context with isolation
            isolation_lock = self._get_run_isolation_lock(run_id)
            with isolation_lock:
                self._execution_contexts.pop(run_id, None)
            
            logger.debug(f"Cleaned up execution resources for run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup execution {run_id}: {str(e)}")
    
    def get_active_executions(self) -> List[str]:
        """
        Get list of currently active execution run IDs.
        
        Returns:
            List of active run IDs
        """
        return list(self._active_executions.keys())
    
    def is_execution_active(self, run_id: str) -> bool:
        """
        Check if a workflow execution is currently active.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            True if execution is active, False otherwise
        """
        return run_id in self._active_executions
    
    def get_execution_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the execution queue.
        
        Returns:
            Dictionary containing queue status information
        """
        return {
            "active_executions": len(self._active_executions),
            "max_concurrent_executions": self._max_concurrent_executions,
            "queued_executions": self._execution_queue.qsize(),
            "queue_capacity": self._execution_queue.maxsize,
            "queue_processor_running": self._queue_processor_running,
            "available_slots": max(0, self._max_concurrent_executions - len(self._active_executions))
        }
    
    def get_concurrent_execution_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about concurrent execution performance.
        
        Returns:
            Dictionary containing execution metrics
        """
        active_runs = list(self._active_executions.keys())
        
        return {
            "total_active_runs": len(active_runs),
            "active_run_ids": active_runs,
            "total_isolation_locks": len(self._run_isolation_locks),
            "total_resource_locks": len(self._resource_locks),
            "execution_contexts": len(self._execution_contexts),
            "queue_size": self._execution_queue.qsize(),
            "max_concurrent_limit": self._max_concurrent_executions
        }
    
    def _start_queue_processor(self) -> None:
        """Start the execution queue processor thread."""
        if not self._queue_processor_running:
            self._queue_processor_running = True
            self._queue_processor_thread = threading.Thread(
                target=self._process_execution_queue,
                daemon=True,
                name="ExecutionQueueProcessor"
            )
            self._queue_processor_thread.start()
            logger.info("Execution queue processor started")
    
    def _process_execution_queue(self) -> None:
        """Process the execution queue in a background thread."""
        while self._queue_processor_running:
            try:
                # Check if we can start more executions
                if len(self._active_executions) >= self._max_concurrent_executions:
                    time.sleep(0.1)  # Wait before checking again
                    continue
                
                # Try to get a queued execution
                try:
                    queued_execution = self._execution_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Start the queued execution
                run_id, graph_id, initial_state = queued_execution
                try:
                    self._start_execution_immediately(run_id, graph_id, initial_state)
                    logger.info(f"Started queued execution: {run_id}")
                except Exception as e:
                    logger.error(f"Failed to start queued execution {run_id}: {str(e)}")
                    # Mark the run as failed
                    try:
                        self.state_manager.finalize_run(
                            run_id, ExecutionStatusEnum.FAILED,
                            error_message=f"Failed to start execution: {str(e)}"
                        )
                    except Exception as finalize_error:
                        logger.error(f"Failed to finalize failed queued run {run_id}: {str(finalize_error)}")
                
            except Exception as e:
                logger.error(f"Error in execution queue processor: {str(e)}")
                time.sleep(1.0)  # Wait before continuing
    
    def _get_run_isolation_lock(self, run_id: str) -> threading.RLock:
        """Get or create an isolation lock for a specific run."""
        with self._isolation_lock_manager:
            if run_id not in self._run_isolation_locks:
                self._run_isolation_locks[run_id] = threading.RLock()
            return self._run_isolation_locks[run_id]
    
    def _cleanup_run_isolation_lock(self, run_id: str) -> None:
        """Clean up the isolation lock for a completed run."""
        with self._isolation_lock_manager:
            self._run_isolation_locks.pop(run_id, None)
    
    def _get_resource_lock(self, resource_name: str) -> threading.RLock:
        """Get or create a lock for a shared resource."""
        with self._resource_lock_manager:
            if resource_name not in self._resource_locks:
                self._resource_locks[resource_name] = threading.RLock()
            return self._resource_locks[resource_name]
    
    def acquire_shared_resource_lock(self, resource_name: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire a lock for a shared resource.
        
        Args:
            resource_name: Name of the shared resource
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            True if lock was acquired, False if timeout occurred
        """
        try:
            resource_lock = self._get_resource_lock(resource_name)
            acquired = resource_lock.acquire(timeout=timeout if timeout else -1)
            if acquired:
                logger.debug(f"Acquired lock for shared resource: {resource_name}")
            else:
                logger.warning(f"Failed to acquire lock for shared resource: {resource_name}")
            return acquired
        except Exception as e:
            logger.error(f"Error acquiring resource lock for {resource_name}: {str(e)}")
            return False
    
    def release_shared_resource_lock(self, resource_name: str) -> None:
        """
        Release a lock for a shared resource.
        
        Args:
            resource_name: Name of the shared resource
        """
        try:
            resource_lock = self._get_resource_lock(resource_name)
            resource_lock.release()
            logger.debug(f"Released lock for shared resource: {resource_name}")
        except Exception as e:
            logger.error(f"Error releasing resource lock for {resource_name}: {str(e)}")
    
    def shutdown(self) -> None:
        """
        Shutdown the execution engine and clean up resources.
        """
        try:
            # Stop queue processor
            self._queue_processor_running = False
            if self._queue_processor_thread and self._queue_processor_thread.is_alive():
                self._queue_processor_thread.join(timeout=5.0)
                logger.info("Execution queue processor stopped")
            
            # Cancel all active executions
            for run_id in list(self._active_executions.keys()):
                self.cancel_execution(run_id)
            
            # Shutdown thread pool
            self._executor.shutdown(wait=True)
            
            # Clean up locks
            with self._resource_lock_manager:
                self._resource_locks.clear()
            
            with self._isolation_lock_manager:
                self._run_isolation_locks.clear()
            
            logger.info("ExecutionEngine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during ExecutionEngine shutdown: {str(e)}")