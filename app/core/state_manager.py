"""State management for workflow executions."""

import json
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..storage.database import get_db
from ..storage.models import WorkflowRunModel, LogEntryModel
from ..models.core import WorkflowState, ExecutionStatusEnum, LogEventType
from .exceptions import StateManagementError, StorageError, TransientError
from .logging import get_logger, set_logging_context, clear_logging_context
from .error_recovery import with_retry, RetryConfig

logger = get_logger(__name__)


class StateSnapshot:
    """Represents a point-in-time snapshot of workflow state."""
    
    def __init__(self, run_id: str, timestamp: datetime, state: WorkflowState, node_id: Optional[str] = None):
        self.run_id = run_id
        self.timestamp = timestamp
        self.state = state
        self.node_id = node_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary representation."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.model_dump(),
            "node_id": self.node_id
        }


class StateManager:
    """Manages workflow state persistence, isolation, and consistency."""
    
    def __init__(self):
        """Initialize the StateManager."""
        self._active_runs: Dict[str, WorkflowState] = {}
        self._run_locks: Dict[str, threading.RLock] = {}
        self._state_lock_manager = threading.RLock()
        logger.info("StateManager initialized")
    
    @with_retry(RetryConfig(max_attempts=3, retryable_exceptions=[StorageError, TransientError]))
    def create_run_state(self, run_id: str, graph_id: str, initial_state: Dict[str, Any]) -> None:
        """
        Create a new workflow run with initial state.
        
        Args:
            run_id: Unique identifier for the workflow run
            graph_id: ID of the graph being executed
            initial_state: Initial state data for the workflow
            
        Raises:
            StateManagementError: If run creation fails
            StorageError: If database operations fail
        """
        # Set logging context
        set_logging_context(run_id=run_id, graph_id=graph_id, operation="create_run_state")
        
        try:
            # Validate inputs
            if not run_id or not run_id.strip():
                raise StateManagementError(
                    "Run ID cannot be empty",
                    run_id=run_id,
                    operation="create_run_state"
                )
            
            if not graph_id or not graph_id.strip():
                raise StateManagementError(
                    "Graph ID cannot be empty",
                    run_id=run_id,
                    operation="create_run_state"
                )
            
            # Check if run already exists
            with self._state_lock_manager:
                if run_id in self._active_runs:
                    raise StateManagementError(
                        f"Run {run_id} already exists",
                        run_id=run_id,
                        operation="create_run_state"
                    )
            
            # Create WorkflowState object
            workflow_state = WorkflowState(
                data=initial_state or {},
                metadata={"created_at": datetime.utcnow().isoformat()},
                current_node=None,
                execution_path=[]
            )
            
            # Store in memory for active runs with proper locking
            with self._state_lock_manager:
                self._active_runs[run_id] = workflow_state
                self._run_locks[run_id] = threading.RLock()
            
            # Persist to database with error handling
            self._persist_run_creation(run_id, graph_id, initial_state, workflow_state)
            
            logger.info(f"Successfully created run state for {run_id}")
            
        except (StateManagementError, StorageError):
            # Clean up on failure
            with self._state_lock_manager:
                self._active_runs.pop(run_id, None)
                self._run_locks.pop(run_id, None)
            raise
        except Exception as e:
            # Clean up on failure
            with self._state_lock_manager:
                self._active_runs.pop(run_id, None)
                self._run_locks.pop(run_id, None)
            logger.error(f"Unexpected error creating run state: {str(e)}", exc_info=True)
            raise StateManagementError(
                f"Failed to create run state: {str(e)}",
                run_id=run_id,
                operation="create_run_state"
            )
        finally:
            clear_logging_context()
    
    def _persist_run_creation(self, run_id: str, graph_id: str, initial_state: Dict[str, Any], workflow_state: WorkflowState) -> None:
        """Persist run creation to database with error handling."""
        db = next(get_db())
        try:
            run_model = WorkflowRunModel(
                id=run_id,
                graph_id=graph_id,
                status=ExecutionStatusEnum.PENDING.value,
                initial_state=initial_state,
                current_state=workflow_state.model_dump(),
                final_state=None,
                current_node=None,
                execution_path=[],
                started_at=datetime.utcnow()
            )
            
            db.add(run_model)
            
            # Log the state creation
            self._log_state_event(
                db, run_id, None, LogEventType.WORKFLOW_START,
                f"Workflow run {run_id} created with initial state",
                workflow_state.model_dump()
            )
            
            db.commit()
            
            logger.info(f"Created workflow run state: {run_id}")
            
        except SQLAlchemyError as e:
            db.rollback()
            raise StorageError(f"Failed to persist run state: {str(e)}")
        except Exception as e:
            db.rollback()
            raise StorageError(f"Failed to persist run state: {str(e)}")
        finally:
            db.close()
    
    def update_state(self, run_id: str, updates: Dict[str, Any], node_id: Optional[str] = None) -> None:
        """
        Update the state of a workflow run.
        
        Args:
            run_id: ID of the workflow run
            updates: Dictionary of state updates to apply
            node_id: ID of the node making the update (optional)
            
        Raises:
            StateManagementError: If state update fails
            StorageError: If database operations fail
        """
        try:
            if not run_id or run_id not in self._active_runs:
                raise StateManagementError(f"Run {run_id} not found or not active")
            
            # Get the run-specific lock
            run_lock = None
            with self._state_lock_manager:
                run_lock = self._run_locks.get(run_id)
                if not run_lock:
                    raise StateManagementError(f"Run {run_id} lock not found")
            
            # Acquire the run-specific lock for this update
            with run_lock:
            
                # Get current state
                current_state = self._active_runs[run_id]
                
                # Apply updates to data
                if updates:
                    current_state.data.update(updates)
                
                # Update metadata
                current_state.metadata["last_updated"] = datetime.utcnow().isoformat()
                if node_id:
                    current_state.metadata["last_updated_by"] = node_id
                
                # Update current node if provided
                if node_id and node_id != current_state.current_node:
                    current_state.current_node = node_id
                    if node_id not in current_state.execution_path:
                        current_state.execution_path.append(node_id)
                
                # Persist to database
                db = next(get_db())
                try:
                    run_model = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
                    if not run_model:
                        raise StateManagementError(f"Run {run_id} not found in database")
                    
                    run_model.current_state = current_state.model_dump()
                    run_model.current_node = current_state.current_node
                    run_model.execution_path = current_state.execution_path
                    
                    # Log the state update
                    self._log_state_event(
                        db, run_id, node_id, LogEventType.STATE_UPDATE,
                        f"State updated by node {node_id}" if node_id else "State updated",
                        current_state.model_dump()
                    )
                    
                    db.commit()
                    
                    logger.debug(f"Updated state for run {run_id}")
                    
                except SQLAlchemyError as e:
                    db.rollback()
                    raise StorageError(f"Failed to persist state update: {str(e)}")
                finally:
                    db.close()
                
        except Exception as e:
            if isinstance(e, (StateManagementError, StorageError)):
                raise
            raise StateManagementError(f"Failed to update state: {str(e)}")
    
    def get_state(self, run_id: str) -> WorkflowState:
        """
        Get the current state of a workflow run.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            Current workflow state
            
        Raises:
            StateManagementError: If run not found
        """
        try:
            # First check active runs (in-memory)
            if run_id in self._active_runs:
                return self._active_runs[run_id].model_copy(deep=True)
            
            # If not in memory, try to load from database
            db = next(get_db())
            try:
                run_model = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
                if not run_model:
                    raise StateManagementError(f"Run {run_id} not found")
                
                # Reconstruct WorkflowState from database
                state_data = run_model.current_state or {}
                workflow_state = WorkflowState(**state_data)
                
                logger.debug(f"Retrieved state for run {run_id} from database")
                return workflow_state
                
            except SQLAlchemyError as e:
                raise StorageError(f"Failed to retrieve state from database: {str(e)}")
            finally:
                db.close()
                
        except Exception as e:
            if isinstance(e, (StateManagementError, StorageError)):
                raise
            raise StateManagementError(f"Failed to get state: {str(e)}")
    
    def get_state_history(self, run_id: str) -> List[StateSnapshot]:
        """
        Get the history of state changes for a workflow run.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            List of state snapshots in chronological order
            
        Raises:
            StateManagementError: If run not found
            StorageError: If database operations fail
        """
        try:
            db = next(get_db())
            try:
                # Get all log entries with state snapshots for this run
                log_entries = (
                    db.query(LogEntryModel)
                    .filter(LogEntryModel.run_id == run_id)
                    .filter(LogEntryModel.state_snapshot.isnot(None))
                    .order_by(LogEntryModel.timestamp)
                    .all()
                )
                
                snapshots = []
                for entry in log_entries:
                    if entry.state_snapshot:
                        try:
                            state = WorkflowState(**entry.state_snapshot)
                            snapshot = StateSnapshot(
                                run_id=run_id,
                                timestamp=entry.timestamp,
                                state=state,
                                node_id=entry.node_id
                            )
                            snapshots.append(snapshot)
                        except Exception as e:
                            logger.warning(f"Failed to reconstruct state snapshot: {str(e)}")
                            continue
                
                logger.debug(f"Retrieved {len(snapshots)} state snapshots for run {run_id}")
                return snapshots
                
            except SQLAlchemyError as e:
                raise StorageError(f"Failed to retrieve state history: {str(e)}")
            finally:
                db.close()
                
        except Exception as e:
            if isinstance(e, (StateManagementError, StorageError)):
                raise
            raise StateManagementError(f"Failed to get state history: {str(e)}")
    
    def finalize_run(self, run_id: str, status: ExecutionStatusEnum, final_state: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None) -> None:
        """
        Finalize a workflow run and clean up resources.
        
        Args:
            run_id: ID of the workflow run
            status: Final execution status
            final_state: Final state data (optional)
            error_message: Error message if run failed (optional)
            
        Raises:
            StateManagementError: If finalization fails
            StorageError: If database operations fail
        """
        try:
            if run_id not in self._active_runs:
                logger.warning(f"Attempting to finalize non-active run: {run_id}")
            
            # Update database with final status
            db = next(get_db())
            try:
                run_model = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
                if not run_model:
                    raise StateManagementError(f"Run {run_id} not found in database")
                
                run_model.status = status.value
                run_model.completed_at = datetime.utcnow()
                run_model.error_message = error_message
                
                if final_state:
                    run_model.final_state = final_state
                elif run_id in self._active_runs:
                    run_model.final_state = self._active_runs[run_id].model_dump()
                
                # Log the completion
                self._log_state_event(
                    db, run_id, None, LogEventType.WORKFLOW_COMPLETE,
                    f"Workflow run {run_id} completed with status: {status.value}",
                    run_model.final_state
                )
                
                db.commit()
                
                logger.info(f"Finalized workflow run {run_id} with status: {status.value}")
                
            except SQLAlchemyError as e:
                db.rollback()
                raise StorageError(f"Failed to finalize run: {str(e)}")
            finally:
                db.close()
            
            # Clean up in-memory state
            with self._state_lock_manager:
                self._active_runs.pop(run_id, None)
                self._run_locks.pop(run_id, None)
            
        except Exception as e:
            if isinstance(e, (StateManagementError, StorageError)):
                raise
            raise StateManagementError(f"Failed to finalize run: {str(e)}")
    
    def validate_state_consistency(self, run_id: str) -> bool:
        """
        Validate the consistency of a workflow run's state.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            True if state is consistent, False otherwise
            
        Raises:
            StateManagementError: If validation fails
        """
        try:
            # Get current state
            current_state = self.get_state(run_id)
            
            # Basic consistency checks
            if not isinstance(current_state.data, dict):
                logger.error(f"State data is not a dictionary for run {run_id}")
                return False
            
            if not isinstance(current_state.metadata, dict):
                logger.error(f"State metadata is not a dictionary for run {run_id}")
                return False
            
            if not isinstance(current_state.execution_path, list):
                logger.error(f"Execution path is not a list for run {run_id}")
                return False
            
            # Check if current_node is in execution_path (if set)
            if current_state.current_node and current_state.current_node not in current_state.execution_path:
                logger.error(f"Current node {current_state.current_node} not in execution path for run {run_id}")
                return False
            
            # Validate against database state
            db = next(get_db())
            try:
                run_model = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
                if not run_model:
                    logger.error(f"Run {run_id} not found in database")
                    return False
                
                # Check if in-memory state matches database state
                if run_id in self._active_runs:
                    db_state = WorkflowState(**run_model.current_state) if run_model.current_state else None
                    if db_state and db_state.model_dump() != current_state.model_dump():
                        logger.error(f"In-memory state does not match database state for run {run_id}")
                        return False
                
                logger.debug(f"State consistency validation passed for run {run_id}")
                return True
                
            except SQLAlchemyError as e:
                raise StorageError(f"Failed to validate state consistency: {str(e)}")
            finally:
                db.close()
                
        except Exception as e:
            if isinstance(e, (StateManagementError, StorageError)):
                raise
            raise StateManagementError(f"Failed to validate state consistency: {str(e)}")
    
    def recover_state(self, run_id: str) -> bool:
        """
        Attempt to recover state consistency for a workflow run.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            True if recovery was successful, False otherwise
            
        Raises:
            StateManagementError: If recovery fails
        """
        try:
            logger.info(f"Attempting state recovery for run {run_id}")
            
            # First, try to load state from database
            db = next(get_db())
            try:
                run_model = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
                if not run_model:
                    logger.error(f"Cannot recover: Run {run_id} not found in database")
                    return False
                
                # Reconstruct state from database
                if run_model.current_state:
                    try:
                        recovered_state = WorkflowState(**run_model.current_state)
                        
                        # Update in-memory state
                        with self._state_lock_manager:
                            self._active_runs[run_id] = recovered_state
                            self._run_locks[run_id] = threading.RLock()
                        
                        # Validate the recovered state
                        if self.validate_state_consistency(run_id):
                            logger.info(f"Successfully recovered state for run {run_id}")
                            return True
                        else:
                            logger.error(f"Recovered state failed consistency validation for run {run_id}")
                            return False
                            
                    except Exception as e:
                        logger.error(f"Failed to reconstruct state from database: {str(e)}")
                        return False
                else:
                    logger.error(f"No current state found in database for run {run_id}")
                    return False
                    
            except SQLAlchemyError as e:
                raise StorageError(f"Failed to recover state: {str(e)}")
            finally:
                db.close()
                
        except Exception as e:
            if isinstance(e, (StateManagementError, StorageError)):
                raise
            raise StateManagementError(f"Failed to recover state: {str(e)}")
    
    def cleanup_completed_runs(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed workflow runs older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for completed runs
            
        Returns:
            Number of runs cleaned up
            
        Raises:
            StorageError: If cleanup fails
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            db = next(get_db())
            try:
                # Find completed runs older than cutoff
                old_runs = (
                    db.query(WorkflowRunModel)
                    .filter(WorkflowRunModel.status.in_([
                        ExecutionStatusEnum.COMPLETED.value,
                        ExecutionStatusEnum.FAILED.value,
                        ExecutionStatusEnum.CANCELLED.value
                    ]))
                    .filter(WorkflowRunModel.completed_at < cutoff_time)
                    .all()
                )
                
                cleanup_count = 0
                for run in old_runs:
                    # Remove from in-memory state if present
                    with self._state_lock_manager:
                        self._active_runs.pop(run.id, None)
                        self._run_locks.pop(run.id, None)
                    
                    # Delete log entries first (foreign key constraint)
                    db.query(LogEntryModel).filter(LogEntryModel.run_id == run.id).delete()
                    
                    # Delete the run
                    db.delete(run)
                    cleanup_count += 1
                
                db.commit()
                
                if cleanup_count > 0:
                    logger.info(f"Cleaned up {cleanup_count} completed workflow runs")
                
                return cleanup_count
                
            except SQLAlchemyError as e:
                db.rollback()
                raise StorageError(f"Failed to cleanup completed runs: {str(e)}")
            finally:
                db.close()
                
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(f"Failed to cleanup completed runs: {str(e)}")
    
    def get_active_runs(self) -> List[str]:
        """
        Get list of currently active workflow run IDs.
        
        Returns:
            List of active run IDs
        """
        return list(self._active_runs.keys())
    
    def is_run_active(self, run_id: str) -> bool:
        """
        Check if a workflow run is currently active.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            True if run is active, False otherwise
        """
        return run_id in self._active_runs
    
    def _log_state_event(self, db: Session, run_id: str, node_id: Optional[str], 
                        event_type: LogEventType, message: str, 
                        state_snapshot: Optional[Dict] = None) -> None:
        """
        Log a state-related event to the database.
        
        Args:
            db: Database session
            run_id: ID of the workflow run
            node_id: ID of the node (optional)
            event_type: Type of event
            message: Log message
            state_snapshot: State snapshot (optional)
        """
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
            # Note: commit is handled by the caller
            
        except Exception as e:
            logger.error(f"Failed to log state event: {str(e)}")
            # Don't raise exception here to avoid breaking the main operation