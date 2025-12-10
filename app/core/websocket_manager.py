"""WebSocket Manager for real-time workflow monitoring."""

import asyncio
import json
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from queue import Queue, Empty
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..models.core import ExecutionStatus, LogEntry, LogEventType, ExecutionStatusEnum
from .logging import get_logger

logger = get_logger(__name__)


class WebSocketEvent(BaseModel):
    """WebSocket event message."""
    event_type: str
    run_id: str
    timestamp: datetime
    data: Dict[str, Any]


class WebSocketConnection:
    """Represents a WebSocket connection with metadata."""
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = datetime.utcnow()
        self.subscribed_runs: Set[str] = set()
        self.is_active = True


class WebSocketManager:
    """Manager for WebSocket connections and real-time event broadcasting."""
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self._connections: Dict[str, WebSocketConnection] = {}
        self._run_subscribers: Dict[str, Set[str]] = {}  # run_id -> set of connection_ids
        self._broadcast_lock = asyncio.Lock()
        
        # Thread-safe queue for broadcast messages from worker threads
        self._broadcast_queue: Queue = Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._processing_broadcasts = False
        
        logger.info("WebSocketManager initialized")
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            
        Returns:
            Connection ID for the new connection
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(websocket, connection_id)
        
        self._connections[connection_id] = connection
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send welcome message
        await self._send_to_connection(connection_id, {
            "event_type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "WebSocket connection established successfully"
        })
        
        return connection_id
    
    async def disconnect(self, connection_id: str) -> None:
        """
        Handle WebSocket disconnection and cleanup.
        
        Args:
            connection_id: ID of the connection to disconnect
        """
        if connection_id not in self._connections:
            return
        
        connection = self._connections[connection_id]
        connection.is_active = False
        
        # Unsubscribe from all runs
        for run_id in list(connection.subscribed_runs):
            await self._unsubscribe_from_run(connection_id, run_id)
        
        # Remove connection
        del self._connections[connection_id]
        
        logger.info(f"WebSocket connection disconnected and cleaned up: {connection_id}")
    
    async def subscribe_to_run(self, connection_id: str, run_id: str) -> bool:
        """
        Subscribe a connection to receive events for a specific workflow run.
        
        Args:
            connection_id: ID of the WebSocket connection
            run_id: ID of the workflow run to subscribe to
            
        Returns:
            True if subscription was successful, False otherwise
        """
        if connection_id not in self._connections:
            logger.warning(f"Attempted to subscribe non-existent connection: {connection_id}")
            return False
        
        connection = self._connections[connection_id]
        if not connection.is_active:
            logger.warning(f"Attempted to subscribe inactive connection: {connection_id}")
            return False
        
        # Add to connection's subscriptions
        connection.subscribed_runs.add(run_id)
        
        # Add to run's subscribers
        if run_id not in self._run_subscribers:
            self._run_subscribers[run_id] = set()
        self._run_subscribers[run_id].add(connection_id)
        
        logger.info(f"Connection {connection_id} subscribed to run {run_id}")
        
        # Send subscription confirmation
        await self._send_to_connection(connection_id, {
            "event_type": "subscription_confirmed",
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Subscribed to workflow run {run_id}"
        })
        
        return True
    
    async def unsubscribe_from_run(self, connection_id: str, run_id: str) -> bool:
        """
        Unsubscribe a connection from a workflow run.
        
        Args:
            connection_id: ID of the WebSocket connection
            run_id: ID of the workflow run to unsubscribe from
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        return await self._unsubscribe_from_run(connection_id, run_id)
    
    async def _unsubscribe_from_run(self, connection_id: str, run_id: str) -> bool:
        """Internal method to unsubscribe from a run."""
        if connection_id not in self._connections:
            return False
        
        connection = self._connections[connection_id]
        
        # Remove from connection's subscriptions
        connection.subscribed_runs.discard(run_id)
        
        # Remove from run's subscribers
        if run_id in self._run_subscribers:
            self._run_subscribers[run_id].discard(connection_id)
            
            # Clean up empty subscriber sets
            if not self._run_subscribers[run_id]:
                del self._run_subscribers[run_id]
        
        logger.info(f"Connection {connection_id} unsubscribed from run {run_id}")
        return True
    
    async def broadcast_workflow_event(self, run_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a workflow event to all subscribers of a run.
        
        Args:
            run_id: ID of the workflow run
            event_type: Type of event
            data: Event data
        """
        if run_id not in self._run_subscribers:
            logger.debug(f"No subscribers for run {run_id}, skipping broadcast")
            return
        
        async with self._broadcast_lock:
            event = {
                "event_type": event_type,
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # Get subscribers for this run
            subscribers = self._run_subscribers[run_id].copy()
            
            # Send to all subscribers
            disconnected_connections = []
            for connection_id in subscribers:
                success = await self._send_to_connection(connection_id, event)
                if not success:
                    disconnected_connections.append(connection_id)
            
            # Clean up disconnected connections
            for connection_id in disconnected_connections:
                await self.disconnect(connection_id)
            
            logger.debug(f"Broadcasted {event_type} event for run {run_id} to {len(subscribers)} subscribers")
    
    async def broadcast_execution_status(self, execution_status: ExecutionStatus) -> None:
        """
        Broadcast execution status update to subscribers.
        
        Args:
            execution_status: Current execution status
        """
        await self.broadcast_workflow_event(
            execution_status.run_id,
            "execution_status_update",
            {
                "status": execution_status.status.value,
                "current_state": execution_status.current_state.model_dump(),
                "started_at": execution_status.started_at.isoformat(),
                "completed_at": execution_status.completed_at.isoformat() if execution_status.completed_at else None,
                "error_message": execution_status.error_message
            }
        )
    
    async def broadcast_log_entry(self, log_entry: LogEntry) -> None:
        """
        Broadcast log entry to subscribers.
        
        Args:
            log_entry: Log entry to broadcast
        """
        await self.broadcast_workflow_event(
            log_entry.run_id,
            "log_entry",
            {
                "node_id": log_entry.node_id,
                "event_type": log_entry.event_type.value,
                "message": log_entry.message,
                "timestamp": log_entry.timestamp.isoformat(),
                "state_snapshot": log_entry.state_snapshot
            }
        )
    
    async def broadcast_node_execution(self, run_id: str, node_id: str, event_type: LogEventType, message: str, state_snapshot: Optional[Dict] = None) -> None:
        """
        Broadcast node execution event to subscribers.
        
        Args:
            run_id: ID of the workflow run
            node_id: ID of the executing node
            event_type: Type of execution event
            message: Event message
            state_snapshot: Optional state snapshot
        """
        await self.broadcast_workflow_event(
            run_id,
            "node_execution",
            {
                "node_id": node_id,
                "execution_event": event_type.value,
                "message": message,
                "state_snapshot": state_snapshot
            }
        )
    
    async def broadcast_error(self, run_id: str, error_message: str, error_details: Optional[Dict] = None) -> None:
        """
        Broadcast error notification to subscribers.
        
        Args:
            run_id: ID of the workflow run
            error_message: Error message
            error_details: Optional error details
        """
        await self.broadcast_workflow_event(
            run_id,
            "execution_error",
            {
                "error_message": error_message,
                "error_details": error_details or {}
            }
        )
    
    async def _send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """
        Send data to a specific WebSocket connection.
        
        Args:
            connection_id: ID of the connection
            data: Data to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if connection_id not in self._connections:
            return False
        
        connection = self._connections[connection_id]
        if not connection.is_active:
            return False
        
        try:
            message = json.dumps(data, default=str)
            await connection.websocket.send_text(message)
            return True
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected during send: {connection_id}")
            connection.is_active = False
            return False
        except Exception as e:
            logger.error(f"Error sending WebSocket message to {connection_id}: {str(e)}")
            connection.is_active = False
            return False
    
    async def send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """
        Public method to send data to a specific connection.
        
        Args:
            connection_id: ID of the connection
            data: Data to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        return await self._send_to_connection(connection_id, data)
    
    def get_connection_count(self) -> int:
        """
        Get the total number of active connections.
        
        Returns:
            Number of active connections
        """
        return len([conn for conn in self._connections.values() if conn.is_active])
    
    def get_run_subscriber_count(self, run_id: str) -> int:
        """
        Get the number of subscribers for a specific run.
        
        Args:
            run_id: ID of the workflow run
            
        Returns:
            Number of subscribers for the run
        """
        return len(self._run_subscribers.get(run_id, set()))
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about all connections.
        
        Returns:
            Dictionary containing connection information
        """
        active_connections = []
        for conn_id, conn in self._connections.items():
            if conn.is_active:
                active_connections.append({
                    "connection_id": conn_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "subscribed_runs": list(conn.subscribed_runs)
                })
        
        return {
            "total_connections": len(active_connections),
            "connections": active_connections,
            "run_subscribers": {
                run_id: len(subscribers) 
                for run_id, subscribers in self._run_subscribers.items()
            }
        }
    
    async def cleanup_inactive_connections(self) -> int:
        """
        Clean up inactive connections.
        
        Returns:
            Number of connections cleaned up
        """
        inactive_connections = [
            conn_id for conn_id, conn in self._connections.items()
            if not conn.is_active
        ]
        
        for conn_id in inactive_connections:
            await self.disconnect(conn_id)
        
        logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")
        return len(inactive_connections)
    
    def start_broadcast_processor(self):
        """Start the broadcast queue processor."""
        if not self._processing_broadcasts:
            self._processing_broadcasts = True
            self._queue_processor_task = asyncio.create_task(self._process_broadcast_queue())
            logger.info("WebSocket broadcast processor started")
    
    def stop_broadcast_processor(self):
        """Stop the broadcast queue processor."""
        self._processing_broadcasts = False
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            logger.info("WebSocket broadcast processor stopped")
    
    async def _process_broadcast_queue(self):
        """Process broadcast messages from the queue."""
        while self._processing_broadcasts:
            try:
                # Check for queued broadcast messages
                try:
                    broadcast_item = self._broadcast_queue.get_nowait()
                    broadcast_type, args, kwargs = broadcast_item
                    
                    # Execute the broadcast
                    if broadcast_type == "workflow_event":
                        await self.broadcast_workflow_event(*args, **kwargs)
                    elif broadcast_type == "execution_status":
                        await self.broadcast_execution_status(*args, **kwargs)
                    elif broadcast_type == "log_entry":
                        await self.broadcast_log_entry(*args, **kwargs)
                    elif broadcast_type == "node_execution":
                        await self.broadcast_node_execution(*args, **kwargs)
                    elif broadcast_type == "error":
                        await self.broadcast_error(*args, **kwargs)
                    
                    self._broadcast_queue.task_done()
                    
                except Empty:
                    # No messages in queue, wait a bit
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing broadcast queue: {str(e)}")
                await asyncio.sleep(0.1)
    
    def queue_workflow_event(self, run_id: str, event_type: str, data: Dict[str, Any]):
        """Queue a workflow event for broadcasting from a thread."""
        try:
            self._broadcast_queue.put(("workflow_event", (run_id, event_type, data), {}))
        except Exception as e:
            logger.error(f"Failed to queue workflow event: {str(e)}")
    
    def queue_node_execution(self, run_id: str, node_id: str, event_type, message: str, state_snapshot: Optional[Dict] = None):
        """Queue a node execution event for broadcasting from a thread."""
        try:
            self._broadcast_queue.put(("node_execution", (run_id, node_id, event_type, message, state_snapshot), {}))
        except Exception as e:
            logger.error(f"Failed to queue node execution event: {str(e)}")
    
    def queue_error(self, run_id: str, error_message: str, error_details: Optional[Dict] = None):
        """Queue an error event for broadcasting from a thread."""
        try:
            self._broadcast_queue.put(("error", (run_id, error_message, error_details), {}))
        except Exception as e:
            logger.error(f"Failed to queue error event: {str(e)}")