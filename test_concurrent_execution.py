#!/usr/bin/env python3
"""
Test script for concurrent execution and isolation functionality.
"""

import time
import threading
from typing import Dict, Any

from app.core.execution_engine import ExecutionEngine
from app.core.state_manager import StateManager
from app.core.graph_manager import GraphManager
from app.core.tool_registry import ToolRegistry
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition


def test_tool_1(state: Dict[str, Any], context=None, execution_engine=None, **kwargs) -> Dict[str, Any]:
    """Test tool that simulates work and updates state."""
    print(f"Tool 1 executing for run {context.run_id if context else 'unknown'}")
    
    # Simulate some work
    time.sleep(0.5)
    
    # Update state
    current_count = state.get('count', 0)
    state['count'] = current_count + 1
    state['tool_1_executed'] = True
    
    return {'count': state['count'], 'tool_1_executed': True}


def test_tool_2(state: Dict[str, Any], context=None, execution_engine=None, **kwargs) -> Dict[str, Any]:
    """Test tool that simulates work and updates state."""
    print(f"Tool 2 executing for run {context.run_id if context else 'unknown'}")
    
    # Simulate some work
    time.sleep(0.3)
    
    # Update state
    current_count = state.get('count', 0)
    state['count'] = current_count + 10
    state['tool_2_executed'] = True
    
    return {'count': state['count'], 'tool_2_executed': True}


def test_shared_resource_tool(state: Dict[str, Any], context=None, execution_engine=None, **kwargs) -> Dict[str, Any]:
    """Test tool that uses shared resources."""
    print(f"Shared resource tool executing for run {context.run_id if context else 'unknown'}")
    
    if execution_engine:
        # Acquire shared resource lock
        if execution_engine.acquire_shared_resource_lock("test_resource", timeout=2.0):
            try:
                print(f"Acquired shared resource lock for run {context.run_id}")
                time.sleep(1.0)  # Simulate work with shared resource
                
                # Update state
                current_count = state.get('shared_count', 0)
                state['shared_count'] = current_count + 1
                
                return {'shared_count': state['shared_count']}
                
            finally:
                execution_engine.release_shared_resource_lock("test_resource")
                print(f"Released shared resource lock for run {context.run_id}")
        else:
            print(f"Failed to acquire shared resource lock for run {context.run_id}")
            return {'error': 'Failed to acquire shared resource'}
    
    return {'error': 'No execution engine provided'}


def create_test_graph() -> GraphDefinition:
    """Create a test graph for concurrent execution."""
    nodes = [
        NodeDefinition(id="start", function_name="test_tool_1"),
        NodeDefinition(id="middle", function_name="test_shared_resource_tool"),
        NodeDefinition(id="end", function_name="test_tool_2")
    ]
    
    edges = [
        EdgeDefinition(from_node="start", to_node="middle"),
        EdgeDefinition(from_node="middle", to_node="end")
    ]
    
    return GraphDefinition(
        name="Test Concurrent Graph",
        description="A test graph for concurrent execution",
        nodes=nodes,
        edges=edges,
        entry_point="start"
    )


def test_concurrent_execution():
    """Test concurrent execution and isolation."""
    print("Testing concurrent execution and isolation...")
    
    # Initialize components
    tool_registry = ToolRegistry()
    state_manager = StateManager()
    graph_manager = GraphManager()  # No parameter needed
    execution_engine = ExecutionEngine(tool_registry, state_manager, graph_manager, max_concurrent_executions=3)
    
    # Register test tools (skip if already registered)
    try:
        tool_registry.register_tool("test_tool_1", test_tool_1, "Test tool 1")
    except Exception:
        pass  # Already registered
    
    try:
        tool_registry.register_tool("test_tool_2", test_tool_2, "Test tool 2")
    except Exception:
        pass  # Already registered
    
    try:
        tool_registry.register_tool("test_shared_resource_tool", test_shared_resource_tool, "Shared resource test tool")
    except Exception:
        pass  # Already registered
    
    # Create and register test graph
    test_graph = create_test_graph()
    graph_id = graph_manager.create_graph(test_graph)
    print(f"Created test graph: {graph_id}")
    
    # Start multiple concurrent executions
    run_ids = []
    for i in range(5):
        initial_state = {'run_number': i, 'count': 0, 'shared_count': 0}
        run_id = execution_engine.execute_workflow(graph_id, initial_state)
        run_ids.append(run_id)
        print(f"Started execution {i+1}: {run_id}")
        time.sleep(0.1)  # Small delay between starts
    
    # Monitor execution progress
    print("\nMonitoring execution progress...")
    completed_runs = set()
    
    while len(completed_runs) < len(run_ids):
        # Check queue status
        queue_status = execution_engine.get_execution_queue_status()
        print(f"Queue status: {queue_status}")
        
        # Check execution metrics
        metrics = execution_engine.get_concurrent_execution_metrics()
        print(f"Execution metrics: {metrics}")
        
        # Check individual run statuses
        for run_id in run_ids:
            if run_id not in completed_runs:
                try:
                    status = execution_engine.get_execution_status(run_id)
                    print(f"Run {run_id}: {status.status}")
                    
                    if status.status in ['completed', 'failed', 'cancelled']:
                        completed_runs.add(run_id)
                        print(f"Run {run_id} completed with status: {status.status}")
                        if status.status == 'completed':
                            print(f"Final state: {status.current_state.data}")
                        elif status.error_message:
                            print(f"Error: {status.error_message}")
                            
                except Exception as e:
                    print(f"Error checking status for {run_id}: {e}")
        
        time.sleep(1.0)
    
    print("\nAll executions completed!")
    
    # Shutdown
    execution_engine.shutdown()
    print("Execution engine shutdown completed")


if __name__ == "__main__":
    test_concurrent_execution()