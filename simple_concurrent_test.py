#!/usr/bin/env python3
"""
Simple test for concurrent execution functionality.
"""

import time
from typing import Dict, Any

from app.core.execution_engine import ExecutionEngine
from app.core.state_manager import StateManager
from app.core.graph_manager import GraphManager
from app.core.tool_registry import ToolRegistry
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition


def simple_tool(state: Dict[str, Any], context=None, execution_engine=None, **kwargs) -> Dict[str, Any]:
    """Simple test tool."""
    print(f"Simple tool executing for run {context.run_id if context else 'unknown'}")
    
    # Simulate some work
    time.sleep(0.2)
    
    # Update state
    current_count = state.get('count', 0)
    state['count'] = current_count + 1
    
    return {'count': state['count'], 'executed': True}


def create_simple_graph() -> GraphDefinition:
    """Create a simple test graph."""
    nodes = [
        NodeDefinition(id="simple_node", function_name="simple_tool")
    ]
    
    return GraphDefinition(
        name="Simple Test Graph",
        description="A simple test graph",
        nodes=nodes,
        edges=[],  # No edges, just one node
        entry_point="simple_node"
    )


def test_simple_concurrent_execution():
    """Test simple concurrent execution."""
    print("Testing simple concurrent execution...")
    
    # Initialize components
    tool_registry = ToolRegistry()
    state_manager = StateManager()
    graph_manager = GraphManager()
    execution_engine = ExecutionEngine(tool_registry, state_manager, graph_manager, max_concurrent_executions=2)
    
    # Register test tool
    try:
        tool_registry.register_tool("simple_tool", simple_tool, "Simple test tool")
    except Exception:
        pass  # Already registered
    
    # Create and register test graph
    test_graph = create_simple_graph()
    graph_id = graph_manager.create_graph(test_graph)
    print(f"Created test graph: {graph_id}")
    
    # Start 3 concurrent executions (should queue one)
    run_ids = []
    for i in range(3):
        initial_state = {'run_number': i, 'count': 0}
        run_id = execution_engine.execute_workflow(graph_id, initial_state)
        run_ids.append(run_id)
        print(f"Started execution {i+1}: {run_id}")
    
    # Wait a bit for executions to start
    time.sleep(0.5)
    
    # Check queue status
    queue_status = execution_engine.get_execution_queue_status()
    print(f"Queue status: {queue_status}")
    
    # Wait for all executions to complete
    completed_runs = set()
    max_wait = 10  # Maximum wait time in seconds
    start_time = time.time()
    
    while len(completed_runs) < len(run_ids) and (time.time() - start_time) < max_wait:
        for run_id in run_ids:
            if run_id not in completed_runs:
                try:
                    status = execution_engine.get_execution_status(run_id)
                    if status.status in ['completed', 'failed', 'cancelled']:
                        completed_runs.add(run_id)
                        print(f"Run {run_id} completed with status: {status.status}")
                        if status.status == 'completed':
                            print(f"Final state: {status.current_state.data}")
                        elif status.error_message:
                            print(f"Error: {status.error_message}")
                except Exception as e:
                    print(f"Error checking status for {run_id}: {e}")
        
        if len(completed_runs) < len(run_ids):
            time.sleep(0.1)
    
    if len(completed_runs) == len(run_ids):
        print("All executions completed successfully!")
    else:
        print(f"Only {len(completed_runs)} out of {len(run_ids)} executions completed")
    
    # Final queue status
    final_queue_status = execution_engine.get_execution_queue_status()
    print(f"Final queue status: {final_queue_status}")
    
    # Shutdown
    execution_engine.shutdown()
    print("Test completed")


if __name__ == "__main__":
    test_simple_concurrent_execution()