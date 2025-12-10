#!/usr/bin/env python3
"""Basic workflow test to isolate the execution engine issue."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.storage.database import create_tables
from app.core.tool_registry import ToolRegistry
from app.core.state_manager import StateManager
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition
from app.tools.test_tools import workflow_test_function

def test_basic_workflow():
    """Test basic workflow execution."""
    print("🧪 Testing Basic Workflow Execution")
    print("=" * 50)
    
    # Initialize database
    create_tables()
    print("✅ Database tables created")
    
    # Initialize components
    tool_registry = ToolRegistry()
    state_manager = StateManager()
    graph_manager = GraphManager()
    execution_engine = ExecutionEngine(
        tool_registry=tool_registry,
        state_manager=state_manager,
        graph_manager=graph_manager,
        max_concurrent_executions=5
    )
    
    try:
        # Register a simple test tool
        tool_registry.register_tool("workflow_test_function", workflow_test_function, "Simple test function")
        print("✅ Tool registered")
        
        # Create a simple workflow
        workflow_def = GraphDefinition(
            name="Basic Test Workflow",
            description="Simple test workflow",
            nodes=[
                NodeDefinition(
                    id="test_node",
                    function_name="workflow_test_function",
                    parameters={"message": "Hello from basic workflow"},
                    timeout=30
                )
            ],
            edges=[],
            entry_point="test_node",
            exit_conditions=[]
        )
        
        # Create the graph
        graph_id = graph_manager.create_graph(workflow_def)
        print(f"✅ Created workflow graph: {graph_id}")
        
        # Execute the workflow
        initial_state = {"test_data": "initial"}
        
        run_id = execution_engine.execute_workflow(graph_id, initial_state)
        print(f"✅ Started workflow execution: {run_id}")
        
        # Wait for completion
        import time
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                status = execution_engine.get_execution_status(run_id)
                print(f"   Status: {status.status.value}")
                
                if status.status.value in ["completed", "failed", "cancelled"]:
                    break
                    
            except Exception as e:
                print(f"   Error getting status: {str(e)}")
                break
            
            time.sleep(1)
        
        # Get final results
        try:
            final_status = execution_engine.get_execution_status(run_id)
            print(f"✅ Final status: {final_status.status.value}")
            
            if final_status.status.value == "completed":
                print(f"✅ Final state: {final_status.current_state.data}")
            else:
                print(f"❌ Error: {final_status.error_message}")
                
        except Exception as e:
            print(f"❌ Error getting final status: {str(e)}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        execution_engine.shutdown()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    test_basic_workflow()