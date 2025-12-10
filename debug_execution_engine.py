#!/usr/bin/env python3
"""Debug script for execution engine parameter passing."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.tool_registry import ToolRegistry
from app.models.core import WorkflowState

# Test code
TEST_CODE = '''
def simple_function(x):
    """A simple function."""
    return x * 2
'''

class MockExecutionContext:
    """Mock execution context."""
    def __init__(self, state, tool_registry, run_id):
        self.state = state
        self.tool_registry = tool_registry
        self.run_id = run_id
        self.execution_metadata = {}

def debug_execution_engine_params():
    """Debug the execution engine parameter passing."""
    print("🐛 Debugging execution engine parameter passing")
    
    # Initialize components
    tool_registry = ToolRegistry()
    
    try:
        # Create workflow state
        workflow_state = WorkflowState(
            data={"code_input": TEST_CODE},
            metadata={},
            current_node=None,
            execution_path=[]
        )
        
        # Create execution context
        context = MockExecutionContext(workflow_state, tool_registry, "test_run")
        
        # Get the tool function
        tool_function = tool_registry.get_tool("extract_functions")
        print("✅ Tool retrieved successfully")
        
        # Prepare function arguments exactly like execution engine does
        node_parameters = {}  # No additional parameters
        kwargs = node_parameters.copy() if node_parameters else {}
        kwargs['state'] = context.state.data  # This is the key line
        kwargs['context'] = context
        kwargs['execution_engine'] = None  # Mock execution engine
        
        print(f"Prepared kwargs: {list(kwargs.keys())}")
        print(f"State type: {type(kwargs['state'])}")
        print(f"State content: {kwargs['state']}")
        
        # Execute the function
        result = tool_function(**kwargs)
        print(f"✅ Tool executed successfully: {result}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_execution_engine_params()