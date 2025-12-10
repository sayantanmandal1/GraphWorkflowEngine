#!/usr/bin/env python3
"""Debug script for state update issues."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.state_manager import StateManager
from app.tools.code_review_tools import extract_functions

# Test code
TEST_CODE = '''
def simple_function(x):
    """A simple function."""
    return x * 2
'''

def debug_state_update():
    """Debug the state update process."""
    print("🐛 Debugging state update")
    
    # Initialize state manager
    state_manager = StateManager()
    
    try:
        # Create a run state
        run_id = "test_run_debug"
        graph_id = "test_graph"
        initial_state = {"code_input": TEST_CODE}
        
        print(f"Creating run state: {run_id}")
        state_manager.create_run_state(run_id, graph_id, initial_state)
        print("✅ Run state created")
        
        # Get the current state
        current_state = state_manager.get_state(run_id)
        print(f"Current state: {current_state}")
        
        # Execute the tool to get result
        result = extract_functions(current_state.data)
        print(f"Tool result: {result}")
        
        # Try to update state with the result
        print("Updating state with tool result...")
        state_manager.update_state(run_id, result, "extract_functions")
        print("✅ State updated successfully")
        
        # Get updated state
        updated_state = state_manager.get_state(run_id)
        print(f"Updated state data keys: {list(updated_state.data.keys())}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_state_update()