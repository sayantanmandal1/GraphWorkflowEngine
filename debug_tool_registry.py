#!/usr/bin/env python3
"""Debug script for tool registry."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.tool_registry import ToolRegistry
from app.tools.code_review_tools import extract_functions

# Test code
TEST_CODE = '''
def simple_function(x):
    """A simple function."""
    return x * 2
'''

def debug_tool_registry():
    """Debug the tool registry with code review tools."""
    print("🐛 Debugging tool registry")
    
    # Initialize tool registry
    tool_registry = ToolRegistry()
    
    try:
        # Get the tool (it should already be registered)
        tool_func = tool_registry.get_tool("extract_functions")
        print("✅ Tool retrieved successfully")
        
        # Test calling the tool directly
        state = {"code_input": TEST_CODE}
        print(f"Calling tool with state: {state}")
        
        result = tool_func(state=state)
        print(f"✅ Tool executed successfully: {result}")
        
        # Test calling through tool registry
        result2 = tool_registry.call_tool("extract_functions", state=state)
        print(f"✅ Tool called through registry: {result2}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tool_registry()