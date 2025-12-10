#!/usr/bin/env python3
"""Debug script for code review tools."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.tools.code_review_tools import extract_functions

# Test code
TEST_CODE = '''
def simple_function(x):
    """A simple function."""
    return x * 2
'''

def debug_extract_functions():
    """Debug the extract_functions tool."""
    print("🐛 Debugging extract_functions")
    
    # Test with proper state format
    state = {"code_input": TEST_CODE}
    
    try:
        print(f"Input state: {state}")
        result = extract_functions(state)
        print(f"Result: {result}")
        
        # Check if result is serializable
        import json
        json_result = json.dumps(result, default=str)
        print(f"JSON serializable: {len(json_result)} characters")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_extract_functions()