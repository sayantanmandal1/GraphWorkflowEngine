#!/usr/bin/env python3
"""
Agent Workflow Engine - Comprehensive Example

This example demonstrates all core capabilities of the workflow engine:
- Tool registration and management
- Graph creation with conditional branching
- Workflow execution with state management
- Real-time monitoring via WebSocket
- Code review workflow implementation

Run this example to see the complete system in action.
"""

import asyncio
import json
import time
from typing import Dict, Any

from app.factory import create_app
from app.config import AppConfig
from app.core.tool_registry import ToolRegistry
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.core.state_manager import StateManager
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition


# ============================================================================
# WORKFLOW TOOLS - Functions that can be used in workflows
# ============================================================================

def math_operation(state: Dict[str, Any], operation: str, value: int, **kwargs) -> Dict[str, Any]:
    """Perform mathematical operations on state data."""
    current_value = state.get('result', 0)
    
    if operation == 'add':
        state['result'] = current_value + value
    elif operation == 'multiply':
        state['result'] = current_value * value
    elif operation == 'subtract':
        state['result'] = current_value - value
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    state['last_operation'] = f"{operation} {value}"
    return state


def quality_check(state: Dict[str, Any], threshold: int = 50, **kwargs) -> Dict[str, Any]:
    """Check if result meets quality threshold."""
    result = state.get('result', 0)
    state['quality_check'] = {
        'threshold': threshold,
        'actual_value': result,
        'passes': result >= threshold,
        'status': 'PASS' if result >= threshold else 'FAIL'
    }
    return state


def generate_report(state: Dict[str, Any], report_type: str = 'summary', **kwargs) -> Dict[str, Any]:
    """Generate different types of reports based on workflow results."""
    if report_type == 'summary':
        state['report'] = {
            'type': 'Summary Report',
            'final_result': state.get('result', 0),
            'operations_performed': state.get('operations_history', []),
            'quality_status': state.get('quality_check', {}).get('status', 'UNKNOWN'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    elif report_type == 'detailed':
        state['report'] = {
            'type': 'Detailed Report',
            'complete_state': dict(state),
            'analysis': 'Complete workflow execution analysis',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    return state


def code_analyzer(state: Dict[str, Any], code: str = None, **kwargs) -> Dict[str, Any]:
    """Analyze Python code for complexity and quality metrics."""
    if code is None:
        code = state.get('code', '')
    
    # Simple code analysis (in real implementation, use AST parsing)
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Basic complexity calculation
    complexity_indicators = ['if ', 'for ', 'while ', 'try:', 'except:', 'elif ']
    complexity = sum(1 for line in non_empty_lines 
                    for indicator in complexity_indicators 
                    if indicator in line)
    
    # Quality scoring
    quality_score = max(0, 100 - (complexity * 5) - max(0, len(non_empty_lines) - 20))
    
    state['code_analysis'] = {
        'lines_of_code': len(non_empty_lines),
        'complexity_score': complexity,
        'quality_score': quality_score,
        'status': 'GOOD' if quality_score >= 70 else 'NEEDS_IMPROVEMENT',
        'suggestions': []
    }
    
    if quality_score < 70:
        state['code_analysis']['suggestions'] = [
            'Consider breaking down complex functions',
            'Add more comments and documentation',
            'Reduce cyclomatic complexity'
        ]
    
    return state


# ============================================================================
# WORKFLOW DEFINITIONS
# ============================================================================

def create_math_workflow() -> GraphDefinition:
    """Create a mathematical processing workflow with conditional branching."""
    import uuid
    return GraphDefinition(
        name=f"Mathematical Processing Workflow {str(uuid.uuid4())[:8]}",
        description="Demonstrates math operations with quality checking and conditional reporting",
        nodes=[
            NodeDefinition(
                id="initialize",
                function_name="math_operation",
                parameters={"operation": "add", "value": 10}
            ),
            NodeDefinition(
                id="multiply",
                function_name="math_operation", 
                parameters={"operation": "multiply", "value": 3}
            ),
            NodeDefinition(
                id="quality_check",
                function_name="quality_check",
                parameters={"threshold": 25}
            ),
            NodeDefinition(
                id="summary_report",
                function_name="generate_report",
                parameters={"report_type": "summary"}
            ),
            NodeDefinition(
                id="detailed_report", 
                function_name="generate_report",
                parameters={"report_type": "detailed"}
            )
        ],
        edges=[
            EdgeDefinition(from_node="initialize", to_node="multiply"),
            EdgeDefinition(from_node="multiply", to_node="quality_check"),
            EdgeDefinition(
                from_node="quality_check", 
                to_node="summary_report",
                condition="state.get('quality_check', {}).get('passes', False) == True"
            ),
            EdgeDefinition(
                from_node="quality_check",
                to_node="detailed_report", 
                condition="state.get('quality_check', {}).get('passes', False) == False"
            )
        ],
        entry_point="initialize"
    )


def create_code_review_workflow() -> GraphDefinition:
    """Create a code review workflow that analyzes code quality."""
    import uuid
    return GraphDefinition(
        name=f"Code Review Workflow {str(uuid.uuid4())[:8]}",
        description="Analyzes Python code and generates quality reports",
        nodes=[
            NodeDefinition(
                id="analyze_code",
                function_name="code_analyzer",
                parameters={}
            ),
            NodeDefinition(
                id="quality_gate",
                function_name="quality_check",
                parameters={"threshold": 70}
            ),
            NodeDefinition(
                id="approval_report",
                function_name="generate_report",
                parameters={"report_type": "summary"}
            ),
            NodeDefinition(
                id="improvement_report",
                function_name="generate_report", 
                parameters={"report_type": "detailed"}
            )
        ],
        edges=[
            EdgeDefinition(from_node="analyze_code", to_node="quality_gate"),
            EdgeDefinition(
                from_node="quality_gate",
                to_node="approval_report",
                condition="state.get('code_analysis', {}).get('status') == 'GOOD'"
            ),
            EdgeDefinition(
                from_node="quality_gate", 
                to_node="improvement_report",
                condition="state.get('code_analysis', {}).get('status') == 'NEEDS_IMPROVEMENT'"
            )
        ],
        entry_point="analyze_code"
    )


# ============================================================================
# EXAMPLE EXECUTION FUNCTIONS
# ============================================================================

def setup_workflow_engine():
    """Initialize and configure the workflow engine with tools."""
    print("🔧 Setting up Workflow Engine...")
    
    # Create configuration
    config = AppConfig()
    
    # Initialize core components
    tool_registry = ToolRegistry()
    graph_manager = GraphManager()
    state_manager = StateManager()
    execution_engine = ExecutionEngine(
        graph_manager=graph_manager,
        state_manager=state_manager,
        tool_registry=tool_registry
    )
    
    # Register workflow tools (handle existing tools gracefully)
    print("📝 Registering workflow tools...")
    tools_to_register = [
        ("math_operation", math_operation, "Perform mathematical operations"),
        ("quality_check", quality_check, "Check quality thresholds"),
        ("generate_report", generate_report, "Generate workflow reports"),
        ("code_analyzer", code_analyzer, "Analyze Python code quality")
    ]
    
    for name, func, desc in tools_to_register:
        try:
            tool_registry.register_tool(name, func, desc)
        except Exception as e:
            if "already registered" in str(e):
                print(f"  ⚠️ Tool '{name}' already registered, skipping...")
            else:
                raise
    
    print(f"✅ Registered {len(tool_registry.list_tools())} tools")
    
    return {
        'config': config,
        'tool_registry': tool_registry,
        'graph_manager': graph_manager,
        'state_manager': state_manager,
        'execution_engine': execution_engine
    }


def run_math_workflow_example(components):
    """Execute the mathematical processing workflow."""
    print("\n" + "="*60)
    print("🧮 MATHEMATICAL PROCESSING WORKFLOW")
    print("="*60)
    
    graph_manager = components['graph_manager']
    execution_engine = components['execution_engine']
    
    # Create and register workflow
    workflow = create_math_workflow()
    graph_id = graph_manager.create_graph(workflow)
    print(f"📊 Created workflow: {workflow.name}")
    
    # Execute workflow
    initial_state = {"result": 5, "operations_history": []}
    print(f"🚀 Starting execution with initial state: {initial_state}")
    
    run_id = execution_engine.execute_workflow(graph_id, initial_state)
    print(f"⏳ Execution started with run ID: {run_id}")
    
    # Wait for completion and show results
    wait_for_completion_and_show_results(execution_engine, run_id)


def run_code_review_workflow_example(components):
    """Execute the code review workflow."""
    print("\n" + "="*60)
    print("🔍 CODE REVIEW WORKFLOW")
    print("="*60)
    
    graph_manager = components['graph_manager']
    execution_engine = components['execution_engine']
    
    # Create and register workflow
    workflow = create_code_review_workflow()
    graph_id = graph_manager.create_graph(workflow)
    print(f"📊 Created workflow: {workflow.name}")
    
    # Sample code to analyze
    sample_code = '''
def complex_function(data, options, config):
    if data is None:
        return None
    
    results = []
    for item in data:
        if item.get('active', False):
            if options.get('filter_enabled', True):
                if item.get('score', 0) > config.get('threshold', 50):
                    try:
                        processed = process_item(item, options)
                        if processed:
                            results.append(processed)
                    except Exception as e:
                        if config.get('ignore_errors', False):
                            continue
                        else:
                            raise e
    return results

def simple_function(x, y):
    return x + y
'''
    
    initial_state = {"code": sample_code}
    print(f"🚀 Starting code review with sample code ({len(sample_code)} characters)")
    
    run_id = execution_engine.execute_workflow(graph_id, initial_state)
    print(f"⏳ Execution started with run ID: {run_id}")
    
    # Wait for completion and show results
    wait_for_completion_and_show_results(execution_engine, run_id)


def wait_for_completion_and_show_results(execution_engine, run_id):
    """Wait for workflow completion and display results."""
    print("⏳ Waiting for workflow completion...")
    
    # Poll for completion
    max_attempts = 30
    for attempt in range(max_attempts):
        status = execution_engine.get_execution_status(run_id)
        
        if status.status.value == 'completed':
            print("✅ Workflow completed successfully!")
            print_workflow_results(status)
            break
        elif status.status.value == 'failed':
            print("❌ Workflow failed!")
            print(f"Error: {status.error_message}")
            break
        else:
            print(f"⏳ Status: {status.status.value} (attempt {attempt + 1}/{max_attempts})")
            time.sleep(1)
    else:
        print("⚠️ Workflow did not complete within timeout")


def print_workflow_results(status):
    """Print formatted workflow execution results."""
    print("\n📋 WORKFLOW RESULTS:")
    print("-" * 40)
    
    state_data = status.current_state.data
    
    # Print final result if available
    if 'result' in state_data:
        print(f"Final Result: {state_data['result']}")
    
    # Print quality check results
    if 'quality_check' in state_data:
        qc = state_data['quality_check']
        print(f"Quality Check: {qc.get('status', 'UNKNOWN')} (threshold: {qc.get('threshold', 'N/A')})")
    
    # Print code analysis results
    if 'code_analysis' in state_data:
        ca = state_data['code_analysis']
        print(f"Code Quality: {ca.get('quality_score', 'N/A')}/100 ({ca.get('status', 'UNKNOWN')})")
        print(f"Complexity: {ca.get('complexity_score', 'N/A')}")
        if ca.get('suggestions'):
            print("Suggestions:")
            for suggestion in ca['suggestions']:
                print(f"  • {suggestion}")
    
    # Print report if available
    if 'report' in state_data:
        report = state_data['report']
        print(f"\n📄 {report.get('type', 'Report')}:")
        if 'final_result' in report:
            print(f"  Final Result: {report['final_result']}")
        if 'quality_status' in report:
            print(f"  Quality Status: {report['quality_status']}")
        print(f"  Generated: {report.get('timestamp', 'N/A')}")
    
    # Print execution path
    execution_path = status.current_state.execution_path
    print(f"\n🛤️ Execution Path: {' → '.join(execution_path)}")


def demonstrate_api_usage():
    """Demonstrate how to use the workflow engine via REST API."""
    print("\n" + "="*60)
    print("🌐 REST API DEMONSTRATION")
    print("="*60)
    print("To use the workflow engine via REST API:")
    print("1. Start the server: python -m app.main")
    print("2. Create graphs: POST /graph/create")
    print("3. Execute workflows: POST /graph/run")
    print("4. Monitor progress: GET /graph/state/{run_id}")
    print("5. Real-time monitoring: WebSocket /ws")
    print("\nExample API calls:")
    print("curl -X POST http://localhost:8000/graph/create -H 'Content-Type: application/json' -d '{...}'")
    print("curl -X POST http://localhost:8000/graph/run -H 'Content-Type: application/json' -d '{...}'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the comprehensive workflow engine example."""
    print("🚀 Agent Workflow Engine - Comprehensive Example")
    print("=" * 60)
    
    try:
        # Setup workflow engine
        components = setup_workflow_engine()
        
        # Run mathematical workflow example
        run_math_workflow_example(components)
        
        # Run code review workflow example  
        run_code_review_workflow_example(components)
        
        # Show API usage information
        demonstrate_api_usage()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The workflow engine is fully operational and ready for use.")
        print("You can now:")
        print("• Create custom workflows using the GraphDefinition model")
        print("• Register your own tools with the ToolRegistry")
        print("• Execute workflows and monitor their progress")
        print("• Use the REST API for integration with other systems")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()