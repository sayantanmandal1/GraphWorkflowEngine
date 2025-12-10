#!/usr/bin/env python3
"""Integration test for code review workflow."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.tool_registry import ToolRegistry
from app.core.state_manager import StateManager
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition
from app.tools.code_review_tools import (
    extract_functions,
    analyze_function_complexity,
    evaluate_quality_threshold,
    generate_improvement_suggestions,
    generate_final_report
)

# Test code
TEST_CODE = '''
def simple_function(x):
    """A simple function."""
    return x * 2

def complex_function(a, b, c, d, e, f):
    if a > 0:
        if b > 0:
            if c > 0:
                return d + e + f
            else:
                return d - e - f
        else:
            return d * e * f
    else:
        return 0
'''

def test_code_review_workflow_integration():
    """Test the code review workflow through the execution engine."""
    print("🧪 Testing Code Review Workflow Integration")
    print("=" * 60)
    
    # Initialize database
    from app.storage.database import create_tables
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
    
    # Register code review tools
    tools = [
        ("extract_functions", extract_functions, "Extract functions from code"),
        ("analyze_function_complexity", analyze_function_complexity, "Analyze function complexity"),
        ("evaluate_quality_threshold", evaluate_quality_threshold, "Evaluate quality threshold"),
        ("generate_improvement_suggestions", generate_improvement_suggestions, "Generate suggestions"),
        ("generate_final_report", generate_final_report, "Generate final report")
    ]
    
    for tool_name, tool_func, tool_desc in tools:
        try:
            tool_registry.register_tool(tool_name, tool_func, tool_desc)
            print(f"✅ Registered tool: {tool_name}")
        except Exception as e:
            print(f"⚠️  Tool registration warning for {tool_name}: {e}")
    
    # Create code review workflow definition with unique name
    import uuid
    unique_name = f"Code Review Workflow Test {str(uuid.uuid4())[:8]}"
    workflow_def = GraphDefinition(
        name=unique_name,
        description="Test workflow for code review functionality",
        nodes=[
            NodeDefinition(
                id="extract_functions",
                function_name="extract_functions",
                parameters={}
            ),
            NodeDefinition(
                id="analyze_complexity",
                function_name="analyze_function_complexity",
                parameters={}
            ),
            NodeDefinition(
                id="evaluate_threshold",
                function_name="evaluate_quality_threshold",
                parameters={"quality_threshold": 7.0}
            ),
            NodeDefinition(
                id="generate_suggestions",
                function_name="generate_improvement_suggestions",
                parameters={}
            ),
            NodeDefinition(
                id="generate_report",
                function_name="generate_final_report",
                parameters={}
            )
        ],
        edges=[
            EdgeDefinition(from_node="extract_functions", to_node="analyze_complexity"),
            EdgeDefinition(from_node="analyze_complexity", to_node="evaluate_threshold"),
            EdgeDefinition(
                from_node="evaluate_threshold", 
                to_node="generate_suggestions",
                condition="state.get('threshold_met', False) == False"
            ),
            EdgeDefinition(
                from_node="evaluate_threshold", 
                to_node="generate_report",
                condition="state.get('threshold_met', False) == True"
            ),
            EdgeDefinition(from_node="generate_suggestions", to_node="generate_report")
        ],
        entry_point="extract_functions",
        exit_conditions=["state.get('report_status') == 'success'"]
    )
    
    try:
        # Create the graph
        graph_id = graph_manager.create_graph(workflow_def)
        print(f"✅ Created workflow graph: {graph_id}")
        
        # Execute the workflow
        initial_state = {
            "code_input": TEST_CODE,
            "workflow_name": "Integration Test"
        }
        
        print(f"Initial state: {list(initial_state.keys())}")
        print(f"Code input length: {len(TEST_CODE)}")
        
        run_id = execution_engine.execute_workflow(graph_id, initial_state)
        print(f"✅ Started workflow execution: {run_id}")
        
        # Wait for completion
        import time
        max_wait = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = execution_engine.get_execution_status(run_id)
            print(f"   Status: {status.status.value}")
            
            if status.status.value in ["completed", "failed", "cancelled"]:
                break
            
            time.sleep(2)
        
        # Get final results
        final_status = execution_engine.get_execution_status(run_id)
        print(f"✅ Final status: {final_status.status.value}")
        
        if final_status.status.value == "completed":
            final_state = final_status.current_state.data
            final_report = final_state.get("final_report", {})
            
            if final_report:
                print("\n📊 Code Review Results:")
                quality_metrics = final_report.get("quality_metrics", {})
                print(f"   Total functions: {quality_metrics.get('total_functions', 0)}")
                print(f"   Average quality: {quality_metrics.get('average_quality_score', 0):.2f}")
                print(f"   Total issues: {quality_metrics.get('total_issues', 0)}")
                print(f"   Overall status: {final_report.get('overall_status', 'UNKNOWN')}")
                
                # Show function details
                function_details = final_report.get("function_details", [])
                for func in function_details:
                    status_emoji = "✅" if func["status"] == "PASS" else "❌"
                    print(f"   {status_emoji} {func['name']}: Quality {func['quality_score']:.1f}, Complexity {func['complexity']}")
                
                print("\n✅ Code review workflow integration test completed successfully!")
            else:
                print("❌ No final report generated")
        else:
            print(f"❌ Workflow failed with status: {final_status.status.value}")
            if final_status.error_message:
                print(f"   Error: {final_status.error_message}")
        
        # Get execution logs
        logs = execution_engine.get_execution_logs(run_id)
        print(f"\n📜 Execution logs ({len(logs)} entries):")
        for log in logs[-5:]:  # Show last 5 logs
            print(f"   [{log.timestamp}] {log.node_id} - {log.event_type.value}: {log.message}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        execution_engine.shutdown()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    test_code_review_workflow_integration()