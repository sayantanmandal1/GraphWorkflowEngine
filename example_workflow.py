"""Example agent workflow demonstrating the workflow engine capabilities."""

from app.models import GraphDefinition, NodeDefinition, EdgeDefinition


def create_code_review_workflow() -> GraphDefinition:
    """
    Create an example code review workflow that demonstrates the engine's capabilities.
    
    This workflow:
    1. Receives code input
    2. Extracts functions from the code
    3. Analyzes each function for complexity
    4. Generates improvement suggestions if needed
    5. Produces a final report
    """
    
    # Define workflow nodes
    nodes = [
        NodeDefinition(
            id="input_handler",
            function_name="handle_code_input",
            parameters={"input_type": "python_code"},
            timeout=30
        ),
        NodeDefinition(
            id="function_extractor", 
            function_name="extract_functions",
            parameters={"language": "python"},
            timeout=60
        ),
        NodeDefinition(
            id="complexity_analyzer",
            function_name="analyze_complexity",
            parameters={"max_complexity": 10},
            timeout=120
        ),
        NodeDefinition(
            id="quality_checker",
            function_name="check_quality_threshold",
            parameters={"min_score": 7.0},
            timeout=30
        ),
        NodeDefinition(
            id="suggestion_generator",
            function_name="generate_suggestions", 
            parameters={"suggestion_type": "improvement"},
            timeout=90
        ),
        NodeDefinition(
            id="report_generator",
            function_name="generate_final_report",
            parameters={"format": "markdown"},
            timeout=60
        )
    ]
    
    # Define workflow edges (execution flow)
    edges = [
        EdgeDefinition(
            from_node="input_handler",
            to_node="function_extractor"
        ),
        EdgeDefinition(
            from_node="function_extractor", 
            to_node="complexity_analyzer"
        ),
        EdgeDefinition(
            from_node="complexity_analyzer",
            to_node="quality_checker"
        ),
        EdgeDefinition(
            from_node="quality_checker",
            to_node="suggestion_generator",
            condition="quality_score < 7.0"  # Only generate suggestions if quality is low
        ),
        EdgeDefinition(
            from_node="quality_checker",
            to_node="report_generator", 
            condition="quality_score >= 7.0"  # Skip to report if quality is good
        ),
        EdgeDefinition(
            from_node="suggestion_generator",
            to_node="complexity_analyzer"  # Loop back for iterative improvement
        )
    ]
    
    # Create the complete workflow graph
    workflow = GraphDefinition(
        name="Code Review Workflow",
        description="Automated code review workflow with iterative improvement suggestions",
        nodes=nodes,
        edges=edges,
        entry_point="input_handler",
        exit_conditions=["quality_score >= 7.0", "max_iterations_reached"]
    )
    
    return workflow


def main():
    """Demonstrate the workflow creation."""
    workflow = create_code_review_workflow()
    
    print("🔧 Agent Workflow Engine - Example Workflow")
    print("=" * 50)
    print(f"Workflow Name: {workflow.name}")
    print(f"Description: {workflow.description}")
    print(f"Entry Point: {workflow.entry_point}")
    print(f"Number of Nodes: {len(workflow.nodes)}")
    print(f"Number of Edges: {len(workflow.edges)}")
    print()
    
    print("📋 Workflow Nodes:")
    for node in workflow.nodes:
        print(f"  • {node.id}: {node.function_name}")
        if node.parameters:
            print(f"    Parameters: {node.parameters}")
    print()
    
    print("🔗 Workflow Edges:")
    for edge in workflow.edges:
        condition_text = f" (if {edge.condition})" if edge.condition else ""
        print(f"  • {edge.from_node} → {edge.to_node}{condition_text}")
    print()
    
    print("✅ Example workflow created successfully!")
    print("This demonstrates the workflow engine's capability to define complex,")
    print("conditional workflows with loops and branching logic.")


if __name__ == "__main__":
    main()