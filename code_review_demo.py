#!/usr/bin/env python3
"""
Code Review Mini-Agent Workflow Demo

This demonstrates Option A: Code Review Mini-Agent with all required steps:
1. Extract functions
2. Check complexity  
3. Detect basic issues
4. Suggest improvements
5. Loop until "quality_score >= threshold"
"""

import ast
import time
from typing import Dict, Any, List

from app.core.tool_registry import ToolRegistry
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.core.state_manager import StateManager
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition


# ============================================================================
# CODE REVIEW WORKFLOW TOOLS
# ============================================================================

def extract_functions(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Step 1: Extract functions from Python code."""
    code = state.get('code', '')
    
    try:
        tree = ast.parse(code)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function info
                func_info = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'args_count': len(node.args.args),
                    'has_docstring': ast.get_docstring(node) is not None
                }
                functions.append(func_info)
        
        state['extracted_functions'] = functions
        state['extraction_status'] = 'SUCCESS'
        print(f"✅ Extracted {len(functions)} functions")
        
    except SyntaxError as e:
        state['extraction_status'] = 'SYNTAX_ERROR'
        state['extraction_error'] = str(e)
        print(f"❌ Syntax error in code: {e}")
    
    return state


def check_complexity(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Step 2: Check complexity of extracted functions."""
    code = state.get('code', '')
    functions = state.get('extracted_functions', [])
    
    if not functions:
        state['complexity_analysis'] = {'total_complexity': 0, 'function_complexities': []}
        return state
    
    try:
        tree = ast.parse(code)
        complexity_results = []
        
        for func_node in ast.walk(tree):
            if isinstance(func_node, ast.FunctionDef):
                # Calculate cyclomatic complexity
                complexity = calculate_cyclomatic_complexity(func_node)
                
                complexity_results.append({
                    'function_name': func_node.name,
                    'complexity': complexity,
                    'complexity_level': get_complexity_level(complexity)
                })
        
        total_complexity = sum(r['complexity'] for r in complexity_results)
        
        state['complexity_analysis'] = {
            'total_complexity': total_complexity,
            'function_complexities': complexity_results,
            'average_complexity': total_complexity / len(complexity_results) if complexity_results else 0
        }
        
        print(f"📊 Analyzed complexity: Total={total_complexity}, Average={state['complexity_analysis']['average_complexity']:.1f}")
        
    except Exception as e:
        state['complexity_error'] = str(e)
        print(f"❌ Complexity analysis failed: {e}")
    
    return state


def detect_issues(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Step 3: Detect basic issues in the code."""
    code = state.get('code', '')
    functions = state.get('extracted_functions', [])
    complexity_analysis = state.get('complexity_analysis', {})
    
    issues = []
    
    # Check for high complexity functions
    for func_complexity in complexity_analysis.get('function_complexities', []):
        if func_complexity['complexity'] > 10:
            issues.append({
                'type': 'HIGH_COMPLEXITY',
                'function': func_complexity['function_name'],
                'severity': 'HIGH',
                'message': f"Function '{func_complexity['function_name']}' has high complexity ({func_complexity['complexity']})"
            })
    
    # Check for functions without docstrings
    for func in functions:
        if not func['has_docstring']:
            issues.append({
                'type': 'MISSING_DOCSTRING',
                'function': func['name'],
                'severity': 'MEDIUM',
                'message': f"Function '{func['name']}' is missing a docstring"
            })
    
    # Check for long functions (>50 lines)
    for func in functions:
        line_count = func['line_end'] - func['line_start'] + 1
        if line_count > 50:
            issues.append({
                'type': 'LONG_FUNCTION',
                'function': func['name'],
                'severity': 'MEDIUM',
                'message': f"Function '{func['name']}' is too long ({line_count} lines)"
            })
    
    # Check for functions with too many parameters
    for func in functions:
        if func['args_count'] > 5:
            issues.append({
                'type': 'TOO_MANY_PARAMS',
                'function': func['name'],
                'severity': 'MEDIUM',
                'message': f"Function '{func['name']}' has too many parameters ({func['args_count']})"
            })
    
    state['detected_issues'] = issues
    state['issue_count'] = len(issues)
    
    print(f"🔍 Detected {len(issues)} issues")
    for issue in issues[:3]:  # Show first 3 issues
        print(f"  • {issue['severity']}: {issue['message']}")
    
    return state


def suggest_improvements(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Step 4: Suggest improvements based on detected issues."""
    issues = state.get('detected_issues', [])
    
    suggestions = []
    
    # Group issues by type and generate suggestions
    issue_types = {}
    for issue in issues:
        issue_type = issue['type']
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue)
    
    # Generate suggestions based on issue types
    if 'HIGH_COMPLEXITY' in issue_types:
        suggestions.append({
            'category': 'COMPLEXITY',
            'priority': 'HIGH',
            'suggestion': 'Break down complex functions into smaller, more focused functions',
            'affected_functions': [i['function'] for i in issue_types['HIGH_COMPLEXITY']]
        })
    
    if 'MISSING_DOCSTRING' in issue_types:
        suggestions.append({
            'category': 'DOCUMENTATION',
            'priority': 'MEDIUM',
            'suggestion': 'Add docstrings to document function purpose, parameters, and return values',
            'affected_functions': [i['function'] for i in issue_types['MISSING_DOCSTRING']]
        })
    
    if 'LONG_FUNCTION' in issue_types:
        suggestions.append({
            'category': 'STRUCTURE',
            'priority': 'MEDIUM',
            'suggestion': 'Split long functions into smaller, single-responsibility functions',
            'affected_functions': [i['function'] for i in issue_types['LONG_FUNCTION']]
        })
    
    if 'TOO_MANY_PARAMS' in issue_types:
        suggestions.append({
            'category': 'PARAMETERS',
            'priority': 'MEDIUM',
            'suggestion': 'Consider using data classes or dictionaries to group related parameters',
            'affected_functions': [i['function'] for i in issue_types['TOO_MANY_PARAMS']]
        })
    
    state['improvement_suggestions'] = suggestions
    state['suggestion_count'] = len(suggestions)
    
    print(f"💡 Generated {len(suggestions)} improvement suggestions")
    for suggestion in suggestions:
        print(f"  • {suggestion['priority']}: {suggestion['suggestion']}")
    
    return state


def calculate_quality_score(state: Dict[str, Any], threshold: int = 80, **kwargs) -> Dict[str, Any]:
    """Step 5: Calculate quality score and determine if threshold is met."""
    functions = state.get('extracted_functions', [])
    issues = state.get('detected_issues', [])
    complexity_analysis = state.get('complexity_analysis', {})
    
    if not functions:
        state['quality_score'] = 0
        state['meets_threshold'] = False
        return state
    
    # Calculate quality score based on various factors
    base_score = 100
    
    # Deduct points for issues
    high_severity_issues = len([i for i in issues if i['severity'] == 'HIGH'])
    medium_severity_issues = len([i for i in issues if i['severity'] == 'MEDIUM'])
    
    score_deduction = (high_severity_issues * 15) + (medium_severity_issues * 5)
    
    # Deduct points for high average complexity
    avg_complexity = complexity_analysis.get('average_complexity', 0)
    if avg_complexity > 5:
        score_deduction += (avg_complexity - 5) * 3
    
    quality_score = max(0, base_score - score_deduction)
    meets_threshold = quality_score >= threshold
    
    state['quality_score'] = quality_score
    state['meets_threshold'] = meets_threshold
    state['quality_threshold'] = threshold
    
    print(f"📈 Quality Score: {quality_score}/100 (threshold: {threshold})")
    print(f"🎯 Meets Threshold: {'✅ YES' if meets_threshold else '❌ NO'}")
    
    return state


def generate_final_report(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Generate final code review report."""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'functions_analyzed': len(state.get('extracted_functions', [])),
            'issues_found': state.get('issue_count', 0),
            'suggestions_provided': state.get('suggestion_count', 0),
            'quality_score': state.get('quality_score', 0),
            'meets_threshold': state.get('meets_threshold', False)
        },
        'details': {
            'extracted_functions': state.get('extracted_functions', []),
            'complexity_analysis': state.get('complexity_analysis', {}),
            'detected_issues': state.get('detected_issues', []),
            'improvement_suggestions': state.get('improvement_suggestions', [])
        }
    }
    
    state['final_report'] = report
    
    print(f"\n📋 FINAL CODE REVIEW REPORT")
    print(f"=" * 50)
    print(f"Functions Analyzed: {report['summary']['functions_analyzed']}")
    print(f"Issues Found: {report['summary']['issues_found']}")
    print(f"Quality Score: {report['summary']['quality_score']}/100")
    print(f"Status: {'✅ APPROVED' if report['summary']['meets_threshold'] else '❌ NEEDS IMPROVEMENT'}")
    
    return state


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_cyclomatic_complexity(func_node):
    """Calculate cyclomatic complexity of a function."""
    complexity = 1  # Base complexity
    
    for node in ast.walk(func_node):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.With, ast.AsyncWith):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    
    return complexity


def get_complexity_level(complexity):
    """Get complexity level description."""
    if complexity <= 5:
        return "LOW"
    elif complexity <= 10:
        return "MEDIUM"
    else:
        return "HIGH"


# ============================================================================
# WORKFLOW DEFINITION
# ============================================================================

def create_code_review_workflow() -> GraphDefinition:
    """Create the Code Review Mini-Agent workflow."""
    return GraphDefinition(
        name="Code Review Mini-Agent",
        description="Comprehensive code review workflow with quality gating",
        nodes=[
            NodeDefinition(
                id="extract_functions",
                function_name="extract_functions",
                parameters={}
            ),
            NodeDefinition(
                id="check_complexity",
                function_name="check_complexity",
                parameters={}
            ),
            NodeDefinition(
                id="detect_issues",
                function_name="detect_issues",
                parameters={}
            ),
            NodeDefinition(
                id="suggest_improvements",
                function_name="suggest_improvements",
                parameters={}
            ),
            NodeDefinition(
                id="calculate_quality",
                function_name="calculate_quality_score",
                parameters={"threshold": 80}
            ),
            NodeDefinition(
                id="generate_report",
                function_name="generate_final_report",
                parameters={}
            ),
            NodeDefinition(
                id="needs_improvement",
                function_name="generate_final_report",
                parameters={}
            )
        ],
        edges=[
            EdgeDefinition(from_node="extract_functions", to_node="check_complexity"),
            EdgeDefinition(from_node="check_complexity", to_node="detect_issues"),
            EdgeDefinition(from_node="detect_issues", to_node="suggest_improvements"),
            EdgeDefinition(from_node="suggest_improvements", to_node="calculate_quality"),
            EdgeDefinition(
                from_node="calculate_quality",
                to_node="generate_report",
                condition="state.get('meets_threshold', False) == True"
            ),
            EdgeDefinition(
                from_node="calculate_quality",
                to_node="needs_improvement",
                condition="state.get('meets_threshold', False) == False"
            )
        ],
        entry_point="extract_functions"
    )


# ============================================================================
# DEMO EXECUTION
# ============================================================================

def run_code_review_demo():
    """Run the Code Review Mini-Agent demo."""
    print("🚀 Code Review Mini-Agent Workflow Demo")
    print("=" * 60)
    
    # Setup workflow engine
    tool_registry = ToolRegistry()
    graph_manager = GraphManager()
    state_manager = StateManager()
    execution_engine = ExecutionEngine(
        graph_manager=graph_manager,
        state_manager=state_manager,
        tool_registry=tool_registry
    )
    
    # Register tools
    tools = [
        ("extract_functions", extract_functions, "Extract functions from Python code"),
        ("check_complexity", check_complexity, "Check cyclomatic complexity"),
        ("detect_issues", detect_issues, "Detect code quality issues"),
        ("suggest_improvements", suggest_improvements, "Suggest code improvements"),
        ("calculate_quality_score", calculate_quality_score, "Calculate quality score"),
        ("generate_final_report", generate_final_report, "Generate final report")
    ]
    
    for name, func, desc in tools:
        try:
            tool_registry.register_tool(name, func, desc)
        except Exception as e:
            if "already registered" in str(e):
                print(f"  ⚠️ Tool '{name}' already registered, skipping...")
    
    # Sample code to review (intentionally has issues)
    sample_code = '''
def complex_function(data, options, config, user_prefs, system_settings, debug_mode):
    """This function does too many things and is overly complex."""
    results = []
    
    if data is None:
        return None
    
    if not isinstance(data, list):
        data = [data]
    
    for item in data:
        if item.get('active', False):
            if options.get('filter_enabled', True):
                if item.get('score', 0) > config.get('threshold', 50):
                    if user_prefs.get('include_metadata', False):
                        if system_settings.get('enhanced_mode', False):
                            try:
                                processed = process_item_enhanced(item, options, config)
                                if processed:
                                    if debug_mode:
                                        print(f"Processed: {processed}")
                                    results.append(processed)
                            except Exception as e:
                                if config.get('ignore_errors', False):
                                    continue
                                else:
                                    raise e
                        else:
                            processed = process_item_basic(item, options)
                            if processed:
                                results.append(processed)
                    else:
                        simple_result = item.get('value', 0) * config.get('multiplier', 1)
                        results.append(simple_result)
    
    return results

def undocumented_function(x, y, z):
    return x + y * z

def simple_function(a, b):
    """A simple, well-written function."""
    return a + b
'''
    
    # Create and execute workflow
    workflow = create_code_review_workflow()
    graph_id = graph_manager.create_graph(workflow)
    
    print(f"📊 Created workflow: {workflow.name}")
    print(f"🔍 Analyzing {len(sample_code)} characters of Python code...")
    
    # Execute workflow
    initial_state = {"code": sample_code}
    run_id = execution_engine.execute_workflow(graph_id, initial_state)
    
    print(f"⏳ Execution started with run ID: {run_id}")
    
    # Wait for completion
    max_attempts = 30
    for attempt in range(max_attempts):
        status = execution_engine.get_execution_status(run_id)
        
        if status.status.value == 'completed':
            print("\n✅ Code Review Workflow completed successfully!")
            
            # Show execution path
            execution_path = status.current_state.execution_path
            print(f"\n🛤️ Execution Path: {' → '.join(execution_path)}")
            
            # Show final results
            final_state = status.current_state.data
            if 'final_report' in final_state:
                report = final_state['final_report']
                print(f"\n📊 WORKFLOW RESULTS:")
                print(f"   Quality Score: {report['summary']['quality_score']}/100")
                print(f"   Issues Found: {report['summary']['issues_found']}")
                print(f"   Functions: {report['summary']['functions_analyzed']}")
                print(f"   Status: {'✅ APPROVED' if report['summary']['meets_threshold'] else '❌ NEEDS IMPROVEMENT'}")
            
            break
            
        elif status.status.value == 'failed':
            print(f"\n❌ Workflow failed: {status.error_message}")
            break
        else:
            print(f"⏳ Status: {status.status.value} (attempt {attempt + 1}/{max_attempts})")
            time.sleep(1)
    else:
        print("⚠️ Workflow did not complete within timeout")


if __name__ == "__main__":
    run_code_review_demo()